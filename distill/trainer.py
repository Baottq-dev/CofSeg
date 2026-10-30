"""Trainer `distill_yolo`: YoloTrainer của benchmark/yolo cộng ba khối `student:`, `outside_box:` và `kd:`.

Kế thừa nguyên YoloTrainer (benchmark/yolo/cofseg/training/yolo.py): cùng mặc định ảnh nadir, cùng cách
chọn best.pt theo mask AP, cùng summary.json, cùng --aug. Phần thêm:

- `student:` dựng trò thay cho model của config (student.py). Khoá `model:` của config vẫn là
  weights/yolo26s-seg.pt: đó là nguồn của phép chọn trọng số, không phải model được train.
- `outside_box:` thay loss mặt nạ bằng loss tính trong hộp nhãn giãn (outside_box.py).
- `kd:` chắt lọc từ ba thầy: dataset train mang đích mặt nạ của thầy (kd_data.py), loss mặt nạ học đích trộn
  và loss PKD trên đặc trưng của YOLO26s thầy (kd_loss.py). Lớp 1×1 khớp kênh gắn vào model lúc train và được
  gỡ khỏi các file trong <run>/weights sau khi train xong. Đi được với `--aug ag1`: dataset train khi đó là
  lớp ghép của kd_data.py và AugDataset của benchmark.

YoloTrainer.fit() gọi `YOLO(...).train(**args)` và chỉ truyền lớp trainer riêng của ultralytics khi có
`mask_head:` hay `--aug`. Để không phải chép fit(), lượt train của distill thay tạm `Model.train` bằng một
lớp bọc truyền lớp trainer của ta vào (ghép với lớp của --aug nếu có), rồi trả lại ngay sau đó.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import distill  # noqa: F401 - đặt sys.path tới benchmark/yolo trước khi import cofseg

from cofseg.registry import register
from cofseg.training.yolo import YoloTrainer

from . import blocks

#: Phần đầu của trò chép Proto26.forward và Segment26.forward của bản này; loss ngoài hộp cài đè
#: single_mask_loss theo chữ ký của bản này. Phiên bản khác thì phải đối chiếu lại trước khi chạy.
ULTRALYTICS_VERSION = "8.4.143"


def make_seg_trainer(student: dict | None, outside_box: dict | None, run_dir, base=None, kd: dict | None = None,
                     fold_root=None):
    """Lớp SegmentationTrainer dựng trò, gắn loss ngoài hộp và/hoặc phần chắt lọc. `base` là lớp trainer mà
    YoloTrainer định truyền (lớp của --aug), để ghép chứ không đè. `fold_root` là thư mục fold (cần cho kd)."""
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    base = base or SegmentationTrainer
    run_dir = Path(run_dir)

    def crit(tr):
        return getattr(unwrap_model(tr.model), "criterion", None)

    class DistillSegTrainer(base):
        STUDENT = student
        OUTSIDE_BOX = outside_box
        KD = kd
        FOLD_ROOT = fold_root

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._kd_teacher = None
            if self.OUTSIDE_BOX:
                from . import kd_loss
                from . import outside_box as ob

                log = ob.EpochLog(run_dir / "loss_outside_box.csv")

                def on_train_start(tr):
                    # Model đã lên GPU và đã có args: hàm loss dựng ở đây mới đúng thiết bị và siêu tham số.
                    m = unwrap_model(tr.model)
                    new = ob.build_criterion(m, self.OUTSIDE_BOX)
                    kd_loss.carry_schedule(getattr(m, "criterion", None), new)
                    m.criterion = new

                self.add_callback("on_train_start", on_train_start)
                self.add_callback("on_train_epoch_start", lambda tr: log.start(crit(tr), tr.epoch))
                self.add_callback("on_train_epoch_end", lambda tr: log.end(crit(tr), tr.epoch))
            if self.KD:
                from . import kd_loss

                log = kd_loss.EpochLog(run_dir / "loss_kd.csv")

                def on_train_start_kd(tr):
                    m = unwrap_model(tr.model)
                    teacher = tr._kd_teacher.to(tr.device) if tr._kd_teacher is not None else None
                    new = kd_loss.build_criterion(m, self.KD, teacher)
                    kd_loss.carry_schedule(getattr(m, "criterion", None), new.base)
                    m.criterion = new

                self.add_callback("on_train_start", on_train_start_kd)
                self.add_callback("on_train_epoch_start", lambda tr: log.start(crit(tr)))
                self.add_callback("on_train_epoch_end", lambda tr: log.end(crit(tr), tr.epoch))

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode != "train" or not (self.KD and self.KD["targets"]):
                return super().build_dataset(img_path, mode, batch)
            import ultralytics.data.build as build

            from .kd_data import make_dataset

            aug = getattr(self, "AUG", None)  # khối --aug, có khi `base` là lớp của aug_train.make_trainer
            # build_yolo_dataset đọc tên YOLODataset từ module của nó lúc gọi (như aug_train.py của benchmark).
            old = build.YOLODataset
            build.YOLODataset = make_dataset(self.KD["targets"], self.FOLD_ROOT, aug, run_dir / "aug" if aug else None)
            try:
                # Lớp của --aug cũng đặt YOLODataset trong build_dataset của nó, sẽ đè lớp vừa đặt. Lớp ở trên đã
                # gồm phần của --aug, nên khi đó gọi thẳng build_dataset của SegmentationTrainer.
                if aug:
                    ds = SegmentationTrainer.build_dataset(self, img_path, mode, batch)
                else:
                    ds = super().build_dataset(img_path, mode, batch)
            finally:
                build.YOLODataset = old
            print(f"Chắt lọc mặt nạ: {ds.kd.describe()}", flush=True)
            return ds

        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            if not self.STUDENT:
                model = super().get_model(cfg, weights, verbose)
            else:
                model = self._student(cfg, weights, verbose)
            if self.KD and self.KD["teacher_runs"]:
                self._attach_adapter(model, weights)
            return model

        def _student(self, cfg, weights, verbose):
            from ultralytics.utils import RANK

            from .student import build_student, source_name

            # `weights` là checkpoint COCO mà Model.train() đã nạp từ khoá `model:` của config: YOLO26n-seg
            # cho trò 0,25, YOLO26s-seg cho trò 0,375.
            teacher = weights if self.STUDENT["init"] == "select" else None
            if teacher is not None and not hasattr(teacher, "state_dict"):
                from ultralytics.nn.tasks import load_checkpoint

                teacher, _ = load_checkpoint(teacher)
            model, counts = build_student(self.STUDENT, cfg, nc=self.data["nc"], ch=self.data["channels"],
                                          teacher=teacher, verbose=verbose and RANK == -1)
            model = self.set_model_names_for_load(model)
            if counts is not None:
                print(f"Trò: lấy trọng số từ {source_name(teacher)}: "
                      + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
            return model

        def _attach_adapter(self, model, weights):
            """Nạp YOLO26s thầy của fold và gắn lớp 1×1 khớp kênh vào model (trước khi dựng optimizer)."""
            from . import kd_loss
            from .teachers import TEACHERS, find_weights, fold_info

            fold = fold_info(self.FOLD_ROOT)["fold"]
            path = find_weights(self.KD["teacher_runs"], TEACHERS["yolo26s"], fold)
            self._kd_teacher = kd_loss.load_teacher(path)
            ch_t = kd_loss.feature_channels(self._kd_teacher)
            ch_s = kd_loss.feature_channels(model)
            model.kd_adapter = kd_loss.KDAdapter(ch_s, ch_t)
            old = getattr(weights, "kd_adapter", None)  # resume từ last.pt: giữ lớp 1×1 đã học
            if old is not None:
                model.kd_adapter.load_state_dict(old.state_dict())
            print(f"Chắt lọc đặc trưng: thầy {path.as_posix()}; kênh P4, P5, feat_fuse của trò {ch_s} -> "
                  f"thầy {ch_t}", flush=True)

    DistillSegTrainer.__name__ = f"DistillSegTrainer_{base.__name__}"
    return DistillSegTrainer


def strip_adapter(path) -> bool:
    """Gỡ kd_adapter (chỉ dùng lúc train) khỏi checkpoint, để best.pt chỉ còn đúng kiến trúc của trò."""
    import torch

    p = Path(path)
    if not p.exists():
        return False
    ck = torch.load(p, map_location="cpu", weights_only=False)
    changed = False
    for key in ("model", "ema"):
        m = ck.get(key) if isinstance(ck, dict) else None
        if m is not None and hasattr(m, "kd_adapter"):
            del m.kd_adapter
            changed = True
    if changed:
        torch.save(ck, p)
    return changed


@contextmanager
def _inject_trainer(make):
    """Thay tạm `Model.train` để lớp trainer của ta đi vào lượt train do YoloTrainer.fit() khởi động."""
    from ultralytics.engine.model import Model

    goc = Model.train

    def train(self, trainer=None, **kw):
        return goc(self, trainer=make(trainer), **kw)

    Model.train = train
    try:
        yield
    finally:
        Model.train = goc


@register("trainer", "distill_yolo")
class DistillYoloTrainer(YoloTrainer):
    """YoloTrainer + khối `student:`, `outside_box:` và `kd:`."""

    @classmethod
    def apply_loss(cls, cfg: dict, spec: str | None) -> str:
        """Kiểm hai khối của distill cùng lúc với --loss và `mask_head:` (train.py gọi bước này cho MỌI lượt,
        trước khi tạo thư mục run). Trả về hậu tố tên run."""
        tag = super().apply_loss(cfg, spec)
        st = blocks.check_student(cfg.get("student"))
        ob = blocks.check_outside_box(cfg.get("outside_box"))
        kd = blocks.check_kd(cfg.get("kd"))
        if (st or ob or kd) and cfg.get("mask_head") is not None:
            raise SystemExit("`student`/`outside_box`/`kd` không đi chung với `mask_head` của benchmark: "
                             "cả hai đều thay đầu hoặc loss mặt nạ.")
        if spec and (ob or kd):
            raise SystemExit("--loss không đi chung với `outside_box`/`kd`: cả hai cùng thay loss mặt nạ.")
        if spec and st:
            raise SystemExit("--loss chưa được kiểm trên trò (đầu hai cấp); chạy biến thể loss trên YOLO26s.")
        if ob and kd:
            raise SystemExit("`outside_box` và `kd` chưa ghép được: cả hai cùng thay hàm loss mặt nạ.")
        for key, block in (("student", st), ("outside_box", ob), ("kd", kd)):
            if block is None:
                cfg.pop(key, None)
            else:
                cfg[key] = block
        return "-".join(p for p in (tag, blocks.student_tag(st), blocks.outside_box_tag(ob), blocks.kd_tag(kd)) if p)

    @classmethod
    def apply_aug(cls, cfg: dict, spec: str | None) -> str:
        """--aug như benchmark; train.py gọi sau apply_loss, nên khối `kd` đã kiểm xong ở đây."""
        tag = super().apply_aug(cfg, spec)
        blocks.check_kd_aug(cfg.get("kd"), cfg.get("aug"))
        return tag

    @classmethod
    def run_name(cls, cfg: dict) -> str:
        """Tên gốc của lượt train (tên config + hậu tố của distill), chưa kèm hậu tố --loss/--aug. Lượt chấm
        của distill/evaluate.py đối chiếu tên này với tên config chấm."""
        tags = [blocks.student_tag(cfg.get("student")), blocks.outside_box_tag(cfg.get("outside_box")),
                blocks.kd_tag(cfg.get("kd"))]
        return "-".join(p for p in (cfg.get("name") or "", *tags) if p)

    def fit(self) -> dict:
        import ultralytics

        st, ob, kd = self.cfg.get("student"), self.cfg.get("outside_box"), self.cfg.get("kd")
        if (st or ob or kd) and ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise SystemExit(f"distill viết cho ultralytics {ULTRALYTICS_VERSION}, môi trường đang có "
                             f"{ultralytics.__version__}: phải đối chiếu lại đoạn chép trước khi chạy.")
        aug = self.cfg.get("aug")
        blocks.check_kd_aug(kd, aug)
        print(blocks.describe_student(st), flush=True)
        print(blocks.describe_outside_box(ob), flush=True)
        print(blocks.describe_kd(kd), flush=True)
        if st or ob or kd:
            fold_root = self.data_yaml.parent
            with _inject_trainer(lambda base: make_seg_trainer(st, ob, self.run_dir, base, kd, fold_root)):
                out = super().fit()
        else:
            out = super().fit()
        if kd and kd["teacher_runs"]:
            for p in sorted((self.run_dir / "weights").glob("*.pt")):
                if strip_adapter(p):
                    print(f"distill: gỡ kd_adapter khỏi {p}", flush=True)
        # evaluate.py của distill đọc khối này: tên gốc và hậu tố --aug để đối chiếu config chấm, hệ số giãn
        # lúc chấm.
        out["distill"] = {"name": self.run_name(self.cfg), "aug": (aug or {}).get("tag") or None, "student": st,
                          "outside_box": ob, "kd": kd}
        return out


def train_distill(weights) -> dict | None:
    """Khối `distill` trong summary.json của lượt train đã sinh ra `weights`; None nếu không có."""
    p = Path(str(weights).replace("\\", "/"))
    for cand in (p.parent.parent / "summary.json", p.parent.parent.parent / "summary.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8")).get("distill")
    return None
