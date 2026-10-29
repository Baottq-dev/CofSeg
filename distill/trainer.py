"""Trainer `distill_yolo`: YoloTrainer của benchmark/yolo cộng hai khối `student:` và `outside_box:`.

Kế thừa nguyên YoloTrainer (benchmark/yolo/cofseg/training/yolo.py): cùng mặc định ảnh nadir, cùng cách
chọn best.pt theo mask AP, cùng summary.json, cùng --aug. Phần thêm:

- `student:` dựng trò thay cho model của config (student.py). Khoá `model:` của config vẫn là
  weights/yolo26s-seg.pt: đó là nguồn của phép chọn trọng số, không phải model được train.
- `outside_box:` thay loss mặt nạ bằng loss tính trong hộp nhãn giãn (outside_box.py).

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


def make_seg_trainer(student: dict | None, outside_box: dict | None, run_dir, base=None):
    """Lớp SegmentationTrainer dựng trò và/hoặc gắn loss ngoài hộp. `base` là lớp trainer mà YoloTrainer
    định truyền (lớp của --aug), để ghép chứ không đè."""
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    base = base or SegmentationTrainer
    run_dir = Path(run_dir)

    class DistillSegTrainer(base):
        STUDENT = student
        OUTSIDE_BOX = outside_box

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.OUTSIDE_BOX:
                from . import outside_box as ob

                log = ob.EpochLog(run_dir / "loss_outside_box.csv")

                def crit(tr):
                    return getattr(unwrap_model(tr.model), "criterion", None)

                def on_train_start(tr):
                    # Model đã lên GPU và đã có args: hàm loss dựng ở đây mới đúng thiết bị và siêu tham số.
                    m = unwrap_model(tr.model)
                    m.criterion = ob.build_criterion(m, self.OUTSIDE_BOX)

                self.add_callback("on_train_start", on_train_start)
                self.add_callback("on_train_epoch_start", lambda tr: log.start(crit(tr), tr.epoch))
                self.add_callback("on_train_epoch_end", lambda tr: log.end(crit(tr), tr.epoch))

        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            if not self.STUDENT:
                return super().get_model(cfg, weights, verbose)
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

    DistillSegTrainer.__name__ = f"DistillSegTrainer_{base.__name__}"
    return DistillSegTrainer


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
    """YoloTrainer + khối `student:` và `outside_box:`."""

    @classmethod
    def apply_loss(cls, cfg: dict, spec: str | None) -> str:
        """Kiểm hai khối của distill cùng lúc với --loss và `mask_head:` (train.py gọi bước này cho MỌI lượt,
        trước khi tạo thư mục run). Trả về hậu tố tên run."""
        tag = super().apply_loss(cfg, spec)
        st = blocks.check_student(cfg.get("student"))
        ob = blocks.check_outside_box(cfg.get("outside_box"))
        if (st or ob) and cfg.get("mask_head") is not None:
            raise SystemExit("`student`/`outside_box` không đi chung với `mask_head` của benchmark: "
                             "cả hai đều thay đầu hoặc loss mặt nạ.")
        if spec and ob:
            raise SystemExit("--loss không đi chung với `outside_box`: cả hai cùng thay loss mặt nạ.")
        if spec and st:
            raise SystemExit("--loss chưa được kiểm trên trò (đầu hai cấp); chạy biến thể loss trên YOLO26s.")
        for key, block in (("student", st), ("outside_box", ob)):
            if block is None:
                cfg.pop(key, None)
            else:
                cfg[key] = block
        return "-".join(p for p in (tag, blocks.student_tag(st), blocks.outside_box_tag(ob)) if p)

    @classmethod
    def run_name(cls, cfg: dict) -> str:
        """Tên gốc của lượt train (tên config + hậu tố của distill), chưa kèm hậu tố --loss/--aug. Lượt chấm
        của distill/evaluate.py đối chiếu tên này với tên config chấm."""
        tags = [blocks.student_tag(cfg.get("student")), blocks.outside_box_tag(cfg.get("outside_box"))]
        return "-".join(p for p in (cfg.get("name") or "", *tags) if p)

    def fit(self) -> dict:
        import ultralytics

        st, ob = self.cfg.get("student"), self.cfg.get("outside_box")
        if (st or ob) and ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise SystemExit(f"distill viết cho ultralytics {ULTRALYTICS_VERSION}, môi trường đang có "
                             f"{ultralytics.__version__}: phải đối chiếu lại đoạn chép trước khi chạy.")
        print(blocks.describe_student(st), flush=True)
        print(blocks.describe_outside_box(ob), flush=True)
        if st or ob:
            with _inject_trainer(lambda base: make_seg_trainer(st, ob, self.run_dir, base)):
                out = super().fit()
        else:
            out = super().fit()
        # evaluate.py của distill đọc khối này: tên gốc để đối chiếu config chấm, hệ số giãn lúc chấm.
        out["distill"] = {"name": self.run_name(self.cfg), "student": st, "outside_box": ob}
        return out


def train_distill(weights) -> dict | None:
    """Khối `distill` trong summary.json của lượt train đã sinh ra `weights`; None nếu không có."""
    p = Path(str(weights).replace("\\", "/"))
    for cand in (p.parent.parent / "summary.json", p.parent.parent.parent / "summary.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8")).get("distill")
    return None
