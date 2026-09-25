"""Trainer detectron2: Mask R-CNN R50-FPN, Cascade Mask R-CNN R50-FPN,
Mask2Former R50 (và PointRend nếu đưa config) — cùng benchmark/mask2former/train.py, cùng
khối `train:` với hai trainer kia.

    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/maskrcnn_r50_d2.yaml --set data.root=data/export/block/f4
    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --probe

Khác biệt có chủ đích với trainer torchvision/YOLO:
- Vòng lặp là DefaultTrainer của detectron2 (hoặc Trainer trong train_net.py
  của Mask2Former); ở đây chỉ dịch epoch/batch/imgsz/lr sang khoá config,
  thêm hook giữ checkpoint tốt nhất theo mask AP trên val, và sau khi xong
  chấm test rồi ghi predictions.json (COCO results) để chấm lại ở nhà.
- Mask2Former giữ tăng cường LSJ của recipe gốc; hai model R-CNN dùng lật
  ngang/dọc + xoay 90° như trainer torchvision (ảnh nadir không có chiều trên).

CHƯA CHẠY THẬT: detectron2 không dựng được trên máy phát triển (Windows).
`build_opts` có test; prepare/probe/fit phải khói trên Linux trước khi tin.
"""

from __future__ import annotations

import contextlib
import csv
import pathlib
import io
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from ..datasets.coco import CocoDataset
from ..datasets.instances import imread
from ..models.detectron2 import ARCHS, base_cfg, class_opts, require_detectron2, size_opts
from ..registry import register
from .. import progress
from . import memory
from .base import Trainer

#: Cùng tên với trainer torchvision ở đâu có thể, để --imgsz/--batch/--epochs
#: nghĩa như nhau trên mọi model. lr None = theo recipe gốc của từng arch,
#: tỉ lệ tuyến tính theo batch (0.02 @16 cho R-CNN, 1e-4 @16 cho Mask2Former).
D2_DEFAULTS: dict = {
    "imgsz": 1024,
    "batch": 4,
    "epochs": 50,
    "lr": None,
    "weight_decay": None,
    "momentum": 0.9,
    "lr_steps": [0.7, 0.9],
    "lr_gamma": 0.1,
    "warmup_iters": 200,
    "amp": True,
    "fliplr": 0.5,
    "flipud": 0.5,
    "rot90": True,
    "val_every": 1,
    "val_conf": 0.05,
    "max_det": 100,
    "num_queries": 100,   # chỉ Mask2Former; ảnh dày nhất có 48 tán
    "workers": 0,
    "seed": 0,
    "log_every": 20,
    # verbose=True trả lại đúng cách detectron2 in mặc định: dump config, dump
    # kiến trúc model, một dòng mỗi 20 iteration. Bật khi cần soi lỗi của
    # chính khung, còn lúc train bình thường thì 71% số dòng là hai dump đó.
    "verbose": False,
}
BASE_LR = {"maskrcnn": 0.02, "cascade": 0.02, "pointrend": 0.02, "mask2former": 1e-4}
#: weight decay của recipe gốc: SGD của R-CNN dùng 1e-4, AdamW của Mask2Former 0.05.
BASE_WD = {"maskrcnn": 1e-4, "cascade": 1e-4, "pointrend": 1e-4, "mask2former": 0.05}


def build_opts(arch: str, n_train: int, args: dict, aspect: float = 9 / 16,
               names: dict | None = None, out_dir: str = "") -> list:
    """Khối train: -> danh sách [khoá, giá trị, ...] cho cfg.merge_from_list.

    Thuần Python, không cần detectron2: đây là phần kiểm được ở nhà.
    """
    if arch not in ARCHS:
        raise ValueError(f"arch {arch!r} không có; có: {ARCHS}")
    batch, epochs = int(args["batch"]), int(args["epochs"])
    if batch < 1 or epochs < 1 or n_train < 1:
        raise ValueError("batch, epochs và số ảnh train đều phải >= 1")
    per_epoch = math.ceil(n_train / batch)
    max_iter = epochs * per_epoch
    lr = args.get("lr")
    if lr is None:
        lr = BASE_LR[arch] * batch / 16
    wd = args.get("weight_decay")
    if wd is None:
        wd = BASE_WD[arch]
    # lr_steps nhận PHẦN của lịch (0.7 = 70% số vòng) hoặc số vòng tuyệt đối,
    # phân biệt bằng < 1. Ghi theo phần thì đổi epochs không phải tính lại mốc.
    steps = tuple(sorted({int(s * max_iter) if 0 < float(s) < 1 else int(s)
                          for s in (args.get("lr_steps") or ())
                          if 0 < (float(s) * max_iter if float(s) < 1 else float(s)) < max_iter}))
    names = names or {}
    opts = [
        "DATASETS.TRAIN", (names.get("train", "coffee_train"),),
        "DATASETS.TEST", (names.get("val", "coffee_val"),),
        "DATALOADER.NUM_WORKERS", int(args["workers"]),
        # Ảnh nền (đã xem, không có tán) vẫn vào train làm mẫu âm, như YOLO.
        "DATALOADER.FILTER_EMPTY_ANNOTATIONS", False,
        "SOLVER.IMS_PER_BATCH", batch,
        "SOLVER.BASE_LR", float(lr),
        "SOLVER.MAX_ITER", max_iter,
        "SOLVER.STEPS", steps,
        "SOLVER.GAMMA", float(args["lr_gamma"]),
        "SOLVER.MOMENTUM", float(args["momentum"]),
        "SOLVER.WEIGHT_DECAY", float(wd),
        "SOLVER.WARMUP_ITERS", min(int(args["warmup_iters"]), max_iter),
        "SOLVER.CHECKPOINT_PERIOD", per_epoch,
        "SOLVER.AMP.ENABLED", bool(args["amp"]),
        "TEST.EVAL_PERIOD", per_epoch * int(args["val_every"]),
        "TEST.DETECTIONS_PER_IMAGE", int(args["max_det"]),
        "SEED", int(args["seed"]),
        "OUTPUT_DIR", str(out_dir),
    ]
    opts += class_opts(arch, int(args["num_queries"]))
    opts += size_opts(int(args["imgsz"]), aspect)
    if arch == "mask2former":
        # LSJ của recipe gốc cắt ô vuông IMAGE_SIZE sau khi co giãn 0.1-2.0.
        opts += ["INPUT.IMAGE_SIZE", int(args["imgsz"])]
    else:
        opts += ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", float(args["val_conf"])]
    return opts


def iters_per_epoch(n_train: int, batch: int) -> int:
    return math.ceil(n_train / max(1, int(batch)))


@register("trainer", "detectron2")
class Detectron2Trainer(Trainer):
    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        m = cfg.get("model") or {}
        self.arch: str = m.get("arch", "maskrcnn")
        if self.arch not in ARCHS:
            raise ValueError(f"model.arch {self.arch!r} không có; có: {ARCHS}")
        self.repo = m.get("repo")
        self.config_file = m.get("config_file")
        self.weights = m.get("weights")
        self.train_args: dict = {**D2_DEFAULTS, **(cfg.get("train") or {})}
        d = cfg.get("data") or {}
        self.root = Path(d.get("root", "data/export/block/f4"))
        self.splits = {k: d.get(k, k) for k in ("train", "val", "test")}
        self.min_area = float(d.get("min_area", 0.0))
        self.limit = d.get("limit")
        self.out_dir = self.run_dir / "d2"

    # -------------------------------------------------------------------- tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        return dict(D2_DEFAULTS)

    TAG_KEYS = (("imgsz", "i"), ("batch", "b"), ("epochs", "e"))

    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        args = {**D2_DEFAULTS, **(cfg.get("train") or {})}
        return "".join(f"{p}{args[k]}" for k, p in cls.TAG_KEYS)

    # ------------------------------------------------------------------ chuẩn bị
    def _dataset_names(self) -> dict[str, str]:
        fold = self.root.name
        return {sp: f"coffee_{fold}_{self.splits[sp]}" for sp in ("train", "val", "test")}

    def prepare(self) -> dict:
        """Kiểm dữ liệu bằng bộ đọc của repo (không cần detectron2), đếm ảnh
        train để đổi epoch -> iteration."""
        out = {}
        for sp in ("train", "val", "test"):
            ds = CocoDataset(self.root, self.splits[sp], min_area=self.min_area)
            out[sp] = ds.summary()
            if sp == "train":
                self.n_train = len(ds)
                first = ds.image_list()[0]
                shape = imread(first.path).shape
                self.aspect = shape[0] / shape[1]
                out["sample"] = {"file": first.file_name, "shape": list(shape)}
        if self.limit:
            self.n_train = min(self.n_train, int(self.limit))
            out["limit"] = int(self.limit)
        out["arch"] = self.arch
        out["iters_per_epoch"] = iters_per_epoch(self.n_train, self.train_args["batch"])
        self.dataset_counts = {sp: out[sp]["images"] for sp in ("train", "val", "test")}
        self.test_fields = sorted(out["test"].get("fields") or {})
        (self.run_dir / "dataset_check.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    # ------------------------------------------------------------ detectron2
    def _register(self) -> dict[str, str]:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances

        names = self._dataset_names()
        for sp, name in names.items():
            if name in DatasetCatalog.list():
                continue
            register_coco_instances(
                name, {},
                str(self.root / "annotations" / f"instances_{self.splits[sp]}.json"),
                str(self.root / "images" / self.splits[sp]),
            )
            MetadataCatalog.get(name).thing_classes = ["canopy"]
        if self.limit:
            # Khói: cắt danh sách train sau khi đăng ký, giữ nguyên val/test.
            full = DatasetCatalog.get(names["train"])
            DatasetCatalog.remove(names["train"])
            DatasetCatalog.register(names["train"], lambda d=full[: int(self.limit)]: d)
            MetadataCatalog.get(names["train"]).thing_classes = ["canopy"]
        return names

    def _cfg(self):
        require_detectron2()
        if not hasattr(self, "n_train"):
            self.prepare()
        names = self._register()
        cfg, m2f = base_cfg(self.arch, repo=self.repo, config_file=self.config_file,
                            weights=self.weights)
        cfg.merge_from_list(build_opts(self.arch, self.n_train, self.train_args,
                                       aspect=self.aspect, names=names,
                                       out_dir=str(self.out_dir)))
        cfg.freeze()
        return cfg, m2f, names

    def _epoch_reporter(self):
        """Hook vẽ thanh tiến trình trong epoch và in một dòng khi epoch xong.

        detectron2 đếm theo ITERATION chứ không theo epoch, nên chữ "epoch"
        không xuất hiện lần nào trong log gốc suốt lúc train. Hook này dịch
        ngược lại: mỗi `per_epoch` iteration là một epoch, đóng thanh, đọc
        segm/AP mà EvalHook vừa ghi vào storage, in một dòng rồi mở thanh mới.

        Một chi tiết dễ sai: EvalHook CỐ TÌNH bỏ qua iteration cuối trong
        `after_step` và chấm val trong `after_train` thay vào đó. Nên epoch
        cuối — epoch duy nhất mà người ta thật sự quan tâm — phải báo cáo ở
        `after_train`, không thì nó in ra dòng thiếu mất AP.
        """
        from detectron2.engine import HookBase

        a = self.train_args
        per_epoch = iters_per_epoch(self.n_train, a["batch"])
        epochs = int(a["epochs"])
        total = per_epoch * epochs

        def value(storage, key):
            """Số mới nhất của một khoá, None nếu khoá chưa từng được ghi."""
            got = storage.latest().get(key)
            return None if got is None else float(got[0])

        def smoothed(storage, key, window=20):
            try:
                return float(storage.history(key).median(window))
            except (KeyError, ValueError):
                return None

        def note(loss):
            return "" if loss is None else f"loss {loss:.3f}"

        def peak_memory():
            try:
                import torch

                if torch.cuda.is_available():
                    # 0 nghĩa là chưa cấp phát gì (chạy CPU, hoặc gọi quá sớm);
                    # in "0.0G" ra chỉ làm người đọc tưởng model không dùng GPU.
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    return peak or None
            except ImportError:
                pass
            return None

        class EpochReporter(HookBase):
            def __init__(self):
                self.bar = None
                self.epoch = 0
                self.best = None
                self.started = None
                self.trained = None      # giây train của epoch, chưa tính val
                self.pending = False

            def _open(self):
                self.epoch += 1
                self.started = time.time()
                self.trained = None
                self.bar = progress.Bar(per_epoch, f"epoch {self.epoch}/{epochs}")

            def close_train_bar(self):
                """Đóng thanh train và chốt thời gian train của epoch này.

                Gọi từ EVALUATOR chứ không phải từ hook. Lý do: EvalHook chấm
                val ngay trong after_step của chính nó, mà hook báo cáo lại
                nằm cuối danh sách — tới lượt nó thì val đã xong. Chỗ duy
                nhất biết "val vừa bắt đầu" là evaluator, lúc reset().

                Gọi nhiều lần vẫn an toàn: lần sau không làm gì.
                """
                if self.bar is not None:
                    # Kéo nốt cho đầy trước khi đóng. EvalHook chấm val NGAY
                    # TRONG after_step của iteration cuối epoch, tức là trước
                    # khi hook này kịp đếm bước đó — không kéo thì thanh đứng
                    # ở 29/30 dù epoch đã xong.
                    self.bar.advance(max(0, self.bar.total - self.bar.n))
                    self.bar.close()
                    self.bar = None
                if self.trained is None and self.started is not None:
                    self.trained = time.time() - self.started

            def _report(self):
                st = self.trainer.storage
                ap = value(st, "segm/AP")
                is_best = ap is not None and (self.best is None or ap > self.best)
                if is_best:
                    self.best = ap
                whole = None if self.started is None else time.time() - self.started
                train_s = self.trained if self.trained is not None else whole
                val_s = None if (whole is None or self.trained is None) else whole - self.trained
                print(progress.epoch_line(
                    self.epoch, epochs, self.epoch * per_epoch, total,
                    loss=smoothed(st, "total_loss"), lr=value(st, "lr"),
                    mem=peak_memory(),
                    metrics={"mAP50-95": ap, "mAP50": value(st, "segm/AP50")},
                    best=is_best, seconds=train_s, val_seconds=val_s,
                ), flush=True)
                self.pending = False

            # ------------------------------------------------------ vòng đời
            def before_train(self):
                self._open()

            def after_step(self):
                done = self.trainer.iter - self.trainer.start_iter + 1
                # Thanh có thể ĐÃ bị đóng ngay trong chính after_step này:
                # EvalHook đứng trước hook này trong danh sách, nó chấm val, và
                # evaluator gọi close_train_bar() lúc reset(). Nên đây không
                # phải phép kiểm phòng xa — nó xảy ra ở MỌI cuối epoch.
                if self.bar is not None:
                    self.bar.advance(1, note(smoothed(self.trainer.storage, "total_loss")))
                if done % per_epoch:
                    return
                self.close_train_bar()
                if done >= self.trainer.max_iter - self.trainer.start_iter:
                    self.pending = True     # AP của epoch cuối chưa có, xem docstring
                else:
                    self._report()
                    self._open()

            def after_train(self):
                self.close_train_bar()
                if self.pending:
                    self._report()

        return EpochReporter()

    def _trainer_cls(self, m2f):
        """DefaultTrainer (R-CNN) hoặc Trainer của Mask2Former, thêm hook giữ
        checkpoint tốt nhất theo segm/AP trên val và mapper tăng cường nadir."""
        from detectron2.data import DatasetMapper, build_detection_train_loader
        from detectron2.data import transforms as T
        from detectron2.engine import DefaultTrainer, hooks
        from detectron2.evaluation import COCOEvaluator
        from detectron2.utils.events import CommonMetricPrinter

        a = self.train_args
        base = m2f.Trainer if m2f is not None else DefaultTrainer
        quiet = not a.get("verbose")
        reporter = None if not quiet else self._epoch_reporter()

        class QuietCOCOEvaluator(COCOEvaluator):
            """Chấm im lặng, kèm thanh tiến trình riêng cho val/test.

            pycocotools in bảng 12 dòng bằng print() thẳng ra stdout. Đó là lý
            do hush() một mình không đủ: nó chỉnh module logging, còn chỗ này
            không đi qua logging. Hứng lại rồi đẩy vào logger để bảng vẫn
            xuống d2/log.txt mà không lên màn hình.

            Thanh tiến trình đặt ở ĐÂY chứ không ở hook, vì đây là chỗ duy
            nhất biết val bắt đầu và kết thúc lúc nào: `reset()` chạy trước
            vòng suy luận, `process()` một lần mỗi ảnh, `evaluate()` sau cùng.
            Hook báo cáo nằm cuối danh sách hook nên tới lượt nó thì val đã
            chấm xong từ lâu.
            """

            label = "val"
            n_images = 0
            on_start = None

            def reset(self):
                super().reset()
                if self.on_start is not None:
                    self.on_start()      # chốt giờ train, đóng thanh train
                self._bar = progress.Bar(self.n_images, self.label) if quiet else None

            def process(self, inputs, outputs):
                super().process(inputs, outputs)
                if getattr(self, "_bar", None) is not None:
                    self._bar.advance(len(inputs))

            def evaluate(self, *args, **kw):
                if getattr(self, "_bar", None) is not None:
                    self._bar.close()
                    self._bar = None
                if not quiet:
                    return super().evaluate(*args, **kw)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = super().evaluate(*args, **kw)
                text = buf.getvalue().strip()
                if text:
                    logging.getLogger("detectron2").info("bảng COCO:\n%s", text)
                return res

        class CoffeeTrainer(base):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if m2f is not None:
                    return base.build_evaluator(cfg, dataset_name, output_folder)
                ev = QuietCOCOEvaluator(
                    dataset_name, tasks=("segm",),
                    output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / "inference"))
                # Tổng số ảnh cho thanh tiến trình. DatasetCatalog trả danh
                # sách đã nạp sẵn nên phép đếm này không đọc lại đĩa.
                from detectron2.data import DatasetCatalog

                ev.n_images = len(DatasetCatalog.get(dataset_name))
                ev.label = "test" if dataset_name.endswith("_test") else "val"
                ev.on_start = None if reporter is None else reporter.close_train_bar
                return ev

            def build_writers(self):
                """Bỏ CommonMetricPrinter — đó chính là thứ in `iter: 19
                total_loss: ...` ra màn hình, và ta thay nó bằng thanh tiến
                trình. JSONWriter phải GIỮ: results.csv dựng từ metrics.json
                mà nó ghi ra, bỏ đi là mất cả bảng kết quả.
                """
                ws = super().build_writers()
                if not quiet:
                    return ws
                return [w for w in ws if not isinstance(w, CommonMetricPrinter)]

            @classmethod
            def build_train_loader(cls, cfg):
                if m2f is not None:
                    return base.build_train_loader(cfg)
                augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                             cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
                if a["fliplr"]:
                    augs.append(T.RandomFlip(prob=float(a["fliplr"]), horizontal=True, vertical=False))
                if a["flipud"]:
                    augs.append(T.RandomFlip(prob=float(a["flipud"]), horizontal=False, vertical=True))
                if a["rot90"]:
                    augs.append(T.RandomRotation([0, 90, 180, 270], sample_style="choice", expand=True))
                return build_detection_train_loader(
                    cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))

            def build_hooks(self):
                ret = super().build_hooks()
                # Sau EvalHook (đọc segm/AP nó vừa ghi), trước PeriodicWriter.
                ret.insert(-1, hooks.BestCheckpointer(
                    self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP",
                    mode="max", file_prefix="model_best"))
                # `log_every` được khai trong D2_DEFAULTS từ đầu nhưng KHÔNG
                # nối vào đâu cả: nhịp 20 trong log là mặc định cứng của
                # detectron2, trùng số nên nhìn như tham số đang có tác dụng.
                for h in ret:
                    if isinstance(h, hooks.PeriodicWriter):
                        h._period = max(1, int(a["log_every"]))
                # Cuối danh sách, sau EvalHook: hook nào đọc segm/AP cũng phải
                # chạy SAU cái hook vừa ghi ra nó.
                if reporter is not None:
                    ret.append(reporter)
                return ret

        return CoffeeTrainer

    # ---------------------------------------------------------------------- dò
    def probe(self) -> dict:
        """Hai iteration thật (forward + backward, AMP) rồi đo đỉnh bộ nhớ, như
        trainer torchvision."""
        import torch

        if not torch.cuda.is_available():
            return {"supported": False, "reason": "không có CUDA"}
        cfg, m2f, _ = self._cfg()
        cls = self._trainer_cls(m2f)
        model = cls.build_model(cfg)
        model.train()
        loader = cls.build_train_loader(cfg)
        opt = cls.build_optimizer(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.SOLVER.AMP.ENABLED)
        torch.cuda.reset_peak_memory_stats()
        t0, n_obj = time.time(), 0
        try:
            it = iter(loader)
            for _ in range(2):
                data = next(it)
                n_obj += sum(len(d["instances"]) for d in data)
                with torch.autocast("cuda", enabled=cfg.SOLVER.AMP.ENABLED):
                    losses = sum(model(data).values())
                opt.zero_grad(set_to_none=True)
                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
        finally:
            peak = torch.cuda.max_memory_allocated() / 2**30
            del model, opt, loader
            torch.cuda.empty_cache()
        wall = memory.wall_gb()
        batch = int(self.train_args["batch"])
        return {
            "supported": True, "arch": self.arch, "imgsz": int(self.train_args["imgsz"]),
            "batch": batch, "peak_gb": round(peak, 2), "under_wall": peak < wall,
            "wall_gb": wall,
            "suggested_batch": max(1, int(batch * wall / max(peak, 1e-6))),
            "objects_in_probe": n_obj, "seconds": round(time.time() - t0, 1),
            "note": "đỉnh đo trên 2 iteration; batch dày vùng hơn sẽ cao hơn một chút",
        }

    @staticmethod
    def _default_setup(default_setup, cfg, quiet: bool):
        """Gọi default_setup mà không để nó đổ nguyên config ra màn hình.

        Đo trên một lượt Mask2Former: 452 / 655 dòng (69%) là bảng môi trường
        và bản dump config, và CẢ HAI in bên trong `default_setup`. Bịt log
        sau khi nó trả về là muộn — đó chính là lỗi đã mắc ở đây trước đó.

        `setup_logger` của detectron2 tạo `StreamHandler(stream=sys.stdout)`
        và GIỮ tham chiếu tới luồng ngay lúc tạo, mà nó chạy bên trong
        `default_setup`, tức là lúc stdout đang bị hứng. Không trỏ handler về
        stdout thật thì mọi dòng về sau rơi vào bộ đệm đã bỏ đi — im lặng
        hoàn toàn, kể cả cảnh báo "Skip loading parameter ...", và người chạy
        không có dấu hiệu gì.

        Không mất gì: default_setup vẫn gắn handler ghi ra d2/log.txt, và bản
        đó nhận đủ những dòng bị hứng ở đây.
        """
        args = SimpleNamespace(config_file="", eval_only=False, opts=[], num_gpus=1)
        if not quiet:
            default_setup(cfg, args)
            return
        real = sys.stdout
        with contextlib.redirect_stdout(io.StringIO()):
            default_setup(cfg, args)
        for name in ("detectron2", "fvcore"):
            for h in logging.getLogger(name).handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                    h.setStream(real)
        progress.hush("detectron2", "fvcore")
        # train_loop bắt ngoại lệ, logger.exception rồi raise lại, nên cùng
        # một traceback ra màn hình hai lần: 90 dòng của detectron2 rồi 96
        # dòng của Python. Bản sau chứa đủ bản trước cộng khung của train.py.
        progress.drop_from_console("detectron2", name="detectron2.engine.train_loop",
                                   startswith="Exception during training")

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.engine import default_setup

        cfg, m2f, names = self._cfg()
        quiet = not self.train_args.get("verbose")
        self._default_setup(default_setup, cfg, quiet)
        if quiet:
            self.warned = progress.warnings_to_file(self.run_dir / "warnings.log")
        cls = self._trainer_cls(m2f)
        per_epoch = iters_per_epoch(self.n_train, self.train_args["batch"])
        print(f"{self.arch}: {self.train_args['epochs']} epoch x {per_epoch} iteration, "
              f"batch {cfg.SOLVER.IMS_PER_BATCH}, imgsz {cfg.INPUT.MAX_SIZE_TRAIN}, "
              f"lr {cfg.SOLVER.BASE_LR:g}, max_iter {cfg.SOLVER.MAX_ITER}")
        t0 = time.time()
        trainer = cls(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        train_seconds = round(time.time() - t0, 1)

        # Trọng số về cùng bố cục với các trainer khác; d2_config.yaml cạnh
        # best.pth để Detectron2Model dựng lại đúng model không cần nhắc tham số.
        weights_dir = self.run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        best_src = self.out_dir / "model_best.pth"
        final_src = self.out_dir / "model_final.pth"
        best = weights_dir / "best.pth"
        shutil.copy2(best_src if best_src.exists() else final_src, best)
        shutil.move(str(final_src), weights_dir / "last.pth")
        (weights_dir / "d2_config.yaml").write_text(cfg.dump(), encoding="utf-8")
        rows = self._results_csv(per_epoch)

        # Chấm TEST bằng checkpoint tốt nhất -> predictions.json (COCO results,
        # image_id là id gốc của bản xuất) để chấm lại ở nhà qua coco_predictions.
        cfg2 = cfg.clone()
        cfg2.defrost()
        cfg2.DATASETS.TEST = (names["test"],)
        cfg2.OUTPUT_DIR = str(self.run_dir / "test")
        cfg2.MODEL.WEIGHTS = str(best)
        cfg2.freeze()
        n_test = (getattr(self, "dataset_counts", None) or {}).get("test")
        where = f" trên {n_test} ảnh" if n_test else ""
        print(f"\nChấm test bằng best.pth{where} ...", flush=True)
        model = cls.build_model(cfg2)
        DetectionCheckpointer(model, save_dir=cfg2.OUTPUT_DIR).resume_or_load(str(best), resume=False)
        test_res = cls.test(cfg2, model)
        preds = Path(cfg2.OUTPUT_DIR) / "inference" / "coco_instances_results.json"
        pred_out = self.run_dir / "predictions.json"
        if preds.exists():
            shutil.copy2(preds, pred_out)
        (self.run_dir / "test_metrics.json").write_text(
            json.dumps(test_res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

        best_rows = [r for r in rows if r.get("segm/AP") not in (None, "")]
        best_row = max(best_rows, key=lambda r: float(r["segm/AP"])) if best_rows else None
        segm = (test_res.get("segm") or {}) if isinstance(test_res, dict) else {}
        self._print_summary(best_row, segm, train_seconds, best)
        return {
            "weights": {"best": str(best), "last": str(weights_dir / "last.pth")},
            "best_epoch": int(best_row["epoch"]) if best_row else -1,
            "best_val_AP": round(float(best_row["segm/AP"]) / 100, 4) if best_row else -1.0,
            "epochs_run": int(self.train_args["epochs"]),
            "train_seconds": train_seconds,
            "results_csv": str(self.run_dir / "results.csv"),
            "predictions": str(pred_out) if pred_out.exists() else None,
            "test": {k: v for k, v in (test_res.get("segm") or {}).items()
                     if k in ("AP", "AP50", "AP75")} if isinstance(test_res, dict) else {},
            "args": dict(self.train_args),
        }

    def _print_summary(self, best_row, test_segm: dict, seconds, weights) -> None:
        """Khối cuối lượt chạy: số nào đáng nhớ thì nằm ở đây, không phải rải
        rác giữa mấy trăm dòng log của khung.

        Dùng tên cột của YOLO (`mAP50-95`, `mAP50`) cho cả bốn model; xem
        cofseg/progress.py để biết vì sao.
        """
        def pair(ap, ap50):
            return f"mAP50-95 {progress.fmt_num(ap)}   mAP50 {progress.fmt_num(ap50)}"

        epochs = int(self.train_args["epochs"])
        rows = [("epoch tốt nhất",
                 f"{best_row['epoch']}/{epochs}" if best_row else "—")]
        if best_row:
            rows.append(("val", pair(best_row.get("segm/AP"), best_row.get("segm/AP50"))))
        fields = getattr(self, "test_fields", []) or []
        label = f"test ({', '.join(fields)})" if fields else "test"
        rows.append((label, pair(test_segm.get("AP"), test_segm.get("AP50"))))
        rows.append(("thời gian", f"train {progress.fmt_time(seconds)}"))
        seen = getattr(getattr(self, "warned", None), "seen", ())
        if seen:
            rows.append(("cảnh báo", f"{len(seen)} loại  ->  warnings.log"))
        # Đường dẫn tương đối với run dir: tên run dir đã nằm ở tiêu đề, lặp
        # lại cả đường dẫn tuyệt đối chỉ kéo khung rộng ra mà không thêm tin.
        try:
            shown = pathlib.Path(weights).relative_to(self.run_dir)
        except ValueError:
            shown = weights
        rows.append(("trọng số", str(shown)))
        print(progress.summary(f"{self.arch} · {self.run_dir.name}", rows), flush=True)

    def _results_csv(self, per_epoch: int) -> list[dict]:
        """metrics.json (một JSON mỗi dòng) của detectron2 -> results.csv có cột
        epoch, cùng tinh thần với các trainer khác."""
        src = self.out_dir / "metrics.json"
        epochs = int(self.train_args["epochs"])
        rows: list[dict] = []
        if not src.exists():
            return rows
        keep = ("iteration", "total_loss", "lr", "time", "data_time",
                "segm/AP", "segm/AP50", "segm/AP75", "bbox/AP")
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "iteration" not in d:
                continue
            # Chặn trên ở số epoch thật. detectron2 tăng self.iter lên
            # max_iter TRƯỚC khi gọi after_train (để after_train phân biệt
            # được "train xong tử tế" với "chết giữa chừng"), mà EvalHook lại
            # cố tình để dành val của epoch CUỐI cho after_train. Nên AP quan
            # trọng nhất của cả lượt chạy được ghi ở iteration = max_iter, và
            # 90 // 30 + 1 ra epoch 4 của một lượt 3 epoch.
            row = {"epoch": min(d["iteration"] // per_epoch + 1, epochs)}
            row.update({k: d[k] for k in keep if k in d})
            rows.append(row)
        out = self.run_dir / "results.csv"
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("epoch", "iteration"), k))
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return rows
