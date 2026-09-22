"""Trainer detectron2: Mask R-CNN R50-FPN, Cascade Mask R-CNN R50-FPN,
Mask2Former R50 (và PointRend nếu đưa config) — cùng scripts/train.py, cùng
khối `train:` với hai trainer kia.

    python scripts/train.py --config members/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --set data.root=data/export/f4
    python scripts/train.py --config members/mask2former/configs/train/mask2former_r50_d2.yaml --probe

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

import csv
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from ..datasets.coco import CocoDataset
from ..datasets.instances import imread
from ..models.detectron2 import ARCHS, base_cfg, class_opts, require_detectron2, size_opts
from ..registry import register
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
}
BASE_LR = {"maskrcnn": 0.02, "cascade": 0.02, "pointrend": 0.02, "mask2former": 1e-4}


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
        "SOLVER.STEPS", (int(0.7 * max_iter), int(0.9 * max_iter)),
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
        self.root = Path(d.get("root", "data/export/f4"))
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

    def _trainer_cls(self, m2f):
        """DefaultTrainer (R-CNN) hoặc Trainer của Mask2Former, thêm hook giữ
        checkpoint tốt nhất theo segm/AP trên val và mapper tăng cường nadir."""
        from detectron2.data import DatasetMapper, build_detection_train_loader
        from detectron2.data import transforms as T
        from detectron2.engine import DefaultTrainer, hooks
        from detectron2.evaluation import COCOEvaluator

        a = self.train_args
        base = m2f.Trainer if m2f is not None else DefaultTrainer

        class CoffeeTrainer(base):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if m2f is not None:
                    return base.build_evaluator(cfg, dataset_name, output_folder)
                return COCOEvaluator(dataset_name, tasks=("segm",),
                                     output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / "inference"))

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

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.engine import default_setup

        cfg, m2f, names = self._cfg()
        cls = self._trainer_cls(m2f)
        default_setup(cfg, SimpleNamespace(config_file="", eval_only=False,
                                           opts=[], num_gpus=1))
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

    def _results_csv(self, per_epoch: int) -> list[dict]:
        """metrics.json (một JSON mỗi dòng) của detectron2 -> results.csv có cột
        epoch, cùng tinh thần với các trainer khác."""
        src = self.out_dir / "metrics.json"
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
            row = {"epoch": d["iteration"] // per_epoch + 1}
            row.update({k: d[k] for k in keep if k in d})
            rows.append(row)
        out = self.run_dir / "results.csv"
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("epoch", "iteration"), k))
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return rows
