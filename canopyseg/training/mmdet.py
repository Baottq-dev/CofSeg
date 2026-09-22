"""Trainer mmdetection: SOLOv2 R50-FPN — cùng scripts/train.py, cùng khối
`train:` với các trainer kia.

    python scripts/train.py --config configs/train/solov2_r50_mm.yaml --set data.root=data/export/f4
    python scripts/train.py --config configs/train/solov2_r50_mm.yaml --probe

Cách nối vào mmdet: nạp config zoo đóng gói trong gói mmdet (không clone repo),
gộp phần ghi đè model ở configs/mmdet/<arch>_coffee.py, rồi tự sinh khối dữ
liệu / lịch học / hook từ `train:` (build_overrides — thuần Python, có test).
Vòng lặp là mmengine Runner; sau khi xong chấm test bằng checkpoint tốt nhất,
ghi predictions.json (COCO results) và test_metrics.json cùng bố cục với
trainer detectron2 để run_fold.sh / score_remote.py dùng chung.

Tăng cường: lật ngang/dọc như hai trainer kia; không có xoay 90° vì mmdet
không có transform sẵn cho mask + box (ghi vào bảng là chênh lệch có chủ đích).

CHƯA CHẠY THẬT: mmcv không cài trên máy phát triển (Windows). build_overrides
và việc gộp config có test (mmengine thuần Python); prepare/probe/fit phải
khói trên Linux trước khi tin.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

from ..datasets.coco import CocoDataset
from ..datasets.instances import imread
from ..registry import register
from ..weights import local_for_url
from . import memory
from .base import Trainer

#: Cùng tên với trainer detectron2/torchvision để --imgsz/--batch/--epochs
#: nghĩa như nhau trên mọi model. lr None = recipe gốc tỉ lệ theo batch.
MM_DEFAULTS: dict = {
    "imgsz": 1024,
    "batch": 4,
    "epochs": 50,
    "lr": None,
    "warmup_iters": 200,
    "amp": True,
    "fliplr": 0.5,
    "flipud": 0.5,
    "val_every": 1,
    "val_conf": 0.05,
    "max_det": 100,
    "workers": 0,
    "seed": 0,
    "log_every": 20,
}
#: arch -> config zoo trong gói mmdet, checkpoint COCO (cùng URL trong
#: configs/weights.yaml), file ghi đè model trong configs/mmdet/, lr gốc @ batch 16.
ZOO = {
    "solov2": dict(
        config="solov2/solov2_r50_fpn_ms-3x_coco.py",
        checkpoint="https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/"
                   "solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth",
        overrides="configs/mmdet/solov2_r50_coffee.py",
        lr=0.01,
    ),
}
ARCHS = tuple(ZOO)
#: Tên lớp trong bản xuất COCO của app/ (categories.name); mmdet ghép theo tên.
CLASSES = ("canopy",)


def require_mmdet():
    try:
        import mmcv  # noqa: F401
        import mmdet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Cần mmcv + mmdet (Linux; xem requirements.txt và scripts/remote/setup.sh). "
            "Ở máy không có, chấm file predictions.json bằng model coco_predictions."
        ) from e


def zoo_config(rel: str) -> Path:
    """Đường dẫn tuyệt đối tới config đóng gói trong gói mmdet (mmdet/.mim/configs).

    Chỉ cần gói mmdet có mặt, không cần import được nó (import mmdet kéo theo
    mmcv, thứ không có ở nhà).
    """
    spec = importlib.util.find_spec("mmdet")
    if spec is None or not spec.origin:
        raise ImportError("Chưa cài gói mmdet (pip install mmdet==3.3.0)")
    p = Path(spec.origin).parent / ".mim" / "configs" / rel
    if not p.exists():
        raise FileNotFoundError(f"Không thấy config zoo {rel} trong gói mmdet: {p}")
    return p


def flip_transform(fliplr: float, flipud: float) -> dict | None:
    """Lật ngang xác suất p, lật dọc xác suất q, độc lập -> một RandomFlip của
    mmdet với ba hướng loại trừ nhau: ngang p(1-q), dọc (1-p)q, chéo pq."""
    p, q = float(fliplr), float(flipud)
    probs = [p * (1 - q), (1 - p) * q, p * q]
    dirs = ["horizontal", "vertical", "diagonal"]
    keep = [(pr, d) for pr, d in zip(probs, dirs) if pr > 0]
    if not keep:
        return None
    return dict(type="RandomFlip", prob=[round(pr, 6) for pr, _ in keep],
                direction=[d for _, d in keep])


def build_overrides(arch: str, root: str | Path, splits: dict, n_train: int, args: dict,
                    aspect: float = 9 / 16, out_dir: str = "", load_from: str = "",
                    limit: int | None = None) -> dict:
    """Khối train: -> dict để Config.merge_from_dict. Thuần Python, không cần mmcv.

    Chỉ những gì phụ thuộc dữ liệu/lịch: dataset ba split, pipeline theo
    imgsz, lịch lr theo epoch, hook checkpoint theo mask AP trên val, AMP.
    """
    if arch not in ZOO:
        raise ValueError(f"arch {arch!r} không có; có: {ARCHS}")
    batch, epochs = int(args["batch"]), int(args["epochs"])
    if batch < 1 or epochs < 1 or n_train < 1:
        raise ValueError("batch, epochs và số ảnh train đều phải >= 1")
    root = Path(root)
    per_epoch = math.ceil(n_train / batch)
    max_iter = epochs * per_epoch
    lr = args.get("lr")
    if lr is None:
        lr = ZOO[arch]["lr"] * batch / 16
    workers = int(args["workers"])
    # imgsz là CẠNH DÀI như YOLO/torchvision/detectron2; mmdet Resize keep_ratio
    # nhận (max cạnh dài, max cạnh ngắn) không phân biệt thứ tự.
    imgsz = int(args["imgsz"])
    scale = (imgsz, int(round(imgsz * aspect)))

    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="Resize", scale=scale, keep_ratio=True),
    ]
    flip = flip_transform(args["fliplr"], args["flipud"])
    if flip:
        train_pipeline.append(flip)
    train_pipeline.append(dict(type="PackDetInputs"))
    test_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="Resize", scale=scale, keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="PackDetInputs",
             meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
    ]

    def dataset(split: str, pipeline: list, test_mode: bool) -> dict:
        d = dict(
            type="CocoDataset",
            data_root=str(root),
            metainfo=dict(classes=tuple(CLASSES)),
            ann_file=f"annotations/instances_{split}.json",
            data_prefix=dict(img=f"images/{split}/"),
            # Ảnh nền (đã xem, không có tán) vẫn vào train làm mẫu âm, như các trainer kia.
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=pipeline,
            backend_args=None,
        )
        if test_mode:
            d["test_mode"] = True
        return d

    train_ds = dataset(splits["train"], train_pipeline, False)
    if limit:
        train_ds["indices"] = int(limit)      # khói: N ảnh đầu
    loader_common = dict(num_workers=workers, persistent_workers=workers > 0)
    out_dir = str(out_dir)

    def ann(split: str) -> str:
        return str(root / "annotations" / f"instances_{split}.json")

    # MultiStepLR theo epoch ở 70 % và 90 %; epoch ít thì không được rơi về 0.
    milestones = sorted({max(1, int(0.7 * epochs)), max(1, int(0.9 * epochs))})

    return dict(
        default_scope="mmdet",
        work_dir=out_dir,
        load_from=str(load_from) if load_from else None,
        resume=False,
        randomness=dict(seed=int(args["seed"])),
        model=dict(test_cfg=dict(score_thr=float(args["val_conf"]), max_per_img=int(args["max_det"]))),
        train_dataloader=dict(
            batch_size=batch, **loader_common,
            sampler=dict(type="DefaultSampler", shuffle=True),
            batch_sampler=dict(type="AspectRatioBatchSampler"),
            dataset=train_ds),
        val_dataloader=dict(
            batch_size=1, **loader_common, drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dataset(splits["val"], test_pipeline, True)),
        test_dataloader=dict(
            batch_size=1, **loader_common, drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dataset(splits["test"], test_pipeline, True)),
        val_evaluator=dict(type="CocoMetric", ann_file=ann(splits["val"]), metric="segm",
                           format_only=False, backend_args=None),
        # Test: chấm VÀ ghi <out_dir>/test/pred.segm.json (COCO results).
        test_evaluator=dict(type="CocoMetric", ann_file=ann(splits["test"]), metric="segm",
                            format_only=False, outfile_prefix=str(Path(out_dir) / "test" / "pred"),
                            backend_args=None),
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=epochs,
                       val_interval=int(args["val_every"])),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        optim_wrapper=dict(
            type="AmpOptimWrapper" if args["amp"] else "OptimWrapper",
            optimizer=dict(type="SGD", lr=float(lr), momentum=0.9, weight_decay=1e-4),
            clip_grad=dict(max_norm=35, norm_type=2)),
        param_scheduler=[
            dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0,
                 end=min(int(args["warmup_iters"]), max_iter)),
            dict(type="MultiStepLR", begin=0, end=epochs, by_epoch=True,
                 milestones=milestones, gamma=0.1),
        ],
        default_hooks=dict(
            logger=dict(type="LoggerHook", interval=int(args["log_every"])),
            checkpoint=dict(type="CheckpointHook", interval=int(args["val_every"]), by_epoch=True,
                            save_best="coco/segm_mAP", rule="greater", max_keep_ckpts=1,
                            save_last=True),
            visualization=dict(type="DetVisualizationHook", draw=False),
        ),
        log_processor=dict(type="LogProcessor", window_size=int(args["log_every"]), by_epoch=True),
    )


def iters_per_epoch(n_train: int, batch: int) -> int:
    return math.ceil(n_train / max(1, int(batch)))


def load_config(arch: str, overrides: dict, config_file: str | None = None):
    """Config zoo + configs/mmdet/<arch>_coffee.py + overrides -> mmengine Config.

    Cần mmengine (thuần Python, có ở nhà); không cần mmcv.
    """
    from mmengine.config import Config

    cfg = Config.fromfile(str(zoo_config(ZOO[arch]["config"])))
    over = Path(config_file or ZOO[arch]["overrides"])
    if not over.exists():
        raise FileNotFoundError(f"Không thấy file ghi đè model {over}")
    cfg.merge_from_dict(Config.fromfile(str(over)).to_dict())
    cfg.merge_from_dict(overrides)
    return cfg


@register("trainer", "mmdet")
class MMDetTrainer(Trainer):
    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        m = cfg.get("model") or {}
        self.arch: str = m.get("arch", "solov2")
        if self.arch not in ARCHS:
            raise ValueError(f"model.arch {self.arch!r} không có; có: {ARCHS}")
        self.config_file = m.get("config")
        self.weights = m.get("weights")
        self.train_args: dict = {**MM_DEFAULTS, **(cfg.get("train") or {})}
        d = cfg.get("data") or {}
        self.root = Path(d.get("root", "data/export/f4"))
        self.splits = {k: d.get(k, k) for k in ("train", "val", "test")}
        self.min_area = float(d.get("min_area", 0.0))
        self.limit = d.get("limit")
        self.out_dir = self.run_dir / "mmdet"

    # -------------------------------------------------------------------- tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        return dict(MM_DEFAULTS)

    TAG_KEYS = (("imgsz", "i"), ("batch", "b"), ("epochs", "e"))

    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        args = {**MM_DEFAULTS, **(cfg.get("train") or {})}
        return "".join(f"{p}{args[k]}" for k, p in cls.TAG_KEYS)

    # ------------------------------------------------------------------ chuẩn bị
    def prepare(self) -> dict:
        """Kiểm dữ liệu bằng bộ đọc của repo (không cần mmdet), đếm ảnh train
        để đổi epoch -> iteration."""
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
        # mmdet ghép lớp theo TÊN (metainfo.classes <-> categories.name); tên
        # lệch là mọi nhãn bị bỏ lặng lẽ, nên kiểm ở đây.
        out["classes"] = self._classes()
        if self.limit:
            self.n_train = min(self.n_train, int(self.limit))
            out["limit"] = int(self.limit)
        out["arch"] = self.arch
        out["iters_per_epoch"] = iters_per_epoch(self.n_train, self.train_args["batch"])
        (self.run_dir / "dataset_check.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _classes(self) -> list[str]:
        ann = self.root / "annotations" / f"instances_{self.splits['train']}.json"
        cats = json.loads(ann.read_text(encoding="utf-8")).get("categories") or []
        names = [c["name"] for c in cats]
        if names != list(CLASSES):
            raise ValueError(f"{ann}: categories {names} != {list(CLASSES)} mà config mmdet dùng")
        return names

    def checkpoint(self) -> str:
        """model.weights nếu đặt; không thì bản trong weights/ theo bản kê, hoặc URL."""
        if self.weights:
            return str(self.weights)
        url = ZOO[self.arch]["checkpoint"]
        local = local_for_url(url)
        return str(local) if local else url

    def _cfg(self):
        if not hasattr(self, "n_train"):
            self.prepare()
        over = build_overrides(self.arch, self.root, self.splits, self.n_train, self.train_args,
                               aspect=self.aspect, out_dir=str(self.out_dir),
                               load_from=self.checkpoint(), limit=self.limit)
        return load_config(self.arch, over, self.config_file)

    # ---------------------------------------------------------------------- dò
    def probe(self) -> dict:
        """Hai iteration thật (forward + backward, AMP) qua Runner rồi đo đỉnh
        bộ nhớ, như hai trainer kia."""
        import torch

        if not torch.cuda.is_available():
            return {"supported": False, "reason": "không có CUDA"}
        require_mmdet()
        from mmengine.runner import Runner

        cfg = self._cfg()
        # Vòng lặp theo iteration, 2 bước, không val, không checkpoint.
        cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=2, val_interval=10**9)
        cfg.param_scheduler = None
        cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
        cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=10**9, by_epoch=False)
        cfg.log_processor = dict(type="LogProcessor", by_epoch=False)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            runner = Runner.from_cfg(cfg)
            runner.train()
            torch.cuda.synchronize()
        finally:
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.empty_cache()
        wall = memory.wall_gb()
        batch = int(self.train_args["batch"])
        return {
            "supported": True, "arch": self.arch, "imgsz": int(self.train_args["imgsz"]),
            "batch": batch, "peak_gb": round(peak, 2), "under_wall": peak < wall,
            "wall_gb": wall,
            "suggested_batch": max(1, int(batch * wall / max(peak, 1e-6))),
            "seconds": round(time.time() - t0, 1),
            "note": "đỉnh đo trên 2 iteration; batch dày vùng hơn sẽ cao hơn một chút",
        }

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        require_mmdet()
        from mmengine.runner import Runner

        cfg = self._cfg()
        per_epoch = iters_per_epoch(self.n_train, self.train_args["batch"])
        print(f"{self.arch}: {self.train_args['epochs']} epoch x {per_epoch} iteration, "
              f"batch {cfg.train_dataloader.batch_size}, imgsz {self.train_args['imgsz']}, "
              f"lr {cfg.optim_wrapper.optimizer.lr:g}, load_from {cfg.load_from}")
        t0 = time.time()
        runner = Runner.from_cfg(cfg)
        runner.train()
        train_seconds = round(time.time() - t0, 1)

        # Trọng số về cùng bố cục với các trainer khác; mmdet_config.py cạnh
        # best.pth để MMDetModel dựng lại đúng model không cần nhắc tham số.
        weights_dir = self.run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        bests = sorted(self.out_dir.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
        lasts = sorted(self.out_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        if not (bests or lasts):
            raise RuntimeError(f"mmdet không để lại checkpoint nào trong {self.out_dir}")
        best_src = bests[-1] if bests else lasts[-1]
        best = weights_dir / "best.pth"
        shutil.copy2(best_src, best)
        if lasts:
            shutil.move(str(lasts[-1]), weights_dir / "last.pth")
        cfg.dump(str(weights_dir / "mmdet_config.py"))
        rows = self._results_csv(per_epoch)

        # Chấm TEST bằng checkpoint tốt nhất -> predictions.json (COCO results,
        # image_id là id gốc của bản xuất) để chấm lại ở nhà qua coco_predictions.
        cfg_t = cfg.copy()
        cfg_t.load_from = str(best)
        cfg_t.work_dir = str(self.out_dir / "test")
        metrics = Runner.from_cfg(cfg_t).test() or {}
        preds = self.out_dir / "test" / "pred.segm.json"
        pred_out = self.run_dir / "predictions.json"
        if preds.exists():
            shutil.copy2(preds, pred_out)
        segm = {"AP": metrics.get("coco/segm_mAP"), "AP50": metrics.get("coco/segm_mAP_50"),
                "AP75": metrics.get("coco/segm_mAP_75")}
        segm = {k: round(100 * float(v), 3) for k, v in segm.items() if v is not None}
        (self.run_dir / "test_metrics.json").write_text(
            json.dumps({"segm": segm, "raw": metrics}, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8")

        val_rows = [r for r in rows if r.get("coco/segm_mAP") not in (None, "")]
        best_row = max(val_rows, key=lambda r: float(r["coco/segm_mAP"])) if val_rows else None
        return {
            "weights": {"best": str(best), "last": str(weights_dir / "last.pth")},
            "best_epoch": int(best_row["epoch"]) if best_row else -1,
            "best_val_AP": round(float(best_row["coco/segm_mAP"]), 4) if best_row else -1.0,
            "epochs_run": int(self.train_args["epochs"]),
            "train_seconds": train_seconds,
            "results_csv": str(self.run_dir / "results.csv"),
            "predictions": str(pred_out) if pred_out.exists() else None,
            "test": segm,
            "args": dict(self.train_args),
        }

    def _results_csv(self, per_epoch: int) -> list[dict]:
        """scalars.json của mmengine (một JSON mỗi dòng; step là iteration với
        loss, là epoch với số đo val) -> results.csv có cột epoch."""
        rows: list[dict] = []
        keep = ("lr", "loss", "loss_mask", "loss_cls", "time", "data_time", "memory",
                "coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75")
        for src in sorted(self.out_dir.glob("*/vis_data/scalars.json")):
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                step = int(d.get("step", 0))
                is_val = any(k.startswith("coco/") for k in d)
                row = {"epoch": step if is_val else step // per_epoch + 1,
                       "iteration": "" if is_val else step}
                row.update({k: d[k] for k in keep if k in d})
                rows.append(row)
        out = self.run_dir / "results.csv"
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("epoch", "iteration"), k))
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys or ["epoch", "iteration"])
            w.writeheader()
            w.writerows(rows)
        return rows
