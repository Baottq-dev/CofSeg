"""Huấn luyện Mask R-CNN R50-FPN bằng torchvision.

torchvision không kèm vòng huấn luyện (khác ultralytics), nên file này viết
vòng đó — cố ý ngắn và không có gì lạ: SGD/AdamW, warmup tuyến tính, cosine,
AMP, chấm val bằng pycocotools mỗi epoch, giữ best.pt theo mask AP. Đây là
recipe tham chiếu của torchvision/detectron2 thu gọn, không phải kỹ thuật mới;
mọi thứ đáng so sánh nằm ở model, không ở vòng lặp.

Ba điểm riêng cho dữ liệu này, cùng lý do với trainer YOLO:
  - lật dọc + xoay 90° bật mặc định: ảnh nadir không có chiều "trên";
  - workers = 0: trên Windows mỗi worker nạp lại torch và paging file không đủ;
  - batch nhỏ, imgsz lớn: nút thắt là độ phân giải đường biên, không phải BN.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from ..datasets.coco import CocoDataset
from ..datasets.instances import InstanceDataset, collate, imread
from ..evaluation.coco_eval import evaluate as coco_evaluate
from ..evaluation.coco_eval import predictions_to_coco
from ..models.maskrcnn import MaskRCNNModel, build_maskrcnn, save_checkpoint
from ..registry import register
from .base import Trainer

#: Toàn bộ tham số khối `train:` nhận, kèm mặc định. Đây cũng là danh sách mà
#: scripts/train.py dùng để bắt lỗi gõ sai và để in --list-params.
MASKRCNN_DEFAULTS: dict = {
    "imgsz": 1024,            # cạnh dài sau khi thu, cùng nghĩa với YOLO
    "batch": 2,
    "epochs": 50,
    "optimizer": "sgd",       # sgd | adamw
    "lr": 0.01,               # SGD; với adamw nên ~1e-4
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup_iters": 200,
    "cos_lr": True,
    "amp": True,
    "workers": 0,
    "fliplr": 0.5,
    "flipud": 0.5,
    "rot90": True,
    "trainable_backbone_layers": 3,   # 0..5, torchvision mặc định 3 khi có trọng số
    "pretrained": True,               # trọng số COCO của torchvision
    "max_det": 100,
    "box_score_thresh": 0.05,
    "val_every": 1,
    "val_conf": 0.05,                 # ngưỡng lúc chấm val: thấp để AP thấy đủ đường PR
    "patience": 0,                    # 0 = không dừng sớm
    "seed": 0,
    "log_every": 20,
}

#: Vách bộ nhớ thực nghiệm trên card 8 GB này: vượt qua thì driver tràn sang
#: RAM hệ thống và chậm đi hàng trăm lần mà không báo OOM (xem trainer YOLO).
MEMORY_WALL_GB = 7.0


@register("trainer", "maskrcnn")
class MaskRCNNTrainer(Trainer):
    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        self.train_args: dict = {**MASKRCNN_DEFAULTS, **(cfg.get("train") or {})}
        d = cfg.get("data") or {}
        self.root = Path(d.get("root", "data/export/f4"))
        self.train_split = d.get("train", "train")
        self.val_split = d.get("val", "val")
        self.min_area = float(d.get("min_area", 0.0))
        self.limit = d.get("limit")

    # -------------------------------------------------------------------- tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        return dict(MASKRCNN_DEFAULTS)

    TAG_KEYS = (("imgsz", "i"), ("batch", "b"), ("epochs", "e"))

    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        args = {**MASKRCNN_DEFAULTS, **(cfg.get("train") or {})}
        return "".join(f"{p}{args[k]}" for k, p in cls.TAG_KEYS)

    # ------------------------------------------------------------------ chuẩn bị
    def prepare(self) -> dict:
        self.train_ds = CocoDataset(self.root, self.train_split, min_area=self.min_area)
        self.val_ds = CocoDataset(self.root, self.val_split, min_area=self.min_area)
        out = {"train": self.train_ds.summary(), "val": self.val_ds.summary()}
        if self.limit:
            out["limit"] = int(self.limit)
        # Kiểm một ảnh đọc được thật, trước khi đụng GPU.
        first = self.train_ds.image_list()[0]
        out["sample"] = {"file": first.file_name, "shape": list(imread(first.path).shape)}
        (self.run_dir / "dataset_check.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    # ------------------------------------------------------------------ dùng chung
    def _seed(self) -> None:
        import torch

        s = int(self.train_args["seed"])
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def _loader(self, coco: CocoDataset, shuffle: bool, limit=None):
        import torch

        a = self.train_args
        ds = InstanceDataset(
            coco, imgsz=a["imgsz"], fliplr=a["fliplr"], flipud=a["flipud"],
            rot90=a["rot90"], limit=limit, seed=a["seed"],
        )
        g = torch.Generator()
        g.manual_seed(int(a["seed"]))
        return torch.utils.data.DataLoader(
            ds, batch_size=int(a["batch"]), shuffle=shuffle, num_workers=int(a["workers"]),
            collate_fn=collate, generator=g, drop_last=False,
        )

    def _model(self, device):
        a = self.train_args
        return build_maskrcnn(
            num_classes=2, pretrained=bool(a["pretrained"]),
            trainable_backbone_layers=int(a["trainable_backbone_layers"]),
            imgsz=int(a["imgsz"]), max_det=int(a["max_det"]),
            box_score_thresh=float(a["box_score_thresh"]),
        ).to(device)

    def _optimizer(self, model):
        import torch

        a = self.train_args
        params = [p for p in model.parameters() if p.requires_grad]
        if a["optimizer"] == "adamw":
            return torch.optim.AdamW(params, lr=a["lr"], weight_decay=a["weight_decay"])
        if a["optimizer"] == "sgd":
            return torch.optim.SGD(
                params, lr=a["lr"], momentum=a["momentum"], weight_decay=a["weight_decay"]
            )
        raise ValueError(f"optimizer phải là sgd hoặc adamw, nhận {a['optimizer']!r}")

    def _lr_at(self, it: int, total: int) -> float:
        """Warmup tuyến tính rồi cosine về 0 (hoặc giữ phẳng nếu cos_lr=false)."""
        a = self.train_args
        base, warm = float(a["lr"]), int(a["warmup_iters"])
        if it < warm:
            return base * (0.001 + 0.999 * it / max(1, warm))
        if not a["cos_lr"]:
            return base
        t = (it - warm) / max(1, total - warm)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    @staticmethod
    def _step(model, images, targets, device, scaler, amp):
        """Một lần forward + backward. Trả về dict loss (float)."""
        import torch

        images = [im.to(device, non_blocking=True) for im in images]
        targets = [
            {k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
            for t in targets
        ]
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hữu hạn: { {k: float(v) for k, v in losses.items()} }")
        scaler.scale(loss).backward()
        return {"loss": float(loss), **{k: float(v) for k, v in losses.items()}}

    # ---------------------------------------------------------------------- dò
    def probe(self) -> dict:
        """Chạy THẬT hai iteration (forward + backward, AMP) rồi đo đỉnh bộ nhớ.

        Đo thật thay vì ước lượng: autobatch của ultralytics chỉ tính chiều thuận
        và lạc quan ~2.5 lần trên card này. Hai iteration đủ để CUDA cấp phát
        xong vùng nhớ đỉnh của cả forward lẫn backward.
        """
        import torch

        if not torch.cuda.is_available():
            return {"supported": False, "reason": "không có CUDA"}
        a = self.train_args
        device = torch.device("cuda")
        self._seed()
        if not hasattr(self, "train_ds"):
            self.prepare()
        loader = self._loader(self.train_ds, shuffle=True, limit=max(2, int(a["batch"]) * 2))
        model = self._model(device)
        opt = self._optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(a["amp"]))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n_obj = 0
        try:
            model.train()
            for i, (images, targets) in enumerate(loader):
                if i >= 2:
                    break
                n_obj += sum(len(t["boxes"]) for t in targets)
                opt.zero_grad(set_to_none=True)
                self._step(model, images, targets, device, scaler, bool(a["amp"]))
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
        finally:
            peak = torch.cuda.max_memory_allocated() / 2**30
            del model, opt
            torch.cuda.empty_cache()
        per_batch = peak
        batch = int(a["batch"])
        return {
            "supported": True,
            "imgsz": int(a["imgsz"]),
            "batch": batch,
            "peak_gb": round(per_batch, 2),
            "under_wall": per_batch < MEMORY_WALL_GB,
            "wall_gb": MEMORY_WALL_GB,
            # Ngoại suy tuyến tính theo batch; chỉ là gợi ý, xác nhận bằng chạy thật.
            "suggested_batch": max(1, int(batch * MEMORY_WALL_GB / max(per_batch, 1e-6))),
            "objects_in_probe": n_obj,
            "seconds": round(time.time() - t0, 1),
            "note": "đỉnh đo trên 2 iteration; batch dày vùng hơn sẽ cao hơn một chút",
        }

    # ----------------------------------------------------------------- huấn luyện
    def _validate(self, model, device) -> dict:
        """Mask AP trên val bằng pycocotools, ở độ phân giải gốc.

        Đi qua đúng lớp bọc MaskRCNNModel mà bước đánh giá sẽ dùng, nên số val
        ở đây và số evaluate.py về sau là cùng một đường tính.
        """
        a = self.train_args
        wrapper = MaskRCNNModel(module=model, imgsz=int(a["imgsz"]), conf=float(a["val_conf"]),
                                max_det=int(a["max_det"]), device=str(device))
        dets, ids = [], []
        cat_id = int(self.val_ds.categories[0]["id"]) if self.val_ds.categories else 1
        for rec in self.val_ds.image_list():
            img = imread(rec.path)
            preds = wrapper.predict(img)
            dets.extend(predictions_to_coco(rec.image_id, preds, cat_id, size=img.shape[:2]))
            ids.append(rec.image_id)
        res = coco_evaluate(str(self.val_ds.ann_file), dets, ids, boundary=False)
        return res.get("mask") or {}

    def fit(self) -> dict:
        import torch

        a = self.train_args
        if not torch.cuda.is_available() and a["amp"]:
            print("Không có CUDA: tắt AMP, huấn luyện trên CPU (rất chậm).")
            a["amp"] = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed()
        if not hasattr(self, "train_ds"):
            self.prepare()

        loader = self._loader(self.train_ds, shuffle=True, limit=self.limit)
        model = self._model(device)
        opt = self._optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(a["amp"]) and device.type == "cuda")

        epochs, total = int(a["epochs"]), int(a["epochs"]) * len(loader)
        weights_dir = self.run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        results_csv = self.run_dir / "results.csv"
        best_ap, best_epoch, bad_epochs, it = -1.0, -1, 0, 0
        meta = dict(imgsz=int(a["imgsz"]), num_classes=2, max_det=int(a["max_det"]),
                    train_args=dict(a))
        print(f"Huấn luyện {epochs} epoch x {len(loader)} iteration, batch {a['batch']}, "
              f"imgsz {a['imgsz']}, {device}")

        rows: list[dict] = []
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            sums: dict[str, float] = {}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for i, (images, targets) in enumerate(loader, 1):
                lr = self._lr_at(it, total)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                part = self._step(model, images, targets, device, scaler, bool(a["amp"]))
                scaler.step(opt)
                scaler.update()
                it += 1
                for k, v in part.items():
                    sums[k] = sums.get(k, 0.0) + v
                if i % int(a["log_every"]) == 0 or i == len(loader):
                    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                    print(f"  epoch {epoch}/{epochs}  {i}/{len(loader)}  loss {part['loss']:.3f}  "
                          f"lr {lr:.2e}  mem {mem:.2f}G  {time.time() - t0:.0f}s", flush=True)

            row = {"epoch": epoch, **{k: round(v / len(loader), 4) for k, v in sums.items()},
                   "lr": float(f"{lr:.4g}"), "seconds": round(time.time() - t0, 1),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                   if device.type == "cuda" else 0.0}

            if epoch % int(a["val_every"]) == 0 or epoch == epochs:
                t1 = time.time()
                stats = self._validate(model, device)
                row.update({f"val_{k}": round(v, 4) for k, v in stats.items()
                            if k in ("AP", "AP50", "AP75", "AR_100")})
                row["val_seconds"] = round(time.time() - t1, 1)
                ap = float(stats.get("AP", -1.0))
                print(f"  epoch {epoch}: val mask AP {ap:.4f}  AP50 {stats.get('AP50', 0):.4f}  "
                      f"AP75 {stats.get('AP75', 0):.4f}", flush=True)
                if ap > best_ap:
                    best_ap, best_epoch, bad_epochs = ap, epoch, 0
                    save_checkpoint(weights_dir / "best.pt", model, epoch=epoch, val=stats, **meta)
                else:
                    bad_epochs += 1
            save_checkpoint(weights_dir / "last.pt", model, epoch=epoch, **meta)

            rows.append(row)
            with results_csv.open("w", encoding="utf-8", newline="") as fh:
                keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "epoch", k))
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

            if a["patience"] and bad_epochs >= int(a["patience"]):
                print(f"Dừng sớm: val AP không cải thiện {bad_epochs} epoch liên tiếp.")
                break

        return {
            "weights": {
                "best": str(weights_dir / "best.pt") if best_epoch > 0 else None,
                "last": str(weights_dir / "last.pt"),
            },
            "best_epoch": best_epoch,
            "best_val_AP": round(best_ap, 4),
            "epochs_run": rows[-1]["epoch"] if rows else 0,
            "results_csv": str(results_csv),
            "args": dict(a),
        }
