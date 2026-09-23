"""Bộ COCO đã xuất -> mẫu huấn luyện cho các model detection kiểu torchvision.

Mỗi mẫu là (ảnh CxHxW float [0,1], target) với target đúng hợp đồng của
torchvision.models.detection: boxes xyxy, labels, masks NxHxW, image_id.

Hai quyết định đáng nói:

1. THU ẢNH TRƯỚC, RASTERISE SAU. Ảnh 2560x1440 với 48 vùng thì mặt nạ đầy đủ
   là 48 x 3.7 MB; thu về cạnh dài 1024 rồi mới vẽ polygon thì còn 48 x 0.6 MB
   và không mất gì vì model cũng chỉ nhìn ảnh đã thu.

2. TĂNG CƯỜNG TRÊN TOẠ ĐỘ POLYGON, không trên mặt nạ. Lật/xoay một polygon là
   vài phép cộng; lật/xoay 48 mặt nạ là 48 lần chép bộ nhớ. Ảnh nadir không có
   chiều "trên", nên lật dọc và xoay 90° đều là mẫu hợp lệ — đây là chỗ rẻ nhất
   để nhân dữ liệu lên 8 lần.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .coco import CocoDataset, ImageRecord


def imread(path: str | Path) -> np.ndarray:
    """cv2.imread không đọc được đường dẫn có ký tự Unicode trên Windows."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh {path}")
    return img


def resize_long_side(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float]:
    """Thu (hoặc phóng) để cạnh dài = imgsz. Trả về (ảnh, hệ số đã nhân)."""
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    if abs(scale - 1.0) < 1e-9:
        return img, 1.0
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    out = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
    return out, scale


@dataclass(frozen=True)
class Ops:
    """Một tổ hợp tăng cường hình học. Thứ tự áp dụng: lật ngang, lật dọc, xoay."""

    hflip: bool = False
    vflip: bool = False
    rot90: int = 0  # số lần xoay 90° ngược chiều kim đồng hồ, 0..3


def apply_ops(img: np.ndarray, polys: list[np.ndarray], ops: Ops) -> tuple[np.ndarray, list[np.ndarray]]:
    """Áp cùng một phép lên ảnh và lên toạ độ polygon.

    Quy ước toạ độ: đỉnh polygon là TÂM điểm ảnh — cv2.fillPoly (và Region.
    rasterize, align_masks trong toàn dự án) vẽ kín cả điểm ảnh mang toạ độ
    đỉnh. Nên lật ngang là x -> (W - 1) - x, đúng như img[:, ::-1] đưa cột c
    về cột W - 1 - c; xoay 90° ngược chiều kim đồng hồ (np.rot90, k=1) là
    (x, y) -> (y, (W - 1) - x) với ảnh mới cỡ (W, H).
    """
    h, w = img.shape[:2]
    polys = [p.astype(np.float64, copy=True) for p in polys]
    if ops.hflip:
        img = img[:, ::-1]
        for p in polys:
            p[:, 0] = (w - 1) - p[:, 0]
    if ops.vflip:
        img = img[::-1, :]
        for p in polys:
            p[:, 1] = (h - 1) - p[:, 1]
    for _ in range(ops.rot90 % 4):
        h, w = img.shape[:2]
        img = np.rot90(img)
        for p in polys:
            x, y = p[:, 0].copy(), p[:, 1].copy()
            p[:, 0], p[:, 1] = y, (w - 1) - x
    return np.ascontiguousarray(img), polys


def rasterize(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """N polygon -> mặt nạ NxHxW uint8 trên khung (h, w)."""
    out = np.zeros((len(polys), h, w), np.uint8)
    for m, p in zip(out, polys):
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return out


class InstanceDataset:
    """torch.utils.data.Dataset, nhưng không import torch ở cấp module để phần
    đọc dữ liệu vẫn test được trên máy không có torch."""

    def __init__(
        self,
        coco: CocoDataset,
        imgsz: int,
        fliplr: float = 0.0,
        flipud: float = 0.0,
        rot90: bool = False,
        limit: int | None = None,
        seed: int = 0,
        label: int = 1,
    ):
        self.coco = coco
        self.imgsz = int(imgsz)
        self.fliplr, self.flipud, self.rot90 = float(fliplr), float(flipud), bool(rot90)
        self.records: list[ImageRecord] = coco.image_list()[:limit]
        self.rng = random.Random(seed)
        self.label = int(label)

    def __len__(self) -> int:
        return len(self.records)

    def sample_ops(self) -> Ops:
        return Ops(
            hflip=self.rng.random() < self.fliplr,
            vflip=self.rng.random() < self.flipud,
            rot90=self.rng.randrange(4) if self.rot90 else 0,
        )

    def load(self, i: int, ops: Ops | None = None) -> tuple[np.ndarray, dict]:
        """Phần thuần numpy của __getitem__, để test không cần torch."""
        rec = self.records[i]
        img, scale = resize_long_side(imread(rec.path), self.imgsz)
        polys = [r.polygon * scale for r in rec.regions]
        img, polys = apply_ops(img, polys, ops or self.sample_ops())
        h, w = img.shape[:2]

        boxes, keep = [], []
        for j, p in enumerate(polys):
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            # Vùng mỏng hơn 1 px sau khi thu: không còn là tán, và torchvision
            # từ chối box có cạnh 0.
            if x1 - x0 >= 1.0 and y1 - y0 >= 1.0:
                boxes.append([x0, y0, x1, y1])
                keep.append(p)
        target = {
            "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
            "labels": np.full(len(keep), self.label, np.int64),
            "masks": rasterize(keep, h, w),
            "image_id": rec.image_id,
            "scale": scale,
        }
        return img, target

    def __getitem__(self, i: int):
        import torch

        img, t = self.load(i)
        # BGR -> RGB: trọng số ImageNet/COCO của torchvision học trên RGB.
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(t["boxes"]),
            "labels": torch.from_numpy(t["labels"]),
            "masks": torch.from_numpy(t["masks"]),
            "image_id": t["image_id"],
        }
        return x, target


def collate(batch):
    """torchvision detection nhận list ảnh (cỡ khác nhau được) và list target."""
    return tuple(zip(*batch))
