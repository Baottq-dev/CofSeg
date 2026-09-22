"""Một vùng tán: polygon, box, và cách dựng mặt nạ.

Mặt nạ luôn được dựng trong KHUNG CỤC BỘ quanh vùng, không phải cả ảnh
2560x1440. Rasterise 6521 vùng ở kích thước đầy đủ tốn ~24 GB bộ nhớ và
không cần thiết: mọi chỉ số ở đây đều cục bộ theo từng vùng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    """Một tán đã gán nhãn."""

    polygon: np.ndarray  # (N, 2) float, toạ độ pixel trong ảnh gốc
    image_file: str  # tên ảnh đã làm phẳng, vd field_1__10__1__DJI_...jpg
    image_id: int
    width: int
    height: int
    area: float
    ann_id: int = -1
    score: float = 1.0
    _meta: dict = _field(default_factory=dict, compare=False)

    # ---------------------------------------------------------------- nguồn gốc
    @property
    def field(self) -> str:
        """field_1, field_2, ... — suy từ tên ảnh đã làm phẳng."""
        return self.image_file.split("__")[0]

    @property
    def flight(self) -> str:
        """Đường bay: mọi thành phần trừ tên file. Đây mới là đơn vị chia tập
        không rò rỉ, vì các khung liền kề trong một đường bay chồng lấn nhau."""
        return "/".join(self.image_file.split("__")[:-1])

    # ------------------------------------------------------------------ hình học
    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        x0, y0 = self.polygon.min(0)
        x1, y1 = self.polygon.max(0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        x0, y0 = self.polygon.min(0)
        x1, y1 = self.polygon.max(0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def equivalent_side(self) -> float:
        """Cạnh hình vuông cùng diện tích — thước đo cỡ tán dùng xuyên suốt.
        Ổn định hơn cạnh bbox với tán dài hoặc nghiêng."""
        return math.sqrt(max(self.area, 0.0))

    def rasterize(self, pad: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
        """Mặt nạ nhị phân trong khung cục bộ.

        Trả về (mask, (x0, y0)) với (x0, y0) là gốc khung trong ảnh gốc, để
        quy chiếu ngược khi cần.
        """
        x0 = math.floor(self.polygon[:, 0].min()) - pad
        y0 = math.floor(self.polygon[:, 1].min()) - pad
        w = math.ceil(self.polygon[:, 0].max()) - x0 + pad
        h = math.ceil(self.polygon[:, 1].max()) - y0 + pad
        m = np.zeros((max(h, 1), max(w, 1)), np.uint8)
        cv2.fillPoly(m, [np.round(self.polygon - (x0, y0)).astype(np.int32)], 1)
        return m.astype(bool), (x0, y0)

    # ------------------------------------------------------------------ dựng từ
    @classmethod
    def from_coco(cls, ann: dict, image: "ImageRecord") -> "Region":  # noqa: F821
        seg = ann["segmentation"][0]
        poly = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
        return cls(
            polygon=poly,
            image_file=image.file_name,
            image_id=int(ann["image_id"]),
            width=image.width,
            height=image.height,
            area=float(ann.get("area", 0.0)),
            ann_id=int(ann.get("id", -1)),
        )

    def normalized_polygon(self) -> np.ndarray:
        """Polygon chuẩn hoá về [0, 1] cho định dạng YOLO.

        Kẹp về [0, 1]: ultralytics loại bỏ nhãn có toạ độ ngoài khoảng này, và
        một đỉnh lệch 0.3 px ở mép ảnh đủ để mất cả vùng.
        """
        p = self.polygon / (self.width, self.height)
        return np.clip(p, 0.0, 1.0)
