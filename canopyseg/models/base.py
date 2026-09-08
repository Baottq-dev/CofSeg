"""Hợp đồng chung cho mọi model phân đoạn.

Đây là thứ khiến vòng lặp đánh giá không cần biết SAM hay YOLO tồn tại, và
khiến hai họ model so sánh được với nhau: cùng đầu vào, cùng đầu ra, cùng
đường đo.

Hợp đồng phải bao được HAI kiểu model khác hẳn nhau:
  - đầu-cuối (YOLO-seg, Mask R-CNN): tự tìm vật thể, needs_prompt = False
  - theo gợi ý (SAM): cần box đưa vào, needs_prompt = True
Cờ needs_prompt cho phép cùng một harness chấm cả hai.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    """Một vật thể được dự đoán.

    `mask` là mặt nạ nhị phân trong khung cục bộ, `origin` là gốc khung trong
    ảnh gốc. Giữ cục bộ vì mặt nạ đầy đủ 2560x1440 cho mỗi vật thể là lãng phí
    (15 tán/ảnh x 3.7 MB) mà không thêm thông tin gì.
    """

    mask: np.ndarray
    origin: tuple[int, int] = (0, 0)
    score: float = 1.0
    polygon: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        return float(np.count_nonzero(self.mask))


class SegmentationModel(ABC):
    """Hợp đồng. Đừng thêm phương thức riêng của một họ model vào đây."""

    #: True nếu model cần box đưa vào mới chạy được (SAM), False nếu tự tìm.
    needs_prompt: bool = False

    @abstractmethod
    def predict(
        self, image: np.ndarray, boxes: np.ndarray | None = None
    ) -> list[Prediction]:
        """Ảnh BGR (và box xyxy nếu needs_prompt) -> danh sách vật thể."""

    def warmup(self) -> None:
        """Chạy trước khi đo thời gian, để lần gọi đầu không gánh chi phí
        khởi tạo CUDA."""

    @property
    def describe(self) -> dict:
        return {"class": type(self).__name__, "needs_prompt": self.needs_prompt}
