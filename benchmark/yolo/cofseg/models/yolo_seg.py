"""Bọc YOLO-seg đã huấn luyện vào hợp đồng SegmentationModel.

Nhờ lớp bọc này, model YOLO đi qua ĐÚNG đường đo với mọi model khác, nên các
con số so sánh được. Không có nó thì mỗi họ model lại có một đường tính chỉ
số hơi khác nhau và bảng so sánh mất ý nghĩa.
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Prediction, SegmentationModel


@register("model", "yolo_seg")
class YoloSegModel(SegmentationModel):
    """Model đầu-cuối: tự tìm tán, không cần gợi ý."""

    needs_prompt = False

    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        device: str | int | None = None,
        retina_masks: bool = True,
    ):
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(weights)
        # retina_masks: xuất mặt nạ ở độ phân giải ảnh thay vì lưới proto.
        # Với dự án lấy đường biên làm trọng tâm thì không có lý do tắt.
        self.kw = dict(
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
            verbose=False,
        )
        if device is not None:
            self.kw["device"] = device

    def warmup(self) -> None:
        self.model.predict(
            np.zeros((self.kw["imgsz"], self.kw["imgsz"], 3), np.uint8), **self.kw
        )

    def predict(
        self, image: np.ndarray, boxes: np.ndarray | None = None
    ) -> list[Prediction]:
        res = self.model.predict(image, **self.kw)[0]
        if res.masks is None:
            return []
        h, w = image.shape[:2]
        scores = (
            res.boxes.conf.cpu().numpy()
            if res.boxes is not None
            else np.ones(len(res.masks))
        )
        out: list[Prediction] = []
        for i, m in enumerate(res.masks.data.cpu().numpy()):
            mask = m.astype(bool)
            if mask.shape != (h, w):
                import cv2

                mask = (
                    cv2.resize(
                        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                )
            poly = None
            if res.masks.xy is not None and i < len(res.masks.xy):
                p = np.asarray(res.masks.xy[i], dtype=np.float64)
                poly = p if p.ndim == 2 and len(p) >= 3 else None
            out.append(
                Prediction(
                    mask=mask,
                    origin=(0, 0),
                    score=float(scores[i]) if i < len(scores) else 1.0,
                    polygon=poly,
                )
            )
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "weights": self.weights, **self.kw}
