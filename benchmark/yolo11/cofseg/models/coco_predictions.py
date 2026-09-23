"""File dự đoán COCO results đóng vai một model.

Model chạy trên máy thuê (detectron2, mmdet: Cascade, Mask2Former, MaskDINO,
PointRend, Mask Transfiner, RSPrompter) xuất một file JSON theo định dạng COCO
results — chính thứ COCO.loadRes() nhận. Lớp này đọc file đó và trả Prediction
cho từng ảnh, nên các dự đoán ngoài đi qua ĐÚNG vòng chấm của evaluate.py và
cả bộ tinh chỉnh cắm thêm (RefineModel) như mọi model chạy tại chỗ.

predictions.json của ultralytics (val save_json=True) cũng đọc được: nó ghi
image_id là tên file không đuôi thay vì id số, nên ở đây khớp được theo cả hai.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from ..registry import register
from .base import Prediction, SegmentationModel


@register("model", "coco_predictions")
class CocoPredictionsModel(SegmentationModel):
    needs_prompt = False

    def __init__(self, file: str, key: str = "auto", conf: float = 0.0,
                 max_det: int | None = None, category_id: int | None = None):
        """`key`: "id" khớp theo image_id số của bộ xuất, "stem" theo tên file
        không đuôi, "auto" nhìn kiểu dữ liệu của image_id trong file."""
        self.file = str(file)
        raw = json.loads(Path(file).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("annotations", [])
        if key == "auto":
            key = "stem" if raw and isinstance(raw[0].get("image_id"), str) else "id"
        if key not in ("id", "stem"):
            raise ValueError(f"key phải là id, stem hoặc auto; nhận {key!r}")
        self.key = key
        self.conf, self.max_det, self.category_id = float(conf), max_det, category_id
        self.by_image: dict = {}
        for d in raw:
            if category_id is not None and int(d.get("category_id", -1)) != category_id:
                continue
            k = d["image_id"] if key == "stem" else int(d["image_id"])
            self.by_image.setdefault(str(k), []).append(d)
        self.n_detections = sum(len(v) for v in self.by_image.values())
        self._current = None

    def set_context(self, record) -> None:
        """Runner gọi trước mỗi ảnh; `record` là ImageRecord của bộ xuất."""
        self._current = record

    def _lookup(self, record) -> list[dict]:
        if self.key == "stem":
            return self.by_image.get(Path(record.file_name).stem, [])
        return self.by_image.get(str(int(record.image_id)), [])

    @staticmethod
    def _decode(seg, h: int, w: int) -> np.ndarray | None:
        if isinstance(seg, dict):                       # RLE (nén hoặc chưa nén)
            rle = seg
            if isinstance(seg.get("counts"), list):
                rle = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.decode(rle).astype(bool)
        if isinstance(seg, list) and seg:               # polygon(s)
            m = np.zeros((h, w), np.uint8)
            for poly in seg:
                pts = np.asarray(poly, np.float64).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
            return m.astype(bool)
        return None

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        if self._current is None:
            raise RuntimeError("CocoPredictionsModel cần set_context(record) trước predict()")
        h, w = image.shape[:2]
        dets = sorted(self._lookup(self._current), key=lambda d: -float(d.get("score", 1.0)))
        out: list[Prediction] = []
        for d in dets:
            score = float(d.get("score", 1.0))
            if score < self.conf:
                continue
            mask = self._decode(d.get("segmentation"), h, w)
            if mask is None or mask.shape != (h, w) or not mask.any():
                continue
            out.append(Prediction(mask=mask, origin=(0, 0), score=score,
                                  meta={"category_id": d.get("category_id")}).cropped())
            if self.max_det is not None and len(out) >= self.max_det:
                break
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "file": self.file, "key": self.key, "conf": self.conf,
                "max_det": self.max_det, "images_in_file": len(self.by_image),
                "detections_in_file": self.n_detections}
