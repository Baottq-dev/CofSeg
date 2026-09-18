"""Ghép model: hai giai đoạn (phát hiện -> gợi ý) và tinh chỉnh cắm thêm.

Hai lớp này không biết SAM là gì. Chúng chỉ biết một model trả Prediction và
một "prompter"/"refiner" nhận box hoặc Prediction rồi trả mặt nạ mới. Nhờ vậy
cùng một lớp dùng được cho YOLO -> SAM 2.1, Mask R-CNN -> SAM 2.1, và về sau
cho file dự đoán từ máy thuê -> bất kỳ bộ tinh chỉnh nào.

Quy ước báo cáo: RefineModel giữ nguyên SỐ dự đoán và SCORE của model gốc,
chỉ thay mặt nạ. Nên chênh lệch giữa "gốc" và "gốc + refine" trên cùng bộ test
là Δ thuần của đường biên — đúng cột Δ trong bảng benchmark.
"""

from __future__ import annotations

import numpy as np

from ..registry import register, resolve
from .base import Prediction, SegmentationModel
from .build import build_model


def _build_refiner(spec):
    if isinstance(spec, dict):
        spec = dict(spec)
        return resolve("refiner", spec.pop("name"))(**spec)
    return spec  # đối tượng đã dựng sẵn (test, hoặc dùng chung backend)


def _build_prompter(spec):
    if isinstance(spec, dict):
        from .sam_prompt import Sam2Backend

        return Sam2Backend(**spec)
    return spec


@register("model", "two_stage")
class TwoStageModel(SegmentationModel):
    """Giai đoạn 1 tìm tán (box), giai đoạn 2 vẽ mặt nạ theo box.

    Score lấy từ detector — đó là thứ quyết định thứ tự trong AP; IoU dự đoán
    của SAM giữ trong meta để phân tích. Nếu SAM trả rỗng cho một box thì giữ
    mặt nạ của detector (fallback=True) để số dự đoán không đổi.
    """

    needs_prompt = False

    def __init__(self, detector: dict | SegmentationModel, prompter: dict | object,
                 fallback: bool = True):
        self.detector = build_model(detector)
        self.prompter = _build_prompter(prompter)
        self.fallback = bool(fallback)

    def warmup(self) -> None:
        self.detector.warmup()

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        dets = self.detector.predict(image)
        pairs = [(d, d.bbox_xyxy) for d in dets]
        pairs = [(d, b) for d, b in pairs if b is not None]
        if not pairs:
            return []
        self.prompter.set_image(image)
        masks, ious = self.prompter.masks_for_boxes(np.array([b for _, b in pairs], np.float32))
        out = []
        for (d, b), m, iou in zip(pairs, masks, ious):
            meta = {"detector_score": d.score, "sam_iou": float(iou), "prompt_box": list(b)}
            if m.any():
                out.append(Prediction(mask=m, origin=(0, 0), score=d.score, meta=meta).cropped())
            elif self.fallback:
                out.append(Prediction(mask=d.mask, origin=d.origin, score=d.score,
                                      polygon=d.polygon, meta={**meta, "fallback": True}))
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "detector": self.detector.describe,
                "prompter": getattr(self.prompter, "describe", {}), "fallback": self.fallback}


@register("model", "refine")
class RefineModel(SegmentationModel):
    """Model gốc + bộ tinh chỉnh biên cắm thêm. Hàng "+X" của bảng benchmark."""

    def __init__(self, base: dict | SegmentationModel, refiner: dict | object):
        self.base = build_model(base)
        self.refiner = _build_refiner(refiner)
        # Model gốc cần gợi ý (oracle) thì model ghép cũng cần.
        self.needs_prompt = bool(self.base.needs_prompt)

    def warmup(self) -> None:
        self.base.warmup()

    def set_context(self, record) -> None:
        """Chuyển ngữ cảnh ảnh (bản ghi COCO) xuống model gốc nếu nó cần —
        CocoPredictionsModel tra dự đoán theo ảnh."""
        fn = getattr(self.base, "set_context", None)
        if fn is not None:
            fn(record)

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        return self.refiner.refine(image, self.base.predict(image, boxes))

    @property
    def describe(self) -> dict:
        return {**super().describe, "base": self.base.describe,
                "refiner": getattr(self.refiner, "describe", {})}
