"""Ghép model: two_stage, refine, build_model — không cần SAM hay GPU.

Model giả và backend giả thay cho YOLO/SAM: cái cần chứng minh là phần ghép
giữ đúng số dự đoán, score, origin và chuyển gợi ý đúng chỗ, chứ không phải
SAM vẽ đẹp đến đâu (việc đó là của tests/test_sam2_real.py, cần trọng số).
"""

import numpy as np
import pytest

from canopyseg.models import build_model, model_param_names
from canopyseg.models.base import Prediction, SegmentationModel
from canopyseg.models.compose import RefineModel, TwoStageModel
from canopyseg.registry import register

H, W = 60, 80


def square(x0, y0, size, score=0.9, full=True):
    if full:
        m = np.zeros((H, W), bool)
        m[y0 : y0 + size, x0 : x0 + size] = True
        return Prediction(mask=m, origin=(0, 0), score=score)
    return Prediction(mask=np.ones((size, size), bool), origin=(x0, y0), score=score)


@register("model", "_test_fixed")
class FixedModel(SegmentationModel):
    """Trả đúng danh sách được cấu hình, bỏ qua ảnh."""

    def __init__(self, boxes=(), scores=(), needs_prompt=False):
        self.boxes, self.scores = list(boxes), list(scores)
        self.needs_prompt = needs_prompt
        self.seen_prompt = None

    def predict(self, image, boxes=None):
        self.seen_prompt = boxes
        return [square(x, y, s, sc) for (x, y, s), sc in zip(self.boxes, self.scores)]


class FakeBackend:
    """Trả mặt nạ = box nới rộng 1 px, IoU = 0.5; nhớ ảnh và box đã nhận."""

    def __init__(self, empty_for=()):
        self.image = None
        self.boxes = None
        self.empty_for = set(empty_for)

    def set_image(self, image):
        self.image = image

    def masks_for_boxes(self, boxes, mask_prompts=None):
        self.boxes = np.asarray(boxes)
        self.mask_prompts = mask_prompts
        out = np.zeros((len(boxes), H, W), bool)
        for i, (x0, y0, x1, y1) in enumerate(self.boxes.astype(int)):
            if i in self.empty_for:
                continue
            out[i, max(0, y0 - 1) : y1 + 2, max(0, x0 - 1) : x1 + 2] = True
        return out, np.full(len(boxes), 0.5, np.float32)


class FakeRefiner:
    def __init__(self):
        self.calls = 0

    def refine(self, image, preds):
        self.calls += 1
        return [Prediction(mask=p.mask, origin=p.origin, score=p.score, meta={"refined": True})
                for p in preds]


# ----------------------------------------------------------------- Prediction
def test_bbox_and_crop_are_consistent():
    p = square(10, 5, 4)
    assert p.bbox_xyxy == (10, 5, 13, 8)
    c = p.cropped()
    assert c.origin == (10, 5) and c.mask.shape == (4, 4) and c.mask.all()
    assert c.bbox_xyxy == p.bbox_xyxy and c.score == p.score
    assert Prediction(mask=np.zeros((3, 3), bool)).bbox_xyxy is None


def test_bbox_is_cached_in_meta():
    p = square(1, 2, 3)
    p.bbox_xyxy
    p.mask[:] = False               # đổi mặt nạ sau khi đã nhớ
    assert p.bbox_xyxy == (1, 2, 3, 4)


# ----------------------------------------------------------------- build
def test_build_model_resolves_name_and_kwargs():
    m = build_model({"name": "_test_fixed", "boxes": [(1, 1, 5)], "scores": [0.7]})
    assert isinstance(m, FixedModel) and m.scores == [0.7]
    assert build_model(m) is m


def test_build_model_requires_name():
    with pytest.raises(ValueError, match="name"):
        build_model({"weights": "x"})


def test_model_param_names_from_signature():
    assert model_param_names("_test_fixed") == {"boxes", "scores", "needs_prompt"}
    assert model_param_names("two_stage") == {"detector", "prompter", "fallback"}


# ----------------------------------------------------------------- two_stage
def test_two_stage_takes_boxes_from_detector_and_masks_from_prompter():
    det = {"name": "_test_fixed", "boxes": [(10, 10, 5), (40, 20, 8)], "scores": [0.9, 0.6]}
    backend = FakeBackend()
    m = TwoStageModel(detector=det, prompter=backend)
    img = np.zeros((H, W, 3), np.uint8)
    out = m.predict(img)
    assert backend.image is img
    np.testing.assert_array_equal(backend.boxes, [[10, 10, 14, 14], [40, 20, 47, 27]])
    assert [p.score for p in out] == [0.9, 0.6]           # score của detector
    assert out[0].meta["sam_iou"] == 0.5 and out[0].meta["detector_score"] == 0.9
    assert out[0].origin == (9, 9) and out[0].mask.shape == (7, 7)   # nới 1 px, đã cắt sát
    assert not m.needs_prompt


def test_two_stage_fallback_keeps_detector_mask_when_prompter_is_empty():
    det = {"name": "_test_fixed", "boxes": [(10, 10, 5), (40, 20, 8)], "scores": [0.9, 0.6]}
    keep = TwoStageModel(detector=det, prompter=FakeBackend(empty_for={1}), fallback=True)
    out = keep.predict(np.zeros((H, W, 3), np.uint8))
    assert len(out) == 2 and out[1].meta.get("fallback") and out[1].mask.shape == (H, W)
    drop = TwoStageModel(detector=det, prompter=FakeBackend(empty_for={1}), fallback=False)
    assert len(drop.predict(np.zeros((H, W, 3), np.uint8))) == 1


def test_two_stage_with_no_detections_returns_empty():
    m = TwoStageModel(detector={"name": "_test_fixed"}, prompter=FakeBackend())
    assert m.predict(np.zeros((H, W, 3), np.uint8)) == []


# ----------------------------------------------------------------- refine
def test_refine_passes_prompt_through_and_keeps_scores():
    base = FixedModel(boxes=[(5, 5, 6)], scores=[0.8], needs_prompt=True)
    ref = FakeRefiner()
    m = RefineModel(base=base, refiner=ref)
    assert m.needs_prompt is True
    gt = np.array([[5, 5, 10, 10]], np.float32)
    out = m.predict(np.zeros((H, W, 3), np.uint8), gt)
    assert base.seen_prompt is gt and ref.calls == 1
    assert len(out) == 1 and out[0].score == 0.8 and out[0].meta["refined"]


def test_refine_builds_nested_specs():
    spec = {"name": "refine", "base": {"name": "_test_fixed", "boxes": [(1, 1, 3)], "scores": [1.0]},
            "refiner": FakeRefiner()}
    m = build_model(spec)
    assert isinstance(m, RefineModel) and isinstance(m.base, FixedModel)
    assert m.describe["base"]["class"] == "FixedModel"
