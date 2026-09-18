"""Vòng chấm chung: mỗi vùng thật ra đúng một hàng, dự đoán thừa ra hàng
riêng, model cần gợi ý nhận box thật, model đọc file được báo ảnh nào.

Model giả trả hình vuông tại chỗ chỉ định, nên đáp án biết trước: khớp, sót,
thừa đếm được bằng tay.
"""

import numpy as np
import pytest

from canopyseg.datasets import CocoDataset
from canopyseg.evaluation import evaluate_split, summarize
from canopyseg.models.base import Prediction, SegmentationModel
from tests.conftest import IMG_H, IMG_W


class SquaresModel(SegmentationModel):
    """Với ảnh đầu: khớp cả hai vùng + một dự đoán thừa. Các ảnh khác: rỗng."""

    needs_prompt = False

    def __init__(self):
        self.calls = []

    def predict(self, image, boxes=None):
        self.calls.append(boxes)
        if len(self.calls) > 1:
            return []
        out = []
        for x, y, s, sc in ((20, 20, 60, 0.9), (150, 100, 80, 0.8), (300, 250, 30, 0.3)):
            m = np.zeros((IMG_H, IMG_W), bool)
            m[y : y + s, x : x + s] = True
            out.append(Prediction(mask=m, score=sc))
        return out


class PromptedModel(SquaresModel):
    needs_prompt = True


class ContextModel(SquaresModel):
    def __init__(self):
        super().__init__()
        self.records = []

    def set_context(self, record):
        self.records.append(record.file_name)


def test_rows_cover_every_region_and_extra_prediction(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    rows, info = evaluate_split(SquaresModel(), ds, progress=False)
    gt_rows = [r for r in rows if r["matched"] in (0, 1)]
    assert len(gt_rows) == ds.n_regions == 3
    assert sum(r["matched"] == 1 for r in gt_rows) == 2      # hai vùng ảnh đầu khớp
    assert sum(r["matched"] == -1 for r in rows) == 1        # một dự đoán thừa
    matched = [r for r in rows if r["matched"] == 1]
    for r in matched:
        # Hình vuông vẽ hai đường (fillPoly bao đỉnh, mặt nạ cắt lát) lệch nhau
        # 1 px; vành 2% cạnh 60 px chỉ rộng 1 px nên Boundary IoU thấp là đúng.
        assert r["iou"] > 0.9 and 0.0 < r["boundary_iou"] <= 1.0
        assert abs(r["area_error_pct"]) < 5.0 and abs(r["signed_median"]) <= 1.0
    missed = [r for r in gt_rows if r["matched"] == 0][0]
    assert missed["iou"] == "" and missed["gt_area"] == 10000.0
    assert len(info["images"]) == 3 and info["images"][0]["n_pred"] == 3
    assert info["images"][0]["count_error"] == 1 and info["images"][2]["n_gt"] == 0
    assert len(info["detections"]) == 3 and info["image_ids"] == [1, 2, 3]
    assert all("ms" in p for p in info["images"])


def test_summary_counts(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    rows, info = evaluate_split(SquaresModel(), ds, progress=False)
    s = summarize(rows, info["images"])
    assert s["n_gt"] == 3 and s["true_positive"] == 2
    assert s["false_negative"] == 1 and s["false_positive"] == 1
    assert s["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["median_iou"] > 0.9 and "by_field" in s
    assert s["by_field"]["field_1"]["recall"] == 1.0 and s["by_field"]["field_2"]["recall"] == 0.0


def test_limit_restricts_images(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    rows, info = evaluate_split(SquaresModel(), ds, limit=1, progress=False)
    assert info["image_ids"] == [1] and len(info["images"]) == 1


def test_prompted_model_receives_ground_truth_boxes(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    m = PromptedModel()
    evaluate_split(m, ds, progress=False)
    first = m.calls[0]
    assert first.shape == (2, 4) and first.dtype == np.float32
    np.testing.assert_allclose(first[0], [20, 20, 80, 80])
    assert m.calls[2].shape == (0, 4)                 # ảnh không vùng: mảng rỗng, không None


def test_unprompted_model_receives_none(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    m = SquaresModel()
    evaluate_split(m, ds, progress=False)
    assert all(b is None for b in m.calls)


def test_set_context_is_called_per_image(tiny_dataset):
    ds = CocoDataset(tiny_dataset, "train")
    m = ContextModel()
    evaluate_split(m, ds, progress=False)
    assert m.records == [im.file_name for im in ds.image_list()]
