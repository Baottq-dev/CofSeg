"""Mask AP và Boundary AP trên hình dựng sẵn có đáp án biết trước.

Điểm cần chứng minh, vì nó là lý do Boundary AP tồn tại trong dự án: mặt nạ
co vào vài px thì Mask AP gần như không đổi còn Boundary AP tụt hẳn. Nếu hai
số đó phản ứng giống nhau thì bộ chấm biên không đo được thứ nó phải đo.
"""

import json

import numpy as np
import pytest

from canopyseg.evaluation import coco_eval
from canopyseg.models.base import Prediction

W, H = 640, 480


def _coco_gt(tmp_path, squares):
    images = [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}]
    anns = []
    for i, (x, y, s) in enumerate(squares, 1):
        anns.append({"id": i, "image_id": 1, "category_id": 1, "iscrowd": 0,
                     "area": float(s * s), "bbox": [x, y, s, s],
                     "segmentation": [[x, y, x + s, y, x + s, y + s, x, y + s]]})
    doc = {"images": images, "annotations": anns,
           "categories": [{"id": 1, "name": "canopy"}]}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _pred(x, y, s, score=0.9):
    m = np.zeros((H, W), bool)
    m[y : y + s, x : x + s] = True
    return Prediction(mask=m, origin=(0, 0), score=score)


SQUARES = [(20, 20, 200), (300, 100, 240), (60, 260, 180)]


def test_perfect_predictions_score_one(tmp_path):
    gt = _coco_gt(tmp_path, SQUARES)
    dets = coco_eval.predictions_to_coco(1, [_pred(*s) for s in SQUARES], 1)
    res = coco_eval.evaluate(gt, dets, [1])
    assert res["mask"]["AP"] == pytest.approx(1.0, abs=1e-3)
    assert res["boundary"]["AP"] == pytest.approx(1.0, abs=1e-3)
    assert "Average Precision" in res["mask_text"] and "Average Precision" in res["boundary_text"]


def test_shrunk_masks_keep_mask_ap_but_lose_boundary_ap(tmp_path):
    """Co 6 px mỗi phía trên tán ~200 px: IoU vẫn ~0.88, vành biên thì hỏng."""
    gt = _coco_gt(tmp_path, SQUARES)
    dets = coco_eval.predictions_to_coco(1, [_pred(x + 6, y + 6, s - 12) for x, y, s in SQUARES], 1)
    res = coco_eval.evaluate(gt, dets, [1])
    assert res["mask"]["AP50"] == pytest.approx(1.0, abs=1e-3)
    assert res["mask"]["AP"] > 0.7
    assert res["boundary"]["AP"] < res["mask"]["AP"] - 0.2


def test_boundary_can_be_skipped(tmp_path):
    gt = _coco_gt(tmp_path, SQUARES)
    dets = coco_eval.predictions_to_coco(1, [_pred(*s) for s in SQUARES], 1)
    res = coco_eval.evaluate(gt, dets, [1], boundary=False)
    assert res["mask"]["AP"] == pytest.approx(1.0, abs=1e-3)
    assert res["boundary"] is None and res["boundary_text"] is None


def test_no_detections_is_reported_not_raised(tmp_path):
    gt = _coco_gt(tmp_path, SQUARES)
    res = coco_eval.evaluate(gt, [], [1])
    assert res["mask"] is None and "error" in res


def test_local_frame_masks_are_pasted_back():
    p = Prediction(mask=np.ones((10, 20), bool), origin=(100, 50), score=0.5)
    full = coco_eval.full_frame(p, (H, W))
    assert full.shape == (H, W) and full.sum() == 200
    assert full[50:60, 100:120].all() and not full[49, 100] and not full[50, 120]
    det = coco_eval.predictions_to_coco(7, [p], 1, size=(H, W))[0]
    assert det["image_id"] == 7 and det["score"] == 0.5
    from pycocotools import mask as mask_utils

    assert mask_utils.area(det["segmentation"]) == 200


def test_local_frame_mask_is_clipped_at_the_border():
    p = Prediction(mask=np.ones((30, 30), bool), origin=(W - 10, H - 5), score=1.0)
    full = coco_eval.full_frame(p, (H, W))
    assert full.sum() == 50


def test_empty_masks_are_dropped():
    assert coco_eval.predictions_to_coco(1, [Prediction(mask=np.zeros((H, W), bool))]) == []
