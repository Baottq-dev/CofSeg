"""File COCO results -> model: mặt nạ phải quay về y hệt, khớp ảnh theo id
hoặc theo tên file, đọc được cả RLE lẫn polygon.
"""

import json

import numpy as np
import pytest

from canopyseg.datasets import CocoDataset
from canopyseg.evaluation import coco_eval, evaluate_split
from canopyseg.models import build_model
from canopyseg.models.base import Prediction
from canopyseg.models.coco_predictions import CocoPredictionsModel
from tests.conftest import IMG_H, IMG_W


def _square(x, y, s, score=0.9):
    m = np.zeros((IMG_H, IMG_W), bool)
    m[y : y + s, x : x + s] = True
    return Prediction(mask=m, score=score)


def _record(ds, i):
    return ds.image_list()[i]


def test_roundtrip_through_file(tiny_dataset, tmp_path):
    ds = CocoDataset(tiny_dataset, "train")
    preds = [_square(20, 20, 60, 0.9), _square(150, 100, 80, 0.4)]
    dets = coco_eval.predictions_to_coco(1, preds, 1, size=(IMG_H, IMG_W))
    f = tmp_path / "preds.json"
    f.write_text(json.dumps(dets), encoding="utf-8")

    m = CocoPredictionsModel(str(f))
    assert m.key == "id" and m.n_detections == 2
    m.set_context(_record(ds, 0))
    out = m.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8))
    assert [p.score for p in out] == [0.9, 0.4]              # sắp theo score giảm dần
    for p, src in zip(out, preds):
        full = coco_eval.full_frame(p, (IMG_H, IMG_W))
        np.testing.assert_array_equal(full, src.mask)
    m.set_context(_record(ds, 1))
    assert m.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8)) == []


def test_conf_and_max_det_filter(tiny_dataset, tmp_path):
    ds = CocoDataset(tiny_dataset, "train")
    dets = coco_eval.predictions_to_coco(
        1, [_square(20, 20, 60, 0.9), _square(150, 100, 80, 0.4), _square(0, 0, 10, 0.1)], 1)
    f = tmp_path / "p.json"
    f.write_text(json.dumps(dets), encoding="utf-8")
    m = CocoPredictionsModel(str(f), conf=0.3)
    m.set_context(_record(ds, 0))
    assert len(m.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8))) == 2
    m = CocoPredictionsModel(str(f), max_det=1)
    m.set_context(_record(ds, 0))
    assert len(m.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8))) == 1


def test_stem_keys_like_ultralytics(tiny_dataset, tmp_path):
    ds = CocoDataset(tiny_dataset, "train")
    rec = _record(ds, 1)
    stem = rec.file_name[:-4]
    dets = [{"image_id": stem, "category_id": 0, "score": 0.7,
             "segmentation": [[40, 40, 140, 40, 140, 140, 40, 140]]}]   # polygon
    f = tmp_path / "ul.json"
    f.write_text(json.dumps(dets), encoding="utf-8")
    m = CocoPredictionsModel(str(f))
    assert m.key == "stem"
    m.set_context(rec)
    out = m.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8))
    assert len(out) == 1 and out[0].origin == (40, 40) and out[0].mask.shape == (101, 101)


def test_predict_without_context_fails_loudly(tmp_path):
    f = tmp_path / "e.json"
    f.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="set_context"):
        CocoPredictionsModel(str(f)).predict(np.zeros((IMG_H, IMG_W, 3), np.uint8))


def test_scores_identically_to_the_live_model(tiny_dataset, tmp_path):
    """Đường vòng qua file phải cho cùng hàng chấm như model chạy trực tiếp."""
    from tests.test_runner import SquaresModel

    ds = CocoDataset(tiny_dataset, "train")
    rows_live, info = evaluate_split(SquaresModel(), ds, progress=False)
    f = tmp_path / "live.json"
    f.write_text(json.dumps(info["detections"]), encoding="utf-8")
    m = build_model({"name": "coco_predictions", "file": str(f)})
    rows_file, info2 = evaluate_split(m, ds, progress=False)
    strip = lambda rs: [{k: v for k, v in r.items()} for r in rs]  # noqa: E731
    assert strip(rows_live) == strip(rows_file)
    assert [p["n_pred"] for p in info["images"]] == [p["n_pred"] for p in info2["images"]]
