"""Bảng model x ruộng từ các lần chấm leave-one-field-out.

Điều phải giữ: Δ% mAP so với mốc tính trên CÙNG ruộng; ruộng thiếu mốc thì
ô trống chứ không gãy; trung bình nhóm chỉ tính khi model có đủ mọi ruộng
của nhóm; cùng (model, fold) chấm hai lần thì lấy lần mới.
"""

from __future__ import annotations

import json

import pytest
import yaml

from canopyseg.evaluation import folds as rep


def _run(root, ts, name, mAP, bap=0.5, biou=0.7, area=3.0, split="test", images=10):
    d = root / f"{ts}_{name}_{split}"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"name": name}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps({
        "data": {"split": split, "images_scored": images},
        "coco": {"mask": {"AP": mAP, "AP50": mAP + 0.2, "AP75": mAP},
                 "boundary": {"AP": bap}},
        "summary": {"mean_boundary_iou": biou, "mean_area_error_pct": area,
                    "recall": 0.9, "precision": 0.8, "ms_per_image": 120.4},
    }), encoding="utf-8")
    return d


@pytest.fixture
def folds_yaml(tmp_path):
    p = tmp_path / "folds.yaml"
    p.write_text(yaml.safe_dump({
        "fields": ["field_1", "field_2", "field_3"],
        "folds": {"f1": {"test": ["field_1"], "val": ["field_2"]},
                  "f2": {"test": ["field_2"], "val": ["field_1"]},
                  "f3": {"test": ["field_3"], "val": ["field_2"]}},
        "interpolation": ["field_1", "field_2"],
        "extrapolation": ["field_3"],
    }))
    return p


def test_delta_is_per_field_and_missing_reference_leaves_a_blank(tmp_path, folds_yaml):
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50)
    _run(ev, "2026-09-30_100100", "cascade_f1", 0.55)
    _run(ev, "2026-09-30_100200", "maskrcnn_f2", 0.40)
    _run(ev, "2026-09-30_100300", "cascade_f2", 0.42)
    _run(ev, "2026-09-30_100400", "cascade_f3", 0.30)          # f3 không có mốc
    _run(ev, "2026-09-30_090000", "maskrcnn_f1", 0.10)         # lần cũ, phải bị đè
    _run(ev, "2026-09-30_100500", "maskrcnn_f1", 0.50, split="val")  # không phải test
    (ev / "rac").mkdir()                                       # thư mục không phải lần chấm

    rows, groups, md = rep.build_report(ev, folds_yaml)
    by = {(r["model"], r["field"]): r for r in rows}
    assert set(by) == {("maskrcnn", "field_1"), ("cascade", "field_1"),
                       ("maskrcnn", "field_2"), ("cascade", "field_2"), ("cascade", "field_3")}
    assert by[("maskrcnn", "field_1")]["mAP"] == 0.5 and by[("maskrcnn", "field_1")]["dmAP_pct"] == 0.0
    assert by[("cascade", "field_1")]["dmAP_pct"] == 10.0
    assert by[("cascade", "field_2")]["dmAP_pct"] == 5.0
    assert by[("cascade", "field_3")]["dmAP_pct"] is None
    assert by[("cascade", "field_1")]["ms_img"] == 120 and by[("cascade", "field_1")]["BIoU"] == 0.7

    # Nhóm nội suy: cả hai model có đủ field_1, field_2; ngoại suy chỉ cascade.
    g = {(r["model"], r["field"]): r for r in groups}
    assert g[("cascade", "TB nội suy")]["mAP"] == pytest.approx(0.485)
    assert g[("cascade", "TB nội suy")]["dmAP_pct"] == pytest.approx(7.5)
    assert g[("maskrcnn", "TB nội suy")]["dmAP_pct"] == 0.0
    assert ("maskrcnn", "TB ngoại suy") not in g and g[("cascade", "TB ngoại suy")]["mAP"] == 0.3

    assert "| field_3 | cascade |" in md and "—" in md          # ô trống là gạch
    out = rep.write_csv(rows + groups, tmp_path / "t.csv")
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("field,fold,model")


def test_run_names():
    assert rep.parse_run_name("mask2former_f4") == ("mask2former", "f4")
    assert rep.parse_run_name("yolo11s_f12") == ("yolo11s", "f12")
    assert rep.parse_run_name("maskrcnn-r50-d2") is None
    assert rep.parse_run_name("f4") is None


def test_older_runs_are_recognised_from_the_directory_name(tmp_path, folds_yaml):
    ev = tmp_path / "eval"
    d = _run(ev, "2026-09-21_090329", "maskrcnn_f4", 0.5)
    (d / "config.yaml").write_text(yaml.safe_dump({"name": "coco-predictions"}), encoding="utf-8")
    # fold f4 không có trong folds.yaml của test -> bị bỏ; thêm fold rồi thử lại.
    assert rep.collect(ev, folds_yaml) == []
    doc = yaml.safe_load(folds_yaml.read_text())
    doc["folds"]["f4"] = {"test": ["field_3"], "val": ["field_2"]}
    folds_yaml.write_text(yaml.safe_dump(doc))
    rows = rep.collect(ev, folds_yaml)
    assert [(r["model"], r["fold"], r["field"]) for r in rows] == [("maskrcnn", "f4", "field_3")]
