"""Bảng model x ruộng từ các lần chấm leave-one-field-out.

Điều phải giữ: Δ% mAP so với mốc tính trên CÙNG ruộng; ruộng thiếu mốc thì
ô trống chứ không gãy; trung bình nhóm chỉ tính khi model có đủ mọi ruộng
của nhóm; cùng khoá chấm hai lần thì lấy lần mới VÀ báo ra.

Khoá gồm cả BỘ FOLD: hai cách chia val cho hai bộ fold cùng tên f1..f6, nên
thiếu chiều đó thì hai thí nghiệm khác nhau đè lên nhau mà không ai biết.
"""

from __future__ import annotations

import json

import pytest
import yaml

from canopyseg.evaluation import folds as rep


def _run(root, ts, name, mAP, bap=0.5, biou=0.7, area=3.0, split="test", images=10,
         data_root=None):
    d = root / f"{ts}_{name}_{split}"
    d.mkdir(parents=True)
    cfg = {"name": name}
    if data_root:
        cfg["data"] = {"root": data_root}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
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
        "folds": {"f1": {"test": ["field_1"]},
                  "f2": {"test": ["field_2"]},
                  "f3": {"test": ["field_3"]}},
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
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith(
        "dataset,field,fold,model")


def test_collect_merges_several_eval_dirs(tmp_path, folds_yaml):
    """Mỗi thành viên ghi vào runs/eval của thư mục mình, nên bảng chung phải
    quét nhiều gốc; trùng (model, fold) giữa hai gốc thì lấy lần mới hơn."""
    a, b = tmp_path / "a", tmp_path / "b"
    _run(a, "2026-09-30_100000", "maskrcnn_f1", 0.50)
    _run(b, "2026-09-30_100100", "cascade_f1", 0.55)
    _run(b, "2026-09-30_110000", "maskrcnn_f1", 0.60)      # mới hơn, đè bản ở a

    rows = rep.collect([a, b], folds_yaml)
    by = {r["model"]: r["mAP"] for r in rows}
    assert by == {"maskrcnn": 0.6, "cascade": 0.55}
    assert rep.collect(a, folds_yaml)[0]["mAP"] == 0.5     # một gốc vẫn chạy như cũ


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


# ------------------------------------------------------------- bộ fold
def test_two_fold_sets_do_not_overwrite_each_other(tmp_path, folds_yaml):
    """Chấm cùng một model trên hai bộ fold: hai hàng, không phải một.

    Trước đây khoá chỉ là (model, fold) nên `maskrcnn_f1` của bộ block và của
    bộ flight trùng khoá, cái chạy sau lặng lẽ thắng — chạy xong 48 lượt mới
    phát hiện mất một nửa bảng.
    """
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50, data_root="data/export/block/f1")
    _run(ev, "2026-09-30_110000", "maskrcnn_f1", 0.42, data_root="data/export/flight/f1")

    rows = rep.collect(ev, folds_yaml, warn=None)
    assert len(rows) == 2
    got = {r["dataset"]: r["mAP"] for r in rows}
    assert got == {"block": 0.50, "flight": 0.42}


def test_a_real_clash_is_reported_not_swallowed(tmp_path, folds_yaml):
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50, data_root="data/export/block/f1")
    _run(ev, "2026-09-30_110000", "maskrcnn_f1", 0.42, data_root="data/export/block/f1")

    said: list[str] = []
    rows = rep.collect(ev, folds_yaml, warn=said.append)
    assert len(rows) == 1 and rows[0]["mAP"] == 0.42        # vẫn lấy lần mới
    assert len(said) == 1 and "block" in said[0] and "f1" in said[0]


def test_dataset_filter_keeps_one_set(tmp_path, folds_yaml):
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50, data_root="data/export/block/f1")
    _run(ev, "2026-09-30_110000", "maskrcnn_f1", 0.42, data_root="data/export/flight/f1")

    rows = rep.collect(ev, folds_yaml, dataset="flight", warn=None)
    assert [r["mAP"] for r in rows] == [0.42]


def test_the_baseline_delta_stays_inside_its_own_fold_set(tmp_path, folds_yaml):
    """Δ% so với mốc phải so trong cùng bộ fold. So chéo là so hai thí nghiệm."""
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50, data_root="data/export/block/f1")
    _run(ev, "2026-09-30_100100", "solov2_f1", 0.55, data_root="data/export/block/f1")
    _run(ev, "2026-09-30_110000", "maskrcnn_f1", 0.40, data_root="data/export/flight/f1")
    _run(ev, "2026-09-30_110100", "solov2_f1", 0.44, data_root="data/export/flight/f1")

    rows = rep.add_reference_delta(rep.collect(ev, folds_yaml, warn=None), "maskrcnn")
    d = {(r["dataset"], r["model"]): r["dmAP_pct"] for r in rows}
    assert d[("block", "solov2")] == pytest.approx(10.0)    # 0.55 vs 0.50
    assert d[("flight", "solov2")] == pytest.approx(10.0)   # 0.44 vs 0.40, KHÔNG phải vs 0.50
    assert d[("block", "maskrcnn")] == 0.0 and d[("flight", "maskrcnn")] == 0.0


def test_group_means_never_mix_two_fold_sets(tmp_path, folds_yaml):
    ev = tmp_path / "eval"
    for ds, base in (("block", 0.50), ("flight", 0.30)):
        for fold, field in (("f1", "field_1"), ("f2", "field_2")):
            _run(ev, f"2026-09-30_1{'01' if ds == 'block' else '11'}0{fold[-1]}00",
                 f"maskrcnn_{fold}", base, data_root=f"data/export/{ds}/{fold}")
    rows = rep.add_reference_delta(rep.collect(ev, folds_yaml, warn=None))
    groups = rep.group_means(rows, ["field_1", "field_2"], "TB nội suy")
    means = {g["dataset"]: g["mAP"] for g in groups}
    assert means == {"block": 0.5, "flight": 0.3}           # không phải 0.4 gộp chung


def test_a_run_without_a_data_root_still_lands_in_the_table(tmp_path, folds_yaml):
    """Lần chấm cũ không ghi data.root: vẫn phải vào bảng, bộ fold để trống."""
    ev = tmp_path / "eval"
    _run(ev, "2026-09-30_100000", "maskrcnn_f1", 0.50)
    rows = rep.collect(ev, folds_yaml, warn=None)
    assert len(rows) == 1 and rows[0]["dataset"] == ""
