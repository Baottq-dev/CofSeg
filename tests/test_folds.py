"""Cắt fold theo ruộng từ bản xuất chưa chia.

Điều phải giữ: id ảnh/annotation không đổi (file dự đoán từ máy khác khớp
theo id), ảnh nền đi theo đúng tập với nhãn rỗng, và một fold cấu hình sai
phải gãy trước khi ghi bất cứ thứ gì.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
import yaml

from canopyseg.datasets import CocoDataset
from canopyseg.datasets import folds as foldmod

FIELDS = ["field_1", "field_2", "field_3"]
W, H = 64, 48


def _square(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


@pytest.fixture
def export(tmp_path):
    """Bản xuất split_by=none: 3 ruộng x 2 ảnh, ảnh thứ hai của field_3 là nền."""
    import cv2

    root = tmp_path / "all"
    (root / "annotations").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "labels").mkdir()
    images, anns = [], []
    iid = aid = 0
    for f in FIELDS:
        for k in ("a", "b"):
            iid += 1
            name = f"{f}__10__1__{k}.jpg"
            images.append({"id": iid, "file_name": name, "width": W, "height": H})
            cv2.imwrite(str(root / "images" / name), np.zeros((H, W, 3), np.uint8))
            lines = []
            if not (f == "field_3" and k == "b"):
                for j in range(2):
                    aid += 1
                    poly = _square(5 + 20 * j, 5, 12)
                    anns.append({"id": aid, "image_id": iid, "category_id": 1,
                                 "segmentation": [poly], "area": 144.0,
                                 "bbox": [poly[0], poly[1], 12, 12], "iscrowd": 0})
                    lines.append("0 " + " ".join("%.6f" % (v / (W if i % 2 == 0 else H))
                                                 for i, v in enumerate(poly)))
            (root / "labels" / (name[:-4] + ".txt")).write_text("\n".join(lines))
    doc = {"info": {"description": "test"}, "licenses": [], "images": images,
           "annotations": anns,
           "categories": [{"id": 1, "name": "canopy", "supercategory": "canopy"}]}
    (root / "annotations" / "instances.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


@pytest.fixture
def folds_yaml(tmp_path):
    p = tmp_path / "folds.yaml"
    p.write_text(yaml.safe_dump({
        "fields": FIELDS,
        "folds": {"f3": {"test": ["field_3"], "val": ["field_2"]},
                  "f1": {"test": ["field_1"], "val": ["field_2"]}},
    }))
    return p


def test_fold_keeps_ids_and_routes_background_images(export, folds_yaml, tmp_path):
    doc = foldmod.load_folds(folds_yaml)
    fields = foldmod.fold_fields(doc, "f3")
    assert fields == {"train": ["field_1"], "val": ["field_2"], "test": ["field_3"]}

    out = tmp_path / "f3"
    s = foldmod.make_fold(export, "f3", fields, out)
    assert s["splits"]["test"] == {"fields": ["field_3"], "images": 2,
                                   "annotations": 2, "empty_images": 1}
    assert s["splits"]["train"]["annotations"] == 4 and s["images_skipped"] == []

    test = json.loads((out / "annotations" / "instances_test.json").read_text(encoding="utf-8"))
    assert [im["id"] for im in test["images"]] == [5, 6]          # id gốc, không đánh lại
    assert [a["image_id"] for a in test["annotations"]] == [5, 5]
    assert test["categories"][0]["id"] == 1 and test["info"]["fold"] == "f3"

    # Bố cục đọc được bằng chính bộ đọc của repo; ảnh nền còn nguyên trong tập.
    ds = CocoDataset(out, "test")
    assert len(ds) == 2 and ds.n_regions == 2
    assert (out / "images" / "test" / "field_3__10__1__b.jpg").exists()
    assert (out / "labels" / "test" / "field_3__10__1__b.txt").read_text() == ""
    assert (out / "labels" / "train" / "field_1__10__1__a.txt").read_text().startswith("0 ")

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "images/train" and data["test"] == "images/test"
    assert data["names"] == {0: "canopy"}

    fold = json.loads((out / "fold.json").read_text(encoding="utf-8"))
    assert fold["source_sha1"] and fold["images_linked"] + fold["images_copied"] == 6


def test_images_are_hardlinked_not_copied(export, folds_yaml, tmp_path):
    doc = foldmod.load_folds(folds_yaml)
    out = tmp_path / "f1"
    s = foldmod.make_fold(export, "f1", foldmod.fold_fields(doc, "f1"), out)
    src = export / "images" / "field_1__10__1__a.jpg"
    dst = out / "images" / "test" / "field_1__10__1__a.jpg"
    # Cùng ổ đĩa (tmp_path) nên phải là hardlink: cùng inode, không thêm byte.
    assert s["images_linked"] == 6 and s["images_copied"] == 0
    assert os.stat(src).st_ino == os.stat(dst).st_ino


def test_copy_flag_and_refusal_to_overwrite(export, folds_yaml, tmp_path):
    doc = foldmod.load_folds(folds_yaml)
    out = tmp_path / "f1"
    s = foldmod.make_fold(export, "f1", foldmod.fold_fields(doc, "f1"), out, copy=True)
    assert s["images_copied"] == 6
    with pytest.raises(FileExistsError):
        foldmod.make_fold(export, "f1", foldmod.fold_fields(doc, "f1"), out)


def test_labels_are_generated_when_the_export_has_none(export, folds_yaml, tmp_path):
    """Bản xuất chỉ có COCO (quên tick ô YOLO) vẫn phải ra fold dùng được.

    Trước đây thiếu labels/ là fold ra file .txt RỖNG: ultralytics đọc được,
    không báo gì, và YOLO train trên không có nhãn nào suốt cả đêm.
    """
    from canopyseg.datasets.yolo import verify_roundtrip

    shutil.rmtree(export / "labels")
    out = tmp_path / "f1"
    s = foldmod.make_fold(export, "f1", foldmod.fold_fields(foldmod.load_folds(folds_yaml), "f1"), out)

    assert s["labels"] == "generate"
    assert sum(v["labels"] for v in s["labels_generated"].values()) == 10   # 5 ảnh có vùng x 2
    assert (out / "data.yaml").exists()
    # Nhãn sinh ra phải khớp polygon trong COCO của chính fold đó.
    r = verify_roundtrip(out, list(foldmod.SPLITS))
    assert r["ok"] and all(v["problems"] == [] for v in r["splits"].values())
    # Ảnh nền vẫn có file rỗng, không phải thiếu file.
    bg = out / "labels" / "train" / "field_3__10__1__b.txt"
    assert bg.exists() and bg.read_text(encoding="utf-8") == ""


def test_label_mode_copy_needs_an_export_that_has_them(export, folds_yaml, tmp_path):
    doc = foldmod.load_folds(folds_yaml)
    fields = foldmod.fold_fields(doc, "f1")
    # Bản xuất CÓ labels/ -> auto chép, giữ nguyên byte của bản xuất.
    s = foldmod.make_fold(export, "f1", fields, tmp_path / "copied")
    assert s["labels"] == "copy" and "labels_generated" not in s
    assert (tmp_path / "copied" / "data.yaml").exists()

    # generate ép sinh lại dù bản xuất có sẵn (dùng khi nghi labels/ đã cũ).
    s = foldmod.make_fold(export, "f1", fields, tmp_path / "gen", labels="generate")
    assert s["labels"] == "generate"

    shutil.rmtree(export / "labels")
    with pytest.raises(FileNotFoundError, match="labels"):
        foldmod.make_fold(export, "f1", fields, tmp_path / "must_copy", labels="copy")
    with pytest.raises(ValueError, match="labels="):
        foldmod.make_fold(export, "f1", fields, tmp_path / "bad", labels="khong_co")


def test_bad_fold_configs_fail_before_writing(tmp_path, export):
    def cfg(folds):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"fields": FIELDS, "folds": folds}))
        return p

    with pytest.raises(ValueError, match="không có trong"):
        foldmod.load_folds(cfg({"x": {"test": ["field_9"], "val": ["field_2"]}}))
    with pytest.raises(ValueError, match="vừa test vừa val"):
        foldmod.load_folds(cfg({"x": {"test": ["field_1"], "val": ["field_1"]}}))
    with pytest.raises(ValueError, match="train"):
        foldmod.load_folds(cfg({"x": {"test": ["field_1", "field_3"], "val": ["field_2"]}}))
    doc = foldmod.load_folds(cfg({"x": {"test": ["field_1"], "val": ["field_2"]}}))
    with pytest.raises(KeyError):
        foldmod.fold_fields(doc, "nope")

    # Ruộng có trong folds.yaml nhưng chưa có ảnh nào trong bản xuất -> tập rỗng.
    out = tmp_path / "empty"
    with pytest.raises(ValueError, match="không có ảnh"):
        foldmod.make_fold(export, "x", {"train": ["field_1"], "val": ["field_2"],
                                         "test": ["field_4"]}, out)
    assert not out.exists() or not any((out / "images").rglob("*.jpg"))


def test_split_export_is_rejected(tmp_path):
    root = tmp_path / "split"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances_train.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="CHƯA chia"):
        foldmod.make_fold(root, "f1", {"train": ["field_1"], "val": ["field_2"],
                                        "test": ["field_3"]}, tmp_path / "out")


def test_shipped_folds_yaml_is_consistent():
    doc = foldmod.load_folds("configs/dataset/folds.yaml")
    tests = [f for name in doc["folds"] for f in doc["folds"][name]["test"]]
    # Leave-one-field-out: mỗi ruộng làm test đúng một lần, val không bao giờ
    # là ruộng ngoại suy.
    assert sorted(tests) == sorted(doc["fields"])
    for name, spec in doc["folds"].items():
        assert not set(spec["val"]) & set(doc["extrapolation"]), name
    assert set(doc["screening"]) <= set(doc["folds"])
