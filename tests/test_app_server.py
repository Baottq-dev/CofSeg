"""Nhãn rỗng "đã xem, không có cây" ở app: được lưu, được liệt kê là đã gán,
và đi vào bộ xuất như ảnh nền (COCO không annotation, YOLO .txt rỗng, mask 0).

Server đọc configs/config.yaml lúc import nên các đường dẫn được thay bằng
thư mục tạm ngay sau đó; SAM không bị nạp (chỉ nạp lười khi bấm SAM).
"""

import json
import os

import cv2
import numpy as np
import pytest

from app import server

W, H = 96, 64
A, B = "field_9/10/1/a.jpg", "field_9/10/1/b.jpg"


@pytest.fixture
def root(tmp_path, monkeypatch):
    base = tmp_path / "images"
    for name in (A, B):
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((H, W, 3), (40, 120, 60), np.uint8)
        cv2.imwrite(str(p), img)
    monkeypatch.setattr(server, "BASE", str(base))
    monkeypatch.setattr(server, "ROOT", str(base))
    monkeypatch.setattr(server, "OUT_DIR", str(tmp_path / "labels"))
    monkeypatch.setattr(server, "EXPORT_DIR", str(tmp_path / "export"))
    return tmp_path


def _save(name, polygons, **kw):
    return server.save(server.SaveReq(name=name, width=W, height=H,
                                      polygons=polygons, **kw))


SQUARE = [10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]


def test_empty_flag_writes_a_label_file_and_counts_as_labeled(root):
    r = _save(A, [], empty=True)
    assert r["count"] == 0 and r["empty"] is True
    jp = server._json_path(A)
    assert os.path.exists(jp)
    doc = json.load(open(jp, encoding="utf-8"))
    assert doc["polygons"] == [] and doc["n_canopy"] == 0 and doc["empty"] is True

    d = server.load(A)
    assert d["empty"] is True and d["polygons"] == []
    lab = server.labeled()
    assert A in lab["labeled"] and A in lab["empty"] and B not in lab["labeled"]


def test_without_the_flag_an_empty_save_still_unlabels(root):
    _save(A, [], empty=True)
    _save(A, [])                       # "Xoá hết" / bỏ đánh dấu -> chưa gán
    assert not os.path.exists(server._json_path(A))
    lab = server.labeled()
    assert A not in lab["labeled"] and A not in lab["empty"]
    assert server.load(A)["empty"] is False


def test_a_polygon_replaces_the_empty_flag(root):
    _save(A, [], empty=True)
    r = _save(A, [SQUARE])
    assert r["count"] == 1
    d = server.load(A)
    assert d["empty"] is False and len(d["polygons"]) == 1
    lab = server.labeled()
    assert A in lab["labeled"] and A not in lab["empty"]


def test_export_carries_reviewed_empty_images_as_background(root):
    _save(A, [SQUARE])
    _save(B, [], empty=True)

    pv = server.export_preview("")
    assert pv["images_total"] == 2 and pv["images_labeled"] == 2
    assert pv["annotations"] == 1 and pv["images_empty"] == 1

    items = list(server._iter_labeled(""))
    assert [n for n, _ in items] == [A, B]
    out = server._export_dataset(items, server.DatasetReq(
        name="ds", out_dir=str(root / "exp"), split_by="none",
        formats=["coco", "yolo", "masks", "masks_instance"]))
    assert out["ok"] and out["problems"] == []
    assert out["splits"]["all"] == {"images": 2, "annotations": 1, "empty_images": 1}

    d = root / "exp" / "ds"
    coco = json.load(open(d / "annotations" / "instances.json", encoding="utf-8"))
    assert [im["file_name"] for im in coco["images"]] == [
        "field_9__10__1__a.jpg", "field_9__10__1__b.jpg"]
    assert len(coco["annotations"]) == 1 and coco["annotations"][0]["image_id"] == 1

    assert (d / "images" / "field_9__10__1__b.jpg").exists()
    assert (d / "labels" / "field_9__10__1__a.txt").read_text().startswith("0 ")
    assert (d / "labels" / "field_9__10__1__b.txt").read_text() == ""
    for sub in ("masks", "masks_instance"):
        m = cv2.imread(str(d / sub / "field_9__10__1__b.png"), cv2.IMREAD_UNCHANGED)
        assert m.shape == (H, W) and int(m.max()) == 0
    m = cv2.imread(str(d / "masks" / "field_9__10__1__a.png"), cv2.IMREAD_UNCHANGED)
    assert int(m.max()) == 1
