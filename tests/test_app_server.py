"""Nhãn rỗng "đã xem, không có cây" ở app: được lưu, được liệt kê là đã gán,
và đi vào bộ xuất như ảnh nền (COCO không annotation, YOLO .txt rỗng, mask 0).

Server đọc configs/config.yaml lúc import nên các đường dẫn được thay bằng
thư mục tạm ngay sau đó; SAM không bị nạp (chỉ nạp lười khi bấm SAM).
"""

import json
import os
import re

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


def _doc(name):
    return json.load(open(server._json_path(name), encoding="utf-8"))


# --- công tắc đếm hoa HSV/Otsu: tắt mặc định ---------------------------------
def test_save_with_flower_off_writes_geometry_only(root):
    r = _save(A, [SQUARE])
    assert r["count"] == 1 and "warning" not in r
    doc = _doc(A)
    assert set(doc["polygons"][0]) == {"points", "conf", "bbox"}
    assert doc["polygons"][0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    for k in ("hsv", "otsu", "flower_method", "flower_stats"):
        assert k not in doc


def test_save_with_flower_off_does_not_read_the_image(root):
    os.remove(root / "images" / A)
    r = _save(A, [SQUARE])
    assert r["count"] == 1 and "warning" not in r
    assert len(server.load(A)["polygons"]) == 1


def test_an_old_client_without_the_flag_saves_geometry_only(root):
    # Trang mở từ trước khi có công tắc vẫn gửi mức hoa và tham số HSV/Otsu.
    _save(A, [SQUARE], classes=[2], manual=[True], sat_max=40, method="otsu")
    assert set(_doc(A)["polygons"][0]) == {"points", "conf", "bbox"}


def test_save_with_flower_on_still_counts_hsv_and_otsu(root):
    r = _save(A, [SQUARE], flower=True)
    assert r["count"] == 1
    doc = _doc(A)
    p = doc["polygons"][0]
    assert {"points", "conf", "bbox", "hsv", "otsu", "flower_ratio", "flower_label"} <= set(p)
    assert doc["flower_method"] == "hsv" and "hsv" in doc and "otsu" in doc


def test_empty_label_carries_flower_fields_only_when_on(root):
    _save(A, [], empty=True)
    assert "flower_stats" not in _doc(A)
    _save(B, [], empty=True, flower=True)
    assert "flower_stats" in _doc(B)


def test_save_writes_through_a_temp_file(root):
    _save(A, [SQUARE])
    labels = os.listdir(server.OUT_DIR)
    assert labels == [os.path.basename(server._json_path(A))]   # không sót file .tmp


def test_a_failed_save_keeps_the_old_label(root, monkeypatch):
    # Server chết giữa lúc ghi (ở đây: json.dump ném lỗi) thì file nhãn cũ phải
    # còn nguyên, không bị thay bằng một file ghi dở.
    _save(A, [SQUARE])
    before = open(server._json_path(A), encoding="utf-8").read()

    def boom(*a, **k):
        raise OSError("disk full")
    real_dump = server.json.dump
    monkeypatch.setattr(server.json, "dump", boom)
    with pytest.raises(OSError):
        _save(A, [SQUARE, [60.0, 10.0, 90.0, 10.0, 90.0, 40.0]])
    # Trả json.dump lại bằng tay: monkeypatch.undo() sẽ gỡ luôn OUT_DIR của
    # fixture và trỏ về thư mục nhãn thật.
    monkeypatch.setattr(server.json, "dump", real_dump)
    assert open(server._json_path(A), encoding="utf-8").read() == before


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


def test_page_loads_its_css_and_js_from_static():
    # CSS/JS nằm ở app/static/css, app/static/js; trang chỉ còn khung HTML.
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    page = c.get("/").text
    assert "<style>" not in page and "<script>\n" not in page and "{{V}}" not in page
    css = re.search(r'href="(/static/css/app\.css\?v=\d+)"', page)
    js = re.findall(r'src="(/static/js/(\w+)\.js\?v=\d+)"', page)
    assert css and js
    # core.js nạp đầu (các file sau dùng hàm của nó), main.js nạp cuối (gắn
    # sự kiện, khởi động); không file nào nạp hai lần.
    names = [n for _, n in js]
    assert names[0] == "core" and names[-1] == "main" and len(set(names)) == len(names)
    for url in [css.group(1)] + [u for u, _ in js]:
        assert c.get(url).status_code == 200


def test_page_has_four_workspaces_and_no_placeholders():
    # Bốn thẻ theo đúng thứ tự phím Alt+1…4; các ô chiếm chỗ mà không làm gì
    # (Train model "Sắp có", avatar tĩnh) đã bỏ.
    from fastapi.testclient import TestClient

    page = TestClient(server.app).get("/").text
    assert re.findall(r'data-tab="(\w+)"', page) == ["crown", "flower", "review", "data"]
    for gone in ("Train model", "Configure Training", 'class="avatar"', 'id="leftcol"'):
        assert gone not in page


def test_overview_counts_fields_crowns_and_flower_checks(root, monkeypatch):
    from app import flower_api
    _save(A, [SQUARE, [60.0, 10.0, 90.0, 10.0, 90.0, 40.0]])
    _save(B, [], empty=True)
    st = flower_api._store()
    flat = server._flat(A)
    st.set_status(flat, 0, "try", state="done", bbox=[10, 10, 51, 41])
    st.set_status(flat, 1, "try", state="doing", bbox=[60, 10, 91, 41])

    d = server.overview()
    assert d["total"] == {"images": 2, "labeled": 2, "empty": 1, "crowns": 2}
    f9 = d["fields"][0]
    assert f9["field"] == "field_9" and f9["flower"] == {"try": {"done": 1, "doing": 1}}
    assert d["checks"] == []

    # Sửa polygon tán 0 sau khi đã gán hoa: thẻ Dữ liệu phải báo.
    _save(A, [[10.0, 10.0, 70.0, 10.0, 70.0, 40.0, 10.0, 40.0], [60.0, 10.0, 90.0, 10.0, 90.0, 40.0]])
    checks = server.overview()["checks"]
    assert [(c["crown"], c["problem"]) for c in checks] == [(0, "polygon tán đã đổi sau khi gán hoa")]
