"""Lớp xem dự đoán trong app: đọc thẳng runs/eval/, và KHÔNG ghi gì.

Hai bất biến đáng test nhất, theo đúng thứ tự mức độ tai hại:

1. Không có đường nào từ lớp này ghi vào thư mục nhãn. App autosave sau 700 ms
   và còn bắn `sendBeacon` lúc rời trang, nên một dự đoán lọt vào `inst` là bị
   ghi thành nhãn thật, im lặng, cho cả 850 ảnh. Test dưới đây gọi hết lối vào
   rồi đòi thư mục nhãn không đổi một byte.

2. Phép ghép phải ra ĐÚNG con số của bảng benchmark, vì màu "khớp" / "sót" trên
   màn hình chính là phán quyết đó. Ghép tham lam theo điểm giảm dần, ngưỡng
   IoU 0,5, mỗi nhãn chỉ nhận một dự đoán — giống
   cofseg/evaluation/matching.py.

Bộ dữ liệu ở đây dựng tại chỗ trong tmp_path: `runs/eval/` thật nằm ngoài git
nên test dựa vào nó sẽ hỏng trên máy người khác.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pycocotools.mask")
cv2 = pytest.importorskip("cv2")

from app import predictions as P  # noqa: E402

W, H = 200, 160
IMG = "field_9__10__a.jpg"


def _vuong(x0, y0, s):
    return [float(x0), float(y0), float(x0 + s), float(y0),
            float(x0 + s), float(y0 + s), float(x0), float(y0 + s)]


def _rle(x0, y0, s):
    from pycocotools import mask as mu

    m = np.zeros((H, W), np.uint8)
    m[y0:y0 + s, x0:x0 + s] = 1
    r = mu.encode(np.asfortranarray(m))
    r["counts"] = r["counts"].decode("ascii")
    return r


@pytest.fixture
def lan_cham(tmp_path, monkeypatch):
    """Một bộ xuất + một lượt chấm nhỏ, đúng bố cục thật của runs/eval/."""
    root = tmp_path / "export" / "field" / "f9"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances_test.json").write_text(json.dumps({
        "images": [{"id": 7, "file_name": IMG, "width": W, "height": H}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 1,
             "segmentation": [_vuong(10, 10, 60)], "area": 3600.0,
             "bbox": [10, 10, 60, 60], "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 1,
             "segmentation": [_vuong(120, 90, 40)], "area": 1600.0,
             "bbox": [120, 90, 40, 40], "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "canopy"}],
    }), encoding="utf-8")

    run = tmp_path / "eval" / "2026-09-30_101010_thu-model_field-f9_test_i1024"
    run.mkdir(parents=True)
    run.joinpath("config.yaml").write_text(
        "data:\n  root: " + root.as_posix() + "\n  split: test\n"
        "name: thu-model\nmodel:\n  name: x\n", encoding="utf-8")
    run.joinpath("metrics.json").write_text(json.dumps(
        {"model": {"imgsz": 1024, "weights": "w/best.pt", "conf": 0.05}}),
        encoding="utf-8")
    # Ba dự đoán: một trùng khít vùng 1, một lệch 3 px (vẫn trên 0,5 IoU nhưng
    # điểm thấp hơn nên bị vùng 1 lấy mất), một nằm chỗ trống = dự đoán thừa.
    # Vùng 2 không ai với tới -> bỏ sót.
    run.joinpath("predictions.json").write_text(json.dumps([
        {"image_id": 7, "category_id": 1, "segmentation": _rle(10, 10, 60), "score": 0.9},
        {"image_id": 7, "category_id": 1, "segmentation": _rle(13, 13, 60), "score": 0.8},
        {"image_id": 7, "category_id": 1, "segmentation": _rle(150, 10, 30), "score": 0.3},
    ]), encoding="utf-8")

    monkeypatch.setattr(P, "RUNS_DIR", str(tmp_path / "eval"))
    P._index = None
    P._preds.clear()
    P._ids.clear()
    P._gt_rle.clear()
    return run.name


def test_chi_muc_tim_dung_anh_va_lan_cham(lan_cham):
    rows = P.runs_for(IMG)
    assert len(rows) == 1
    assert rows[0]["model"] == "thu-model"
    assert rows[0]["dataset"] == "field" and rows[0]["fold"] == "f9"
    assert rows[0]["imgsz"] == 1024
    assert rows[0]["n_gt_export"] == 2
    assert P.runs_for("khong-co-anh-nay.jpg") == []


def test_ghep_tham_lam_dung_luat_cua_bang(lan_cham):
    gt = [_vuong(10, 10, 60), _vuong(120, 90, 40)]
    d = P.polygons_for(IMG, lan_cham, gt, conf=0.05)
    assert d["ok"], d.get("error")
    s = d["summary"]
    # Dự đoán 0 (điểm cao nhất) lấy vùng 1; dự đoán 1 chồng lên chính vùng đó
    # nhưng vùng đã bị lấy -> thành thừa; dự đoán 2 ở chỗ trống -> thừa.
    assert (s["tp"], s["fp"], s["fn"], s["n_gt"]) == (1, 2, 1, 2)
    kh = [p for p in d["polygons"] if p["matched"]]
    assert len(kh) == 1 and kh[0]["conf"] == pytest.approx(0.9)
    # Nhãn được tô bằng `cv2.fillPoly`, đúng phép mà matching.py dùng: đa giác
    # (10,10)-(70,70) ra ô 61x61 vì đỉnh biên cũng được tô, trong khi mặt nạ dự
    # đoán `m[10:70, 10:70]` là 60x60. IoU = 3600/3721. Con số này được chốt ở
    # đây vì nó chứng minh nhãn KHÔNG bị tô bằng `frPyObjects` của pycocotools
    # — phép đó rasterise theo luật khác và IoU sẽ lệch vài phần nghìn so với
    # bảng benchmark, tức màu "khớp"/"sót" trên màn hình nói khác bảng.
    assert kh[0]["iou"] == pytest.approx(3600 / 3721, abs=1e-4)
    assert [g["matched"] for g in d["gt"]] == [True, False]


def test_nguong_diem_loc_dung_va_khong_doi_ket_qua_cu(lan_cham):
    gt = [_vuong(10, 10, 60), _vuong(120, 90, 40)]
    cao = P.polygons_for(IMG, lan_cham, gt, conf=0.5)
    assert cao["summary"]["n_pred_all"] == 3 and cao["summary"]["n_pred"] == 2
    # Bỏ dự đoán điểm 0,3 đi thì chỉ còn một thừa.
    assert cao["summary"]["fp"] == 1
    assert cao["summary"]["tp"] == 1


def test_bao_khi_nhan_da_sua_sau_luc_xuat(lan_cham):
    # Bộ xuất ghi 2 vùng; nhãn hiện tại chỉ còn 1 -> phần khớp/sót ở đây không
    # còn so được với bảng benchmark, và điều đó phải NÓI RA.
    d = P.polygons_for(IMG, lan_cham, [_vuong(10, 10, 60)], conf=0.05)
    assert d["ok"] and d["warn"] and "bộ xuất" in d["warn"]
    khop = P.polygons_for(IMG, lan_cham, [_vuong(10, 10, 60), _vuong(120, 90, 40)])
    assert not khop["warn"]


def test_nhieu_vong_va_toa_do_nam_trong_khung(lan_cham):
    gt = [_vuong(10, 10, 60)]
    d = P.polygons_for(IMG, lan_cham, gt, conf=0.05)
    for p in d["polygons"]:
        xs, ys = p["points"][0::2], p["points"][1::2]
        assert len(p["points"]) >= 6 and len(p["points"]) % 2 == 0
        assert 0 <= min(xs) and max(xs) <= W
        assert 0 <= min(ys) and max(ys) <= H


def test_mat_na_roi_thanh_hai_manh_ra_hai_vong():
    """Tới 42,7% dự đoán của YOLOv8s có hơn một biên ngoài; giữ mỗi mảnh lớn
    nhất là bỏ mất gần một nửa hình."""
    from pycocotools import mask as mu

    m = np.zeros((H, W), np.uint8)
    m[20:40, 20:40] = 1
    m[20:40, 80:100] = 1
    rle = mu.encode(np.asfortranarray(m))
    assert len(P.rle_to_polygons(rle)) == 2


def test_loi_ro_rang_khi_anh_khong_thuoc_lan_cham(lan_cham):
    d = P.polygons_for("field_1__10__khac.jpg", lan_cham, [], conf=0.5)
    assert not d["ok"] and "split" in d["error"]
    assert not P.polygons_for(IMG, "lan-cham-khong-co", [])["ok"]


def test_lop_xem_khong_ghi_gi_vao_thu_muc_nhan(lan_cham, tmp_path, monkeypatch):
    """Bất biến quan trọng nhất: đọc dự đoán KHÔNG được chạm vào nhãn."""
    from app import server

    nhan = tmp_path / "labels"
    nhan.mkdir()
    (nhan / "cu.json").write_text('{"giu": 1}', encoding="utf-8")
    monkeypatch.setattr(server, "OUT_DIR", str(nhan))
    monkeypatch.setattr(server, "BASE", str(tmp_path / "images"))
    monkeypatch.setattr(server, "ROOT", str(tmp_path / "images"))
    truoc = {p.name: p.read_bytes() for p in nhan.iterdir()}

    server.pred_runs(name="field_9/10/a.jpg")
    server.pred_data(name="field_9/10/a.jpg", run=lan_cham, conf=0.5)

    assert {p.name: p.read_bytes() for p in nhan.iterdir()} == truoc


def test_giao_dien_khoa_sua_o_che_do_xem():
    """Khoá là một phần của tính năng, không phải chi tiết hiển thị: nếu ai đó
    gỡ `predLocked()` khỏi các lối vào sửa thì phải gãy ở đây."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
    s = src.read_text(encoding="utf-8")
    assert "function predLocked()" in s
    # Chế độ 'view' chặn mọi nhánh sửa trong handler chuột.
    assert "function setMode(m){if(predOn&&m!=='view')return;" in s
    # Ba lối vào KHÔNG đi qua handler chuột, nên phải hỏi riêng.
    assert s.count("predLocked()") >= 4, "thiếu chốt ở phím tắt / danh sách vùng"
    # Dự đoán không bao giờ được đi vào `inst` (thứ mà payload() lưu xuống đĩa).
    than = s[s.index("function drawPred()"):s.index("function predPick(")]
    assert "inst" not in than, "drawPred chạm vào inst — dự đoán sẽ bị lưu thành nhãn"
    assert "predData" not in s[s.index("function payload()"):s.index("function payload()") + 400]
