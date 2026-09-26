"""Boundary AP phải tính trên cửa sổ cắt sát, và không giữ lại gì giữa các ảnh.

Bản trước cache vành của MỌI vùng thật và MỌI dự đoán tới hết lượt chấm. Vành
là mảng bool full-frame 2560x1440 = 3,5 MiB, nên 280 ảnh x (14 vùng + tới 100
dự đoán) là hàng chục GB và kernel giết tiến trình giữa chừng — đo thật trên
15 ảnh: 5746 MiB với bản cũ, 10 MiB với bản này, cùng một con số AP.

Cache đó còn không bao giờ trúng: `COCOeval.evaluate()` gọi `computeIoU` đúng
một lần cho mỗi cặp (ảnh, lớp), mà bộ này chỉ có một lớp.

Test dựng mặt nạ tổng hợp ở đúng kích thước ảnh thật, nên nó không cần data/.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo")

H, W = 1440, 2560
D = 0.02 * math.sqrt(H * H + W * W)      # ~58,7 px, đúng định nghĩa bài báo

mask_utils = pytest.importorskip("pycocotools.mask")


def _ce(model: str):
    d = BENCH / model
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for k in [m for m in sys.modules if m == "cofseg" or m.startswith("cofseg.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cofseg"] = m
    spec.loader.exec_module(m)
    from cofseg.evaluation import coco_eval
    return coco_eval


def _tron(cx: int, cy: int, r: int):
    m = np.zeros((H, W), np.uint8)
    yy, xx = np.ogrid[:H, :W]
    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
    return mask_utils.encode(np.asfortranarray(m))


def _hop(x0: int, y0: int, x1: int, y1: int):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_utils.encode(np.asfortranarray(m))


#: Cặp mặt nạ phủ các trường hợp mà phép cắt cửa sổ dễ sai nhất.
CAP = {
    "chồng một phần": (_tron(800, 700, 162), _tron(830, 720, 170)),
    "trùng khít": (_tron(600, 600, 200), _tron(600, 600, 200)),
    "rời hẳn": (_tron(300, 300, 120), _tron(2000, 1000, 120)),
    "sát mép ảnh": (_hop(0, 100, 260, 500), _hop(0, 110, 250, 505)),
    "tán nhỏ hơn vành": (_tron(1500, 800, 20), _tron(1505, 803, 22)),
    "tán lớn hơn vành": (_tron(1200, 700, 600), _tron(1210, 705, 590)),
    "lệch nhiều": (_tron(700, 700, 160), _tron(900, 760, 150)),
}


def _iou_full_frame(ce, ra, rb) -> float:
    """Cách cũ: vành trên cả khung rồi | và &. Đây là định nghĩa gốc."""
    a = ce._boundary_band(mask_utils.decode(ra), D)
    b = ce._boundary_band(mask_utils.decode(rb), D)
    u = np.count_nonzero(a | b)
    return 0.0 if u == 0 else np.count_nonzero(a & b) / u


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ten", sorted(CAP))
def test_cat_cua_so_cho_dung_ket_qua_nhu_ca_khung(model, ten):
    """Cắt cửa sổ là tối ưu, không phải xấp xỉ: phải trùng tới từng bit.

    Nới bbox ra d px là đủ, vì điểm nào có khoảng cách tới nền <= d thì điểm
    nền gần nhất của nó nằm trong bán kính d, tức vẫn trong cửa sổ.
    """
    ce = _ce(model)
    ra, rb = CAP[ten]
    goc = _iou_full_frame(ce, ra, rb)
    moi = ce._band_iou(ce._band_of(ra, D, (H, W)), ce._band_of(rb, D, (H, W)))
    assert moi == pytest.approx(goc, abs=1e-12), f"{model}/{ten}"


@pytest.mark.parametrize("model", MODELS)
def test_vanh_khong_con_la_mang_full_frame(model):
    """Vành của một tán phải nhỏ hơn hẳn khung ảnh.

    Tán trung vị rộng 324 px; cộng vành 59 px mỗi bên thì cửa sổ ~442 px, tức
    dưới 6% diện tích khung. Trả về full-frame nghĩa là phép cắt đã hỏng.
    """
    ce = _ce(model)
    band = ce._band_of(_tron(1200, 700, 162), D, (H, W))
    assert band.m.size < 0.1 * H * W, f"{model}: vành vẫn là {band.m.shape}"
    assert band.area > 0


@pytest.mark.parametrize("model", MODELS)
def test_khong_con_cache_vanh(model):
    """Cache vành là thứ đã làm kernel giết tiến trình; nó không được quay lại."""
    src = (BENCH / model / "cofseg" / "evaluation" / "coco_eval.py").read_text(
        encoding="utf-8")
    assert "_band_cache" not in src, f"{model}: cache vành đã quay lại"
    assert "id(rle)" not in src, f"{model}: id() không phải khoá cache an toàn"


@pytest.mark.parametrize("model", MODELS)
def test_buoc_cham_co_thanh_tien_trinh(model):
    """Bước Boundary AP chạy vài phút; trước đây nó không in gì nên nhìn từ
    ngoài không phân biệt được 'đang chạy' với 'đã treo'."""
    ce = _ce(model)
    src = (BENCH / model / "cofseg" / "evaluation" / "coco_eval.py").read_text(
        encoding="utf-8")
    assert "on_image" in src, f"{model}: không có móc đếm theo ảnh"
    assert "progress.Bar" in src, f"{model}: bước chấm COCO không có thanh"
    # Thanh phải dựng TRƯỚC _run: tqdm giữ sys.stdout lúc khởi tạo, mà _run
    # bọc cả lượt chấm trong redirect_stdout.
    i_bar, i_run = src.index("progress.Bar"), src.index("BoundaryCOCOeval(gt, dt")
    assert i_bar < i_run, f"{model}: thanh dựng bên trong redirect_stdout"

    run_src = (BENCH / model / "cofseg" / "evaluation" / "runner.py").read_text(
        encoding="utf-8")
    assert "progress.Bar" in run_src, f"{model}: vòng suy luận không có thanh"


@pytest.mark.parametrize("model", MODELS)
def test_moc_dem_chay_dung_mot_lan_moi_anh(model):
    """`computeIoU` là chỗ duy nhất đếm được, và nó chạy một lần mỗi ảnh —
    kể cả ảnh không có dự đoán nào, nên thanh không bị kẹt."""
    ce = _ce(model)
    src = (BENCH / model / "cofseg" / "evaluation" / "coco_eval.py").read_text(
        encoding="utf-8")
    than = src[src.index("def computeIoU"):]
    than = than[:than.index("\n\n")]
    dau = than.index("self.on_image")
    assert dau < than.index("if not gt or not dt"), \
        f"{model}: ảnh không có dự đoán sẽ không được đếm"
