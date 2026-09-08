"""Kiểm phần đo bằng hình có đáp án tính được bằng tay.

Vì sao bắt buộc: nếu boundary_iou hay signed_boundary_error sai, mọi bảng kết
quả về sau vẫn in ra số trông rất hợp lý và không có gì báo động. Phần đo phải
đúng TRƯỚC khi có con số nào từ model.

Hình dùng để kiểm: đĩa tròn bán kính r, "dự đoán" là chính nó co vào k px.
    - độ phủ    = ((r-k)/r)^2
    - sai số biên có dấu = -k tại mọi điểm
    - ASSD      = k
"""

import numpy as np
import pytest

from canopyseg.metrics import (
    assd,
    boundary_iou,
    coverage,
    dice,
    excess,
    hd95,
    iou,
    normalized_surface_dice,
    signed_boundary_error,
)
from canopyseg.metrics.boundary import band_width, edge

R = 120
PAD = 30


def disk(r: float, size: int = 2 * (R + PAD)) -> np.ndarray:
    c = size / 2.0
    y, x = np.ogrid[:size, :size]
    return ((x - c) ** 2 + (y - c) ** 2) <= r * r


# ---------------------------------------------------------------- chồng lấn
def test_identical_masks():
    m = disk(R)
    assert iou(m, m) == 1.0
    assert dice(m, m) == 1.0
    assert coverage(m, m) == 1.0
    assert excess(m, m) == 0.0


def test_disjoint_masks():
    a = np.zeros((50, 50), bool)
    b = np.zeros((50, 50), bool)
    a[:10, :10] = True
    b[30:40, 30:40] = True
    assert iou(a, b) == 0.0
    assert coverage(a, b) == 0.0


def test_empty_masks_are_equal():
    z = np.zeros((20, 20), bool)
    assert iou(z, z) == 1.0
    assert coverage(z, z) == 1.0


@pytest.mark.parametrize("k", [2, 5, 10])
def test_coverage_matches_analytic_area_ratio(k):
    gt = disk(R)
    pred = disk(R - k)
    expected = ((R - k) / R) ** 2
    assert coverage(pred, gt) == pytest.approx(expected, abs=0.01)
    # Co vào thì không có gì lòi ra ngoài.
    assert excess(pred, gt) == pytest.approx(0.0, abs=1e-3)


def test_excess_detects_over_segmentation():
    gt = disk(R)
    pred = disk(R + 10)
    assert coverage(pred, gt) == pytest.approx(1.0, abs=1e-3)
    assert excess(pred, gt) > 0.15


# ------------------------------------------------------------------ có dấu
@pytest.mark.parametrize("k", [2, 5, 10])
def test_signed_error_is_negative_k_when_shrunk(k):
    gt, pred = disk(R), disk(R - k)
    d = signed_boundary_error(pred, gt)
    assert d.size > 0
    # ĐÂY là quy ước cả dự án dựa vào: phủ thiếu -> dấu âm.
    assert np.median(d) == pytest.approx(-k, abs=1.0)
    assert (d < 0).mean() > 0.95


@pytest.mark.parametrize("k", [2, 5, 10])
def test_signed_error_is_positive_when_dilated(k):
    gt, pred = disk(R), disk(R + k)
    d = signed_boundary_error(pred, gt)
    assert np.median(d) == pytest.approx(+k, abs=1.0)
    assert (d > 0).mean() > 0.95


def test_signed_error_spread_separates_uniform_from_uneven():
    """Hai lỗi có cùng sai số trung bình nhưng cần hai cách chữa khác nhau:
    co đều thì nở bù là xong, co lệch thì phải huấn luyện."""
    gt = disk(R)
    uniform = disk(R - 5)
    uneven = disk(R).copy()
    uneven[:, : uneven.shape[1] // 2] = disk(R - 10)[:, : uneven.shape[1] // 2]
    su = signed_boundary_error(uniform, gt)
    sv = signed_boundary_error(uneven, gt)
    assert su.std() < sv.std()


# ------------------------------------------------------------- khoảng cách
@pytest.mark.parametrize("k", [2, 5, 10])
def test_assd_equals_shift(k):
    gt, pred = disk(R), disk(R - k)
    assert assd(pred, gt) == pytest.approx(k, abs=1.0)


def test_hd95_at_least_assd():
    gt, pred = disk(R), disk(R - 5)
    assert hd95(pred, gt) >= assd(pred, gt) - 1e-6


def test_nsd_tolerance_behaviour():
    gt, pred = disk(R), disk(R - 5)
    assert normalized_surface_dice(pred, gt, tau=1.0) < 0.5
    assert normalized_surface_dice(pred, gt, tau=10.0) > 0.95


def test_distance_metrics_nan_on_empty_prediction():
    gt = disk(R)
    empty = np.zeros_like(gt)
    assert np.isnan(assd(empty, gt))
    assert np.isnan(hd95(empty, gt))


# ------------------------------------------------------- boundary iou / vành
def test_boundary_iou_perfect_and_degraded():
    gt = disk(R)
    d = band_width(np.pi * R * R)
    assert boundary_iou(gt, gt, d) == 1.0
    assert boundary_iou(disk(R - 5), gt, d) < boundary_iou(disk(R - 1), gt, d)


def test_boundary_iou_is_stricter_than_mask_iou():
    """Tính chất khiến cả dự án chọn chỉ số này: với tán lớn, IoU vẫn cao
    trong khi đường biên đã hỏng."""
    gt, pred = disk(R), disk(R - 5)
    d = band_width(np.pi * R * R)
    assert iou(pred, gt) > 0.9
    assert boundary_iou(pred, gt, d) < 0.5


def test_band_width_scales_with_area():
    small = band_width(128.0**2)
    big = band_width(1000.0**2)
    assert big > small
    assert band_width(1.0) >= 1.0


def test_edge_is_one_pixel_thin():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    e = edge(m)
    # Vuông 20x20 -> viền trong dày 1 px = 4*20 - 4 = 76 pixel.
    assert e.sum() == 76


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        iou(np.zeros((10, 10), bool), np.zeros((10, 11), bool))
