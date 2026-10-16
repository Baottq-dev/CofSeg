"""TC2: nhãn dải tiếp giáp và khoảng cách tới biên, trên hình dựng tại chỗ."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cofseg.training.contact_targets import boundary_distance, contact_band, to_semseg_grid  # noqa: E402

H = 40


def _rect(m, idx, y0, y1, x0, x1):
    m[y0:y1, x0:x1] = idx
    return m


def _disc(m, idx, cy, cx, r):
    y, x = torch.meshgrid(torch.arange(m.shape[0]), torch.arange(m.shape[1]), indexing="ij")
    m[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = idx
    return m


def test_hai_tan_cham_nhau_chi_co_dai_quanh_cho_cham():
    m = torch.zeros(H, H)
    _rect(m, 1, 10, 30, 5, 20)
    _rect(m, 2, 10, 30, 20, 35)          # chạm tán 1 ở cột 19|20
    band = contact_band(m[None], radius=2)[0]
    ys, xs = band.nonzero(as_tuple=True)
    assert len(xs) > 0
    assert xs.min() >= 18 and xs.max() <= 21           # chỉ quanh đường chạm, bán kính 2
    assert ys.min() >= 8 and ys.max() <= 31
    assert not band[:, :10].any() and not band[:, 30:].any()   # biên tán-đất không có dải


def test_tan_dung_rieng_khong_co_dai():
    m = _disc(torch.zeros(H, H), 1, 20, 20, 12)
    assert not contact_band(m[None], radius=2).any()


@pytest.mark.parametrize("gap,co_dai", [(1, True), (3, True), (4, False), (6, False)])
def test_khe_hep_thuoc_dai_khe_rong_thi_khong(gap, co_dai):
    m = torch.zeros(H, H)
    _rect(m, 1, 10, 30, 5, 15)
    _rect(m, 2, 10, 30, 15 + gap, 30)
    assert bool(contact_band(m[None], radius=2).any()) is co_dai


def test_tan_nho_chong_len_tan_lon_co_dai_o_ranh_gioi():
    m = torch.zeros(H, H)
    _rect(m, 1, 5, 35, 5, 35)
    _rect(m, 2, 15, 25, 15, 25)          # tán nhỏ nằm trên, như cách mã hoá overlap_mask
    band = contact_band(m[None], radius=1)[0]
    assert band[15, 20] and band[20, 15]                # ngay ranh giới
    assert not band[20, 20] and not band[8, 8]          # lõi tán nhỏ, góc xa của tán lớn


def test_ban_kinh_va_hinh_dau_vao_sai_bi_chan():
    m = torch.zeros(1, H, H)
    with pytest.raises(ValueError, match="radius"):
        contact_band(m, radius=0)
    with pytest.raises(ValueError, match="B, H, W"):
        contact_band(m[0], radius=2)


def test_khoang_cach_mot_tan():
    m = _disc(torch.zeros(H, H), 1, 20, 20, 12)
    d = boundary_distance(m[None])[0]
    assert d.min() >= 0 and d.max() == 1.0
    assert d[20, 20] == 1.0                              # lõi
    assert 0 < d[20, 8] < 0.2                            # mép
    assert (d[m == 0] == 0).all()                        # ngoài tán


def test_khoang_cach_hai_tan_cham_nhau_thanh_hai_dinh():
    m = torch.zeros(H, 2 * H)
    _disc(m, 1, 20, 20, 12)
    _disc(m, 2, 20, 43, 12)                              # vẽ đè cột 31-32: tán 1 hết ở cột 30
    d = boundary_distance(m[None])[0]
    assert d[20, 20] == 1.0 and d[20, 43] == 1.0
    assert m[20, 30] == 1 and m[20, 31] == 2
    assert d[20, 30] < 0.2 and d[20, 31] < 0.2           # chỗ hai tán gặp nhau là biên của cả hai


def test_chuan_hoa_theo_tung_anh_va_tung_tan():
    big = _disc(torch.zeros(H, H), 1, 20, 20, 15)
    small = _disc(torch.zeros(H, H), 1, 20, 20, 4)       # cùng chỉ số 1 nhưng ở ảnh khác
    d = boundary_distance(torch.stack([big, small]))
    assert d[0].max() == 1.0 and d[1].max() == 1.0


def test_mep_anh_goc_khong_phai_bien():
    m = torch.zeros(H, H)
    _rect(m, 1, 0, H, 0, 20)                             # tán chạm mép trái, trên, dưới
    d = boundary_distance(m[None])[0]
    assert d[20, 0] > d[20, 19]                          # mép ảnh xa biên thật (cột 19) hơn


def test_thu_ve_luoi_semseg_giu_dai_mong():
    band = torch.zeros(1, H, H, dtype=torch.bool)
    band[0, :, 21] = True                                # dải rộng 1 pixel ở cột lẻ
    dist = torch.rand(1, H, H)
    fg = torch.rand(1, H, H) > 0.5
    b, d, f = to_semseg_grid(band, dist, fg, (H // 2, H // 2))
    assert b.shape == (1, 1, H // 2, H // 2)
    assert b[0, 0, :, 10].all()                          # max-pool giữ dải
    assert torch.allclose(d[0, 0, 0, 0], dist[0, :2, :2].mean())
    ref = torch.nn.functional.interpolate(fg.float()[:, None], (H // 2, H // 2), mode="nearest")
    assert torch.equal(f, ref)


def test_cung_co_thi_khong_doi():
    band = torch.zeros(2, H, H, dtype=torch.bool)
    dist = torch.rand(2, H, H)
    fg = dist > 0.5
    b, d, f = to_semseg_grid(band, dist, fg, (H, H))
    assert torch.equal(d[:, 0], dist) and torch.equal(f[:, 0], fg.float())
