"""Phần CPU của --aug pa1/ag4 (aug_paste.py), trên cảnh tổng hợp: đĩa tròn làm tán.

Khung 1024 x 576 với r = 0,4 như khung nạp thật (ảnh 2560 x 1440, imgsz 1024),
nên các hằng đo bằng px ảnh gốc đổi đúng như lúc train: chạm = 2 px, bước = 2 px,
tán tối thiểu đường kính 60 px.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cofseg.training import aug_paste as ap

H, W, R = 576, 1024, 0.4
PA1 = {"p": 1.0, "k_min": 2, "k_max": 4, "c_lo": 0.2, "c_hi": 0.5, "hid": 0.5}
AG4 = {"p": 1.0, "k_min": 2, "k_max": 4, "gap": 25.0}


def disk(cx, cy, rad, n=120):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rad * np.cos(a), cy + rad * np.sin(a)], 1).astype(np.float32)


def ellipse(cx, cy, ra, rb, ang, n=160):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = ra * np.cos(a), rb * np.sin(a)
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1).astype(np.float32)


def full_raster(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly, np.float64) * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    return m


def scene(seed=0, n=9):
    """Lưới đĩa thưa cộng vài cặp chạm nhau; ảnh có màu khác nhau trên mỗi tán."""
    rng = np.random.default_rng(seed)
    segs = []
    for i in range(n):
        cx = 110 + (i % 5) * 200 + rng.uniform(-15, 15)
        cy = 120 + (i // 5) * 300 + rng.uniform(-15, 15)
        segs.append(ellipse(cx, cy, rng.uniform(38, 52), rng.uniform(38, 52), rng.uniform(0, np.pi)))
    img = np.full((H, W, 3), (40, 90, 120), np.uint8)
    for k, s in enumerate(segs):
        cv2.fillPoly(img, [np.round(s).astype(np.int32)], (30 + 20 * k % 200, 160, 60))
    return img, segs, rng


def ref_place_contact(U, tgt, sm, rng, cstar, r, c_min, hid_max):
    """Cách thử thẳng từng bước, đúng như phép đo gây hại trên R2."""
    t, step = ap.px(ap.TOUCH, r), ap.px(ap.STEP, r)
    K = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    tm = tgt.full(H, W)
    b = ap.boundary_local(tgt.m, tgt.x0, tgt.y0, H, W)
    ty, tx = np.nonzero(b)
    nb = len(ty)
    if not nb:
        return None
    ty, tx = ty + tgt.y0, tx + tgt.x0
    ys, xs = np.nonzero(tm)
    tc = np.array([xs.mean(), ys.mean()])
    tr = np.sqrt((xs - tc[0]) ** 2 + (ys - tc[1]) ** 2).max()
    h, w = sm.shape
    ys, xs = np.nonzero(sm)
    cs = np.array([xs.mean(), ys.mean()])
    area = len(ys)
    rs = np.sqrt((xs - cs[0]) ** 2 + (ys - cs[1]) ** 2).max()
    smb, best = sm > 0, None
    for dg in rng.permutation(8) * 45:
        u = np.array([np.cos(np.radians(dg)), np.sin(np.radians(dg))])
        d = tr + rs + 2 * t
        while d >= 0:
            x0, y0 = np.round(tc + d * u - cs).astype(int)
            d -= step
            if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
                continue
            vis = smb & (U[y0:y0 + h, x0:x0 + w] == 0)
            hid = 1 - vis.sum() / area
            if hid > hid_max:
                continue
            sel = (ty >= y0 - t) & (ty < y0 + h + t) & (tx >= x0 - t) & (tx < x0 + w + t)
            if not sel.any():
                continue
            vd = cv2.dilate(np.pad(vis.astype(np.uint8), t), K)
            c = vd[ty[sel] - y0 + t, tx[sel] - x0 + t].sum() / nb
            if c >= c_min and (best is None or abs(c - cstar) < abs(best[0] - cstar)):
                best = (float(c), float(hid), int(x0), int(y0), int(dg))
    return best


# ------------------------------------------------------------------ bảng tán

def test_px_doi_theo_khung():
    assert ap.px(ap.TOUCH, 0.4) == 2 and ap.px(ap.STEP, 0.4) == 2
    assert ap.px(ap.TOUCH, 1.0) == 5 and ap.px(1.0, 0.1) == 1


def test_to_cuc_bo_trung_to_ca_khung():
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = ellipse(rng.uniform(-30, W + 30), rng.uniform(-30, H + 30), rng.uniform(5, 80), rng.uniform(5, 80),
                    rng.uniform(0, 3))
        m, x0, y0 = ap.raster_local(s, H, W)
        full = np.zeros((H, W), np.uint8)
        if m.size:
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        assert np.array_equal(full, full_raster(s))


def test_bang_tan_tiep_xuc_va_mep():
    # 0 và 1 vừa chạm nhau; 4 kẹp giữa 5 và 6, chồng mỗi bên 10 px; 2 đứng riêng; 3 bị khung cắt.
    segs = [disk(200, 150, 50), disk(301, 150, 50), disk(600, 300, 50), disk(20, 300, 50),
            disk(300, 420, 50), disk(210, 420, 50), disk(390, 420, 50)]
    crowns, U = ap.crown_table(segs, H, W, R)
    assert 0.02 < crowns[0].contact < 0.2 and 0.02 < crowns[1].contact < 0.2
    assert crowns[4].contact >= 0.2
    assert crowns[2].contact == 0.0
    assert crowns[3].border and not crowns[2].border
    assert abs(crowns[2].diam - 100) < 3
    union = np.zeros((H, W), np.uint8)
    for s_ in segs:
        union |= full_raster(s_)
    assert np.array_equal(U, union)
    assert ap.is_target(crowns[2], R) and ap.is_donor(crowns[2], R)
    assert ap.is_target(crowns[4], R) and not ap.is_donor(crowns[4], R)
    assert not ap.is_target(crowns[3], R)
    small, _ = ap.crown_table([disk(500, 300, 25)], H, W, R)
    assert not ap.is_target(small[0], R)          # đường kính 50 px < 60 px (150 px gốc)


def test_tan_nguon_cat_dung_manh():
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), np.uint8)
    crowns, _ = ap.crown_table([disk(300, 200, 40)], H, W, R)
    sp, sm = ap.crop_donor(img, crowns[0])
    y0, y1, x0, x1 = ap.donor_box(crowns[0], H, W)
    assert sp.shape[:2] == sm.shape == (y1 - y0, x1 - x0)
    assert np.array_equal(sp, img[y0:y1, x0:x1])
    assert sm.sum() == crowns[0].area
    assert sm[:ap.PAD].sum() == 0 and sm[-ap.PAD:].sum() == 0


# ------------------------------------------------------------------ chỗ dán

@pytest.mark.parametrize("seed", range(6))
def test_cho_cham_trung_cach_thu_thang(seed):
    rng = np.random.default_rng(seed)
    segs = [ellipse(rng.uniform(200, 800), rng.uniform(150, 420), rng.uniform(35, 60), rng.uniform(35, 60),
                    rng.uniform(0, 3)) for _ in range(6)]
    crowns, U = ap.crown_table(segs, H, W, R)
    donor, _ = ap.crown_table([ellipse(120, 120, rng.uniform(35, 60), rng.uniform(35, 60), 0.4)], H, W, R)
    for ti in range(len(crowns)):
        if not crowns[ti].area:
            continue
        sm = ap.d4(ap.donor_mask(donor[0], H, W), seed % 4, ti % 2)
        cstar = float(rng.uniform(0.2, 0.5))
        got = ap.place_contact(U, crowns[ti], sm, np.random.default_rng(ti), cstar, r=R, c_min=0.2, hid_max=0.5)
        want = ref_place_contact(U, crowns[ti], sm, np.random.default_rng(ti), cstar, R, 0.2, 0.5)
        assert got == want
        if got is not None:
            c, hid, x0, y0, _ = got
            assert 0.2 <= c <= 1 and hid <= 0.5


def test_cho_xa_cach_moi_tan():
    segs = [disk(200, 200, 50), disk(700, 300, 60)]
    _, U = ap.crown_table(segs, H, W, R)
    sm = ap.donor_mask(ap.crown_table([disk(100, 100, 45)], H, W, R)[0][0], H, W)
    state = ap.far_state(U, 10)
    dist = cv2.distanceTransform((U == 0).astype(np.uint8), cv2.DIST_L2, 5)
    for s in range(20):
        x0, y0 = ap.place_far(state, sm, np.random.default_rng(s), ap.inradius(sm))
        h, w = sm.shape
        assert x0 >= 0 and y0 >= 0 and x0 + w <= W and y0 + h <= H
        assert dist[y0:y0 + h, x0:x0 + w][sm > 0].min() >= 10


def test_cho_xa_het_cho():
    segs = [disk(x, y, 70) for x in range(60, W, 120) for y in range(60, H, 120)]
    _, U = ap.crown_table(segs, H, W, R)
    sm = ap.donor_mask(ap.crown_table([disk(300, 300, 60)], H, W, R)[0][0], H, W)
    assert ap.place_far(ap.far_state(U, 10), sm, np.random.default_rng(0), ap.inradius(sm)) is None


# ------------------------------------------------------------------ dán và nhãn

def test_phan_nhin_thay_va_lo():
    U = np.zeros((H, W), np.uint8)
    U[100:110, 100:110] = 1                    # tán nhỏ nằm lọt trong tán dán
    sm = np.zeros((60, 60), np.uint8)
    cv2.circle(sm, (30, 30), 28, 1, -1)
    keep, hole = ap.visible_part(U, sm, 75, 75)
    assert hole == 100
    keep, hole = ap.visible_part(np.zeros((H, W), np.uint8), sm, 75, 75)
    assert hole == 0 and keep.sum() == sm.sum()
    U2 = np.zeros((H, W), np.uint8)
    U2[75:135, 104:106] = 1                    # cắt tán dán làm hai nửa
    keep, _ = ap.visible_part(U2, sm, 75, 75)
    n, _ = cv2.connectedComponents(keep)
    assert n == 2 and keep.sum() < sm.sum() / 2 + 60


def test_dan_sau_giu_nguyen_tan_co_san():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), np.uint8)
    crowns, U = ap.crown_table([disk(300, 300, 50)], H, W, R)
    sp = rng.integers(0, 255, (110, 110, 3), np.uint8)
    sm = np.zeros((110, 110), np.uint8)
    cv2.circle(sm, (55, 55), 50, 1, -1)
    keep, _ = ap.visible_part(U, sm, 300, 250)
    out = img.copy()
    ap.paste_behind(out, U, sp, sm, keep, 300, 250)
    assert np.array_equal(out[U > 0], img[U > 0])
    far = np.ones((H, W), bool)
    far[250 - 4:250 + 110 + 4, 300 - 4:300 + 110 + 4] = False
    assert np.array_equal(out[far], img[far])
    inner = cv2.erode(keep, np.ones((5, 5), np.uint8)) > 0
    assert np.abs(out[250:360, 300:410][inner].astype(int) - sp[inner].astype(int)).max() <= 1


def test_duong_vien_to_lai_khop():
    keep = np.zeros((80, 90), np.uint8)
    cv2.ellipse(keep, (45, 40), (38, 30), 20, 0, 360, 1, -1)
    poly = ap.outline(keep, 100, 200)
    back = full_raster(poly)[200:280, 100:190]
    iou = (back & keep).sum() / (back | keep).sum()
    assert iou > 0.95


def _check_pool(mode, p, seed):
    img, segs, _ = scene(seed)
    crowns, U = ap.crown_table(segs, H, W, R)
    bank = ap.image_donors(img, segs, R, "khac.jpg")
    specs, rows = ap.image_pool(segs, H, W, R, mode, p, np.random.default_rng(seed), [d.mask for d in bank],
                                [d.diam for d in bank], [d.src for d in bank], "nay.jpg")
    assert len(rows) >= len(specs)
    assert sum(r_["ok"] for r_ in rows) == len(specs)
    return img, segs, crowns, U, bank, specs


@pytest.mark.parametrize("seed", range(3))
def test_kho_pa1_va_dan(seed):
    img, segs, crowns, U, bank, specs = _check_pool("pa1", PA1, seed)
    assert specs, "cảnh thưa phải có chỗ dán"
    for s in specs:
        assert 0.2 <= s.c and s.hid <= 0.5
    chosen = ap.choose(specs, 4, np.random.default_rng(1), ap.px(ap.TOUCH, R) + 1)
    assert len({s.tgt for s in chosen}) == len(chosen)
    out, polys, done = ap.apply_specs(img, segs, R, chosen, bank)
    assert len(polys) == len(done) == len(chosen)
    assert np.array_equal(out[U > 0], img[U > 0])           # tán có sẵn giữ nguyên
    for poly, s in zip(polys, done):
        m = full_raster(poly)
        assert (m & U).sum() <= 0.02 * m.sum()               # nhãn tán dán không đè tán có sẵn
        tgt = crowns[s.tgt]
        by, bx = ap.boundary_local(tgt.m, tgt.x0, tgt.y0, H, W).nonzero()
        c = (cv2.dilate(m, np.ones((5, 5), np.uint8))[by + tgt.y0, bx + tgt.x0] > 0).mean()
        assert c >= 0.15                                     # vẫn chạm tán đích sau khi tô lại


@pytest.mark.parametrize("seed", range(3))
def test_kho_ag4_khong_cham(seed):
    img, segs, crowns, U, bank, specs = _check_pool("ag4", AG4, seed)
    assert specs
    gap = ap.px(AG4["gap"], R)
    chosen = ap.choose(specs, 4, np.random.default_rng(1), gap)
    out, polys, done = ap.apply_specs(img, segs, R, chosen, bank)
    dist = cv2.distanceTransform((U == 0).astype(np.uint8), cv2.DIST_L2, 5)
    masks = [full_raster(p_) for p_ in polys]
    for m, s in zip(masks, done):
        assert s.c == 0 and s.hid <= ap.AG4_FRAG
        assert dist[m > 0].min() >= gap - 1
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            d = cv2.distanceTransform((masks[i] == 0).astype(np.uint8), cv2.DIST_L2, 5)
            assert d[masks[j] > 0].min() >= gap - 1         # tán dán cũng không chạm nhau


def test_chon_khung_cach_nhau():
    S = ap.Spec
    specs = [S(0, ("same", 1), 0, 0, 0, 0, .3, 0, .3, 10, (0, 0, 10, 10)),
             S(1, ("same", 1), 0, 0, 0, 0, .3, 0, .3, 10, (12, 0, 20, 10)),
             S(0, ("same", 2), 0, 0, 0, 0, .3, 0, .3, 10, (100, 0, 110, 10)),
             S(2, ("same", 2), 0, 0, 0, 0, .3, 0, .3, 10, (200, 0, 210, 10))]
    for seed in range(20):
        got = ap.choose(specs, 4, np.random.default_rng(seed), 3)
        assert len({s.tgt for s in got}) == len(got)
        assert not ({id(specs[0]), id(specs[1])} <= {id(s) for s in got})   # cách nhau 2 px < 3
        assert len(ap.choose(specs, 1, np.random.default_rng(seed), 3)) == 1


def test_kho_khong_phu_thuoc_so_tien_trinh():
    img, segs, _ = scene(1)
    bank = {"field_9": ap.image_donors(img, segs, R, "field_9__khac.jpg")}
    tasks = [(i, f"field_9__{i}.jpg", np.stack(segs), H, W, R) for i in range(3)]
    a, ra = ap.build_pools(tasks, bank, "pa1", PA1, 7, workers=0)
    b, rb = ap.build_pools(tasks, bank, "pa1", PA1, 7, workers=2)
    assert a == b and ra == rb
    assert a[0] != a[1] or a[0] == []                        # mỗi ảnh một bộ số riêng


def test_doc_ruong_va_khung():
    assert ap.field_of("field_1__10__2__DJI_x.jpg") == "field_1"
    assert ap.frame_size(1440, 2560, 1024) == (576, 1024, 0.4)
    assert ap.frame_size(640, 640, 640) == (640, 640, 1.0)
