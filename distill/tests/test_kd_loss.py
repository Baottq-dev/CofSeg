"""Loss chắt lọc: α = 0, đích bằng nhãn hay tán không có thầy đều trùng loss gốc; trộn đích tuyến tính; PKD không
lệ thuộc độ lớn; chắt lọc đặc trưng đi tới lớp 1×1 và trò, không tới thầy; khối config; gỡ lớp 1×1."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from distill import blocks  # noqa: E402
from distill import kd_loss as kl  # noqa: E402

H = 32


@pytest.fixture(scope="module")
def yolo26():
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    m = SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False)
    m.args = get_cfg()
    m.train()
    torch.manual_seed(1)
    img = torch.rand(2, 3, 128, 128)
    return m, img, m(img)


def _stock(m):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    return E2ELoss(m, v8SegmentationLoss)


def _kd(batch, soft="gt", has=True):
    """Thêm kd_map/kd_ids/kd_has vào batch: đích bằng nhãn ('gt'), hay một hình khác ('other')."""
    b = dict(batch)
    masks = b["masks"].long()
    n = len(b["cls"])
    kd_map = torch.zeros(2, 4, H, H, dtype=torch.uint8)
    kd_map[:, 0] = kd_map[:, 2] = 255
    ids = torch.zeros(n, dtype=torch.long)
    k = 0
    for i in range(2):
        for j in range(int((b["batch_idx"] == i).sum())):
            own = masks[i] == j + 1
            if soft == "other":                                   # đích lệch 3 ô sang phải, giá trị 0,6
                own = torch.roll(own, 3, dims=1)
            kd_map[i, 0][own] = 10 + j
            kd_map[i, 1][own] = 255 if soft == "gt" else 153
            ids[k] = 10 + j
            k += 1
    b.update(kd_map=kd_map, kd_ids=ids, kd_has=torch.full((n,), has))
    return b


def _same(a, b):
    assert torch.equal(a[0], b[0])
    for k in b[1]:
        assert torch.equal(a[1][k], b[1][k]), k


def test_alpha_0_trung_tung_so_voi_loss_goc(yolo26, batch):
    m, _, preds = yolo26
    _same(kl.build_seg_criterion(m, 0.0, 1.0)(preds, _kd(batch, "other")), _stock(m)(preds, batch))


def test_dich_bang_nhan_hoac_khong_co_thay_trung_loss_goc(yolo26, batch):
    m, _, preds = yolo26
    goc = _stock(m)(preds, batch)
    _same(kl.build_seg_criterion(m, 0.5, 1.0)(preds, _kd(batch, "gt")), goc)
    _same(kl.build_seg_criterion(m, 0.7, 1.0)(preds, _kd(batch, "other", has=False)), goc)
    _same(kl.build_seg_criterion(m, 0.5, 1.0)(preds, batch), goc)        # batch val: không có khoá kd


def test_tron_dich_tuyen_tinh_chi_doi_loss_mat_na(yolo26, batch):
    m, _, preds = yolo26
    b = _kd(batch, "other")
    seg = {a: kl.build_seg_criterion(m, a, 1.0)(preds, b)[1] for a in (0.0, 0.3, 1.0)}
    want = 0.7 * seg[0.0]["seg_loss"] + 0.3 * seg[1.0]["seg_loss"]
    assert torch.allclose(seg[0.3]["seg_loss"], want, rtol=1e-5)
    assert not torch.allclose(seg[0.0]["seg_loss"], seg[1.0]["seg_loss"])
    for k in seg[0.0]:
        if k != "seg_loss":
            assert torch.equal(seg[0.3][k], seg[0.0][k]), k


def test_dem_neo_co_dich_thay(yolo26, batch):
    m, _, preds = yolo26
    crit = kl.build_seg_criterion(m, 0.5, 1.0)
    crit(preds, _kd(batch, "other"))
    for name, part in kl.seg_parts(crit):
        n, nk = part.stats.tolist()
        assert n > 0 and nk == n, name
    crit2 = kl.build_seg_criterion(m, 0.5, 1.0)
    crit2(preds, _kd(batch, "other", has=False))
    assert all(p.stats[1] == 0 for _, p in kl.seg_parts(crit2))


def test_vung_gian_cong_them(yolo26, batch):
    m, _, preds = yolo26
    b = _kd(batch, "other")
    a = kl.build_seg_criterion(m, 0.5, 1.0)(preds, b)[1]["seg_loss"]
    c = kl.build_seg_criterion(m, 0.5, 1.25)(preds, b)[1]["seg_loss"]
    assert c > a


# ---------------------------------------------------------------- PKD
def test_pkd_khong_le_thuoc_do_lon_va_dau():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(2, 8, 6, 6, generator=g)
    fg = torch.zeros(2, 6, 6, dtype=torch.bool)
    fg[:, 1:4, 1:4] = True
    scale = torch.rand(1, 8, 1, 1, generator=g) * 3 + 0.5
    assert kl.pkd_loss(scale * f + 7.0, f, fg, 0.5) < 1e-5
    # tương quan −1: (ŝ + ŝ)² = 4ŝ², TB ŝ² = (N − 1)/N với độ lệch chuẩn không chệch
    n = 2 * 6 * 6
    want = 0.5 * (1 + 0.5) * 4 * (n - 1) / n
    assert abs(float(kl.pkd_loss(-f, f, fg, 0.5)) - want) < 0.15


def test_pkd_trong_ngoai_hop_tinh_rieng():
    f = torch.zeros(1, 1, 4, 4)
    f[0, 0, 0, 0] = 1.0
    t = f.clone()
    t[0, 0, 3, 3] = 1.0                    # chỉ lệch ở ô (3, 3)
    fg = torch.zeros(1, 4, 4, dtype=torch.bool)
    fg[0, 2:, 2:] = True
    inside_only = kl.pkd_loss(f, t, fg, 0.0)
    assert inside_only > 0
    fg2 = torch.zeros_like(fg)
    fg2[0, :2, :2] = True                  # ô lệch nằm ngoài hộp
    assert kl.pkd_loss(f, t, fg2, 0.0) < inside_only


def test_box_mask():
    b = torch.tensor([[0.5, 0.5, 0.5, 0.25], [0.125, 0.125, 0.25, 0.25]])
    m = kl.box_mask(b, torch.tensor([0.0, 1.0]), 2, 8, 8)
    assert m[0].sum() == 4 * 2 and m[0, 3:5, 2:6].all()
    assert m[1].sum() == 4 and m[1, :2, :2].all()


# ---------------------------------------------------------------- chắt lọc đặc trưng
def test_kenh_dac_trung(teacher, yolo26):
    m, _, _ = yolo26
    assert kl.feature_channels(teacher) == [256, 512, 128]
    assert kl.feature_channels(m) == [128, 256, 64]
    assert m.training                       # trả lại chế độ train


def test_chat_loc_dac_trung_toi_lop_1x1_va_tro_khong_toi_thay(teacher, batch):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(3)
    m = SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False)
    m.args = get_cfg()
    m.kd_adapter = kl.KDAdapter(kl.feature_channels(m), kl.feature_channels(teacher))
    m.train()
    block = blocks.check_kd({"targets": "x", "teacher_runs": "y"})
    crit = kl.build_criterion(m, block, teacher)
    try:
        b = _kd(batch, "other")
        b["img"] = torch.rand(2, 3, 128, 128)
        loss, items = crit(m(b["img"]), b)
        assert loss.shape == (6,) and "kd_loss" in items and items["kd_loss"] > 0
        assert torch.allclose(loss[-1], items["kd_loss"] * 2)              # nhân cỡ batch như ultralytics
        loss.sum().backward()
        assert all(p.grad is not None for p in m.kd_adapter.parameters())
        assert m.model[0].conv.weight.grad is not None
        assert all(p.grad is None for p in teacher.parameters())
        assert crit.s_neck.out is None and crit.t_fuse.out is None          # không giữ đồ thị sang bước sau
        assert crit.feat_stats[0] == 1
        # EMA dựng lại bằng deepcopy chỉ mang loss mặt nạ, không mang thầy và hook
        cp = copy.deepcopy(crit)
        assert type(cp).__name__ == "E2ELoss"
        assert hasattr(crit, "update") and crit.o2m == crit.base.o2m
    finally:
        crit.remove()
    assert not teacher.model[-1]._forward_pre_hooks


def test_chi_mat_na_thi_khong_can_thay(yolo26, batch):
    m, _, preds = yolo26
    block = blocks.check_kd({"targets": "x"})
    crit = kl.build_criterion(m, block, None)
    loss, items = crit(preds, _kd(batch, "other"))
    assert loss.shape == (5,) and "kd_loss" not in items


def test_giu_lich_one2many_khi_thay_ham_loss(yolo26):
    m, _, _ = yolo26
    old, new = _stock(m), _stock(m)
    for _ in range(5):
        old.update()
    kl.carry_schedule(old, new)
    assert new.updates == old.updates and new.o2m == old.o2m


def test_nhat_ky_epoch(tmp_path, yolo26, batch):
    import csv

    m, _, preds = yolo26
    crit = kl.build_criterion(m, blocks.check_kd({"targets": "x"}), None)
    log = kl.EpochLog(tmp_path / "loss_kd.csv")
    log.start(crit)
    crit(preds, _kd(batch, "other"))
    log.end(crit, 0)
    row = list(csv.DictReader(open(tmp_path / "loss_kd.csv", encoding="utf-8")))[0]
    assert row["epoch"] == "1" and int(row["one2many_pos"]) > 0 and row["one2many_pos"] == row["one2many_pos_kd"]


def test_go_lop_1x1_khoi_checkpoint(tmp_path):
    from distill.trainer import strip_adapter

    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
    m.kd_adapter = kl.KDAdapter([4], [8])
    p = tmp_path / "best.pt"
    torch.save({"model": None, "ema": m, "epoch": 3}, p)
    assert strip_adapter(p)
    ck = torch.load(p, weights_only=False)
    assert not hasattr(ck["ema"], "kd_adapter") and ck["epoch"] == 3
    assert not strip_adapter(p)


# ---------------------------------------------------------------- khối config
def test_khoi_kd_mac_dinh_va_hau_to():
    b = blocks.check_kd({"targets": "t", "teacher_runs": "r"})
    assert b["mask_alpha"] == 0.5 and b["feat_weight"] == 6.0 and blocks.kd_tag(b) == ""
    assert blocks.kd_tag(blocks.check_kd({"targets": "t"})) == "nofeat"
    assert blocks.kd_tag(blocks.check_kd({"teacher_runs": "r"})) == "nomask"
    assert blocks.kd_tag(blocks.check_kd({"targets": "t", "teacher_runs": "r", "mask_alpha": 0.3,
                                          "mask_expand": 1.25, "feat_weight": 2, "feat_bg": 1})) == "a0.3-mx1.25-fw2-bg1"
    assert "đặc trưng PKD" in blocks.describe_kd(b) and blocks.describe_kd(None) == "Chắt lọc: không"


@pytest.mark.parametrize("block,match", [
    ({}, "ít nhất một"), ({"targets": "t", "mask_alpha": 1.5}, r"\[0, 1\]"),
    ({"targets": "t", "mask_expand": 0.9}, ">= 1"), ({"teacher_runs": "r", "feat_weight": -1}, ">= 0"),
    ({"targets": "t", "alpha": 0.5}, "mask_alpha"), ({"targets": ""}, "đường dẫn"),
])
def test_khoi_kd_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        blocks.check_kd(block)
