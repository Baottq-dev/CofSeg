"""T1 (GPR + NCB): toán tử lấy mẫu lại đúng, chuyển Proto giữ trọng số, loss NCB
trùng loss gốc khi tắt phần mới, validator chấm đúng lưới, checkpoint nạp lại.

Model dựng từ yaml của gói ultralytics (yolo26n-seg) với trọng số ngẫu nhiên,
ảnh và nhãn dựng tại chỗ — không đụng data/ hay weights/.
"""

from __future__ import annotations

import copy
import csv
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch.nn.functional as F  # noqa: E402

from cofseg.training import mask_head  # noqa: E402

IMG = 128
CFG = mask_head.check({"type": "gpr"})


def _batch(size: int):
    """Ba tán hình chữ nhật không chồng nhau, nhãn ở lưới `size`: hai ở ảnh 0, một ở ảnh 1."""
    boxes = torch.tensor([[0.25, 0.25, 0.30, 0.30], [0.70, 0.70, 0.35, 0.40], [0.50, 0.50, 0.50, 0.50]])
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    masks = torch.zeros(2, size, size)
    for k, (b, bi) in enumerate(zip(boxes, batch_idx)):
        j = int((batch_idx[:k] == bi).sum())
        cx, cy, w, h = (b * size).tolist()
        masks[int(bi), int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)] = j + 1
    return {"batch_idx": batch_idx, "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros_like(masks)}


def _stock(seed=0):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(seed)
    m = SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False)
    m.args = get_cfg()
    return m


def _gpr(cfg=None, seed=0):
    from cofseg.training.gpr_head import to_gpr

    m = to_gpr(_stock(seed), dict(cfg or CFG))
    m.train()
    return m


def _cfg(**kw):
    return mask_head.check({"type": "gpr", **kw})


# ------------------------------------------------------------------ toán tử
@pytest.mark.parametrize("groups", [1, 2])
def test_do_lech_0_la_phong_song_tuyen(groups):
    from cofseg.training.gpr_head import GuidedResample

    torch.manual_seed(0)
    op = GuidedResample(8, 4, 4, groups=groups).eval()
    x, g = torch.randn(2, 8, 5, 7), torch.randn(2, 4, 10, 14)
    ref = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(op(x, g), ref, atol=1e-5)


def test_do_lech_dich_dung_huong_va_do_lon():
    """Δx = +0,5 ô thô (bias của lớp cuối = atanh 0,5) trên một dốc theo cột:
    điểm trong lòng nhận đúng giá trị của dốc dịch nửa ô."""
    from cofseg.training.gpr_head import GuidedResample

    op = GuidedResample(1, 1, 2).eval()
    with torch.no_grad():
        op.off.bias.copy_(torch.tensor([math.atanh(0.5), 0.0]))
    x = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 1, 6, 8).contiguous()
    out = op(x, torch.zeros(1, 1, 12, 16))
    ref = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(out[..., 2:-3], ref[..., 2:-3] + 0.5, atol=1e-5)


@pytest.mark.parametrize("detach", [True, False])
def test_dan_huong_tach_gradient(detach):
    from cofseg.training.gpr_head import GuidedResample

    op = GuidedResample(4, 3, 4, detach_guide=detach)
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    g = torch.randn(1, 3, 8, 8, requires_grad=True)
    op(x, g).sum().backward()
    assert x.grad is not None
    assert (g.grad is None) == detach


# ------------------------------------------------------------------ chuyển Proto
def test_chuyen_chi_doi_proto():
    from cofseg.training.gpr_head import GPRSegmentationModel, Proto26GPR, Segment26GPR, find_proto

    goc = _stock()
    sd0 = goc.state_dict()
    m = _gpr()
    assert type(m) is GPRSegmentationModel and type(m.model[-1]) is Segment26GPR
    assert type(find_proto(m)) is Proto26GPR and find_proto(m).gpr["pstride"] == 2.0
    sd = m.state_dict()
    bo = sorted(set(sd0) - set(sd))
    them = sorted(set(sd) - set(sd0))
    assert bo == ["model.23.proto.upsample.bias", "model.23.proto.upsample.weight"]
    assert them and all(k.startswith(("model.23.proto.gpr1.", "model.23.proto.gpr2.")) for k in them)
    assert all(torch.equal(sd0[k], sd[k]) for k in set(sd0) & set(sd))
    assert m.model[-1].f == [16, 19, 22, 2, 0] and {0, 2} <= set(m.save)


def test_chi_nhan_segment26():
    from ultralytics.nn.tasks import SegmentationModel

    from cofseg.training.gpr_head import to_gpr

    with pytest.raises(SystemExit, match="Segment26"):
        to_gpr(SegmentationModel("yolov8n-seg.yaml", nc=1, verbose=False), dict(CFG))


def test_nap_trong_so_sau_khi_chuyen():
    """Thứ tự của trainer: dựng chuẩn -> chuyển -> nạp. Lớp cũ nhận trọng số,
    lớp mới của GPR giữ khởi tạo (lớp độ lệch = 0)."""
    from cofseg.training.gpr_head import find_proto

    nguon = _stock(seed=7)
    m = _gpr(seed=0)
    m.load(nguon, verbose=False)
    sd, sn = m.state_dict(), nguon.state_dict()
    for k in ("model.23.proto.cv3.conv.weight", "model.23.proto.cv1.conv.weight", "model.0.conv.weight"):
        assert torch.equal(sd[k], sn[k])
    p = find_proto(m)
    assert p.gpr1.off.weight.abs().sum() == 0 and p.gpr2.off.weight.abs().sum() == 0


@pytest.mark.parametrize("stages,cells", [([1, 2], IMG // 2), ([2], IMG // 2), ([], IMG // 4)])
def test_dau_ra_train_va_suy_luan(stages, cells):
    m = _gpr(_cfg(stages=stages))
    preds = m(torch.rand(2, 3, IMG, IMG))
    p, sem = preds["one2many"]["proto"]
    assert p.shape == (2, 32, cells, cells) and sem.shape[-1] == IMG // 8
    m.eval()
    with torch.no_grad():
        (y, proto), _ = m(torch.rand(2, 3, IMG, IMG))
    assert proto.shape == (2, 32, cells, cells) and y.shape[1] == 4 + 1 + 32     # box, điểm, 32 hệ số


def test_stages_rong_va_chi_gpr2_khop_proto_goc():
    """stages [] cho đúng proto gốc; stages [2] với độ lệch 0 cho đúng proto gốc
    phóng song tuyến x2 — tức mọi khác biệt về sau là do GPR học được."""
    goc = _stock(seed=3).eval()
    x = torch.rand(1, 3, IMG, IMG)
    with torch.no_grad():
        ref = goc(x)[0][1]
        for stages, want in (([], ref), ([2], F.interpolate(ref, scale_factor=2, mode="bilinear",
                                                             align_corners=False))):
            m = _gpr(_cfg(stages=stages), seed=0)
            m.load(goc, verbose=False)
            assert torch.allclose(m.eval()(x)[0][1], want, atol=1e-5)


# ------------------------------------------------------------------ loss NCB
def test_nhan_mem():
    from cofseg.training.gpr_train import gaussian_kernel, soft_target

    gt = torch.zeros(1, 8, 8)
    gt[0, 1:5, 2:4] = 1                       # 4x2 px, lệch pha so với ô 2x2
    cov = soft_target(gt, 2, None)
    assert cov.shape == (1, 4, 4)
    assert torch.allclose(cov[0, 0], torch.tensor([0.0, 0.5, 0.0, 0.0]))
    assert torch.allclose(cov[0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.isclose(cov.sum() * 4, gt.sum())             # tỉ lệ phủ giữ diện tích

    k = gaussian_kernel(1.2)
    assert gaussian_kernel(0.0) is None and torch.isclose(k.sum(), torch.tensor(1.0))
    assert torch.allclose(soft_target(torch.ones(1, 6, 6), 1, k), torch.ones(1, 6, 6))  # mép ảnh không thành biên giả
    half = torch.zeros(1, 9, 9)
    half[0, :, :4] = 1
    s = soft_target(half, 1, k)[0, 4]
    assert s[0] > 0.99 and s[-1] < 0.01 and 0.3 < s[3] < 0.7 and torch.all(s[:-1] >= s[1:])


def test_ncb_sigma0_trung_loss_goc():
    """σ = 0, nhãn ở đúng lưới proto, không chặn số tán: NCB phải cho đúng từng
    số loss của ultralytics — mọi khác biệt về sau là do tỉ lệ phủ và σ."""
    goc = _stock(seed=1)
    m = _gpr(_cfg(stages=[], sigma=0.0, max_pos=10_000), seed=0)
    m.load(goc, verbose=False)
    preds = m(torch.rand(2, 3, IMG, IMG))
    b = _batch(IMG // 4)
    _, a = goc.init_criterion()(preds, copy.deepcopy(b))
    _, n = m.init_criterion()(preds, copy.deepcopy(b))
    for k in a:
        assert torch.isclose(a[k], n[k], rtol=1e-5, atol=1e-6), k


def test_loss_huu_han_va_gradient():
    from cofseg.training.gpr_head import find_proto

    m = _gpr()
    loss, items = m.init_criterion()(m(torch.rand(2, 3, IMG, IMG)), _batch(IMG))
    assert torch.isfinite(loss).all() and items["seg_loss"] > 0
    loss.sum().backward()
    p = find_proto(m)
    for g in (p.gpr2.off.weight.grad, p.gpr1.off.weight.grad, p.cv3.conv.weight.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_mask_ratio_sai_bi_bao():
    m = _gpr()
    with pytest.raises(RuntimeError, match="mask_ratio"):
        m.init_criterion()(m(torch.rand(2, 3, IMG, IMG)), _batch(48))


def test_sigma_lam_mem_nhan():
    """Cùng dự đoán, σ khác nhau thì loss khác nhau; σ chỉ đi vào loss mặt nạ."""
    preds_m = _gpr(_cfg(sigma=0.0))
    preds = preds_m(torch.rand(2, 3, IMG, IMG))
    a = preds_m.init_criterion()(preds, _batch(IMG))[1]
    m2 = _gpr(_cfg(sigma=4.0))
    m2.load(preds_m, verbose=False)
    b = m2.init_criterion()(preds, _batch(IMG))[1]
    assert a["seg_loss"] != b["seg_loss"]
    assert torch.equal(a["box_loss"], b["box_loss"]) and torch.equal(a["cls_loss"], b["cls_loss"])


def test_gioi_han_so_tan_moi_anh():
    m = _gpr(_cfg(max_pos=2))
    crit = m.init_criterion()
    crit(m(torch.rand(2, 3, IMG, IMG)), _batch(IMG))
    assert crit.one2many.stats[0].item() <= 4     # 2 ảnh x tối đa 2
    assert crit.one2one.stats[0].item() == 3      # một anchor dương mỗi tán, dưới mức chặn


def test_nhat_ky_epoch(tmp_path):
    from cofseg.training.gpr_head import find_proto
    from cofseg.training.gpr_train import EpochLog

    m = _gpr()
    preds = m(torch.rand(2, 3, IMG, IMG))
    crit = m.init_criterion()
    log = EpochLog(tmp_path / "mask_head_gpr.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, _batch(IMG))
        log.end(crit, ep, find_proto(m))
    rows = list(csv.DictReader((tmp_path / "mask_head_gpr.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["one2one_n"] == "3" and 0.0 <= float(rows[0]["one2one_iou"]) <= 1.0
    assert float(rows[0]["gpr2_offset"]) == 0.0        # lớp độ lệch khởi tạo 0


# ------------------------------------------------------------------ chấm
def test_validator_dung_luoi_cua_proto(monkeypatch):
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionValidator

    from cofseg.training.gpr_head import GPRSegValidator

    m = _gpr().eval()
    m.names = {0: "crown"}
    v = GPRSegValidator(args=get_cfg(overrides={"task": "segment", "conf": 0.0, "max_det": 5}))
    v.training, v.data = True, {"val": ""}
    v.init_metrics(m)
    with torch.no_grad():
        out = v.postprocess(m(torch.rand(1, 3, IMG, IMG)))
    assert out[0]["masks"].shape[1:] == (IMG // 2, IMG // 2)

    monkeypatch.setattr(DetectionValidator, "_prepare_batch",
                        lambda self, si, batch: {"cls": torch.zeros(2), "imgsz": (IMG, IMG)})
    got = v._prepare_batch(0, _batch(IMG))["masks"]
    assert got.shape == (2, IMG // 2, IMG // 2) and got.sum() > 0


def test_checkpoint_nap_lai_va_yolosegmodel(tmp_path):
    """Lưu như trainer, nạp bằng YoloSegModel: nhận ra GPR, mặt nạ ở cỡ ảnh gốc."""
    import numpy as np

    from cofseg.models.yolo_seg import YoloSegModel

    m = _gpr().eval()
    m.names = {0: "crown"}
    p = tmp_path / "gpr.pt"
    torch.save({"model": copy.deepcopy(m).half(), "train_args": {"task": "segment", "imgsz": IMG}}, p)
    ym = YoloSegModel(str(p), imgsz=IMG, conf=0.0, max_det=3, device="cpu")
    assert ym.mask_head == "gpr" and ym.describe["mask_head"] == "gpr"
    img = (np.random.default_rng(0).random((96, 160, 3)) * 255).astype(np.uint8)
    res = ym.model.predict(img, conf=0.0, max_det=3, imgsz=IMG, retina_masks=True, verbose=False)[0]
    assert res.masks is not None and res.masks.data.shape[1:] == (96, 160)
    ym.predict(img)
