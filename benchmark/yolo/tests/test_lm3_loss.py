"""LM3 (`--loss lm3`): α-CIoU có cổng theo tán. α = 1, hoặc cổng không bao giờ bật,
cho đúng loss gốc; cổng luôn bật cho đúng TH5; chỉ loss hộp đổi."""

from __future__ import annotations

import csv

import pytest

from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32
DEFAULTS = {"alpha": 3.0, "tau_s": 0.6, "tau_u": 0.7, "kappa": 0.05}


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "lm3") == "lm3"
    assert cfg["loss"]["lm3"] == DEFAULTS
    assert YoloTrainer.apply_loss(_cfg(loss={"lm3": {"tau_s": 0.5}}), "lm3") == "lm3-ts0.5"


@pytest.mark.parametrize("bad", [{"alpha": 0.5}, {"tau_s": 1.2}, {"tau_u": -0.1}, {"kappa": 0.0}])
def test_tham_so_sai_bi_chan(bad):
    with pytest.raises(SystemExit, match="loss.lm3"):
        YoloTrainer.apply_loss(_cfg(loss={"lm3": bad}), "lm3")


def test_cong_theo_tan():
    """Neo của cùng một tán dùng chung một cổng, tính từ điểm và CIoU LỚN NHẤT trong tán."""
    from cofseg.training.lm3_loss import crown_gate

    s = torch.tensor([0.9, 0.2, 0.3, 0.1, 0.95])
    u = torch.tensor([0.5, 0.9, 0.8, 0.95, 0.6])
    group = torch.tensor([0, 0, 1, 1, 2])
    gam = crown_gate(s, u, group, tau_s=0.6, tau_u=0.7, kappa=0.05)
    sig = lambda x: 1 / (1 + torch.exp(torch.tensor(-x)))  # noqa: E731
    assert gam[0] == gam[1]                                 # tán 0: điểm 0,9 (neo 0), CIoU 0,9 (neo 1)
    assert gam[0].item() == pytest.approx((sig(6.0) * sig(4.0)).item(), rel=1e-5)
    assert gam[2] == gam[3]                                 # tán 1: điểm lớn nhất 0,3 → tắt dù hộp tốt
    assert gam[2].item() == pytest.approx((sig(-6.0) * sig(5.0)).item(), rel=1e-4) and gam[2] < 0.01
    assert gam[4].item() == pytest.approx((sig(7.0) * sig(-2.0)).item(), rel=1e-5)  # tán 2: CIoU 0,6 < 0,7
    assert gam[4] < gam[0] / 5


# ------------------------------------------------------------------ gắn vào loss
def _batch():
    masks = torch.zeros(2, H, H)
    masks[0, 6:22, 4:16] = 1
    masks[0, 6:22, 16:28] = 2
    masks[1, 8:24, 8:24] = 1
    boxes = torch.tensor([[10 / H, 14 / H, 12 / H, 16 / H],
                          [22 / H, 14 / H, 12 / H, 16 / H],
                          [16 / H, 16 / H, 16 / H, 16 / H]])
    return {"batch_idx": torch.tensor([0.0, 0.0, 1.0]), "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, H, H)}


def _model(yaml_name="yolo26n-seg.yaml"):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    m = SegmentationModel(yaml_name, nc=1, verbose=False)
    m.args = get_cfg()
    m.train()
    return m


@pytest.fixture(scope="module")
def yolo26():
    m = _model()
    torch.manual_seed(1)
    return m, _batch(), m(torch.rand(2, 3, 128, 128))


def _p(**kw):
    return {**DEFAULTS, **kw}


def _losses(crit, preds, batch):
    return {h: getattr(crit, h).loss(preds[h], batch)[1] for h in ("one2one", "one2many")}


def test_alpha_1_trung_loss_goc(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.lm3_loss import build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)(preds, batch)
    moi = build_criterion(m, _p(alpha=1.0, tau_s=0.0, tau_u=0.0))(preds, batch)
    assert torch.equal(moi[0], goc[0])
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_cong_tat_trung_loss_goc_cong_bat_trung_th5(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.alpha_ciou_loss import build_criterion as th5
    from cofseg.training.lm3_loss import build_criterion

    m, batch, preds = yolo26
    parsed = E2ELoss(m, v8SegmentationLoss).one2many.parse_output(preds)
    goc = _losses(E2ELoss(m, v8SegmentationLoss), parsed, batch)
    alpha = _losses(th5(m, {"alpha": 3.0}), parsed, batch)
    tat = _losses(build_criterion(m, _p(tau_s=1.0, tau_u=1.0, kappa=1e-3)), parsed, batch)   # cổng không bao giờ bật
    # cổng luôn bật: điểm thuộc (0, 1) và CIoU thuộc [−1, 1], nên đặt ngưỡng dưới cả hai miền
    bat = _losses(build_criterion(m, _p(tau_s=-0.5, tau_u=-1.5, kappa=1e-3)), parsed, batch)
    for h in ("one2one", "one2many"):
        for k in goc[h]:
            assert torch.allclose(tat[h][k], goc[h][k], rtol=1e-6, atol=1e-7), (h, k)
            assert torch.allclose(bat[h][k], alpha[h][k], rtol=1e-6, atol=1e-7), (h, k)


def test_chi_doi_loss_hop(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.lm3_loss import build_criterion

    m, batch, preds = yolo26
    parsed = E2ELoss(m, v8SegmentationLoss).one2many.parse_output(preds)
    goc = _losses(E2ELoss(m, v8SegmentationLoss), parsed, batch)
    moi = _losses(build_criterion(m, _p(tau_s=0.0, tau_u=0.0)), parsed, batch)
    for h in ("one2one", "one2many"):
        for k in goc[h]:
            if k == "box_loss":
                assert not torch.equal(goc[h][k], moi[h][k]), h
            else:
                assert torch.equal(goc[h][k], moi[h][k]), (h, k)


def test_gradient_huu_han():
    from cofseg.training.lm3_loss import build_criterion

    m = _model()
    torch.manual_seed(2)
    build_criterion(m, _p(tau_s=0.0, tau_u=0.0))(m(torch.rand(2, 3, 128, 128)), _batch())[0].sum().backward()
    grads = [p.grad for p in m.model[-1].cv2.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau():
    from cofseg.training.lm3_loss import LM3SegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, _p())
    assert isinstance(crit, LM3SegLoss)
    assert torch.isfinite(crit(m(torch.rand(2, 3, 128, 128)), _batch())[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.lm3_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, _p(tau_s=0.0, tau_u=0.0))
    log = EpochLog(tmp_path / "loss_lm3.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_lm3.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["one2one_n"] == rows[1]["one2one_n"]
    for col in ("one2one_gate", "one2many_gate_on", "one2many_crown_on", "one2many_iou_08"):
        assert 0.0 <= float(rows[0][col]) <= 1.0, col
