"""TH3 (`--loss boundary`): khoảng cách có dấu đúng, cắt đúng, phạt xa nặng hơn
gần, và trước khi λ bật thì loss trùng gốc từng số."""

from __future__ import annotations

import csv

import pytest

from cofseg.training import losses
from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


# ------------------------------------------------------------------ cờ
def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "boundary") == "boundary"
    assert cfg["loss"]["boundary"] == {"weight": 0.1, "clip": 5, "start_epoch": 10, "ramp_epochs": 10}
    cfg = _cfg(loss={"boundary": {"clip": 3}})
    assert YoloTrainer.apply_loss(cfg, "boundary") == "boundary-c3"


@pytest.mark.parametrize("bad", [{"clip": 0}, {"weight": -0.1}, {"start_epoch": 1.5}, {"ramp_epochs": -1}])
def test_tham_so_sai_bi_chan(bad):
    with pytest.raises(SystemExit, match="loss.boundary"):
        YoloTrainer.apply_loss(_cfg(loss={"boundary": bad}), "boundary")


# ------------------------------------------------------------------ khoảng cách
def _vuong(r0, r1, c0, c1):
    t = torch.zeros(1, H, H)
    t[0, r0:r1, c0:c1] = 1.0
    return t


def test_khoang_cach_co_dau_va_cat():
    from cofseg.training.boundary_loss import signed_distance

    phi = signed_distance(_vuong(8, 24, 8, 24), 5)[0] * 5      # về đơn vị ô
    assert phi[16, 8].item() == pytest.approx(-0.5)           # ô biên, trong tán
    assert phi[16, 7].item() == pytest.approx(0.5)            # ô sát biên, ngoài tán
    assert phi[16, 5].item() == pytest.approx(2.5)            # cách 3 ô
    assert phi[16, 0].item() == pytest.approx(5.0)            # quá 5 ô: cắt
    assert phi[16, 10].item() == pytest.approx(-2.5)
    assert phi[16, 16].item() == pytest.approx(-5.0)          # sâu trong tán: cắt
    assert phi.abs().max().item() == pytest.approx(5.0)


def test_gop_mat_na_trung_cho_dung_ket_qua():
    """Mười anchor dương cùng một tán: gộp rồi trải lại phải ra đúng từng hàng."""
    from cofseg.training.boundary_loss import signed_distance, signed_distance_dedup

    a, b = _vuong(8, 24, 8, 24), _vuong(2, 12, 18, 30)
    gt = torch.cat([a, b, a, a, b, torch.zeros(1, H, H)])
    assert torch.equal(signed_distance_dedup(gt, 5), signed_distance(gt, 5))


def test_mep_anh_khong_thanh_bien():
    """Tán bị khung ảnh cắt vẫn kéo dài ra ngoài ảnh: ô sát mép ảnh không phải ô biên."""
    from cofseg.training.boundary_loss import signed_distance

    phi = signed_distance(_vuong(0, H, 0, 16), 5)[0] * 5
    assert phi[16, 0].item() == pytest.approx(-5.0)
    assert phi[0, 10].item() == pytest.approx(-5.0)
    assert phi[16, 15].item() == pytest.approx(-0.5)


def test_bl_bang_0_khi_dung_va_phat_xa_nang_hon_gan():
    """Cùng 16 ô thừa: thừa sát biên (φ 0,5) bị phạt nhẹ hơn thừa cách 4 ô trống (φ 4,5)."""
    from cofseg.training.boundary_loss import boundary_term

    gt = _vuong(8, 24, 8, 24)
    box = torch.tensor([[0.0, 0.0, float(H), float(H)]])
    area = torch.tensor([1.0])
    dung, _, _ = boundary_term(gt * 40 - 20, gt, box, area, 5)
    assert dung.item() == pytest.approx(0.0, abs=1e-6)
    gan = gt.clone()
    gan[0, 8:24, 24] = 1                                      # ngay ngoài biên phải
    xa = gt.clone()
    xa[0, 8:24, 28] = 1                                       # 4 ô trống giữa biên và cột thừa
    bl_gan, _, _ = boundary_term(gan * 40 - 20, gt, box, area, 5)
    bl_xa, _, _ = boundary_term(xa * 40 - 20, gt, box, area, 5)
    assert 0 < bl_gan.item() < bl_xa.item()
    assert bl_xa.item() / bl_gan.item() == pytest.approx(4.5 / 0.5, rel=1e-3)


# ------------------------------------------------------------------ lịch λ
def _crit(yolo26, **p):
    from cofseg.training.boundary_loss import build_criterion

    m, _, _ = yolo26
    return build_criterion(m, {"weight": 0.1, "clip": 5, "start_epoch": 10, "ramp_epochs": 10, **p})


def test_lich_lambda(yolo26):
    crit = _crit(yolo26).one2one
    for ep, want in ((0, 0.0), (9, 0.0), (10, 0.0), (15, 0.05), (20, 0.1), (60, 0.1)):
        crit.epoch = ep
        assert crit.lam == pytest.approx(want), ep
    step = _crit(yolo26, ramp_epochs=0).one2one
    step.epoch = 10
    assert step.lam == pytest.approx(0.1)


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


def _set_epoch(crit, ep):
    for head in ("one2one", "one2many"):
        getattr(crit, head).epoch = ep


def test_truoc_khi_lambda_bat_trung_loss_goc(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)(preds, batch)
    crit = _crit(yolo26)
    _set_epoch(crit, 5)
    moi = crit(preds, batch)
    assert torch.equal(moi[0], goc[0])
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_sau_khi_lambda_bat_chi_doi_loss_mat_na(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    crit = _crit(yolo26)
    _set_epoch(crit, 30)
    for head in ("one2one", "one2many"):
        a = getattr(goc, head).loss(preds[head], batch)[1]
        b = getattr(crit, head).loss(preds[head], batch)[1]
        for k in a:
            if k == "seg_loss":
                assert not torch.equal(a[k], b[k]), head
            else:
                assert torch.equal(a[k], b[k]), (head, k)


def test_gradient_toi_prototype():
    from cofseg.training.boundary_loss import build_criterion

    m = _model()
    crit = build_criterion(m, {"weight": 0.1, "clip": 5, "start_epoch": 0, "ramp_epochs": 0})
    torch.manual_seed(2)
    crit(m(torch.rand(2, 3, 128, 128)), _batch())[0].sum().backward()
    grads = [p.grad for p in m.model[-1].proto.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau():
    from cofseg.training.boundary_loss import BoundarySegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"weight": 0.1, "clip": 5, "start_epoch": 0, "ramp_epochs": 0})
    assert isinstance(crit, BoundarySegLoss)
    assert torch.isfinite(crit(m(torch.rand(2, 3, 128, 128)), _batch())[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.boundary_loss import EpochLog

    _, batch, preds = yolo26
    crit = _crit(yolo26)
    log = EpochLog(tmp_path / "loss_boundary.csv")
    for ep in (0, 15):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_boundary.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "16"]
    assert [float(r["lambda"]) for r in rows] == [0.0, 0.05]
    assert int(rows[0]["one2one_n"]) == 3 and rows[0]["one2one_n"] == rows[1]["one2one_n"]
    assert float(rows[0]["one2one_bl"]) >= 0.0          # BL ghi cả trước khi λ bật
    assert 0.0 <= float(rows[0]["one2one_far"]) <= 1.0
