"""TH2 (`--loss vfl`): chỉ loss điểm của đầu one-to-many đổi, và đổi đúng công thức Varifocal.

Model dựng từ yaml của gói ultralytics với trọng số ngẫu nhiên, ảnh và nhãn
dựng tại chỗ, như test của L1/L2.
"""

from __future__ import annotations

import csv

import pytest

from cofseg.training import losses
from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32  # lưới proto của ảnh 128 px


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


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


# ------------------------------------------------------------------ cờ
def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "vfl") == "vfl"
    assert cfg["loss"]["vfl"] == {"alpha": 0.75, "gamma": 2.0}
    assert losses.describe_active(cfg["loss"]) == "Loss: vfl (alpha 0.75, gamma 2)"
    cfg = _cfg(loss={"vfl": {"gamma": 1.5}})
    assert YoloTrainer.apply_loss(cfg, "vfl") == "vfl-g1.5"


@pytest.mark.parametrize("bad", [{"alpha": -0.1}, {"gamma": "2"}, {"gamma": True}])
def test_tham_so_sai_bi_chan(bad):
    with pytest.raises(SystemExit, match="loss.vfl"):
        YoloTrainer.apply_loss(_cfg(loss={"vfl": bad}), "vfl")


# ------------------------------------------------------------------ công thức
def test_trong_so_dung_cong_thuc_varifocal_cua_thu_vien():
    """Từng phần tử khớp công thức; rút gọn kiểu của thư viện thì khớp VarifocalLoss."""
    from ultralytics.utils.loss import VarifocalLoss

    from cofseg.training.vfl_loss import VarifocalBCE

    torch.manual_seed(3)
    pred = torch.randn(2, 50, 1) * 3
    target = torch.zeros(2, 50, 1)
    target[0, :7, 0] = torch.rand(7)
    target[1, 10:13, 0] = torch.tensor([0.9, 0.2, 0.55])
    label = (target > 0).float()
    got = VarifocalBCE(alpha=0.75, gamma=2.0)(pred, target)
    w = 0.75 * pred.sigmoid().pow(2) * (1 - label) + target * label
    want = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none") * w
    assert torch.allclose(got, want)
    lib = VarifocalLoss(gamma=2.0, alpha=0.75)(pred, target, label)
    assert torch.allclose(got.mean(1).sum(), lib)


def test_alpha_1_gamma_0_dich_nhi_phan_la_bce():
    """Âm tính trọng số 1, dương tính đích 1 trọng số 1: đúng BCE gốc."""
    from cofseg.training.vfl_loss import VarifocalBCE

    torch.manual_seed(4)
    pred = torch.randn(1, 40, 1)
    target = (torch.rand(1, 40, 1) > 0.8).float()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")(pred, target)
    assert torch.allclose(VarifocalBCE(alpha=1.0, gamma=0.0)(pred, target), bce)


def test_am_tinh_de_gan_nhu_khong_con_tinh():
    """p = 0,01 ở âm tính: BCE ~0,01, VFL nhân thêm 0,75·1e-4."""
    from cofseg.training.vfl_loss import VarifocalBCE

    logit = torch.logit(torch.tensor([[[0.01]], [[0.8]]]))
    out = VarifocalBCE()(logit, torch.zeros(2, 1, 1))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.zeros(2, 1, 1), reduction="none")
    assert out[0].item() == pytest.approx(bce[0].item() * 0.75 * 1e-4, rel=1e-3)
    assert out[1].item() == pytest.approx(bce[1].item() * 0.75 * 0.64, rel=1e-3)   # âm tính tự tin: 0,48


# ------------------------------------------------------------------ gắn vào loss
def test_chi_doi_loss_diem_cua_dau_one2many(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.vfl_loss import VarifocalBCE, build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    moi = build_criterion(m, {"alpha": 0.75, "gamma": 2.0})
    assert isinstance(moi.one2many.bce, VarifocalBCE)
    assert not isinstance(moi.one2one.bce, VarifocalBCE)
    o2o_goc, o2o_moi = goc.one2one.loss(preds["one2one"], batch), moi.one2one.loss(preds["one2one"], batch)
    assert torch.equal(o2o_goc[0], o2o_moi[0])
    o2m_goc, o2m_moi = goc.one2many.loss(preds["one2many"], batch), moi.one2many.loss(preds["one2many"], batch)
    for k in o2m_goc[1]:
        if k == "cls_loss":
            assert not torch.isclose(o2m_moi[1][k], o2m_goc[1][k])
        else:
            assert torch.equal(o2m_moi[1][k], o2m_goc[1][k]), k


def test_gradient_chay_va_huu_han():
    from cofseg.training.vfl_loss import build_criterion

    m = _model()
    torch.manual_seed(2)
    loss = build_criterion(m, {"alpha": 0.75, "gamma": 2.0})(m(torch.rand(2, 3, 128, 128)), _batch())[0]
    loss.sum().backward()
    grads = [p.grad for p in m.model[-1].cv3.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau_doi_o_dau_do():
    from ultralytics.utils.loss import v8SegmentationLoss

    from cofseg.training.vfl_loss import VarifocalBCE, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"alpha": 0.75, "gamma": 2.0})
    assert isinstance(crit, v8SegmentationLoss) and isinstance(crit.bce, VarifocalBCE)
    assert torch.isfinite(crit(m(torch.rand(2, 3, 128, 128)), _batch())[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.vfl_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"alpha": 0.75, "gamma": 2.0})
    log = EpochLog(tmp_path / "loss_vfl.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_vfl.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["n_pos"]) > 3 and rows[0]["n_pos"] == rows[1]["n_pos"]   # start() xoá số cũ
    assert -1.0 <= float(rows[0]["r_pos"]) <= 1.0
    assert 0.0 <= float(rows[0]["neg_share"]) <= 1.0
    assert 0.0 < float(rows[0]["target_pos"]) <= 1.0
