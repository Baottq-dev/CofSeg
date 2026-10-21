"""TH4 (`--loss repgt`): RepGT tính đúng trên hộp dựng tay, chỉ đổi loss hộp của
đầu one-to-many, và λ = 0 thì trùng gốc từng số."""

from __future__ import annotations

import csv
import math

import pytest

from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


# ------------------------------------------------------------------ cờ
def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "repgt") == "repgt"
    assert cfg["loss"]["repgt"] == {"weight": 0.5, "sigma": 0.5}


@pytest.mark.parametrize("bad", [{"sigma": 1.0}, {"sigma": -0.1}, {"weight": -1}])
def test_tham_so_sai_bi_chan(bad):
    with pytest.raises(SystemExit, match="loss.repgt"):
        YoloTrainer.apply_loss(_cfg(loss={"repgt": bad}), "repgt")


# ------------------------------------------------------------------ công thức
def test_smooth_ln():
    from cofseg.training.repgt_loss import smooth_ln

    x = torch.tensor([0.0, 0.3, 0.5, 0.8, 1.0], requires_grad=True)
    y = smooth_ln(x, 0.5)
    want = [0.0, -math.log(0.7), -math.log(0.5), 0.6 - math.log(0.5), 1.0 - math.log(0.5)]
    assert torch.allclose(y.detach(), torch.tensor(want), atol=1e-6)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()                 # IoG = 1 không làm gradient nổ
    assert x.grad[3].item() == pytest.approx(2.0)       # độ dốc 1/(1 − sigma) trên ngưỡng


def test_repgt_tren_hop_dung_tay():
    """Hộp dự đoán trùng tán mục tiêu [0,0,10,10]; tán bên cạnh [5,0,15,10]
    bị phủ một nửa -> IoG 0,5. Anchor thứ hai không chồng ai -> 0."""
    from cofseg.training.repgt_loss import repgt_terms

    gts = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [40.0, 40.0, 50.0, 50.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0]], requires_grad=True)
    valid = torch.tensor([[False, True, True], [True, True, False]])
    loss, iog, has = repgt_terms(pred, gts[None].expand(2, -1, -1), valid, 0.5)
    assert has.tolist() == [True, False]
    assert iog[0].item() == pytest.approx(0.5)
    assert loss[0].item() == pytest.approx(-math.log(0.5)) and loss[1].item() == 0.0
    loss.sum().backward()
    assert pred.grad[0, 2].item() > 0                   # kéo cạnh phải ra khỏi tán bên cạnh
    assert pred.grad[1].abs().sum().item() == 0


def test_gt_dem_va_gt_muc_tieu_khong_bi_xet():
    from cofseg.training.repgt_loss import repgt_terms

    gts = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]]])
    loss, _, has = repgt_terms(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), gts, torch.tensor([[False, False]]), 0.5)
    assert not has.any() and loss.item() == 0.0


# ------------------------------------------------------------------ gắn vào loss
def _batch():
    """Hai tán sát nhau ở ảnh 0, một tán ở ảnh 1."""
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


def test_weight_0_trung_goc(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.repgt_loss import build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    moi = build_criterion(m, {"weight": 0.0, "sigma": 0.5})
    for head in ("one2many", "one2one"):
        a = getattr(goc, head).loss(preds[head], batch)
        b = getattr(moi, head).loss(preds[head], batch)
        assert torch.equal(a[0], b[0]), head


def test_chi_doi_loss_hop_cua_dau_one2many(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.repgt_loss import RepGTSegLoss, build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    moi = build_criterion(m, {"weight": 0.5, "sigma": 0.5})
    assert isinstance(moi.one2many, RepGTSegLoss) and not isinstance(moi.one2one, RepGTSegLoss)
    assert moi.one2many.assigner.topk == goc.one2many.assigner.topk
    a = goc.one2one.loss(preds["one2one"], batch)
    b = moi.one2one.loss(preds["one2one"], batch)
    assert torch.equal(a[0], b[0])
    a = goc.one2many.loss(preds["one2many"], batch)[1]
    b = moi.one2many.loss(preds["one2many"], batch)[1]
    assert moi.one2many.stats[1].item() > 0             # có anchor chồng tán bên cạnh
    for k in a:
        if k == "box_loss":
            assert b[k] > a[k]
        else:
            assert torch.equal(a[k], b[k]), k


def test_gradient_huu_han():
    from cofseg.training.repgt_loss import build_criterion

    m = _model()
    torch.manual_seed(2)
    build_criterion(m, {"weight": 0.5, "sigma": 0.5})(m(torch.rand(2, 3, 128, 128)), _batch())[0].sum().backward()
    grads = [p.grad for p in m.model[-1].cv2.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau():
    from cofseg.training.repgt_loss import RepGTSegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"weight": 0.5, "sigma": 0.5})
    assert isinstance(crit, RepGTSegLoss)
    assert torch.isfinite(crit(m(torch.rand(2, 3, 128, 128)), _batch())[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.repgt_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"weight": 0.5, "sigma": 0.5})
    log = EpochLog(tmp_path / "loss_repgt.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_repgt.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["n_pos"]) > 3 and rows[0]["n_pos"] == rows[1]["n_pos"]
    for col in ("share_overlap", "share_iog_05"):
        assert 0.0 <= float(rows[0][col]) <= 1.0
    assert float(rows[0]["repgt"]) >= 0.0
