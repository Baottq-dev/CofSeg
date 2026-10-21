"""TH5 (`--loss alpha_ciou`): α = 1 trùng CIoU gốc từng số, α = 3 dồn gradient
vào hộp IoU cao, và chỉ loss hộp đổi."""

from __future__ import annotations

import csv

import pytest

from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "alpha_ciou") == "alpha-ciou"
    assert cfg["loss"]["alpha_ciou"] == {"alpha": 3.0}
    assert YoloTrainer.apply_loss(_cfg(loss={"alpha_ciou": {"alpha": 2.0}}), "alpha_ciou") == "alpha-ciou-a2"


def test_alpha_duoi_1_bi_chan():
    with pytest.raises(SystemExit, match="loss.alpha_ciou.alpha"):
        YoloTrainer.apply_loss(_cfg(loss={"alpha_ciou": {"alpha": 0.5}}), "alpha_ciou")


# ------------------------------------------------------------------ công thức
def _boxes(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 50
    wh = torch.rand(n, 2, generator=g) * 30 + 5
    a = torch.cat([xy, xy + wh], 1)
    b = a + torch.randn(n, 4, generator=g) * 4
    b[:, 2:] = torch.maximum(b[:, 2:], b[:, :2] + 1)
    return a, b


def test_alpha_1_trung_ciou_goc():
    from ultralytics.utils.metrics import bbox_iou

    from cofseg.training.alpha_ciou_loss import alpha_ciou

    a, b = _boxes()
    assert torch.equal(alpha_ciou(a, b, 1.0), bbox_iou(a, b, xywh=False, CIoU=True))


def test_alpha_3_don_gradient_vao_hop_iou_cao():
    """Hộp 10x10 dịch ngang so với đích: IoU 0,9 (dịch 10/19) và 0,5 (dịch 10/3).

    Phần IoU của 1 − IoU^3 có gradient 3·IoU² theo IoU, tức nặng hơn 3,24 lần
    ở IoU 0,9 so với 0,5. Đo theo toạ độ hộp (có cả phần tâm), tỉ số gradient
    giữa hai mức là 1,37 với α = 1 và 5,19 với α = 3: tăng 3,8 lần."""
    from cofseg.training.alpha_ciou_loss import alpha_ciou

    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    def grad_at(shift, alpha):
        p = torch.tensor([[shift, 0.0, 10.0 + shift, 10.0]], requires_grad=True)
        (1 - alpha_ciou(p, target, alpha)).sum().backward()
        return p.grad.abs().sum().item()

    r1 = grad_at(10 / 19, 1.0) / grad_at(10 / 3, 1.0)
    r3 = grad_at(10 / 19, 3.0) / grad_at(10 / 3, 3.0)
    assert r3 / r1 > 3.0


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


def test_alpha_1_trung_loss_goc(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.alpha_ciou_loss import build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)(preds, batch)
    moi = build_criterion(m, {"alpha": 1.0})(preds, batch)
    assert torch.equal(moi[0], goc[0])
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_alpha_3_chi_doi_loss_hop(yolo26):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.alpha_ciou_loss import build_criterion

    m, batch, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    moi = build_criterion(m, {"alpha": 3.0})
    for head in ("one2one", "one2many"):
        a = getattr(goc, head).loss(preds[head], batch)[1]
        b = getattr(moi, head).loss(preds[head], batch)[1]
        for k in a:
            if k == "box_loss":
                assert not torch.equal(a[k], b[k]), head
            else:
                assert torch.equal(a[k], b[k]), (head, k)


def test_gradient_huu_han():
    from cofseg.training.alpha_ciou_loss import build_criterion

    m = _model()
    torch.manual_seed(2)
    build_criterion(m, {"alpha": 3.0})(m(torch.rand(2, 3, 128, 128)), _batch())[0].sum().backward()
    grads = [p.grad for p in m.model[-1].cv2.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau():
    from cofseg.training.alpha_ciou_loss import AlphaCIoUSegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"alpha": 3.0})
    assert isinstance(crit, AlphaCIoUSegLoss)
    assert torch.isfinite(crit(m(torch.rand(2, 3, 128, 128)), _batch())[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.alpha_ciou_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"alpha": 3.0})
    log = EpochLog(tmp_path / "loss_alpha_ciou.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_alpha_ciou.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["one2one_n"]) == 3 and rows[0]["one2one_n"] == rows[1]["one2one_n"]
    for col in ("one2one_iou", "one2many_iou_08"):
        assert 0.0 <= float(rows[0][col]) <= 1.0
