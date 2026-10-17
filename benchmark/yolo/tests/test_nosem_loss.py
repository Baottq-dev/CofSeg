"""Đối chứng của TC2 (`--loss nosem`): chỉ loss ngữ nghĩa phụ về 0, còn lại y nguyên."""

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


def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "nosem") == "nosem"
    assert cfg["loss"]["use"] == ["nosem"]
    assert losses.describe_active(cfg["loss"]) == "Loss: nosem"
    assert "nosem" in YoloTrainer.describe_losses()


def test_khong_nhan_tham_so():
    with pytest.raises(SystemExit, match="không có tham số"):
        YoloTrainer.apply_loss(_cfg(loss={"nosem": {"weight": 1}}), "nosem")


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


def test_chi_sem_loss_ve_0_va_semseg_khong_nhan_gradient():
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.nosem_loss import build_criterion

    m = _model()
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    batch = _batch()
    goc = E2ELoss(m, v8SegmentationLoss)(preds, batch)
    moi = build_criterion(m, {})(preds, batch)
    assert goc[1]["sem_loss"] > 0 and moi[1]["sem_loss"] == 0 and moi[0][4] == 0
    for k in goc[1]:
        if k != "sem_loss":
            assert torch.equal(moi[1][k], goc[1][k]), k
    assert torch.equal(moi[0][:4], goc[0][:4])
    moi[0].sum().backward()
    semseg = m.model[-1].proto.semseg
    grads = [p.grad for p in semseg.parameters()]
    assert all(g is not None and g.abs().sum() == 0 for g in grads)   # trong đồ thị, gradient 0


def test_chi_cho_yolo26():
    from cofseg.training.nosem_loss import check_model

    check_model(_model(), {})
    with pytest.raises(SystemExit, match="yolo26"):
        check_model(_model("yolov8n-seg.yaml"), {})


def test_nhat_ky_epoch(tmp_path):
    from cofseg.training.nosem_loss import EpochLog, build_criterion

    m = _model()
    crit = build_criterion(m, {})
    log = EpochLog(tmp_path / "loss_nosem.csv")
    log.start(crit, 0)
    torch.manual_seed(1)
    crit(m(torch.rand(2, 3, 128, 128)), _batch())
    log.end(crit, 0)
    rows = list(csv.DictReader((tmp_path / "loss_nosem.csv").open(encoding="utf-8")))
    assert rows == [{"epoch": "1", "batches_sem_off": "1"}]
