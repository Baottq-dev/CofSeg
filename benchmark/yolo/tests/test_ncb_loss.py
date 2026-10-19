"""TH1 (`--loss ncb`): NCB với Proto gốc — cần nhãn đủ độ phân giải, trùng gốc
khi biên thẳng hàng với lưới, và mềm đúng chỗ biên cắt ngang ô."""

from __future__ import annotations

import csv

import pytest

from cofseg.training import losses
from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

S = 128   # ảnh đầu vào
H = 32    # lưới proto (stride 4)


def _cfg(**train):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024, **train}}


# ------------------------------------------------------------------ cờ
def test_can_mask_ratio_1():
    """Thiếu --set train.mask_ratio=1 thì dừng, kèm đúng lệnh cần thêm."""
    with pytest.raises(SystemExit, match=r"--set train.mask_ratio=1"):
        YoloTrainer.apply_loss(_cfg(mask_ratio=4), "ncb")
    with pytest.raises(SystemExit, match=r"--set train.mask_ratio=1"):
        YoloTrainer.apply_loss(_cfg(), "ncb")


def test_bat_bang_co_va_nhan():
    cfg = _cfg(mask_ratio=1)
    assert YoloTrainer.apply_loss(cfg, "ncb") == "ncb"
    assert cfg["loss"]["ncb"] == {"sigma": 2.4}
    cfg = {**_cfg(mask_ratio=1), "loss": {"ncb": {"sigma": 0.0}}}
    assert YoloTrainer.apply_loss(cfg, "ncb") == "ncb-s0"
    assert "cần --set train.mask_ratio=1" in losses.describe()


def test_sigma_am_bi_chan():
    with pytest.raises(SystemExit, match="loss.ncb.sigma"):
        YoloTrainer.apply_loss({**_cfg(mask_ratio=1), "loss": {"ncb": {"sigma": -1}}}, "ncb")


# ------------------------------------------------------------------ loss
def _rects(scale: int):
    """Ba tán chữ nhật, toạ độ ở lưới proto; scale = 4 cho nhãn đủ độ phân giải."""
    m = torch.zeros(2, H * scale, H * scale)
    m[0, 6 * scale:22 * scale, 4 * scale:16 * scale] = 1
    m[0, 6 * scale:22 * scale, 16 * scale:28 * scale] = 2
    m[1, 8 * scale:24 * scale, 8 * scale:24 * scale] = 1
    return m


def _batch(masks):
    boxes = torch.tensor([[10 / H, 14 / H, 12 / H, 16 / H],
                          [22 / H, 14 / H, 12 / H, 16 / H],
                          [16 / H, 16 / H, 16 / H, 16 / H]])
    n = masks.shape[-1]
    return {"batch_idx": torch.tensor([0.0, 0.0, 1.0]), "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, n, n)}


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
    return m, m(torch.rand(2, 3, S, S))


def test_bien_thang_hang_luoi_sigma_0_trung_loss_goc(yolo26):
    """Biên nằm đúng mép ô 4x4: tỉ lệ phủ chỉ có 0 và 1, nên NCB (nhãn 128²)
    phải cho đúng loss gốc (nhãn 32²) ở cả hai đầu, kể cả loss ngữ nghĩa."""
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.ncb_loss import build_criterion

    m, preds = yolo26
    goc = E2ELoss(m, v8SegmentationLoss)
    moi = build_criterion(m, {"sigma": 0.0})
    for head in ("one2many", "one2one"):
        a = getattr(goc, head).loss(preds[head], _batch(_rects(1)))
        b = getattr(moi, head).loss(preds[head], _batch(_rects(4).to(torch.uint8)))
        for k in a[1]:
            assert torch.allclose(a[1][k], b[1][k], rtol=1e-5, atol=1e-6), (head, k)


def test_bien_cat_ngang_o_thi_dich_mem(yolo26):
    """Lệch biên 2 px đầu vào (nửa ô): ô biên có đích 0,5, và loss mặt nạ đổi."""
    from cofseg.training.gpr_train import soft_target

    from cofseg.training.ncb_loss import build_criterion

    m, preds = yolo26
    full = _rects(4)
    full[0, :, 16 * 4:16 * 4 + 2] = 1          # tán 1 lấn 2 px sang ô của tán 2
    gt = (full[0] == 1)[None]
    cov = soft_target(gt, 4, None)[0]
    assert cov[10, 16].item() == pytest.approx(0.5)
    assert set(torch.unique(cov).tolist()) == {0.0, 0.5, 1.0}
    crit = build_criterion(m, {"sigma": 0.0})
    a = crit.one2one.loss(preds["one2one"], _batch(_rects(4).to(torch.uint8)))[1]["seg_loss"]
    b = crit.one2one.loss(preds["one2one"], _batch(full.to(torch.uint8)))[1]["seg_loss"]
    assert not torch.isclose(a, b)


def test_sigma_lam_mem_them(yolo26):
    from cofseg.training.ncb_loss import build_criterion

    m, preds = yolo26
    b = _batch(_rects(4).to(torch.uint8))
    a = build_criterion(m, {"sigma": 0.0}).one2one.loss(preds["one2one"], b)[1]
    c = build_criterion(m, {"sigma": 2.4}).one2one.loss(preds["one2one"], b)[1]
    assert not torch.isclose(a["seg_loss"], c["seg_loss"])
    for k in ("box_loss", "cls_loss"):
        assert torch.equal(a[k], c[k])


def test_khong_gioi_han_neo_duong(yolo26):
    """T1 lấy mẫu 64 anchor mỗi ảnh; TH1 dùng hết."""
    from cofseg.training.ncb_loss import build_criterion

    m, preds = yolo26
    crit = build_criterion(m, {"sigma": 2.4})
    assert crit.one2many.max_pos == 0 and crit.one2many.sigma == 2.4
    crit.one2many.loss(preds["one2many"], _batch(_rects(4).to(torch.uint8)))
    assert crit.one2many.stats[0].item() > 3     # đầu one-to-many có nhiều anchor dương mỗi tán


def test_gradient_toi_prototype():
    from cofseg.training.ncb_loss import build_criterion

    m = _model()
    torch.manual_seed(2)
    loss = build_criterion(m, {"sigma": 2.4})(m(torch.rand(2, 3, S, S)), _batch(_rects(4).to(torch.uint8)))[0]
    loss.sum().backward()
    grads = [p.grad for p in m.model[-1].proto.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau():
    from cofseg.training.gpr_train import NCBSegLoss
    from cofseg.training.ncb_loss import build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"sigma": 2.4})
    assert isinstance(crit, NCBSegLoss)
    assert torch.isfinite(crit(m(torch.rand(2, 3, S, S)), _batch(_rects(4).to(torch.uint8)))[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    from cofseg.training.ncb_loss import EpochLog, build_criterion

    m, preds = yolo26
    crit = build_criterion(m, {"sigma": 2.4})
    log = EpochLog(tmp_path / "loss_ncb.csv")
    b = _batch(_rects(4).to(torch.uint8))
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, b)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_ncb.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["one2one_n"]) == 3 and rows[0]["one2one_n"] == rows[1]["one2one_n"]
    assert 0.0 <= float(rows[0]["one2one_iou"]) <= 1.0 and float(rows[0]["one2one_bce"]) > 0.0
