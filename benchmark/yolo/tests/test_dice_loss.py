"""L2 (`--loss dice`): chỉ loss mặt nạ đổi, tắt thì không đổi gì, và Dice tính đúng.

Model dựng từ yaml của gói ultralytics với trọng số ngẫu nhiên, ảnh và nhãn
dựng tại chỗ — cùng cách với test của L1, không đụng data/ hay weights/.
"""

from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32  # lưới proto của ảnh 128 px (stride 4)


def _batch():
    """Ba tán hình chữ nhật không chồng nhau: hai ở ảnh 0, một ở ảnh 1 (giống
    test_mask_iou_loss; thư mục test không phải package nên không import chéo)."""
    boxes = torch.tensor([[0.25, 0.25, 0.30, 0.30], [0.70, 0.70, 0.35, 0.40], [0.50, 0.50, 0.50, 0.50]])
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    masks = torch.zeros(2, H, H)
    for k, (b, bi) in enumerate(zip(boxes, batch_idx)):
        j = int((batch_idx[:k] == bi).sum())
        cx, cy, w, h = (b * H).tolist()
        masks[int(bi), int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)] = j + 1
    return {"batch_idx": batch_idx, "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, H, H)}


def _model(yaml_name: str):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    m = SegmentationModel(yaml_name, nc=1, verbose=False)
    m.args = get_cfg()
    m.train()
    return m


def _stock(m):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    return E2ELoss(m, v8SegmentationLoss)


@pytest.fixture(scope="module")
def yolo26():
    m = _model("yolo26n-seg.yaml")
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    return m, _batch(), preds


def test_weight_0_trung_tung_so_voi_loss_goc(yolo26):
    """Phép kiểm hồi quy quan trọng nhất: tắt Dice thì không lệch một bit nào."""
    from cofseg.training.dice_loss import build_criterion

    m, batch, preds = yolo26
    goc = _stock(m)(preds, batch)
    moi = build_criterion(m, {"weight": 0.0})(preds, batch)
    assert torch.equal(moi[0], goc[0])
    assert goc[1].keys() == moi[1].keys()
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_weight_1_chi_doi_loss_mat_na(yolo26):
    """Hộp, phân loại, ngữ nghĩa giữ nguyên; loss mặt nạ tăng thêm phần Dice."""
    from cofseg.training.dice_loss import build_criterion

    m, batch, preds = yolo26
    goc = _stock(m)(preds, batch)[1]
    moi = build_criterion(m, {"weight": 1.0})(preds, batch)[1]
    for k in goc:
        if k == "seg_loss":
            assert moi[k] > goc[k]
        else:
            assert torch.equal(moi[k], goc[k]), k


def _vuong(r0, r1, c0, c1, H_=H):
    t = torch.zeros(1, H_, H_)
    t[0, r0:r1, c0:c1] = 1.0
    return t


def test_dice_va_iou_tren_mat_na_dung_tay():
    """Hai hình vuông 10x10 (100 px): trùng, lệch nửa, rời nhau.

    Logit ±20 nên sigmoid gần như đúng 0/1; smooth = 1 ở cả tử và mẫu.
    """
    from cofseg.training.dice_loss import dice_and_iou

    pred = _vuong(4, 14, 4, 14) * 40 - 20
    ca_luoi = torch.tensor([[0.0, 0.0, float(H), float(H)]])

    d, i = dice_and_iou(pred, _vuong(4, 14, 4, 14), ca_luoi)
    assert d.item() == pytest.approx(0.0, abs=1e-6) and i.item() == 1.0

    d, i = dice_and_iou(pred, _vuong(4, 14, 9, 19), ca_luoi)     # giao 50
    assert d.item() == pytest.approx(1 - 101 / 201, abs=1e-6)
    assert i.item() == pytest.approx(50 / 150)

    d, i = dice_and_iou(pred, _vuong(20, 30, 20, 30), ca_luoi)   # rời nhau
    assert d.item() == pytest.approx(1 - 1 / 201, abs=1e-6) and i.item() == 0.0


def test_phan_tran_ngoai_box_khong_bi_phat():
    """Vùng giám sát là box GT, như BCE gốc: mặt nạ dự đoán tràn sang phải box
    không làm Dice tăng. Nới vùng là việc của L3."""
    from cofseg.training.dice_loss import dice_and_iou

    gt = _vuong(4, 14, 4, 14)
    tran = _vuong(4, 14, 4, 24) * 40 - 20
    box = torch.tensor([[4.0, 4.0, 14.0, 14.0]])
    d, i = dice_and_iou(tran, gt, box)
    assert d.item() == pytest.approx(0.0, abs=1e-6) and i.item() == 1.0
    # cùng mặt nạ, box phủ cả lưới: phần tràn bị tính
    d, _ = dice_and_iou(tran, gt, torch.tensor([[0.0, 0.0, float(H), float(H)]]))
    assert d.item() == pytest.approx(1 - 201 / 301, abs=1e-6)


def test_fp16_box_ca_luoi_khong_tran():
    """Lưới proto 256x256 của ảnh 1024: tổng 65 536 > 65 504 của fp16."""
    from cofseg.training.dice_loss import dice_and_iou

    n = 256
    logit = torch.full((1, n, n), 20.0).half()
    d, i = dice_and_iou(logit, torch.ones(1, n, n), torch.tensor([[0.0, 0.0, float(n), float(n)]]))
    assert torch.isfinite(d).all() and d.item() == pytest.approx(0.0, abs=1e-6)
    assert i.item() == 1.0


def test_gradient_chay_qua_dice():
    """crop_mask nhân tại chỗ: cắt nhầm lên sigmoid hay gt là autograd báo lỗi
    ở lượt lùi. Chạy backward thật và gradient phải tới được prototype."""
    from cofseg.training.dice_loss import build_criterion

    m = _model("yolo26n-seg.yaml")
    torch.manual_seed(2)
    loss = build_criterion(m, {"weight": 1.0})(m(torch.rand(2, 3, 128, 128)), _batch())[0]
    loss.sum().backward()
    proto = m.model[-1].proto
    grads = [p.grad for p in proto.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_mot_dau_dung_thang_lop_loss():
    """YOLOv8/11 không có đầu one-to-one: hàm loss là DiceSegLoss trần."""
    from cofseg.training.dice_loss import DiceSegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    crit = build_criterion(m, {"weight": 1.0})
    assert isinstance(crit, DiceSegLoss)
    out = crit(m(torch.rand(2, 3, 128, 128)), _batch())
    assert torch.isfinite(out[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    """Mỗi epoch một dòng: số anchor dương, BCE, 1 − Dice, IoU cứng của từng đầu."""
    from cofseg.training.dice_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"weight": 1.0})
    log = EpochLog(tmp_path / "loss_dice.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_dice.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["one2one_n"]) == 3          # một anchor dương mỗi tán
    assert int(rows[0]["one2many_n"]) > 3
    assert rows[0]["one2one_n"] == rows[1]["one2one_n"]   # start() xoá số của epoch trước
    for col in ("one2one_dice", "one2one_iou"):
        assert 0.0 <= float(rows[0][col]) <= 1.0
    assert float(rows[0]["one2one_bce"]) > 0.0
