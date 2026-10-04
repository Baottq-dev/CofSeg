"""L1 (`--loss mask_iou`): đổi đúng một thứ, và không đổi gì khi tắt.

Model dựng từ yaml của gói ultralytics (yolo26n-seg, yolov8n-seg) với trọng số
ngẫu nhiên, ảnh và nhãn dựng tại chỗ — không đụng data/ hay weights/.
"""

from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32  # lưới proto của ảnh 128 px (stride 4)


def _batch():
    """Ba tán hình chữ nhật không chồng nhau: hai ở ảnh 0, một ở ảnh 1.

    overlap_mask=True nên mặt nạ GT là bản đồ chỉ số: tán thứ j của một ảnh
    mang giá trị j+1, đúng thứ tự gt trong batch — y như Format của ultralytics.
    """
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
    m.args = get_cfg()  # box, cls, dfl, overlap_mask, epochs: đúng chỗ hàm loss đọc
    m.train()
    return m


@pytest.fixture(scope="module")
def yolo26():
    m = _model("yolo26n-seg.yaml")
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    return m, _batch(), preds


def _stock(m):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    return E2ELoss(m, v8SegmentationLoss)


def test_mix_0_trung_tung_so_voi_loss_goc(yolo26):
    """Phép kiểm hồi quy quan trọng nhất: tắt L1 thì không lệch một bit nào."""
    from cofseg.training.mask_iou_loss import build_criterion

    m, batch, preds = yolo26
    goc = _stock(m)(preds, batch)
    moi = build_criterion(m, {"mix": 0.0, "warmup_epochs": 0, "heads": "both"})(preds, batch)
    assert torch.equal(moi[0], goc[0])
    assert goc[1].keys() == moi[1].keys()
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_mix_1_chi_doi_loss_phan_loai(yolo26):
    """Hộp, mặt nạ, ngữ nghĩa giữ nguyên; chỉ loss phân loại khác."""
    from cofseg.training.mask_iou_loss import build_criterion

    m, batch, preds = yolo26
    goc = _stock(m)(preds, batch)[1]
    moi = build_criterion(m, {"mix": 1.0, "warmup_epochs": 0, "heads": "both"})(preds, batch)[1]
    for k in goc:
        if k == "cls_loss":
            assert not torch.equal(moi[k], goc[k])
        else:
            assert torch.equal(moi[k], goc[k]), k


def test_warmup_epoch_0_la_loss_goc(yolo26):
    """w tăng tuyến tính từ 0: epoch đầu tiên vẫn là loss gốc."""
    from cofseg.training.mask_iou_loss import build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"mix": 1.0, "warmup_epochs": 5, "heads": "both"})
    assert crit.one2one.weight == 0.0
    assert torch.equal(crit(preds, batch)[0], _stock(m)(preds, batch)[0])
    crit.one2one.epoch = 2
    assert crit.one2one.weight == pytest.approx(0.4)
    crit.one2one.epoch = 9
    assert crit.one2one.weight == 1.0


def test_iou_mat_na_cat_bang_hop_du_doan():
    """Hai hình vuông 10x10 lệch nhau 5 cột: IoU 50/150. Hộp dự đoán cắt bớt mặt
    nạ thì IoU phải tính trên phần còn lại — đúng như lúc suy luận."""
    from cofseg.training.mask_iou_loss import mask_iou

    pred = torch.full((1, H, H), -5.0)
    pred[0, 4:14, 4:14] = 5.0
    gt = torch.zeros(1, H, H)
    gt[0, 4:14, 9:19] = 1.0
    hop = lambda x2: torch.tensor([[0.0, 0.0, x2, float(H)]])
    assert mask_iou(pred, hop(H), gt).item() == pytest.approx(50 / 150)
    # cắt còn cột 4..11: 80 px, giao 30 px -> 30 / (80 + 100 - 30)
    assert mask_iou(pred, hop(12.0), gt).item() == pytest.approx(30 / 150)
    # cắt còn cột 4..8: không giao
    assert mask_iou(pred, hop(9.0), gt).item() == 0.0


def test_heads_o2o_tra_dau_one_to_many_ve_loss_goc(yolo26):
    from ultralytics.utils.loss import v8SegmentationLoss

    from cofseg.training.mask_iou_loss import MaskIoUSegLoss, build_criterion

    m, _, _ = yolo26
    crit = build_criterion(m, {"mix": 1.0, "warmup_epochs": 0, "heads": "o2o"})
    assert type(crit.one2many) is v8SegmentationLoss
    assert isinstance(crit.one2one, MaskIoUSegLoss)
    assert crit.one2many.assigner.topk == 10 and crit.one2one.assigner.topk2 == 1


def test_model_mot_dau_dung_thang_lop_loss():
    """YOLOv8/11 không có đầu one-to-one: hàm loss là MaskIoUSegLoss trần."""
    from cofseg.training.mask_iou_loss import MaskIoUSegLoss, build_criterion, is_end2end

    m = _model("yolov8n-seg.yaml")
    assert not is_end2end(m)
    crit = build_criterion(m, {"mix": 1.0, "warmup_epochs": 0, "heads": "both"})
    assert isinstance(crit, MaskIoUSegLoss)
    out = crit(m(torch.rand(2, 3, 128, 128)), _batch())
    assert torch.isfinite(out[0]).all()


def test_nhat_ky_epoch(yolo26, tmp_path):
    """Mỗi epoch một dòng, có số anchor dương, q, điểm và tương quan."""
    from cofseg.training.mask_iou_loss import EpochLog, build_criterion

    m, batch, preds = yolo26
    crit = build_criterion(m, {"mix": 1.0, "warmup_epochs": 0, "heads": "both"})
    log = EpochLog(tmp_path / "loss_mask_iou.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, batch)
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_mask_iou.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert int(rows[0]["one2one_n"]) == 3          # một anchor dương mỗi tán
    assert int(rows[0]["one2many_n"]) > 3
    assert rows[0]["one2one_n"] == rows[1]["one2one_n"]   # start() xoá số của epoch trước
    assert 0.0 <= float(rows[0]["one2one_q"]) <= 1.0
