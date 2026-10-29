"""Giám sát mặt nạ ngoài hộp: hệ số 1 trùng loss gốc, chỉ loss mặt nạ đổi, vành được giám sát, ngoài hộp giãn
thì không, chia diện tích hộp gốc, nhật ký epoch."""

from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from distill.outside_box import EpochLog, OutsideBoxSegLoss, _axes, build_criterion  # noqa: E402

H = 32


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
    return m, m(torch.rand(2, 3, 128, 128))


def _stock(m):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    return E2ELoss(m, v8SegmentationLoss)


def test_he_so_1_trung_tung_so_voi_loss_goc(yolo26, batch):
    m, preds = yolo26
    goc = _stock(m)(preds, batch)
    moi = build_criterion(m, {"train_expand": 1.0, "infer_expand": 1.0})(preds, batch)
    assert torch.equal(moi[0], goc[0])
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_gian_hop_chi_doi_loss_mat_na(yolo26, batch):
    m, preds = yolo26
    goc = _stock(m)(preds, batch)[1]
    moi = build_criterion(m, {"train_expand": 1.25, "infer_expand": 1.1})(preds, batch)[1]
    for k in goc:
        if k == "seg_loss":
            assert moi[k] > goc[k]          # vành cộng thêm số hạng không âm
        else:
            assert torch.equal(moi[k], goc[k]), k


def _single(expand, logit, gt, box):
    crit = OutsideBoxSegLoss.__new__(OutsideBoxSegLoss)
    crit.expand = expand
    crit.stats = torch.zeros(4, dtype=torch.float64)
    proto = logit[None].clone().requires_grad_(True)          # 1 prototype = chính bản đồ logit
    pred = torch.ones(1, 1)
    area = ((box[0, 2] - box[0, 0]) * (box[0, 3] - box[0, 1]) / (H * H))[None]
    return crit, crit.single_mask_loss(gt[None], pred, proto, box, area), proto


def test_tong_trong_hop_gian_chia_dien_tich_hop_goc():
    import torch.nn.functional as F

    logit = torch.randn(H, H, generator=torch.Generator().manual_seed(0))
    gt = torch.zeros(H, H)
    gt[10:18, 10:18] = 1
    box = torch.tensor([[10.0, 10.0, 18.0, 18.0]])            # cạnh 8 ô, tâm 14
    _, loss, _ = _single(1.25, logit, gt, box)
    bce = F.binary_cross_entropy_with_logits(logit, gt, reduction="none")
    want = bce[9:19, 9:19].sum() / (H * H) / (64 / (H * H))   # hộp giãn: cạnh 10 ô, 9..19
    assert loss.item() == pytest.approx(want.item(), rel=1e-5)


def test_gradient_co_o_vanh_khong_co_ngoai_hop_gian():
    logit = torch.zeros(H, H)
    gt = torch.zeros(H, H)
    gt[10:18, 10:18] = 1
    box = torch.tensor([[10.0, 10.0, 18.0, 18.0]])
    _, loss, proto = _single(1.25, logit, gt, box)
    loss.backward()
    g = proto.grad[0].abs()
    assert g[9, 14] > 0 and g[14, 18] > 0                     # vành (ngoài hộp nhãn, trong hộp giãn)
    assert g[8, 14] == 0 and g[14, 19] == 0 and g[0, 0] == 0  # ngoài hộp giãn
    _, loss1, proto1 = _single(1.0, logit, gt, box)
    loss1.backward()
    assert proto1.grad[0][9, 14] == 0                         # loss gốc: vành tự do


def test_hop_gian_ket_trong_luoi():
    logit = torch.zeros(H, H)
    gt = torch.zeros(H, H)
    gt[0:8, 0:8] = 1
    box = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    _, loss, proto = _single(1.25, logit, gt, box)
    loss.backward()
    assert proto.grad[0][0, 8] > 0 and proto.grad[0][0, 9] == 0   # hộp giãn [-1, 9) kẹp về [0, 9)


def test_truc_hang_cot_dung_quy_uoc_crop_mask():
    from ultralytics.utils.ops import crop_mask

    box = torch.tensor([[2.5, 3.0, 7.2, 9.9]])
    rows, cols = _axes(box, 12, 10)
    want = crop_mask(torch.ones(1, 12, 10), box)
    assert torch.equal(rows[:, :, None] * cols[:, None, :], want)


def test_nhat_ky_ghi_trong_vanh_va_tran(yolo26, batch, tmp_path):
    m, preds = yolo26
    crit = build_criterion(m, {"train_expand": 1.25, "infer_expand": 1.1})
    log = EpochLog(tmp_path / "loss_outside_box.csv")
    log.start(crit, 0)
    crit(preds, batch)
    log.end(crit, 0)
    rows = list(csv.DictReader(open(tmp_path / "loss_outside_box.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["epoch"] == "1"
    for head in ("one2one", "one2many"):
        assert int(rows[0][f"{head}_n"]) > 0
        assert float(rows[0][f"{head}_ring"]) > 0
        assert 0.0 <= float(rows[0][f"{head}_leak"]) <= 1.0


def test_dung_duoc_tren_tro(teacher, batch):
    from ultralytics.cfg import get_cfg

    from distill.blocks import check_student
    from distill.student import build_student

    m, _ = build_student(check_student({}), teacher.yaml, nc=1, teacher=teacher)
    m.args = get_cfg()
    m.train()
    preds = m(torch.rand(2, 3, 128, 128))
    loss, _ = build_criterion(m, {"train_expand": 1.25, "infer_expand": 1.1})(preds, batch)
    assert torch.isfinite(loss).all()
