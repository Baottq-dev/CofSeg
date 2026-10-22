"""LM1 (`--loss lm1`): term xếp hạng một chiều, chỉ đẩy logit của ứng viên vẽ đúng
ở tán đang thua mức nền thật; λ = 0 trùng loss gốc từng số.

Phần chọn neo và mức nền kiểm bằng số dựng tay; phần gắn vào loss kiểm trên
model dựng từ yaml của gói ultralytics với trọng số ngẫu nhiên, như test của TH1–TH5.
"""

from __future__ import annotations

import csv
import math

import pytest

from cofseg.training.yolo import YoloTrainer

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32  # lưới proto của ảnh 128 px
DEFAULTS = {"weight": 0.5, "margin": 1.0, "temp": 0.25, "k": 3, "start_epoch": 5, "ramp_epochs": 5,
            "bg_iou": 0.1, "bg_top": 300, "q_min": 0.5}


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


def test_bat_bang_co_va_nhan():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "lm1") == "lm1"
    assert cfg["loss"]["lm1"] == DEFAULTS
    assert YoloTrainer.apply_loss(_cfg(loss={"lm1": {"weight": 1.0, "k": 5}}), "lm1") == "lm1-w1-k5"


@pytest.mark.parametrize("bad", [{"weight": -1.0}, {"q_min": 1.5}, {"bg_iou": -0.1}, {"k": 0},
                                 {"bg_top": 2}, {"temp": 0.0}])
def test_tham_so_sai_bi_chan(bad):
    with pytest.raises(SystemExit, match="loss.lm1"):
        YoloTrainer.apply_loss(_cfg(loss={"lm1": bad}), "lm1")


# ------------------------------------------------------------------ chọn neo, cổng, term
def test_chon_i_sao_cong_va_gradient():
    """Bốn tán, mức nền τ = 0, Δ = 1:
    - tán 0: neo điểm cao nhất (z 3) vẽ sai (q 0,3), neo vẽ đúng nhất (q 0,8) điểm −5 → lệch điểm–mặt nạ, bật;
    - tán 1: một neo q 0,9, z 4 → đã vượt nền, không bật;
    - tán 2: hai neo cùng q 0,7, z −1 và −2 → i* là neo z −1, thua nền, bật;
    - tán 3: q 0,2 < q_min → không đẩy."""
    from cofseg.training.lm1_loss import crown_terms

    z = torch.tensor([-5.0, 2.0, 3.0, 4.0, -2.0, -1.0, -6.0], requires_grad=True)
    q = torch.tensor([0.8, 0.6, 0.3, 0.9, 0.7, 0.7, 0.2])
    g = torch.tensor([0, 0, 0, 1, 2, 2, 3])
    term, c = crown_terms(z, q, g, 0.0, margin=1.0, temp=0.25, q_min=0.5)
    want = sum(0.25 * math.log1p(math.exp((1.0 - zz) / 0.25)) for zz in (-5.0, -1.0))
    assert term.item() == pytest.approx(want, rel=1e-6)
    assert c == {"n": 4, "r": 3, "gate": 2, "mis": 1, "lose0": 1, "gap": pytest.approx(6.0 + 2.0)}
    term.backward()
    nz = torch.nonzero(z.grad)[:, 0].tolist()
    assert nz == [0, 5] and (z.grad[[0, 5]] < 0).all()  # chỉ đẩy lên, và chỉ ở i* của tán bị bật


def test_tan_vuot_nen_xa_thi_khong_co_gradient():
    from cofseg.training.lm1_loss import crown_terms

    z = torch.tensor([5.0, 6.0], requires_grad=True)
    term, c = crown_terms(z, torch.tensor([0.9, 0.8]), torch.tensor([0, 1]), 0.0, margin=1.0, temp=0.25, q_min=0.5)
    assert term.item() == 0.0 and c["gate"] == 0


def _scene():
    """Lưới 16x16, mặt nạ của mỗi neo = đúng hộp của nó (proto kênh 0 = 1, hệ số [1, 0]).
    Tán nhãn ở [0:6, 0:6]."""
    boxes = torch.tensor([[0, 0, 6, 6],        # 0 neo dương, z 5
                          [0, 0, 6, 5.9],      # 1 bóng NMS của neo dương, z 4
                          [3, 3, 9, 9],        # 2 mảnh chạm tán (IoU mặt nạ 0,14), z 3,5
                          [10, 10, 14, 14],    # 3 nền, z 3
                          [10, 0, 14, 4],      # 4 nền, z 2
                          [0, 10, 4, 14],      # 5 nền, z 1
                          [12, 12, 16, 16]],   # 6 nền, z 0,5
                         dtype=torch.float)
    z = torch.tensor([5.0, 4.0, 3.5, 3.0, 2.0, 1.0, 0.5])
    pos = torch.tensor([True, False, False, False, False, False, False])
    coef = torch.tensor([[1.0, 0.0]]).repeat(7, 1)
    proto = torch.stack([torch.ones(16, 16), -torch.ones(16, 16)])
    gt = torch.zeros(1, 16, 16)
    gt[0, :6, :6] = 1
    return z, boxes, pos, coef, proto, gt, torch.ones(4)


def test_muc_nen_bo_bong_nms_va_manh_tan():
    from cofseg.training.lm1_loss import background_level

    z, boxes, pos, coef, proto, gt, scale = _scene()
    assert background_level(z, boxes, pos, coef, proto, gt, scale, k=3, top=300, bg_iou=0.1) == pytest.approx(1.0)
    # không lọc mảnh chạm tán thì mảnh (3,5) chen lên đầu: 3,5 / 3 / 2
    assert background_level(z, boxes, pos, coef, proto, gt, scale, k=3, top=300, bg_iou=1.01) == pytest.approx(2.0)
    # chỉ xét 3 neo âm sạch điểm cao nhất (mảnh, 3, 2) thì chỉ còn 2 neo nền: không đủ k = 3
    assert background_level(z, boxes, pos, coef, proto, gt, scale, k=3, top=3, bg_iou=0.1) is None


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


def _p(**kw):
    return {**DEFAULTS, **kw}


# Đẩy mạnh để trên model ngẫu nhiên chắc chắn có tán bị bật: mọi tán vào tập R (q_min 0),
# biên lớn, λ đầy ngay từ epoch 0.
FORCE = dict(q_min=0.0, margin=20.0, start_epoch=0, ramp_epochs=0, weight=1.0)


def test_lambda_0_trung_loss_goc():
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.lm1_loss import build_criterion

    m = _model()
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    goc = E2ELoss(m, v8SegmentationLoss)(preds, _batch())
    for p in (_p(weight=0.0), _p()):  # λ = 0, và λ chưa tăng (epoch 0 < start_epoch 5)
        moi = build_criterion(m, p)(preds, _batch())
        assert torch.equal(moi[0], goc[0])
        for k in goc[1]:
            assert torch.equal(moi[1][k], goc[1][k]), k


def test_chi_cong_vao_loss_diem_va_chi_lan_qua_nhanh_diem():
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    from cofseg.training.lm1_loss import build_criterion

    m = _model()
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    goc = E2ELoss(m, v8SegmentationLoss)(preds, _batch())
    crit = build_criterion(m, _p(**FORCE))
    moi = crit(preds, _batch())
    extra = crit.one2many.extra
    assert extra is not None and extra.item() > 0
    diff = moi[0] - goc[0]
    assert diff[2].item() == pytest.approx(extra.item(), rel=1e-5)        # cộng vào phần điểm
    assert torch.equal(diff[[0, 1, 3, 4]], torch.zeros(4))                # hộp, mặt nạ, dfl, ngữ nghĩa giữ nguyên

    m.zero_grad()
    extra.backward()
    head = m.model[-1]

    def grads(mod):
        return [p.grad for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0]

    assert grads(head.cv3)                                                  # nhánh điểm one-to-many nhận gradient
    for name in ("cv2", "cv4", "proto", "one2one_cv2", "one2one_cv3", "one2one_cv4"):
        assert not grads(getattr(head, name)), name                          # hộp, hệ số mặt nạ, Proto, đầu one2one: không
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_model_mot_dau():
    from ultralytics.utils.loss import v8SegmentationLoss

    from cofseg.training.lm1_loss import LM1SegLoss, build_criterion

    m = _model("yolov8n-seg.yaml")
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    crit = build_criterion(m, _p(**FORCE))
    assert isinstance(crit, LM1SegLoss) and crit.standalone
    moi = crit(preds, _batch())
    goc = v8SegmentationLoss(m)(preds, _batch())
    assert torch.isfinite(moi[0]).all()
    assert (moi[0] - goc[0])[2].item() == pytest.approx(crit.extra.item(), rel=1e-5)


def test_nhat_ky_epoch(tmp_path):
    from cofseg.training.lm1_loss import EpochLog, build_criterion

    m = _model()
    torch.manual_seed(1)
    preds = m(torch.rand(2, 3, 128, 128))
    crit = build_criterion(m, _p(q_min=0.0, margin=20.0, start_epoch=1, ramp_epochs=2, weight=0.5))
    log = EpochLog(tmp_path / "loss_lm1.csv")
    for ep in range(3):
        log.start(crit, ep)
        crit(preds, _batch())
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "loss_lm1.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert [float(r["lambda"]) for r in rows] == [0.0, 0.0, 0.25]
    assert int(rows[0]["n_crowns"]) == 3
    for col in ("share_r", "share_gate", "share_mis", "share_lose0"):
        assert 0.0 <= float(rows[0][col]) <= 1.0, col
    assert float(rows[0]["share_gate"]) == 1.0 and float(rows[0]["lm1"]) > 0
