"""A1 (đầu mặt nạ động): phép dựng đúng, chuyển đầu giữ trọng số, loss chỉ nhìn
trong cửa sổ, checkpoint nạp lại được và predictor không cắt sát box.

Model dựng từ yaml của gói ultralytics (yolo26n-seg) với trọng số ngẫu nhiên,
ảnh và nhãn dựng tại chỗ — không đụng data/ hay weights/.
"""

from __future__ import annotations

import copy
import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch.nn.functional as F  # noqa: E402

H = 32  # lưới proto của ảnh 128 px (stride 4)
CFG = {"dims": [8, 8], "coords": True, "window": 1.5, "max_pos": 64}


def _batch():
    """Ba tán hình chữ nhật không chồng nhau: hai ở ảnh 0, một ở ảnh 1."""
    boxes = torch.tensor([[0.25, 0.25, 0.30, 0.30], [0.70, 0.70, 0.35, 0.40], [0.50, 0.50, 0.50, 0.50]])
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    masks = torch.zeros(2, H, H)
    for k, (b, bi) in enumerate(zip(boxes, batch_idx)):
        j = int((batch_idx[:k] == bi).sum())
        cx, cy, w, h = (b * H).tolist()
        masks[int(bi), int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)] = j + 1
    return {"batch_idx": batch_idx, "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, H, H)}


def _stock(seed=0):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(seed)
    m = SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False)
    m.args = get_cfg()
    return m


def _dyn(cfg=CFG, seed=0):
    from cofseg.training.dyn_head import to_dynamic

    m = to_dynamic(_stock(seed), dict(cfg))
    m.train()
    return m


# ------------------------------------------------------------------ phép dựng
def test_so_tham_so_dong():
    from cofseg.training.dyn_head import n_params

    assert n_params(32, [8, 8], True) == 361          # (34*8+8) + (8*8+8) + (8+1)
    assert n_params(32, [], False) == 33              # đầu gốc: 32 hệ số + 1 bias


def _conv_tay(theta, feats, points, strides, dims, coords, fstride):
    """Từng tán một, ghép [F, toạ độ] rồi chạy conv 1x1 thật."""
    from cofseg.training.dyn_head import COORD_SCALE

    c, h, w = feats.shape
    chans = [c + (2 if coords else 0), *dims, 1]
    out = []
    for t, (px, py), s in zip(theta, points, strides):
        x = feats
        if coords:
            xs = ((torch.arange(w) + 0.5) * fstride - px) / (s * COORD_SCALE)
            ys = ((torch.arange(h) + 0.5) * fstride - py) / (s * COORD_SCALE)
            x = torch.cat([feats, xs[None, None, :].expand(1, h, w), ys[None, :, None].expand(1, h, w)])
        k = 0
        ws = []
        for i, o in zip(chans[:-1], chans[1:]):
            ws.append(t[k:k + i * o].view(o, i, 1, 1))
            k += i * o
        x = x[None]
        for li, (wl, o) in enumerate(zip(ws, chans[1:])):
            x = F.conv2d(x, wl, t[k:k + o])
            k += o
            if li < len(ws) - 1:
                x = F.relu(x)
        out.append(x[0, 0])
    return torch.stack(out)


@pytest.mark.parametrize("dims,coords", [([8, 8], True), ([4], False), ([], True)])
def test_dung_conv_viet_tay(dims, coords):
    from cofseg.training.dyn_head import dynamic_logits, n_params

    torch.manual_seed(0)
    c = 5
    feats = torch.randn(c, 6, 7)
    theta = torch.randn(3, n_params(c, dims, coords))
    points = torch.tensor([[3.0, 9.0], [20.0, 1.0], [10.0, 10.0]])
    strides = torch.tensor([8.0, 16.0, 32.0])
    got = dynamic_logits(theta, feats, points, strides, dims=dims, coords=coords, fstride=4.0)
    want = _conv_tay(theta, feats, points, strides, dims, coords, 4.0)
    assert torch.allclose(got, want, atol=1e-5)


def test_tuyen_tinh_khong_toa_do_la_to_hop_prototype():
    """dims=[], coords=False: đúng phép `hệ số @ proto` của YOLO-seg gốc (thêm bias)."""
    from cofseg.training.dyn_head import dynamic_logits

    torch.manual_seed(0)
    feats = torch.randn(32, H, H)
    theta = torch.randn(4, 33)
    got = dynamic_logits(theta, feats, torch.zeros(4, 2), torch.ones(4), dims=[], coords=False, fstride=4.0)
    want = (theta[:, :32] @ feats.view(32, -1)).view(4, H, H) + theta[:, 32, None, None]
    assert torch.allclose(got, want, atol=1e-4)


def test_window_boxes():
    from cofseg.training.dyn_head import window_boxes

    b = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    assert torch.equal(window_boxes(b, 1.5), torch.tensor([[5.0, 10.0, 35.0, 70.0]]))
    assert window_boxes(b, 1.0) is b


# ------------------------------------------------------------------ chuyển đầu
def test_chuyen_dau_chi_doi_lop_sinh_tham_so():
    from cofseg.training.dyn_head import DynSegmentationModel, Segment26Dyn, find_head

    goc = _stock()
    sd0 = {k: v.clone() for k, v in goc.state_dict().items()}
    m = _dyn()
    assert type(m) is DynSegmentationModel and type(find_head(m)) is Segment26Dyn
    sd = m.state_dict()
    doi = sorted(k for k in sd0 if sd0[k].shape != sd[k].shape or not torch.equal(sd0[k], sd[k]))
    assert doi == sorted(f"model.23.{h}.{i}.2.{p}" for h in ("cv4", "one2one_cv4")
                         for i in range(3) for p in ("weight", "bias"))
    assert find_head(m).dyn["n_params"] == 361 and find_head(m).dyn["fstride"] == 4.0


def test_chi_nhan_segment26():
    from ultralytics.nn.tasks import SegmentationModel

    from cofseg.training.dyn_head import to_dynamic

    v8 = SegmentationModel("yolov8n-seg.yaml", nc=1, verbose=False)
    with pytest.raises(SystemExit, match="Segment26"):
        to_dynamic(v8, dict(CFG))


def test_nap_trong_so_sau_khi_chuyen():
    """Thứ tự của trainer: dựng chuẩn -> chuyển -> nạp. Proto26 và mọi lớp trùng
    hình dạng nhận trọng số; chỉ bộ sinh θ giữ khởi tạo mới."""
    from cofseg.training.dyn_head import find_head

    nguon = _stock(seed=7)
    m = _dyn(seed=0)
    theta_truoc = find_head(m).cv4[0][-1].weight.clone()
    m.load(nguon, verbose=False)
    sd, sn = m.state_dict(), nguon.state_dict()
    assert torch.equal(sd["model.23.proto.cv3.conv.weight"], sn["model.23.proto.cv3.conv.weight"])
    assert torch.equal(sd["model.0.conv.weight"], sn["model.0.conv.weight"])
    assert torch.equal(find_head(m).cv4[0][-1].weight, theta_truoc)


def test_dau_ra_train_va_suy_luan():
    m = _dyn()
    preds = m(torch.rand(2, 3, 128, 128))
    assert preds["one2many"]["mask_coefficient"].shape[1] == 361
    assert preds["one2one"]["mask_coefficient"].shape[1] == 361
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(2, 3, 128, 128))[0][0]
    assert y.shape[1] == 4 + 1 + 361 + 3        # box, điểm, θ, x, y, stride


# ------------------------------------------------------------------ loss
def test_loss_huu_han_va_gradient_toi_theta_va_proto():
    from cofseg.training.dyn_head import find_head

    m = _dyn()
    loss, items = m.init_criterion()(m(torch.rand(2, 3, 128, 128)), _batch())
    assert torch.isfinite(loss).all() and items["seg_loss"] > 0
    loss.sum().backward()
    head = find_head(m)
    for g in (head.cv4[0][-1].weight.grad, head.one2one_cv4[0][-1].weight.grad, head.proto.cv3.conv.weight.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_loss_chi_nhin_trong_cua_so():
    """Đổi nhãn ở pixel ngoài mọi cửa sổ thì loss mặt nạ không đổi."""
    m = _dyn()
    preds = m(torch.rand(2, 3, 128, 128))
    crit = m.init_criterion()
    a = crit(preds, _batch())[1]["seg_loss"]
    b = _batch()
    b["masks"][1, 0:3, 0:3] = 1     # gán góc ảnh 1 cho chính tán của nó: ngoài cửa sổ x1.5 (ô 4..28)
    assert crit(preds, b)[1]["seg_loss"] == a
    b["masks"][1, 14:18, 14:18] = 0  # giữa tán: trong cửa sổ, phải đổi
    assert crit(preds, b)[1]["seg_loss"] != a


def test_gioi_han_so_tan_moi_anh():
    m = _dyn({**CFG, "max_pos": 2})
    crit = m.init_criterion()
    crit(m(torch.rand(2, 3, 128, 128)), _batch())
    assert crit.one2many.stats[0].item() <= 4    # 2 ảnh x tối đa 2
    assert crit.one2one.stats[0].item() == 3     # một anchor dương mỗi tán, dưới mức chặn


def test_nhat_ky_epoch(tmp_path):
    from cofseg.training.dyn_train import EpochLog

    m = _dyn()
    preds = m(torch.rand(2, 3, 128, 128))
    crit = m.init_criterion()
    log = EpochLog(tmp_path / "mask_head_dyn.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, _batch())
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "mask_head_dyn.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["one2one_n"] == rows[1]["one2one_n"] == "3"
    assert 0.0 <= float(rows[0]["one2one_iou"]) <= 1.0 and float(rows[0]["one2one_bce"]) > 0


# ------------------------------------------------------------------ suy luận
def _theta_luon_duong(dyn, n):
    """θ cho logit = +10 ở mọi pixel: mọi trọng số 0, bias lớp cuối 10."""
    t = torch.zeros(n, dyn["n_params"])
    t[:, -1] = 10.0
    return t


def test_mat_na_vuot_box_toi_mep_cua_so():
    """Không cắt sát box: logit dương khắp nơi thì mặt nạ đúng bằng cửa sổ x1.5."""
    from cofseg.training.dyn_head import find_head, process_mask_dyn, process_mask_native_dyn

    dyn = find_head(_dyn()).dyn
    proto = torch.randn(32, H, H)
    box = torch.tensor([[40.0, 40.0, 80.0, 80.0]])               # 10x10 ô lưới
    extra = torch.cat([_theta_luon_duong(dyn, 1), torch.tensor([[60.0, 60.0, 8.0]])], 1)
    m = process_mask_dyn(proto, extra, box, (128, 128), dyn=dyn)
    assert m.shape == (1, H, H) and int(m.sum()) == 15 * 15     # cửa sổ 30..90 px = ô 7.5..22.5
    m = process_mask_native_dyn(proto, extra, box, (128, 128), dyn=dyn)
    assert int(m.sum()) == 60 * 60
    assert m[0, 35, 60] == 1 and m[0, 25, 60] == 0             # ngoài box, trong cửa sổ / ngoài cửa sổ


def test_checkpoint_nap_lai_va_predictor(tmp_path):
    """Lưu như trainer, nạp bằng YOLO(), predictor A1 ra mặt nạ ở cỡ ảnh gốc."""
    import numpy as np
    from ultralytics import YOLO

    from cofseg.training.dyn_head import DynSegPredictor, is_dynamic

    m = _dyn()
    m.names = {0: "crown"}
    p = tmp_path / "a1.pt"
    torch.save({"model": copy.deepcopy(m).half(), "train_args": {"task": "segment", "imgsz": 128}}, p)
    y = YOLO(str(p))
    assert is_dynamic(y.model) and y.task == "segment"
    img = (np.random.default_rng(0).random((96, 160, 3)) * 255).astype(np.uint8)
    r = y.predict(img, predictor=DynSegPredictor, conf=0.0, max_det=5, imgsz=128,
                  retina_masks=True, verbose=False)[0]
    assert type(y.predictor) is DynSegPredictor
    assert r.masks is not None and r.masks.data.shape[1:] == (96, 160)
