"""KT4 (toạ độ tương đối trong phép ghép mặt nạ): phép dựng đúng, chuyển đầu giữ
trọng số và trạng thái ngẫu nhiên, khởi đầu trùng R2 từng số, hộp dự đoán không
nhận gradient từ loss mặt nạ, checkpoint nạp lại được và predictor dựng đúng.

Model dựng từ yaml của gói ultralytics (yolo26n-seg) với trọng số ngẫu nhiên,
ảnh và nhãn dựng tại chỗ — không đụng data/ hay weights/.
"""

from __future__ import annotations

import copy
import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

H = 32  # lưới proto của ảnh 128 px (stride 4)


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


def _coord(box="pred", seed=0):
    from cofseg.training.coord_head import to_coord

    m = to_coord(_stock(seed), {"type": "coord", "box": box})
    m.train()
    return m


# ------------------------------------------------------------------ phép dựng
def test_toa_do_trong_hop():
    from cofseg.training.coord_head import coord_axes

    box = torch.tensor([[40.0, 20.0, 80.0, 100.0]])            # tâm (60, 60), nửa cạnh 20 x 40
    u, v = coord_axes(box, 32, 32, 4.0)
    xs = (torch.arange(32) + 0.5) * 4
    assert torch.allclose(u[0], ((xs - 60) / 20).clamp(-2, 2))
    assert torch.allclose(v[0], ((xs - 60) / 40).clamp(-2, 2))
    assert u.abs().max() == 2.0                                # kẹp
    # hộp suy biến: nửa cạnh không nhỏ hơn một ô
    u, _ = coord_axes(torch.tensor([[50.0, 50.0, 50.0, 50.0]]), 32, 32, 4.0)
    assert torch.isfinite(u).all() and u.abs().max() == 2.0


def test_phan_toa_do_dung_ban_do():
    from cofseg.training.coord_head import coord_axes, coord_maps, coord_term

    torch.manual_seed(0)
    boxes = torch.tensor([[10.0, 30.0, 70.0, 90.0], [0.0, 0.0, 20.0, 50.0], [64.0, 8.0, 120.0, 40.0]])
    c = torch.randn(3, 5)
    u, v = coord_axes(boxes, 24, 40, 4.0)
    got = coord_term(c, u, v)
    want = torch.einsum("nk,nkhw->nhw", c, coord_maps(boxes, 24, 40, 4.0))
    assert got.shape == (3, 24, 40) and torch.allclose(got, want, atol=1e-5)


def test_he_so_toa_do_bang_0_la_phep_dung_goc():
    from cofseg.training.coord_head import mask_logits

    torch.manual_seed(0)
    proto = torch.randn(32, H, H)
    coefs = torch.cat([torch.randn(4, 32), torch.zeros(4, 5)], 1)
    boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0]]).repeat(4, 1)
    want = (coefs[:, :32] @ proto.view(32, -1)).view(4, H, H)
    assert torch.allclose(mask_logits(proto, coefs, boxes, 4.0), want, atol=1e-5)


# ------------------------------------------------------------------ chuyển đầu
def test_chuyen_dau_giu_trong_so_va_trang_thai_ngau_nhien():
    from cofseg.training.coord_head import CoordSegmentationModel, Segment26Coord, find_head, to_coord

    goc = _stock()
    sd0 = {k: v.clone() for k, v in goc.state_dict().items()}
    rng = torch.get_rng_state()
    m = to_coord(goc, {"type": "coord", "box": "pred"})
    assert torch.equal(torch.get_rng_state(), rng)
    assert type(m) is CoordSegmentationModel and type(find_head(m)) is Segment26Coord
    sd = m.state_dict()
    doi = sorted(k for k in sd0 if sd0[k].shape != sd[k].shape or not torch.equal(sd0[k], sd[k]))
    assert doi == sorted(f"model.23.{h}.{i}.2.{p}" for h in ("cv4", "one2one_cv4")
                         for i in range(3) for p in ("weight", "bias"))
    for k in doi:
        assert sd[k].shape[0] == 37
        assert torch.equal(sd[k][:32], sd0[k]) and not sd[k][32:].any()
    assert find_head(m).kt4["fstride"] == 4.0 and find_head(m).kt4["box"] == "pred"


def test_chi_nhan_segment26():
    from ultralytics.nn.tasks import SegmentationModel

    from cofseg.training.coord_head import to_coord

    v8 = SegmentationModel("yolov8n-seg.yaml", nc=1, verbose=False)
    with pytest.raises(SystemExit, match="Segment26"):
        to_coord(v8, {"type": "coord", "box": "pred"})


def test_dau_ra_train_va_suy_luan():
    m = _coord()
    preds = m(torch.rand(2, 3, 128, 128))
    assert preds["one2many"]["mask_coefficient"].shape[1] == 37
    assert preds["one2one"]["mask_coefficient"].shape[1] == 37
    m.eval()
    with torch.no_grad():
        (y, proto), _ = m(torch.rand(2, 3, 128, 128))
    assert y.shape[1] == 4 + 1 + 37 and proto.shape[1] == 32


# ------------------------------------------------------------------ loss
@pytest.mark.parametrize("box", ["pred", "gt"])
def test_khoi_dau_trung_loss_goc(box):
    """5 hệ số toạ độ bằng 0 lúc khởi đầu: loss của KT4 bằng loss gốc từng số."""
    goc = _stock()
    m = _coord(box)
    goc.train()
    x = torch.rand(2, 3, 128, 128)
    a, ia = goc.init_criterion()(goc(x), _batch())
    b, ib = m.init_criterion()(m(x), _batch())
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    assert ia.keys() == ib.keys()
    for k in ia:
        assert torch.allclose(ia[k], ib[k], rtol=1e-5, atol=1e-6), k


def test_he_so_toa_do_nhan_gradient():
    from cofseg.training.coord_head import find_head

    m = _coord()
    loss, items = m.init_criterion()(m(torch.rand(2, 3, 128, 128)), _batch())
    assert torch.isfinite(loss).all() and items["seg_loss"] > 0
    loss.sum().backward()
    head = find_head(m)
    for conv in (head.cv4[0][-1], head.one2one_cv4[0][-1]):
        g = conv.weight.grad
        assert g is not None and torch.isfinite(g).all() and g[32:].abs().sum() > 0


def test_hop_du_doan_khong_nhan_gradient_tu_loss_mat_na():
    """box: pred — toạ độ lấy từ hộp dự đoán đã tách gradient: phần loss mặt nạ
    không đẩy gì vào đầu ra hộp."""
    from cofseg.training.coord_head import find_head

    m = _coord("pred")
    head = find_head(m)
    with torch.no_grad():                                      # cho toạ độ thật sự góp vào logit
        for conv in (head.cv4[0][-1], head.cv4[1][-1], head.cv4[2][-1]):
            conv.bias[32:] = 1.0
    preds = m(torch.rand(2, 3, 128, 128))
    crit = m.init_criterion()
    o2m = crit.one2many.parse_output(preds)["one2many"]
    seg = crit.one2many.loss(o2m, _batch())[0][1]
    g = torch.autograd.grad(seg, o2m["boxes"], allow_unused=True)[0]
    assert g is None or not g.any()


def test_hop_that_va_hop_du_doan_cho_loss_khac_nhau():
    from cofseg.training.coord_head import find_head

    x = torch.rand(2, 3, 128, 128)
    out = {}
    for box in ("pred", "gt"):
        m = _coord(box)
        head = find_head(m)
        with torch.no_grad():                                  # seg_loss ghi log là của nhánh one2one
            for conv in (*head.cv4, *head.one2one_cv4):
                conv[-1].bias[32:] = 1.0
        out[box] = m.init_criterion()(m(x), _batch())[1]["seg_loss"]
    assert out["pred"] != out["gt"]


def test_nhat_ky_epoch(tmp_path):
    from cofseg.training.coord_train import EpochLog

    m = _coord()
    preds = m(torch.rand(2, 3, 128, 128))
    crit = m.init_criterion()
    log = EpochLog(tmp_path / "mask_head_kt4.csv")
    for ep in range(2):
        log.start(crit, ep)
        crit(preds, _batch())
        log.end(crit, ep)
    rows = list(csv.DictReader((tmp_path / "mask_head_kt4.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["one2one_n"] == rows[1]["one2one_n"] == "3"
    assert float(rows[0]["one2many_coord_share"]) == 0.0        # hệ số toạ độ còn bằng 0
    assert 0.0 <= float(rows[0]["one2many_box_iou"]) <= 1.0 and float(rows[0]["one2many_bce"]) > 0


# ------------------------------------------------------------------ suy luận
def test_dung_mat_na_trung_goc_khi_he_so_toa_do_bang_0():
    from ultralytics.utils import ops

    from cofseg.training.coord_head import process_mask_coord, process_mask_native_coord

    torch.manual_seed(0)
    proto = torch.randn(32, H, H)
    coefs = torch.cat([torch.randn(3, 32), torch.zeros(3, 5)], 1)
    boxes = torch.tensor([[8.0, 8.0, 60.0, 70.0], [40.0, 30.0, 120.0, 100.0], [0.0, 64.0, 64.0, 128.0]])
    for up in (False, True):
        assert torch.equal(process_mask_coord(proto, coefs, boxes, (128, 128), upsample=up),
                           ops.process_mask(proto, coefs[:, :32], boxes, (128, 128), upsample=up))
    assert torch.equal(process_mask_native_coord(proto, coefs, boxes, (128, 128), boxes, (128, 128)),
                       ops.process_mask_native(proto, coefs[:, :32], boxes, (128, 128)))


def test_toa_do_doi_mat_na_theo_hop():
    """Chỉ dùng toạ độ: c_u = +10 tô nửa phải của hộp, bỏ nửa trái."""
    from cofseg.training.coord_head import process_mask_coord

    proto = torch.zeros(32, H, H)
    coefs = torch.zeros(1, 37)
    coefs[0, 32] = 10.0
    box = torch.tensor([[40.0, 40.0, 80.0, 80.0]])             # ô 10..20
    m = process_mask_coord(proto, coefs, box, (128, 128))[0]
    assert m[15, 16:20].all() and not m[15, 10:15].any()


def test_checkpoint_nap_lai_va_predictor(tmp_path):
    """Lưu như trainer, nạp bằng YOLO(), predictor KT4 ra mặt nạ ở cỡ ảnh gốc."""
    import numpy as np
    from ultralytics import YOLO

    from cofseg.training.coord_head import CoordSegPredictor, is_coord

    m = _coord()
    m.names = {0: "crown"}
    p = tmp_path / "kt4.pt"
    torch.save({"model": copy.deepcopy(m).half(), "train_args": {"task": "segment", "imgsz": 128}}, p)
    y = YOLO(str(p))
    assert is_coord(y.model) and y.task == "segment"
    img = (np.random.default_rng(0).random((96, 160, 3)) * 255).astype(np.uint8)
    r = y.predict(img, predictor=CoordSegPredictor, conf=0.0, max_det=5, imgsz=128,
                  retina_masks=True, verbose=False)[0]
    assert type(y.predictor) is CoordSegPredictor
    assert r.masks is not None and r.masks.data.shape[1:] == (96, 160)


def test_trainer_nap_coco_roi_moi_them_kenh():
    """Thứ tự của trainer: dựng chuẩn -> nạp trọng số 32 kênh -> thêm 5 kênh 0.
    Khi resume (checkpoint đã 37 kênh): chuyển trước rồi nạp."""
    from cofseg.training.coord_head import find_head
    from cofseg.training.coord_train import make_trainer

    nguon = _stock(seed=7)
    T = make_trainer({"type": "coord", "box": "pred"})
    tr = T.__new__(T)                                          # không dựng dataloader: chỉ thử get_model
    tr.args = nguon.args
    tr.data = {"nc": 1, "names": {0: "crown"}, "channels": 3}
    m = T.get_model(tr, cfg="yolo26n-seg.yaml", weights=nguon, verbose=False)
    w = find_head(m).cv4[0][-1].weight
    assert w.shape[0] == 37
    assert torch.equal(w[:32], nguon.model[-1].cv4[0][-1].weight)
    assert not w[32:].any()
    with torch.no_grad():
        find_head(m).cv4[0][-1].weight[32:] = 0.5
    m2 = T.get_model(tr, cfg="yolo26n-seg.yaml", weights=m, verbose=False)
    assert torch.equal(find_head(m2).cv4[0][-1].weight, find_head(m).cv4[0][-1].weight)
