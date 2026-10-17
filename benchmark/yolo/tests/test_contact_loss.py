"""TC2: đổi lớp cuối của `semseg`, hàm loss, trainer, nhật ký.

Model dựng từ yaml của gói ultralytics với trọng số ngẫu nhiên, nhãn dựng tại
chỗ, như test của L1 và L2; không đụng data/ hay weights/.
"""

from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from cofseg.training import contact_train as ct  # noqa: E402

H = 32  # lưới nhãn của ảnh 128 px (mask_ratio 4)
BLOCK = {"type": "contact", "band": 2, "w_band": 0.25, "w_dist": 0.25}


def _batch():
    """Ảnh 0: hai tán chữ nhật CHẠM nhau (có dải tiếp giáp). Ảnh 1: một tán riêng."""
    masks = torch.zeros(2, H, H)
    masks[0, 6:22, 4:16] = 1
    masks[0, 6:22, 16:28] = 2
    masks[1, 8:24, 8:24] = 1
    boxes = torch.tensor([[10 / H, 14 / H, 12 / H, 16 / H],
                          [22 / H, 14 / H, 12 / H, 16 / H],
                          [16 / H, 16 / H, 16 / H, 16 / H]])
    return {"batch_idx": torch.tensor([0.0, 0.0, 1.0]), "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, H, H)}


def _model(yaml_name: str = "yolo26n-seg.yaml", overlap: bool = True):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    m = SegmentationModel(yaml_name, nc=1, verbose=False)
    m.args = get_cfg(overrides={"overlap_mask": overlap})
    m.train()
    return m


def _expanded(**kw):
    m = _model(**kw)
    ct.expand_semseg(m)
    return m


def _forward(m):
    torch.manual_seed(1)
    return m(torch.rand(2, 3, 128, 128))


def _stock_preds(preds, nc=1):
    """Cùng dự đoán, nhưng đầu semseg chỉ còn kênh tiền cảnh: đầu vào của loss gốc."""
    out = {}
    for k, v in preds.items():
        p, s = v["proto"]
        out[k] = {**v, "proto": (p, s[:, :nc])}
    return out


def _stock(m):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    return E2ELoss(m, v8SegmentationLoss)


# ------------------------------------------------------------------ expand_semseg
def test_expand_semseg_khoi_tao_dung_va_khong_lam_lan_hai():
    m = _model()
    nc = 1
    last = ct.find_proto26(m).semseg[-1]
    assert last.out_channels == nc
    new = ct.expand_semseg(m)
    assert new.out_channels == nc + 2 and new.in_channels == last.in_channels
    assert torch.equal(new.weight[nc:], torch.zeros_like(new.weight[nc:]))
    assert new.bias[nc] == ct.BAND_BIAS and new.bias[nc + 1] == 0
    assert ct.expand_semseg(m) is new                     # lần hai: không thay


def test_expand_semseg_khong_dong_toi_rng_chung():
    m = _model()
    before = torch.get_rng_state()
    ct.expand_semseg(m, seed=3)
    assert torch.equal(torch.get_rng_state(), before)


def test_expand_semseg_cung_seed_cung_trong_so():
    a, b = _model(), _model()
    assert torch.equal(ct.expand_semseg(a, seed=5).weight, ct.expand_semseg(b, seed=5).weight)


def test_model_khong_phai_yolo26_bi_chan():
    with pytest.raises(SystemExit, match="Proto26"):
        ct.expand_semseg(_model("yolov8n-seg.yaml"))


# ------------------------------------------------------------------------ loss
def test_trong_so_0_trung_tung_so_voi_loss_goc():
    """Tắt hai kênh mới thì vector loss và dict loss không lệch một bit nào."""
    m = _expanded()
    preds, batch = _forward(m), _batch()
    goc = _stock(m)(_stock_preds(preds), batch)
    moi = ct.build_criterion(m, {**BLOCK, "w_band": 0.0, "w_dist": 0.0})(preds, batch)
    assert torch.equal(moi[0], goc[0])
    assert moi[1].keys() == goc[1].keys()
    for k in goc[1]:
        assert torch.equal(moi[1][k], goc[1][k]), k


def test_trong_so_duong_chi_doi_sem_loss_va_co_gradient_vao_kenh_moi():
    m = _expanded()
    preds, batch = _forward(m), _batch()
    goc = _stock(m)(_stock_preds(preds), batch)
    moi = ct.build_criterion(m, BLOCK)(preds, batch)
    for k in goc[1]:
        if k == "sem_loss":
            assert moi[1][k] > goc[1][k]
        else:
            assert torch.equal(moi[1][k], goc[1][k]), k
    assert torch.equal(moi[0][:4], goc[0][:4]) and moi[0][4] > goc[0][4]
    moi[0].sum().backward()
    g = ct.find_proto26(m).semseg[-1].weight.grad
    assert g is not None and g[1].abs().sum() > 0 and g[2].abs().sum() > 0


def test_nhanh_one2one_khong_dua_gradient_vao_semseg():
    m = _expanded()
    preds, batch = _forward(m), _batch()
    crit = ct.build_criterion(m, BLOCK)
    total, _ = crit.one2one.loss(preds["one2one"], batch)
    total.sum().backward()
    for p in ct.find_proto26(m).semseg.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_che_do_eval_di_thang_vao_loss_goc():
    m = _expanded()
    m.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        out = m(torch.rand(2, 3, 128, 128))
    batch = _batch()
    goc = _stock(m)(out, batch)
    moi = ct.build_criterion(m, BLOCK)(out, batch)
    assert torch.equal(moi[0], goc[0])


def test_khong_co_overlap_mask_bi_chan():
    m = _expanded(overlap=False)
    preds, batch = _forward(m), _batch()
    with pytest.raises(RuntimeError, match="overlap_mask"):
        ct.build_criterion(m, BLOCK)(preds, batch)


def test_model_chua_doi_semseg_bi_chan():
    m = _model()
    preds, batch = _forward(m), _batch()
    with pytest.raises(RuntimeError, match="expand_semseg"):
        ct.build_criterion(m, BLOCK)(preds, batch)


# -------------------------------------------------------------- gắn loss, resume
def test_attach_criterion_giu_tien_do_progloss():
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    m = _expanded()
    old = E2ELoss(m, v8SegmentationLoss)
    old.updates = 39
    old.update()                                          # như resume_training ở epoch 41
    m.criterion = old
    new = ct.attach_criterion(m, BLOCK)
    assert m.criterion is new and isinstance(new.one2many, ct.ContactSemLoss)
    assert (new.updates, new.o2m, new.o2o) == (old.updates, old.o2m, old.o2o)
    assert new.o2m < 0.8


def test_attach_criterion_tren_model_chua_doi_semseg_bi_chan():
    with pytest.raises(RuntimeError, match="ContactSegTrainer"):
        ct.attach_criterion(_model(), BLOCK)


def _trainer():
    tr = object.__new__(ct.make_trainer(BLOCK))
    tr.data = {"nc": 1, "channels": 3, "names": {0: "canopy"}}
    tr.args = SimpleNamespace(seed=0, cls_remap=True)
    return tr


def test_trainer_doi_semseg_roi_moi_nap_trong_so():
    """Nạp từ model gốc (semseg 1 kênh): lớp mới giữ khởi tạo của TC2. Nạp từ
    checkpoint TC2 (3 kênh, như --resume): lớp mới nhận đúng trọng số đã học."""
    tr = _trainer()
    goc = _model()
    m = tr.get_model(cfg="yolo26n-seg.yaml", weights=goc, verbose=False)
    last = ct.find_proto26(m).semseg[-1]
    assert last.out_channels == 3 and last.bias[1] == ct.BAND_BIAS

    hoc = _expanded()
    with torch.no_grad():
        ct.find_proto26(hoc).semseg[-1].weight.normal_()
    m2 = tr.get_model(cfg="yolo26n-seg.yaml", weights=hoc, verbose=False)
    assert torch.equal(ct.find_proto26(m2).semseg[-1].weight, ct.find_proto26(hoc).semseg[-1].weight)
    assert type(tr).CONTACT == BLOCK


# ------------------------------------------------------------ checkpoint, fuse
def test_fuse_bo_semseg_nen_so_tham_so_bang_model_goc():
    a, b = _model(), _expanded()
    a.eval().fuse(verbose=False)
    b.eval().fuse(verbose=False)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(a) == n(b)


def test_checkpoint_nap_duoc_ma_khong_can_goi_cofseg(tmp_path):
    m = _expanded()
    path = tmp_path / "tc2.pt"
    torch.save({"model": m}, path)
    code = ("import torch; m = torch.load(r'%s', weights_only=False)['model']; "
            "print(type(m).__name__, m.model[-1].proto.semseg[-1].out_channels)") % path
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["SegmentationModel", "3"]


# ------------------------------------------------------------------- nhật ký
def test_nhat_ky_epoch(tmp_path):
    m = _expanded()
    crit = ct.build_criterion(m, BLOCK)
    log = ct.EpochLog(tmp_path / "mask_head_tc2.csv")
    log.start(crit, 0)
    crit(_forward(m), _batch())
    log.end(crit, 0)
    rows = list(csv.DictReader((tmp_path / "mask_head_tc2.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert r["epoch"] == "1" and r["one2many_batches"] == "1"
    assert float(r["one2many_band"]) > 0 and float(r["one2many_dist"]) > 0
    assert 0 < float(r["one2many_p_band_in"]) < 1
    assert float(r["w_o2m"]) == pytest.approx(0.8)


def test_nhat_ky_ghi_trong_so_one2many_cua_chinh_epoch_do(tmp_path):
    """Trainer gọi update() trước on_train_epoch_end; cột w_o2m vẫn phải là
    trọng số đã dùng trong epoch, không phải trọng số của epoch sau."""
    m = _expanded()
    m.args.epochs = 3
    crit = ct.build_criterion(m, BLOCK)
    log = ct.EpochLog(tmp_path / "mask_head_tc2.csv")
    for epoch in range(2):
        log.start(crit, epoch)
        crit(_forward(m), _batch())
        crit.update()                                     # như trainer.py, trước callback cuối epoch
        log.end(crit, epoch)
    rows = list(csv.DictReader((tmp_path / "mask_head_tc2.csv").open(encoding="utf-8")))
    assert [float(r["w_o2m"]) for r in rows] == pytest.approx([0.8, 0.45])
