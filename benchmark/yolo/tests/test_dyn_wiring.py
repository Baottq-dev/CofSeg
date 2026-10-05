"""A1 nối vào benchmark: khối `mask_head` của config, tên run và lượt chấm,
validator lúc train, và YoloSegModel tự dùng predictor A1."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofseg.training import mask_head
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


# ------------------------------------------------------------------ khối mask_head
def test_mac_dinh_va_nhan():
    mh = mask_head.check({})
    assert mh == {"type": "dynamic", "dims": [8, 8], "coords": True, "window": 1.5, "max_pos": 64}
    assert mask_head.tag(mh) == "dyn"
    assert mask_head.check(None) is None


def test_tham_so_khac_mac_dinh_vao_nhan():
    """Thang ablation A1a/A1b ra những nhãn khác nhau."""
    assert mask_head.tag(mask_head.check({"dims": [], "coords": False})) == "dyn-d0-nocoord"
    assert mask_head.tag(mask_head.check({"dims": [16], "window": 2})) == "dyn-d16-win2"
    assert mask_head.tag(mask_head.check({"window": 1.5, "max_pos": 32})) == "dyn-mp32"


@pytest.mark.parametrize("block,match", [
    ({"dim": [8]}, "dims"),                       # gõ sai khoá: gợi ý
    ({"type": "static"}, "dynamic"),
    ({"dims": 8}, "danh sách"),
    ({"dims": [8, 0]}, "danh sách"),
    ({"dims": [True]}, "danh sách"),
    ({"coords": 1}, "true/false"),
    ({"window": 0.9}, ">= 1"),
    ({"window": True}, ">= 1"),
    ({"max_pos": 0}, ">= 1"),
    ("dynamic", "khối"),
])
def test_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        mask_head.check(block)


def test_a1_khong_di_chung_voi_loss():
    with pytest.raises(SystemExit, match="mask_head"):
        YoloTrainer.apply_loss(_cfg(mask_head={}), "dice")
    cfg = _cfg(mask_head={"window": 2})
    assert YoloTrainer.apply_loss(cfg, None) == ""
    assert cfg["mask_head"]["window"] == 2.0 and cfg["mask_head"]["dims"] == [8, 8]


def test_print_config_cua_file_config_rieng():
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s-dyn.yaml", "--print-config"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("mask_head:"))
    assert json.loads(line.split(":", 1)[1]) == mask_head.check({})
    assert "imgsz" in r.stdout and "1024" in r.stdout   # phần còn lại lấy từ yolo26s.yaml


# ------------------------------------------------------------------ tên lượt chấm
def _train_run(tmp_path, mh=None, loss=None):
    run = tmp_path / "2026-10-05_160000_yolo26s-seg-dyn_field-f1_i1024b16e100"
    (run / "weights").mkdir(parents=True)
    doc = {"weights": {}, "args": {"imgsz": 1024}, "loss": loss,
           "mask_head": {**mh, "tag": mask_head.tag(mh)} if mh else None}
    (run / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    return run


def test_cham_mang_hau_to_dyn(tmp_path):
    from cofseg import artifacts

    run = _train_run(tmp_path, mh=mask_head.check({}))
    w = run / "weights/best.pt"
    assert artifacts.eval_run_name("yolo26s-seg", w, explicit=False) == ("yolo26s-seg-dyn", None)
    assert artifacts.eval_run_name("yolo26s-seg-dyn", w, explicit=False)[0] == "yolo26s-seg-dyn"
    assert artifacts.eval_run_name("tu-dat", w, explicit=True)[0] == "tu-dat"
    assert artifacts.train_mask_head(w)["tag"] == "dyn"


def test_cham_luot_dau_goc_giu_ten(tmp_path):
    from cofseg import artifacts

    w = _train_run(tmp_path) / "weights/best.pt"
    assert artifacts.eval_run_name("yolo26s-seg", w, explicit=False) == ("yolo26s-seg", None)
    assert artifacts.train_mask_head(w) is None


# ------------------------------------------------------------------ validator, chấm
torch = pytest.importorskip("torch")


def _dyn_model(theta_bias_last=None):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    from cofseg.training.dyn_head import find_head, to_dynamic

    torch.manual_seed(0)
    m = to_dynamic(SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False), mask_head.check({}))
    m.args = get_cfg()
    m.names = {0: "crown"}
    if theta_bias_last is not None:
        # θ hằng: mọi trọng số động 0, bias lớp cuối dương -> mặt nạ = cả cửa sổ
        for seq in (find_head(m).cv4, find_head(m).one2one_cv4):
            for br in seq:
                torch.nn.init.zeros_(br[-1].weight)
                torch.nn.init.zeros_(br[-1].bias)
                br[-1].bias.data[-1] = theta_bias_last
    return m.eval()


def test_validator_dung_dau_a1_va_tra_ham_goc_cho_nhan(monkeypatch):
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import ops

    from cofseg.training.dyn_head import DynSegValidator

    m = _dyn_model(theta_bias_last=10.0)
    v = DynSegValidator(args=get_cfg(overrides={"task": "segment", "conf": 0.0, "max_det": 5}))
    v.training, v.data = True, {"val": ""}
    v.init_metrics(m)
    with torch.no_grad():
        out = v.postprocess(m(torch.rand(1, 3, 128, 128)))
    masks = out[0]["masks"]
    assert masks.shape[0] == len(out[0]["bboxes"]) > 0 and masks.shape[1:] == (32, 32)
    assert (masks.flatten(1).sum(1) > 0).all()       # không im lặng trả mặt nạ rỗng

    seen = {}
    monkeypatch.setattr(SegmentationValidator, "_prepare_batch",
                        lambda self, si, batch: seen.setdefault("process", self.process))
    dyn_process = v.process
    v._prepare_batch(0, {})
    assert seen["process"] is ops.process_mask and v.process is dyn_process


def test_yolosegmodel_tu_dung_predictor_a1(tmp_path):
    import numpy as np

    from cofseg.models.yolo_seg import YoloSegModel

    p = tmp_path / "a1.pt"
    torch.save({"model": copy.deepcopy(_dyn_model(theta_bias_last=10.0)).half(),
                "train_args": {"task": "segment", "imgsz": 128}}, p)
    ym = YoloSegModel(str(p), imgsz=128, conf=0.0, max_det=3, device="cpu")
    assert ym.mask_head == "dyn" and ym.describe["mask_head"] == "dyn"
    preds = ym.predict((np.random.default_rng(0).random((96, 160, 3)) * 255).astype(np.uint8))
    assert 0 < len(preds) <= 3
    assert type(ym.model.predictor).__name__ == "DynSegPredictor"
