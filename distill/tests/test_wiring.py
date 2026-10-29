"""Nối vào benchmark/yolo: trainer `distill_yolo`, file config riêng, tên run, ghép với lớp trainer của --aug,
và phép đối chiếu của distill/evaluate.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from distill import trainer as dtr

REPO = Path(__file__).resolve().parents[2]


def _cfg(**extra):
    return {"trainer": "distill_yolo", "name": "tro-seg", "model": "weights/yolo26s-seg.pt",
            "train": {"imgsz": 1024}, **extra}


def test_trainer_da_dang_ky():
    from cofseg.registry import resolve

    assert resolve("trainer", "distill_yolo") is dtr.DistillYoloTrainer


def test_apply_loss_kiem_va_dat_hau_to():
    cfg = _cfg(student={})
    assert dtr.DistillYoloTrainer.apply_loss(cfg, None) == ""
    assert cfg["student"]["nm"] == 16
    cfg = _cfg(student={"nm": 32, "drop_p3": False})
    assert dtr.DistillYoloTrainer.apply_loss(cfg, None) == "nm32-p3"
    assert dtr.DistillYoloTrainer.run_name(cfg) == "tro-seg-nm32-p3"
    cfg = _cfg(name="yolo26s-seg-ka", outside_box={})
    assert dtr.DistillYoloTrainer.apply_loss(cfg, None) == ""
    assert dtr.DistillYoloTrainer.run_name(cfg) == "yolo26s-seg-ka"
    cfg = _cfg()
    assert dtr.DistillYoloTrainer.apply_loss(cfg, None) == "" and "student" not in cfg


@pytest.mark.parametrize("cfg,spec,match", [
    (_cfg(outside_box={}), "dice", "--loss"),
    (_cfg(student={}), "dice", "trò"),
    (_cfg(student={}, mask_head={"type": "coord"}), None, "mask_head"),
    (_cfg(student={"nm": 64}), None, r"\[1, 32\]"),
])
def test_to_hop_khong_hop_bi_chan(cfg, spec, match):
    with pytest.raises(SystemExit, match=match):
        dtr.DistillYoloTrainer.apply_loss(cfg, spec)


@pytest.mark.parametrize("config,want", [
    ("distill/configs/train/tro.yaml", ["Tham số cuối cùng", "imgsz              1024", "epochs             100"]),
    ("distill/configs/train/tro-025.yaml", ["Tham số cuối cùng", "imgsz              1024"]),
    ("distill/configs/train/yolo26s-ka.yaml", ["Tham số cuối cùng", "mosaic             0.0"]),
])
def test_print_config_cua_file_config_rieng(config, want):
    r = subprocess.run([sys.executable, "distill/train.py", "--config", config, "--print-config"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    for w in want:
        assert w in r.stdout


#: config train -> (khối của distill, checkpoint nguồn, config chấm)
CAP = {
    "tro.yaml": ("student", "weights/yolo26s-seg.pt", "tro.yaml"),
    "tro-025.yaml": ("student", "weights/yolo26n-seg.pt", "tro-025.yaml"),
    "yolo26s-ka.yaml": ("outside_box", "weights/yolo26s-seg.pt", "yolo26s-ka.yaml"),
}


@pytest.mark.parametrize("f", sorted(CAP))
def test_config_ke_thua_cong_thuc_chuan_va_khop_config_cham(f):
    """Ba lượt đợt 1 train đúng công thức của mô hình gốc, chỉ thêm khối của distill (và đổi checkpoint
    nguồn ở trò 0,25); tên run khớp tên config chấm cùng cặp."""
    from cofseg import config as cfgmod

    key, src, ev = CAP[f]
    goc = cfgmod.load(REPO / "benchmark/yolo/configs/train/yolo26s.yaml")
    c = cfgmod.load(REPO / "distill/configs/train" / f)
    assert c["train"] == goc["train"] and c["trainer"] == "distill_yolo" and c["model"] == src
    assert set(c) - set(goc) == {key}
    dtr.DistillYoloTrainer.apply_loss(c, None)
    assert dtr.DistillYoloTrainer.run_name(c) == cfgmod.load(REPO / "distill/configs/eval" / ev)["name"]


def test_ghep_voi_lop_trainer_cua_aug(tmp_path):
    from ultralytics.models.yolo.segment import SegmentationTrainer

    class AugLike(SegmentationTrainer):
        def build_dataset(self, *a, **k):
            return "aug"

    cls = dtr.make_seg_trainer({"init": "scratch"}, None, tmp_path, base=AugLike)
    assert issubclass(cls, AugLike) and cls.build_dataset(None) == "aug"
    assert dtr.make_seg_trainer(None, {}, tmp_path).__mro__[1] is SegmentationTrainer


def test_thay_tam_model_train_roi_tra_lai():
    from ultralytics.engine.model import Model

    goc = Model.train
    seen = []
    with dtr._inject_trainer(lambda base: seen.append(base) or "lop"):
        assert Model.train is not goc
    assert Model.train is goc


def _run(tmp_path, block):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best.pt").write_bytes(b"")
    (run / "summary.json").write_text(json.dumps({"distill": block}), encoding="utf-8")
    return str(run / "weights/best.pt")


def _evaluate_module():
    """distill/evaluate.py là script, không phải module của gói: nạp theo đường dẫn, không chạy phần chấm."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("distill_evaluate_script", REPO / "distill/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_doi_chieu_ten_va_he_so_cat(tmp_path):
    ev = _evaluate_module()
    ka = {"name": "yolo26s-seg-ka", "student": None, "outside_box": {"train_expand": 1.25, "infer_expand": 1.1}}
    w = _run(tmp_path, ka)
    ok = ["--config", str(REPO / "distill/configs/eval/yolo26s-ka.yaml"), "--weights", w]
    assert "khớp" in ev.check(ok)
    with pytest.raises(SystemExit, match="mang tên"):
        ev.check(["--config", str(REPO / "distill/configs/eval/tro.yaml"), "--weights", w])
    with pytest.raises(SystemExit, match="×1.1"):
        ev.check([*ok, "--crop_expand", "1.0"])
    # Gõ --name thì bỏ qua đối chiếu tên, nhưng hệ số cắt vẫn phải đúng.
    with pytest.raises(SystemExit, match="×1.1"):
        ev.check(["--config", str(REPO / "distill/configs/eval/tro.yaml"), "--weights", w, "--name", "x"])
    tro = {"name": "tro-seg", "student": {"nm": 16}, "outside_box": None}
    assert "khớp" in ev.check(["--config", str(REPO / "distill/configs/eval/tro.yaml"),
                                "--weights", _run(tmp_path / "t", tro)])
