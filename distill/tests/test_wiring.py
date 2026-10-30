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
    cfg = _cfg(name="tro-seg-kd", student={"width": 0.25}, kd={"targets": "t", "mask_alpha": 0.3})
    assert dtr.DistillYoloTrainer.apply_loss(cfg, None) == "w0.25-a0.3-nofeat"
    assert cfg["kd"]["feat_weight"] == 6.0 and cfg["kd"]["teacher_runs"] is None


@pytest.mark.parametrize("cfg,spec,match", [
    (_cfg(outside_box={}), "dice", "--loss"),
    (_cfg(student={}), "dice", "trò"),
    (_cfg(student={}, mask_head={"type": "coord"}), None, "mask_head"),
    (_cfg(student={"nm": 64}), None, r"\[1, 32\]"),
    (_cfg(kd={"targets": "t"}), "dice", "--loss"),
    (_cfg(name="yolo26s-seg-ka", outside_box={}, kd={"targets": "t"}), None, "chưa ghép"),
    (_cfg(student={}, kd={"targets": "t"}, mask_head={"type": "coord"}), None, "mask_head"),
])
def test_to_hop_khong_hop_bi_chan(cfg, spec, match):
    with pytest.raises(SystemExit, match=match):
        dtr.DistillYoloTrainer.apply_loss(cfg, spec)


@pytest.mark.parametrize("config,want", [
    ("distill/configs/train/tro.yaml", ["Tham số cuối cùng", "imgsz              1024", "epochs             100"]),
    ("distill/configs/train/tro-025.yaml", ["Tham số cuối cùng", "imgsz              1024"]),
    ("distill/configs/train/yolo26s-ka.yaml", ["Tham số cuối cùng", "mosaic             0.0"]),
    ("distill/configs/train/tro-025-kd.yaml", ["Tham số cuối cùng", "imgsz              1024"]),
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
    "tro-025-kd.yaml": (("student", "kd"), "weights/yolo26n-seg.pt", "tro-025-kd.yaml"),
}


@pytest.mark.parametrize("f", sorted(CAP))
def test_config_ke_thua_cong_thuc_chuan_va_khop_config_cham(f):
    """Các lượt đợt 1, 2 train đúng công thức của mô hình gốc, chỉ thêm khối của distill (và đổi checkpoint
    nguồn ở trò 0,25); tên run khớp tên config chấm cùng cặp."""
    from cofseg import config as cfgmod

    key, src, ev = CAP[f]
    goc = cfgmod.load(REPO / "benchmark/yolo/configs/train/yolo26s.yaml")
    c = cfgmod.load(REPO / "distill/configs/train" / f)
    assert c["train"] == goc["train"] and c["trainer"] == "distill_yolo" and c["model"] == src
    assert set(c) - set(goc) == ({key} if isinstance(key, str) else set(key))
    dtr.DistillYoloTrainer.apply_loss(c, None)
    assert dtr.DistillYoloTrainer.run_name(c) == cfgmod.load(REPO / "distill/configs/eval" / ev)["name"]


def test_ghep_voi_lop_trainer_cua_aug(tmp_path):
    from ultralytics.models.yolo.segment import SegmentationTrainer

    class AugLike(SegmentationTrainer):
        def build_dataset(self, *a, **k):
            return "aug"

    cls = dtr.make_seg_trainer({"init": "scratch"}, None, tmp_path, base=AugLike)
    assert issubclass(cls, AugLike) and cls.__new__(cls).build_dataset("img") == "aug"
    # có kd nhưng chỉ chắt lọc đặc trưng: dataset train vẫn là của lớp --aug
    cls = dtr.make_seg_trainer(None, None, tmp_path, base=AugLike, kd={"targets": None, "teacher_runs": "r"})
    assert cls.__new__(cls).build_dataset("img") == "aug"
    assert dtr.make_seg_trainer(None, {}, tmp_path).__mro__[1] is SegmentationTrainer


def test_kd_voi_aug_ag1_dung_dataset_ghep(tmp_path, monkeypatch):
    """Lớp của --aug cũng thay dataset trong build_dataset của nó; có chắt lọc mặt nạ thì dataset phải là lớp
    ghép (đích thầy + phép của --aug), và lớp của --aug không được đè lên."""
    import types

    import ultralytics.data.build as build
    from cofseg.training.aug_train import AugDataset
    from ultralytics.models.yolo.segment import SegmentationTrainer

    from distill.kd_data import KDDataset

    seen = []

    def fake(self, img_path, mode="train", batch=None):
        seen.append(build.YOLODataset)
        return types.SimpleNamespace(kd=types.SimpleNamespace(describe=lambda: "đích"))

    owner = next(c for c in SegmentationTrainer.__mro__ if "build_dataset" in vars(c))
    monkeypatch.setattr(owner, "build_dataset", fake)
    goc = build.YOLODataset

    class AugLike(SegmentationTrainer):
        AUG = {"use": ["ag1"], "ag1": {"p": 0.5}, "tag": "ag1"}

        def build_dataset(self, *a, **k):
            return "aug"

    kd = {"targets": "t", "teacher_runs": None}
    cls = dtr.make_seg_trainer(None, None, tmp_path, base=AugLike, kd=kd, fold_root=tmp_path)
    assert cls.__new__(cls).build_dataset("img") != "aug"
    assert issubclass(seen[-1], KDDataset) and issubclass(seen[-1], AugDataset)
    assert seen[-1].AUG == AugLike.AUG and seen[-1].LOG_DIR == str(tmp_path / "aug")
    cls = dtr.make_seg_trainer(None, None, tmp_path, kd=kd, fold_root=tmp_path)
    cls.__new__(cls).build_dataset("img")
    assert issubclass(seen[-1], KDDataset) and not issubclass(seen[-1], AugDataset)
    assert build.YOLODataset is goc


def test_kd_chi_ghep_voi_chuyen_vi():
    from distill import blocks

    mask = blocks.check_kd({"targets": "t"})
    blocks.check_kd_aug(mask, {"use": ["ag1"]})
    blocks.check_kd_aug(blocks.check_kd({"teacher_runs": "r"}), {"use": ["pa1"]})  # chỉ đặc trưng: không cần ảnh đích
    with pytest.raises(SystemExit, match="ag4"):
        blocks.check_kd_aug(mask, {"use": ["ag1", "ag4"]})
    cfg = _cfg(name="tro-seg-kd", student={"width": 0.25}, kd={"targets": "t", "teacher_runs": "r"})
    dtr.DistillYoloTrainer.apply_loss(cfg, None)
    assert dtr.DistillYoloTrainer.apply_aug(cfg, "ag1") == "ag1" and cfg["aug"]["use"] == ["ag1"]
    cfg = _cfg(name="tro-seg-kd", student={"width": 0.25}, kd={"targets": "t"})
    dtr.DistillYoloTrainer.apply_loss(cfg, None)
    with pytest.raises(SystemExit, match="pa1"):
        dtr.DistillYoloTrainer.apply_aug(cfg, "pa1")


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
