"""Cờ --aug: bật, ghép, tên run, kiểm tham số, và hậu tố của lượt chấm."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofseg import artifacts
from cofseg.training import aug
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]
PA1 = {"p": 0.5, "k_min": 2, "k_max": 4, "c_lo": 0.2, "c_hi": 0.5, "hid": 0.5}
AG4 = {"p": 0.5, "k_min": 2, "k_max": 4, "gap": 25.0}


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


def test_khong_co_co_la_chuoi_goc():
    cfg = _cfg()
    assert YoloTrainer.apply_aug(cfg, None) == ""
    assert "aug" not in cfg
    assert aug.describe_active(None).startswith("Tăng cường thêm: không")


@pytest.mark.parametrize("spec,tag,use", [
    ("ag1", "ag1", ["ag1"]),
    ("pa1", "pa1", ["pa1"]),
    ("ag4", "ag4", ["ag4"]),
    ("ag1+pa1", "ag1-pa1", ["ag1", "pa1"]),
    ("pa1+ag1", "ag1-pa1", ["ag1", "pa1"]),      # thứ tự gõ không đổi tên
    ("ag4,ag1", "ag1-ag4", ["ag1", "ag4"]),
    (" PA1 + AG1 ", "ag1-pa1", ["ag1", "pa1"]),
])
def test_bat_va_ghep(spec, tag, use):
    cfg = _cfg()
    assert YoloTrainer.apply_aug(cfg, spec) == tag
    assert cfg["aug"]["use"] == use
    assert cfg["aug"]["tag"] == tag


def test_mac_dinh():
    cfg = _cfg()
    YoloTrainer.apply_aug(cfg, "ag1+pa1")
    assert cfg["aug"]["ag1"] == {"p": 0.5}
    assert cfg["aug"]["pa1"] == PA1
    cfg = _cfg()
    YoloTrainer.apply_aug(cfg, "ag4")
    assert cfg["aug"]["ag4"] == AG4


def test_tham_so_khac_mac_dinh_vao_ten():
    cfg = _cfg(aug={"pa1": {"p": 0.3, "c_hi": 0.4}})
    assert YoloTrainer.apply_aug(cfg, "pa1") == "pa1-p0.3-ch0.4"
    cfg = _cfg(aug={"ag4": {"gap": 75.0}, "ag1": {"p": 1.0}})
    assert YoloTrainer.apply_aug(cfg, "ag1+ag4") == "ag1-p1-ag4-g75"


def test_pa1_va_ag4_loai_tru_nhau():
    with pytest.raises(SystemExit, match="cùng một phép so"):
        YoloTrainer.apply_aug(_cfg(), "pa1+ag4")


@pytest.mark.parametrize("spec,match", [
    ("pa2", "Ý bạn là: pa1"),
    ("ag9", "không có"),
])
def test_ten_sai(spec, match):
    with pytest.raises(SystemExit, match=match):
        YoloTrainer.apply_aug(_cfg(), spec)


@pytest.mark.parametrize("block,match", [
    ({"pa1": {"p": 1.5}}, r"aug.pa1.p"),
    ({"pa1": {"k_min": 0}}, r"aug.pa1.k_min"),
    ({"pa1": {"k_min": 3, "k_max": 2}}, r"k_max"),
    ({"pa1": {"c_lo": 0.4, "c_hi": 0.3}}, r"c_hi"),
    ({"pa1": {"hid": 0.99}}, r"aug.pa1.hid"),
    ({"pa1": {"p": True}}, r"aug.pa1.p"),
    ({"pa1": {"cc": 0.1}}, r"không có tham số"),
])
def test_tham_so_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        YoloTrainer.apply_aug(_cfg(aug=block), "pa1")


def test_ag4_gap_sai():
    with pytest.raises(SystemExit, match="aug.ag4.gap"):
        YoloTrainer.apply_aug(_cfg(aug={"ag4": {"gap": 0}}), "ag4")


def test_tham_so_khong_co_co_bi_chan():
    with pytest.raises(SystemExit, match="không có --aug"):
        YoloTrainer.apply_aug(_cfg(aug={"pa1": {"p": 0.3}}), None)
    with pytest.raises(SystemExit, match="không bật 'ag4'"):
        YoloTrainer.apply_aug(_cfg(aug={"ag4": {"gap": 30.0}}), "pa1")
    with pytest.raises(SystemExit, match="chỉ bật bằng cờ"):
        YoloTrainer.apply_aug(_cfg(aug={"use": ["pa1"]}), "pa1")


def test_khong_ghep_voi_mask_head():
    with pytest.raises(SystemExit, match="mask_head"):
        YoloTrainer.apply_aug(_cfg(mask_head={"type": "contact"}), "pa1")


def test_paste_mode():
    assert aug.paste_mode(None) is None
    assert aug.paste_mode({"use": ["ag1"]}) is None
    assert aug.paste_mode({"use": ["ag1", "pa1"]}) == "pa1"
    assert aug.paste_mode({"use": ["ag4"]}) == "ag4"


def test_dong_nhat_ky():
    cfg = _cfg()
    YoloTrainer.apply_aug(cfg, "ag1+pa1")
    line = aug.describe_active(cfg["aug"])
    assert line.startswith("Tăng cường thêm: ag1 (p 0.5); pa1 (p 0.5, k_min 2")


def test_luot_cham_mang_hau_to(tmp_path):
    run = tmp_path / "runs" / "train" / "x_yolo26s-seg-ag1-pa1_field-f1_i1024"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"")
    (run / "summary.json").write_text(json.dumps({"aug": {"use": ["ag1", "pa1"], "tag": "ag1-pa1"}}),
                                      encoding="utf-8")
    name, _ = artifacts.eval_run_name("yolo26s-seg", str(run / "weights" / "best.pt"), explicit=False)
    assert name == "yolo26s-seg-ag1-pa1"
    name, _ = artifacts.eval_run_name("tay", str(run / "weights" / "best.pt"), explicit=True)
    assert name == "tay"


def test_train_py_in_config_va_liet_ke():
    py = sys.executable
    base = [py, "benchmark/yolo/train.py", "--config", "benchmark/yolo/configs/train/yolo26s.yaml"]
    out = subprocess.run(base + ["--list-augs"], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert "ag1:" in out.stdout and "pa1:" in out.stdout and "ag4:" in out.stdout
    out = subprocess.run(base + ["--aug", "pa1+ag1", "--print-config"], cwd=REPO, capture_output=True,
                         text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert '"tag": "ag1-pa1"' in out.stdout
    out = subprocess.run(base + ["--print-config"], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    assert "aug: chuỗi gốc của framework" in out.stdout
