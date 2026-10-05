"""T1 nối vào benchmark: khối `mask_head` loại gpr, ràng buộc mask_ratio, file
config riêng, và tên lượt chấm."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofseg.training import mask_head
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]


def _cfg(mask_ratio=1, **extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt",
            "train": {"imgsz": 1024, "mask_ratio": mask_ratio}, **extra}


def test_mac_dinh_va_nhan():
    mh = mask_head.check({"type": "gpr"})
    assert mh == {"type": "gpr", "stages": [1, 2], "detach_guide": True, "groups": 4,
                  "scope": 1.0, "sigma": 2.4, "max_pos": 64}
    assert mask_head.tag(mh) == "gpr"


@pytest.mark.parametrize("block,tag", [
    ({"stages": [2]}, "gpr-s2"),
    ({"stages": []}, "gpr-s0"),
    ({"stages": [2, 1]}, "gpr"),                  # thứ tự không quan trọng
    ({"detach_guide": False}, "gpr-grad"),
    ({"sigma": 0}, "gpr-sig0"),
    ({"groups": 1, "scope": 0.5, "max_pos": 32}, "gpr-g1-sc0.5-mp32"),
])
def test_tham_so_khac_mac_dinh_vao_nhan(block, tag):
    assert mask_head.tag(mask_head.check({"type": "gpr", **block})) == tag


@pytest.mark.parametrize("block,match", [
    ({"sigm": 1.0}, "sigma"),                     # gõ sai khoá: gợi ý
    ({"window": 1.5}, "không có khoá"),          # khoá của A1 không lọt sang GPR
    ({"stages": [3]}, r"\[1, 2\]"),
    ({"stages": [1, 1]}, r"\[1, 2\]"),
    ({"stages": "12"}, r"\[1, 2\]"),
    ({"detach_guide": 1}, "true/false"),
    ({"groups": 0}, ">= 1"),
    ({"scope": 0}, "> 0"),
    ({"sigma": -1}, ">= 0"),
    ({"sigma": True}, ">= 0"),
    ({"max_pos": 0}, ">= 1"),
])
def test_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        mask_head.check({"type": "gpr", **block})


def test_loai_khong_co_bi_chan_va_a1_khong_doi():
    with pytest.raises(SystemExit, match="gpr"):
        mask_head.check({"type": "p2"})
    assert mask_head.check({})["type"] == "dynamic"     # khối không có type vẫn là A1


def test_can_mask_ratio_1_va_khong_di_chung_voi_loss():
    with pytest.raises(SystemExit, match="mask_ratio: 1"):
        YoloTrainer.apply_loss(_cfg(mask_ratio=4, mask_head={"type": "gpr"}), None)
    with pytest.raises(SystemExit, match="mask_head"):
        YoloTrainer.apply_loss(_cfg(mask_head={"type": "gpr"}), "dice")
    cfg = _cfg(mask_head={"type": "gpr", "sigma": 0})
    assert YoloTrainer.apply_loss(cfg, None) == ""
    assert cfg["mask_head"]["sigma"] == 0.0 and cfg["mask_head"]["stages"] == [1, 2]


def test_print_config_cua_file_config_rieng():
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s-gpr.yaml", "--print-config"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("mask_head:"))
    assert json.loads(line.split(":", 1)[1]) == mask_head.check({"type": "gpr"})
    assert "mask_ratio" in r.stdout and "1024" in r.stdout   # phần còn lại lấy từ yolo26s.yaml


def test_cham_mang_hau_to_gpr(tmp_path):
    from cofseg import artifacts

    run = tmp_path / "2026-10-06_090000_yolo26s-seg-gpr_field-f1_i1024b16e100"
    (run / "weights").mkdir(parents=True)
    mh = mask_head.check({"type": "gpr"})
    doc = {"weights": {}, "args": {"imgsz": 1024}, "loss": None, "mask_head": {**mh, "tag": mask_head.tag(mh)}}
    (run / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    w = run / "weights/best.pt"
    assert artifacts.eval_run_name("yolo26s-seg", w, explicit=False) == ("yolo26s-seg-gpr", None)
    assert artifacts.eval_run_name("yolo26s-seg-gpr", w, explicit=False)[0] == "yolo26s-seg-gpr"
