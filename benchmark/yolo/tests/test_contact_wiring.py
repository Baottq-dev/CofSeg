"""TC2 nối vào benchmark: khối `mask_head` loại contact, ràng buộc với khối
train, file config riêng, và tên lượt chấm."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofseg.training import mask_head
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]


def _cfg(mask_ratio=4, overlap=True, **extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt",
            "train": {"imgsz": 1024, "mask_ratio": mask_ratio, "overlap_mask": overlap}, **extra}


def test_mac_dinh_va_nhan():
    mh = mask_head.check({"type": "contact"})
    assert mh == {"type": "contact", "band": 2, "w_band": 0.25, "w_dist": 0.25}
    assert mask_head.tag(mh) == "tc2"


@pytest.mark.parametrize("block,tag", [
    ({"band": 3}, "tc2-b3"),
    ({"w_band": 0}, "tc2-wb0"),
    ({"w_dist": 0.5}, "tc2-wd0.5"),
    ({"band": 1, "w_band": 0.5, "w_dist": 0}, "tc2-b1-wb0.5-wd0"),
])
def test_tham_so_khac_mac_dinh_vao_nhan(block, tag):
    assert mask_head.tag(mask_head.check({"type": "contact", **block})) == tag


@pytest.mark.parametrize("block,match", [
    ({"bnd": 2}, "band"),                         # gõ sai khoá: gợi ý
    ({"max_pos": 64}, "không có khoá"),          # khoá của A1/T1 không lọt sang
    ({"window": 1.5}, "không có khoá"),
    ({"band": 0}, ">= 1"),
    ({"band": 1.5}, ">= 1"),
    ({"band": True}, ">= 1"),
    ({"w_band": -0.1}, ">= 0"),
    ({"w_dist": "0.25"}, ">= 0"),
    ({"w_dist": False}, ">= 0"),
])
def test_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        mask_head.check({"type": "contact", **block})


def test_a1_va_t1_khong_doi():
    assert mask_head.check({})["type"] == "dynamic"
    assert mask_head.check({"type": "gpr"})["max_pos"] == 64
    with pytest.raises(SystemExit, match=">= 1"):
        mask_head.check({"type": "gpr", "max_pos": 0})


def test_rang_buoc_voi_khoi_train_va_voi_loss():
    with pytest.raises(SystemExit, match="mask_ratio: 4"):
        YoloTrainer.apply_loss(_cfg(mask_ratio=1, mask_head={"type": "contact"}), None)
    with pytest.raises(SystemExit, match="overlap_mask"):
        YoloTrainer.apply_loss(_cfg(overlap=False, mask_head={"type": "contact"}), None)
    with pytest.raises(SystemExit, match="mask_head"):
        YoloTrainer.apply_loss(_cfg(mask_head={"type": "contact"}), "dice")
    cfg = _cfg(mask_head={"type": "contact", "w_band": 1})
    assert YoloTrainer.apply_loss(cfg, None) == ""
    assert cfg["mask_head"]["w_band"] == 1.0 and isinstance(cfg["mask_head"]["w_band"], float)


def test_mo_ta_cho_run_log():
    line = mask_head.describe_active(mask_head.check({"type": "contact"}))
    assert "TC2" in line and "2 ô" in line


def test_print_config_cua_file_config_rieng():
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s-tc2.yaml", "--print-config"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("mask_head:"))
    assert json.loads(line.split(":", 1)[1]) == mask_head.check({"type": "contact"})
    # phần còn lại lấy từ yolo26s.yaml
    assert "mask_ratio" in r.stdout and "overlap_mask" in r.stdout and "1024" in r.stdout


def test_cham_mang_hau_to_tc2(tmp_path):
    from cofseg import artifacts

    run = tmp_path / "2026-10-16_090000_yolo26s-seg-tc2_field-f1_i1024b16e100"
    (run / "weights").mkdir(parents=True)
    mh = mask_head.check({"type": "contact"})
    doc = {"weights": {}, "args": {"imgsz": 1024}, "loss": None, "mask_head": {**mh, "tag": mask_head.tag(mh)}}
    (run / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    w = run / "weights/best.pt"
    assert artifacts.eval_run_name("yolo26s-seg", w, explicit=False) == ("yolo26s-seg-tc2", None)
    assert artifacts.eval_run_name("yolo26s-seg-tc2", w, explicit=False)[0] == "yolo26s-seg-tc2"
