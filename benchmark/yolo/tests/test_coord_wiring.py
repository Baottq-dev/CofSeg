"""KT4 nối vào benchmark: khối `mask_head` loại coord, file config riêng, tên
lượt chấm, và lớp bọc của đường chấm chọn đúng predictor."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofseg.training import mask_head
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt",
            "train": {"imgsz": 1024, "mask_ratio": 4, "overlap_mask": True}, **extra}


def test_mac_dinh_va_nhan():
    mh = mask_head.check({"type": "coord"})
    assert mh == {"type": "coord", "box": "pred"}
    assert mask_head.tag(mh) == "kt4"
    assert mask_head.tag(mask_head.check({"type": "coord", "box": "gt"})) == "kt4-gt"


@pytest.mark.parametrize("block,match", [
    ({"box": "GT"}, "pred"),
    ({"box": True}, "pred"),
    ({"boxx": "pred"}, "box"),                    # gõ sai khoá: gợi ý
    ({"band": 2}, "không có khoá"),               # khoá của TC2 không lọt sang
    ({"max_pos": 64}, "không có khoá"),
])
def test_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        mask_head.check({"type": "coord", **block})


def test_khong_di_chung_voi_loss():
    with pytest.raises(SystemExit, match="mask_head"):
        YoloTrainer.apply_loss(_cfg(mask_head={"type": "coord"}), "dice")
    cfg = _cfg(mask_head={"type": "coord", "box": "gt"})
    assert YoloTrainer.apply_loss(cfg, None) == ""
    assert cfg["mask_head"] == {"type": "coord", "box": "gt"}


def test_mo_ta_cho_run_log():
    line = mask_head.describe_active(mask_head.check({"type": "coord"}))
    assert "KT4" in line and "dự đoán" in line
    assert "thật" in mask_head.describe_active(mask_head.check({"type": "coord", "box": "gt"}))


def test_print_config_cua_file_config_rieng():
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s-kt4.yaml", "--print-config"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("mask_head:"))
    assert json.loads(line.split(":", 1)[1]) == mask_head.check({"type": "coord"})
    assert "mask_ratio" in r.stdout and "1024" in r.stdout


def test_set_doi_sang_hop_that():
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s-kt4.yaml", "--set", "mask_head.box=gt",
         "--print-config"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("mask_head:"))
    assert json.loads(line.split(":", 1)[1])["box"] == "gt"


def test_cham_mang_hau_to_kt4(tmp_path):
    from cofseg import artifacts

    run = tmp_path / "2026-10-20_090000_yolo26s-seg-kt4_field-f1_i1024b16e100"
    (run / "weights").mkdir(parents=True)
    mh = mask_head.check({"type": "coord"})
    doc = {"weights": {}, "args": {"imgsz": 1024}, "loss": None, "mask_head": {**mh, "tag": mask_head.tag(mh)}}
    (run / "summary.json").write_text(json.dumps(doc), encoding="utf-8")
    w = run / "weights/best.pt"
    assert artifacts.eval_run_name("yolo26s-seg", w, explicit=False) == ("yolo26s-seg-kt4", None)


def test_lop_boc_cham_chon_predictor_kt4(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    import copy

    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    from cofseg.models.yolo_seg import YoloSegModel
    from cofseg.training.coord_head import CoordSegPredictor, to_coord

    torch.manual_seed(0)
    m = SegmentationModel("yolo26n-seg.yaml", nc=1, verbose=False)
    m.args = get_cfg()
    to_coord(m, {"type": "coord", "box": "pred"})
    m.names = {0: "crown"}
    p = tmp_path / "kt4.pt"
    torch.save({"model": copy.deepcopy(m).half(), "train_args": {"task": "segment", "imgsz": 128}}, p)
    y = YoloSegModel(str(p), imgsz=128, conf=0.0, max_det=5)
    assert y.mask_head == "kt4" and y._predict_kw["predictor"] is CoordSegPredictor
    with pytest.raises(SystemExit, match="KT4"):
        YoloSegModel(str(p), imgsz=128, crop_expand=1.1)
