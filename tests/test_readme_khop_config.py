"""Lệnh trong README phải khớp config, từng giá trị một.

Lệnh mẫu ghi ra giá trị mặc định để người đọc thấy ngay có gì chỉnh được. Điều
đó chỉ có ích khi giá trị ĐÚNG — một con số cũ nằm lại trong README còn tệ hơn
không ghi gì, vì nó trông như đã được kiểm.

Ba khoá trong config để `null` cho trainer tự tính (`lr`, `weight_decay`,
`val_batch`); lệnh mẫu ghi giá trị HIỆU DỤNG của chúng, nên test đối chiếu với
hằng mặc định của trainer chứ không với `null`.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"

#: model -> (config train, file trainer, hằng mặc định)
CAI_DAT = {
    "maskrcnn": ("maskrcnn_r50_d2.yaml", "training/detectron2.py", "D2_DEFAULTS"),
    "mask2former": ("mask2former_r50_d2.yaml", "training/detectron2.py", "D2_DEFAULTS"),
    "solov2": ("solov2_r50_mm.yaml", "training/mmdet.py", "MM_DEFAULTS"),
    "yolo11": ("yolo11s.yaml", None, None),
}
MODELS = tuple(CAI_DAT)

#: Cờ của train.py, không phải siêu tham số.
CUA_SCRIPT = {"config", "data", "runs", "name", "set"}

#: Giá trị hiệu dụng khi config để `null`. Lấy từ chính recipe, xem
#: BASE_LR/BASE_WD trong trainer.
HIEU_DUNG = {
    "maskrcnn": {"lr": 0.02, "weight_decay": 1e-4},
    "mask2former": {"lr": 1e-4, "weight_decay": 0.05},
    "solov2": {"lr": 0.01, "weight_decay": 1e-4},
}

#: workers phụ thuộc hệ điều hành (0 trên Windows, 8 trên Linux) nên lệnh mẫu
#: ghi giá trị cho máy Linux — chỗ các lượt chạy thật diễn ra.
BO_QUA = {"workers", "verbose", "log_every"}


def _tai_config(model: str) -> dict:
    """Config đã gộp `base:`, nạp bằng chính cofseg.config của thư mục đó."""
    duong = BENCH / model / "cofseg" / "config.py"
    spec = importlib.util.spec_from_file_location(f"_cfg_{model}", duong)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load(BENCH / model / "configs" / "train" / CAI_DAT[model][0])


def _mac_dinh(model: str) -> dict:
    rel, ten = CAI_DAT[model][1], CAI_DAT[model][2]
    if ten is None:
        return {}
    cay = ast.parse((BENCH / model / "cofseg" / rel).read_text(encoding="utf-8"))
    for node in cay.body:
        dich = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(dich, ast.Name) and dich.id == ten:
            ra = {}
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    ra[k.value] = ast.literal_eval(v)
                except ValueError:
                    pass      # giá trị tính lúc chạy, vd DEFAULT_WORKERS
            return ra
    raise AssertionError(f"{model}: không thấy {ten}")


def _co_trong_readme(model: str, lenh: str) -> dict:
    doc = (BENCH / model / "README.md").read_text(encoding="utf-8")
    i = doc.index("### Lệnh đầy đủ")
    muc = doc[i:doc.index("### Đổi backbone")]
    khoi = next(k for k in re.findall(r"```bash\n(.*?)```", muc, re.S) if lenh in k)
    ra = {}
    for m in re.finditer(r'--([a-z0-9_-]+)(?:\s+"([^"]*)"|\s+([^\s\\]+))?', khoi):
        ten, nhay, tho = m.group(1), m.group(2), m.group(3)
        ra[ten.replace("-", "_")] = nhay if nhay is not None else tho
    return ra


def _bang(a, b) -> bool:
    na, nb = str(a).strip(), str(b).strip()
    for ky in "[] '":
        na, nb = na.replace(ky, ""), nb.replace(ky, "")
    try:
        return abs(float(na) - float(nb)) < 1e-12
    except ValueError:
        return na.lower() == nb.lower()


@pytest.mark.parametrize("model", MODELS)
def test_moi_gia_tri_trong_lenh_train_khop_config(model):
    cfg = {**_mac_dinh(model), **(_tai_config(model).get("train") or {})}
    cfg.update({k: v for k, v in HIEU_DUNG.get(model, {}).items() if cfg.get(k) is None})
    if cfg.get("val_batch") is None and "batch" in cfg:
        cfg["val_batch"] = cfg["batch"]

    doc = _co_trong_readme(model, "train.py")
    for co, gia_tri in doc.items():
        if co in CUA_SCRIPT or co in BO_QUA or gia_tri is None:
            continue
        assert co in cfg, f"{model}: --{co} không có trong config"
        assert _bang(cfg[co], gia_tri), \
            f"{model}: --{co}={gia_tri} nhưng config là {cfg[co]}"


@pytest.mark.parametrize("model", MODELS)
def test_lenh_train_khong_bo_sot_tham_so_dang_ke(model):
    """Những tham số quyết định kết quả phải có mặt, không để người đọc đoán."""
    doc = _co_trong_readme(model, "train.py")
    can = {"imgsz", "batch", "epochs"}
    if model != "yolo11":
        can |= {"lr", "weight_decay", "lr_steps", "warmup_iters", "amp",
                "val_conf", "val_batch", "max_det"}
    thieu = can - set(doc)
    assert not thieu, f"{model}: lệnh mẫu thiếu {sorted(thieu)}"


@pytest.mark.parametrize("model", MODELS)
def test_lenh_cham_tro_dung_trong_so_cua_lan_train(model):
    doc = (BENCH / model / "README.md").read_text(encoding="utf-8")
    duoi = "best.pt" if model == "yolo11" else "best.pth"
    assert f"weights/{duoi}" in doc, f"{model}: lệnh chấm không trỏ vào best của run"


def test_mask2former_khong_quang_cao_co_vo_hieu():
    """Bốn cờ không có tác dụng với model này; README phải nói, và lệnh mẫu
    không được chứa chúng."""
    doc = (BENCH / "mask2former" / "README.md").read_text(encoding="utf-8")
    lenh = _co_trong_readme("mask2former", "train.py")
    for co in ("fliplr", "flipud", "rot90", "mask_resolution"):
        assert co not in lenh, f"mask2former: --{co} vô hiệu mà vẫn nằm trong lệnh mẫu"
        assert co in doc, f"mask2former: không nói --{co} vô hiệu"
