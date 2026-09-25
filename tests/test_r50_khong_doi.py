"""Lượt r50 phải cho ra ĐÚNG config như sáu lượt đã chạy trong runs/.

Thêm đường backbone mới mà làm lệch mốc số 0 thì sáu lượt `field/f1..f6` đã
chạy thành vô nghĩa: không so được với bất cứ lượt nào chạy sau. Test này
không tin vào việc đọc code — nó dựng lại danh sách tuỳ chọn cho r50 rồi đối
chiếu với `weights/d2_config.yaml` mà detectron2 đã dump ra trong chính lượt
chạy thật.

Giá trị đối chiếu chép từ:
    runs/train/2026-09-25_142157_maskrcnn-r50-d2_field-f6_i1024b16e50/
        weights/d2_config.yaml
(fold f6: 470 ảnh train, batch 16, 50 epoch, imgsz 1024.)
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"

#: Lượt f6 đã chạy: 1650 iteration = 50 epoch x 33, nên n_train nằm trong
#: (32*16, 33*16] = (512, 528]. Số thật đọc từ dataset_check.json của lượt đó.
N_TRAIN = 522
IMGSZ, BATCH, EPOCHS = 1024, 16, 50

#: Nguyên văn từ d2_config.yaml của lượt chạy thật.
DA_CHAY = {
    "SOLVER.BASE_LR": 0.02,
    "SOLVER.WEIGHT_DECAY": 0.0001,
    "SOLVER.MOMENTUM": 0.9,
    "SOLVER.MAX_ITER": 1650,
    "SOLVER.STEPS": (1155, 1485),
    "SOLVER.GAMMA": 0.1,
    "SOLVER.WARMUP_ITERS": 49,
    "SOLVER.IMS_PER_BATCH": 16,
    "SOLVER.CHECKPOINT_PERIOD": 33,
    "SOLVER.AMP.ENABLED": True,
    "TEST.EVAL_PERIOD": 33,
    "TEST.DETECTIONS_PER_IMAGE": 100,
    "MODEL.ROI_HEADS.NUM_CLASSES": 1,
    "MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.05,
    "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 14,
    "INPUT.MIN_SIZE_TRAIN": (576,),
    "INPUT.MAX_SIZE_TRAIN": 1024,
    "INPUT.MIN_SIZE_TEST": 576,
    "INPUT.MAX_SIZE_TEST": 1024,
    "DATALOADER.NUM_WORKERS": 8,
}


def _mod(model: str):
    """training/detectron2.py của thư mục đó, nạp thẳng — phần build_opts là
    thuần Python nên không cần detectron2."""
    d = BENCH / model
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for k in [m for m in sys.modules if m == "cofseg" or m.startswith("cofseg.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cofseg"] = m
    spec.loader.exec_module(m)
    from cofseg.training import detectron2 as t
    return t


def _opts(model: str, backbone=None) -> dict:
    t = _mod(model)
    args = {**t.D2_DEFAULTS, "imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS,
            "workers": 8, "lr_steps": [0.7, 0.9], "warmup_iters": 0.03}
    flat = t.build_opts("maskrcnn", N_TRAIN, args, aspect=9 / 16,
                        names={"train": "coffee_train", "val": "coffee_val"},
                        out_dir="", backbone=backbone)
    return dict(zip(flat[::2], flat[1::2]))


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_r50_ra_dung_config_da_chay(model):
    """Không truyền backbone = đường của sáu lượt đã chạy."""
    got = _opts(model)
    for k, v in DA_CHAY.items():
        assert k in got, f"{model}: mất khoá {k}"
        assert got[k] == v, f"{model}: {k} = {got[k]!r}, lượt đã chạy là {v!r}"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_r50_khong_dat_input_format(model):
    """INPUT.FORMAT phải KHÔNG xuất hiện khi chưa đổi backbone: lượt đã chạy
    dùng mặc định BGR của detectron2, đặt lại nó là đổi thứ đang đúng."""
    assert "INPUT.FORMAT" not in _opts(model)


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_backbone_yaml_khong_dung_toi_lich_hoc(model):
    """Đổi sang một backbone CNN chỉ đổi kiến trúc, không đổi lr hay lịch —
    nếu không thì `r50` và `r101` không so được với nhau."""
    t = _mod(model)
    r101 = t.Detectron2Trainer.BACKBONES["maskrcnn"]["r101"]
    got = _opts(model, backbone=r101)
    for k, v in DA_CHAY.items():
        assert got[k] == v, f"{model}: --backbone r101 làm lệch {k}"
    assert "INPUT.FORMAT" not in got


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_vitdet_doi_dung_ba_thu_recipe_bat_buoc(model):
    """ViTDet phải đổi lr/weight_decay/kênh ảnh — và CHỈ ba thứ đó.

    AdamW 1e-4 chứ không phải SGD 0.02, weight_decay 0.1 chứ không phải 1e-4,
    RGB chứ không phải BGR. Lịch (max_iter, steps, warmup) giữ nguyên để lượt
    ViT vẫn cùng ngân sách với ba model kia.
    """
    t = _mod(model)
    vit = t.Detectron2Trainer.BACKBONES["maskrcnn"]["vit-b"]
    got = _opts(model, backbone=vit)
    assert got["SOLVER.BASE_LR"] == pytest.approx(1e-4)
    assert got["SOLVER.WEIGHT_DECAY"] == pytest.approx(0.1)
    assert got["INPUT.FORMAT"] == "RGB"
    for k in ("SOLVER.MAX_ITER", "SOLVER.STEPS", "SOLVER.WARMUP_ITERS",
              "SOLVER.IMS_PER_BATCH", "TEST.EVAL_PERIOD", "INPUT.MAX_SIZE_TRAIN"):
        assert got[k] == DA_CHAY[k], f"{model}: ViTDet làm lệch {k}"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_lr_van_scale_theo_batch(model):
    """Luật scale tuyến tính áp cho cả backbone mới, không riêng r50."""
    t = _mod(model)
    vit = t.Detectron2Trainer.BACKBONES["maskrcnn"]["vit-b"]
    args = {**t.D2_DEFAULTS, "imgsz": IMGSZ, "batch": 8, "epochs": EPOCHS,
            "workers": 8, "lr_steps": [0.7, 0.9], "warmup_iters": 0.03}
    flat = t.build_opts("maskrcnn", N_TRAIN, args, aspect=9 / 16, backbone=vit)
    got = dict(zip(flat[::2], flat[1::2]))
    assert got["SOLVER.BASE_LR"] == pytest.approx(1e-4 * 8 / 16)


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_moi_o_vitdet_du_thong_tin_de_dung_model(model):
    """Ô LazyConfig phải đủ: config, kích thước ViT, recipe optimizer, kênh
    ảnh và checkpoint COCO. Thiếu một cái là hỏng ở tận máy thuê."""
    t = _mod(model)
    bang = t.Detectron2Trainer.BACKBONES["maskrcnn"]
    lazy = {k: v for k, v in bang.items() if v.get("lazy")}
    assert len(lazy) == 3, f"{model}: chờ vit-b/l/h, thấy {sorted(lazy)}"
    for ten, spec in lazy.items():
        assert spec["lazy"].endswith(".py")
        assert spec["input_format"] == "RGB", ten
        assert spec["checkpoint"].startswith("https://"), ten
        v = spec["vit"]
        assert {"embed_dim", "depth", "num_heads", "drop_path_rate", "global_at"} <= set(v), ten
        # Block attention toàn cục phải nằm trong phạm vi độ sâu.
        assert max(v["global_at"]) < v["depth"], ten
        o = spec["optim"]
        assert o["num_layers"] == v["depth"], f"{ten}: lr decay đếm sai số tầng"
        assert 0 < o["lr_decay_rate"] < 1, ten


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_duong_lazy_khong_cham_vao_duong_yaml(model):
    """`build_model`/`build_optimizer` phải trả về bản gốc khi backbone không
    phải LazyConfig — đó là điều giữ cho lượt r50 chạy y như cũ."""
    src = (BENCH / model / "cofseg" / "training" / "detectron2.py").read_text(encoding="utf-8")
    cay = ast.parse(src)
    ten = {n.name for n in ast.walk(cay) if isinstance(n, ast.FunctionDef)}
    assert {"lazy_model", "lazy_optimizer", "weights_report"} <= ten
    for ham in ("build_model", "build_optimizer"):
        i = src.index(f"def {ham}(cls, cfg")
        than = src[i:i + 400]
        assert 'if not bb.get("lazy"):' in than, f"{model}: {ham} thiếu nhánh quay về bản gốc"
        assert f"return base.{ham}(" in than, f"{model}: {ham} không gọi lại bản gốc"
