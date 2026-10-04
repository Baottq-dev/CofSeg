"""Cờ --loss: mặc định là loss gốc, biến thể chỉ bật bằng cờ, tên run nói thật."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofseg.training import losses
from cofseg.training.base import Trainer
from cofseg.training.yolo import YoloTrainer

REPO = Path(__file__).resolve().parents[3]


def _cfg(**extra):
    return {"trainer": "yolo", "model": "weights/yolo26s-seg.pt", "train": {"imgsz": 1024}, **extra}


# --------------------------------------------------------------- mặc định
def test_khong_co_co_la_loss_goc():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, None) == ""
    assert "loss" not in cfg
    assert losses.describe_active(cfg.get("loss")) == "Loss: mặc định của ultralytics"


def test_tham_so_loss_khong_kem_co_bi_chan():
    """--set loss.mask_iou.mix=0.5 mà quên --loss: báo lỗi, không lặng lẽ train
    bằng loss gốc trong khi config.yaml ghi như thể đã dùng mix 0.5."""
    with pytest.raises(SystemExit, match="--loss"):
        YoloTrainer.apply_loss(_cfg(loss={"mask_iou": {"mix": 0.5}}), None)


@pytest.mark.parametrize("spec", [None, "mask_iou"])
def test_bat_bien_the_tu_config_bi_chan(spec):
    """Biến thể chỉ bật bằng cờ: `loss.use` trong config hay --set đều bị chặn."""
    with pytest.raises(SystemExit, match="chỉ bật bằng cờ --loss"):
        YoloTrainer.apply_loss(_cfg(loss={"use": ["mask_iou"]}), spec)


def test_ho_khong_co_bien_the_thi_bao_loi():
    with pytest.raises(SystemExit, match="chưa có biến thể"):
        Trainer.apply_loss(_cfg(), "mask_iou")
    assert Trainer.apply_loss(_cfg(), None) == ""


# --------------------------------------------------------------- bật và đặt tên
def test_mask_iou_mac_dinh():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, "mask_iou") == "mask-iou"
    assert cfg["loss"] == {"use": ["mask_iou"], "tag": "mask-iou",
                           "mask_iou": {"mix": 1.0, "warmup_epochs": 5, "heads": "both"}}
    assert losses.describe_active(cfg["loss"]) == "Loss: mask_iou (mix 1, warmup_epochs 5, heads both)"


def test_tham_so_khac_mac_dinh_vao_ten():
    """Hai cấu hình khác nhau không bao giờ ra cùng một tên thư mục."""
    cfg = _cfg(loss={"mask_iou": {"mix": 0.5, "heads": "o2o"}})
    assert YoloTrainer.apply_loss(cfg, "mask_iou") == "mask-iou-mix0.5-ho2o"
    cfg = _cfg(loss={"mask_iou": {"warmup_epochs": 10}})
    assert YoloTrainer.apply_loss(cfg, "mask_iou") == "mask-iou-wu10"
    # Ghi lại đúng giá trị mặc định thì không đổi tên.
    cfg = _cfg(loss={"mask_iou": {"mix": 1}})
    assert YoloTrainer.apply_loss(cfg, "mask_iou") == "mask-iou"


def test_lap_ten_va_khoang_trang():
    cfg = _cfg()
    assert YoloTrainer.apply_loss(cfg, " mask_iou , mask_iou ") == "mask-iou"
    assert cfg["loss"]["use"] == ["mask_iou"]


@pytest.mark.parametrize("spec,block,match", [
    ("mask_iuo", None, "mask_iou"),                        # gõ sai tên: gợi ý
    ("mask_iou", {"mask_iou": {"mixx": 0.5}}, "mix"),      # gõ sai tham số: gợi ý
    ("mask_iou", {"mask_iou": {"mix": 1.5}}, r"\[0, 1\]"),
    ("mask_iou", {"mask_iou": {"mix": True}}, r"\[0, 1\]"),
    ("mask_iou", {"mask_iou": {"warmup_epochs": -1}}, ">= 0"),
    ("mask_iou", {"mask_iou": {"warmup_epochs": 2.5}}, ">= 0"),
    ("mask_iou", {"mask_iou": {"heads": "o2m"}}, "o2o"),
    ("mask_iou", {"dice": {"weight": 1}}, "không bật"),   # tham số cho biến thể chưa bật
])
def test_gia_tri_sai_bi_chan(spec, block, match):
    cfg = _cfg(**({"loss": block} if block else {}))
    with pytest.raises(SystemExit, match=match):
        YoloTrainer.apply_loss(cfg, spec)


def test_list_losses_liet_ke_tham_so():
    out = YoloTrainer.describe_losses()
    for s in ("mask_iou", "mask_iou.mix", "mask_iou.warmup_epochs", "mask_iou.heads", "--loss mask_iou"):
        assert s in out


# --------------------------------------------------------------- gắn vào lượt train
torch = pytest.importorskip("torch")


def _yolo(yaml_name):
    from ultralytics import YOLO

    return YOLO(yaml_name, task="segment")


def test_install_gan_loss_vao_dung_model_dang_train(tmp_path):
    """Callback thay hàm loss trên model đang train; lớp model không đổi và bản
    chụp để lưu checkpoint (deepcopy + criterion=None) vẫn là model chuẩn."""
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.loss import E2ELoss

    from cofseg.training.mask_iou_loss import MaskIoUSegLoss

    y = _yolo("yolo26n-seg.yaml")
    cfg = _cfg()
    YoloTrainer.apply_loss(cfg, "mask_iou")
    losses.install(y, cfg["loss"], tmp_path / "loss_mask_iou.csv")

    from ultralytics.cfg import get_cfg

    model = y.model
    model.args = get_cfg()  # trainer gán args dạng namespace trước on_train_start (set_model_attributes)
    trainer = SimpleNamespace(model=model, epoch=0)
    for cb in y.callbacks["on_train_start"]:
        cb(trainer)
    assert type(model) is SegmentationModel
    assert isinstance(model.criterion, E2ELoss)
    assert isinstance(model.criterion.one2one, MaskIoUSegLoss)
    assert isinstance(model.criterion.one2many, MaskIoUSegLoss)

    trainer.epoch = 3
    for cb in y.callbacks["on_train_epoch_start"]:
        cb(trainer)
    assert model.criterion.one2one.epoch == 3 and model.criterion.one2many.epoch == 3

    snap = copy.deepcopy(model)
    snap.criterion = None  # đúng việc trainer.save_model làm trước khi lưu
    assert type(snap) is SegmentationModel


def test_khong_co_co_thi_khong_gan_callback():
    """Đường mặc định không đụng vào callback nào của ultralytics."""
    y = _yolo("yolo26n-seg.yaml")
    before = {k: len(v) for k, v in y.callbacks.items()}
    cfg = _cfg()
    YoloTrainer.apply_loss(cfg, None)
    assert not cfg.get("loss")
    assert {k: len(v) for k, v in y.callbacks.items()} == before


def test_sai_phien_ban_ultralytics_thi_dung(monkeypatch, tmp_path):
    import ultralytics

    cfg = _cfg()
    YoloTrainer.apply_loss(cfg, "mask_iou")
    monkeypatch.setattr(ultralytics, "__version__", "8.5.0")
    with pytest.raises(SystemExit, match="8.4.143"):
        losses.install(_yolo("yolo26n-seg.yaml"), cfg["loss"], tmp_path / "x.csv")


def test_heads_o2o_voi_model_mot_dau_bi_chan(tmp_path):
    cfg = _cfg(loss={"mask_iou": {"heads": "o2o"}})
    YoloTrainer.apply_loss(cfg, "mask_iou")
    with pytest.raises(SystemExit, match="o2o"):
        losses.install(_yolo("yolov8n-seg.yaml"), cfg["loss"], tmp_path / "x.csv")


# --------------------------------------------------------------- dòng lệnh
def _train_py(*args):
    return subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s.yaml", *args],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )


def test_cli_print_config_hien_khoi_loss():
    goc = _train_py("--print-config")
    assert goc.returncode == 0, goc.stderr
    assert "loss: gốc của framework" in goc.stdout
    moi = _train_py("--print-config", "--loss", "mask_iou", "--set", "loss.mask_iou.mix=0.5")
    assert moi.returncode == 0, moi.stderr
    assert '"tag": "mask-iou-mix0.5"' in moi.stdout


def test_cli_set_loss_thieu_co_bao_loi():
    r = _train_py("--print-config", "--set", "loss.mask_iou.mix=0.5")
    assert r.returncode != 0
    assert "--loss" in (r.stdout + r.stderr)


def test_cli_list_losses():
    r = _train_py("--list-losses")
    assert r.returncode == 0, r.stderr
    assert "mask_iou" in r.stdout
