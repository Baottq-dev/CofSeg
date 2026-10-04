"""Cờ --data và việc ép kiểu giá trị dòng lệnh.

Hai thứ nhỏ nhưng hay cắn: ba model đọc thư mục fold ở `data.root` còn
ultralytics đọc `data.yaml` bên trong nó, và PyYAML không nhận "1e-4" là số.
"""

from __future__ import annotations

import pytest

from cofseg import cli, training  # noqa: F401 - nạp để đăng ký trainer
from cofseg.registry import resolve

TRAINERS = ["detectron2", "mmdet", "yolo"]


def trainer_or_skip(name: str):
    try:
        return resolve("trainer", name)
    except Exception:
        pytest.skip(f"thư mục này không có trainer {name}")


@pytest.mark.parametrize("name", TRAINERS)
def test_data_arg_gives_the_key_that_trainer_actually_reads(name):
    T = trainer_or_skip(name)
    key, val = T.data_arg("data/export/block/f1")
    if T.DATA_KEY == "yaml":
        assert (key, val) == ("data.yaml", "data/export/block/f1/data.yaml")
    else:
        assert (key, val) == ("data.root", "data/export/block/f1")


@pytest.mark.parametrize("name", TRAINERS)
def test_data_arg_takes_windows_paths_and_trailing_slashes(name):
    """Người chạy trên Windows chép đường dẫn từ Explorer ra là có dấu \\ và
    hay kèm dấu / cuối. Cả hai đều phải nuốt được, không thì config nhận một
    đường dẫn hỏng mà mãi sau mới biết."""
    T = trainer_or_skip(name)
    _, val = T.data_arg("data" + "\\" + "export" + "\\" + "flight" + "\\" + "f6" + "\\")
    assert val.startswith("data/export/flight/f6")
    assert "\\" not in val and "//" not in val


def test_scientific_notation_becomes_a_number():
    """PyYAML theo YAML 1.1 trả "1e-4" về CHUỖI. Mà 1e-4 đúng là cách viết tự
    nhiên nhất cho lr của Mask2Former, và chuỗi đó đi thẳng vào ultralytics
    thì hỏng."""
    got = cli.parse_overrides(["--lr", "1e-4", "--weight-decay", "5e-5"], None)
    assert got == {"lr": 1e-4, "weight_decay": 5e-5}
    assert all(isinstance(v, float) for v in got.values())


def test_ordinary_values_keep_their_types():
    got = cli.parse_overrides(
        ["--epochs", "60", "--lr", "0.01", "--amp", "true",
         "--lr-steps", "[0.7,0.9]", "--optimizer", "auto"], None)
    assert got == {"epochs": 60, "lr": 0.01, "amp": True,
                   "lr_steps": [0.7, 0.9], "optimizer": "auto"}
    assert isinstance(got["epochs"], int) and isinstance(got["amp"], bool)


def test_train_khong_con_co_limit():
    """--limit từng được nhận rồi bị trainer YOLO lờ đi: lệnh khói train cả
    fold mà không báo gì. Giờ nó phải bị chặn ngay ở dòng lệnh."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        [sys.executable, "benchmark/yolo/train.py",
         "--config", "benchmark/yolo/configs/train/yolo26s.yaml", "--print-config", "--limit", "5"],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert r.returncode != 0
    assert "limit" in (r.stdout + r.stderr)
