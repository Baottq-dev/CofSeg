"""Trainer YOLO của thư mục này: hợp đồng tham số và vách bộ nhớ.

Bản sao lõi trong cofseg/ là của riêng thư mục, nên test cũng ở đây: sửa
trainer mà quên chạy test thì không ai khác biết.
"""

from __future__ import annotations

import pytest


def test_yolo_trainer_declares_ultralytics_params():
    """Danh sách sống của ultralytics: phải có các tham số hay dùng và bốn khoá."""
    from cofseg.training.yolo import YoloTrainer

    names = YoloTrainer.param_names()
    assert names is not None and {"imgsz", "batch", "epochs", "mask_ratio"} <= names
    assert YoloTrainer.locked_params() == {"data", "project", "name", "exist_ok"}
    assert YoloTrainer.locked_params() <= names


def test_memory_wall_scales_with_the_card(monkeypatch):
    """0.88 x VRAM: vách 7.0 GB trên card 8 GB, 21.1 GB trên card 24 GB."""
    from cofseg.training import memory

    monkeypatch.setattr(memory, "total_gb", lambda device=0: 8.0)
    assert memory.wall_gb() == pytest.approx(7.0)
    monkeypatch.setattr(memory, "total_gb", lambda device=0: 24.0)
    assert memory.wall_gb() == pytest.approx(21.1)
    monkeypatch.setattr(memory, "total_gb", lambda device=0: None)
    assert memory.wall_gb() is None
