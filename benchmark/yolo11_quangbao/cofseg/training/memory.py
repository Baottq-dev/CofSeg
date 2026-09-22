"""Vách bộ nhớ GPU: ngưỡng mà một lần chạy không được vượt.

Đo trên card 8 GB của máy này: vượt ~7.0 GB thì driver NVIDIA (Windows)
không báo OOM mà âm thầm tràn sang RAM hệ thống, tốc độ rơi hàng trăm lần
(1536/batch 2 -> 6.99 GB, 1.25 it/s; 1280/batch 4 -> 7.22 GB, 179 s/it).
7.0 / 8.0 = 0.875, làm tròn thành 0.88 và áp cho mọi card: trên 24 GB vách là
~21 GB. Trên Linux driver OOM thẳng thay vì tràn, nhưng cùng biên an toàn
đó vẫn là chỗ nên dừng — phần còn lại là cho pha val và cho batch dày vùng.
"""

from __future__ import annotations

WALL_FRACTION = 0.88


def total_gb(device: int = 0) -> float | None:
    """Tổng VRAM của card, GB; None nếu không có CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device).total_memory / 2**30


def wall_gb(device: int = 0) -> float | None:
    """Ngưỡng an toàn (GB) cho card đang dùng; None nếu không có CUDA."""
    total = total_gb(device)
    return None if total is None else round(total * WALL_FRACTION, 1)
