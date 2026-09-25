"""Huấn luyện. Mỗi họ model một trainer, tất cả thoả cùng hợp đồng.

Import ở đây để registry biết chúng tồn tại — nhờ vậy benchmark/yolo11/train.py chỉ
cần đọc tên trong config, không cần biết YOLO là gì.
"""

from .base import Trainer
from . import yolo  # noqa: F401

__all__ = ["Trainer"]
