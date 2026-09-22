"""Huấn luyện. Mỗi họ model một trainer, tất cả thoả cùng hợp đồng.

Import ở đây để registry biết chúng tồn tại — nhờ vậy scripts/train.py chỉ
cần đọc tên trong config, không cần biết YOLO là gì.

Bốn model của bảng benchmark KHÔNG nằm ở đây: mỗi model có bản riêng trong
benchmark/<model>_<người>/cofseg/, do người phụ trách sở hữu. Chỗ này là
đường chạy nhanh ở nhà và nhánh promptable.
"""

from .base import Trainer
from . import maskrcnn, yolo  # noqa: F401 - import để @register chạy

__all__ = ["Trainer"]
