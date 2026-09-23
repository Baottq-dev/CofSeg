"""Chỉ số đo. Hàm thuần: vào hai mặt nạ, ra số. Không GPU, không I/O.

Tách riêng vì mọi kết luận của dự án treo trên đây, nên phần này phải được
chứng minh đúng độc lập với model — xem tests/test_metrics.py.
"""

from .boundary import (
    assd,
    boundary_band,
    boundary_iou,
    hd95,
    normalized_surface_dice,
    signed_boundary_error,
)
from .mask import coverage, dice, excess, iou

__all__ = [
    "iou",
    "dice",
    "coverage",
    "excess",
    "boundary_band",
    "boundary_iou",
    "signed_boundary_error",
    "assd",
    "hd95",
    "normalized_surface_dice",
]
