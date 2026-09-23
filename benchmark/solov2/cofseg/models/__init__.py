"""Thí sinh. Mỗi họ model một file, tất cả thoả cùng một hợp đồng.

Import ở đây để registry biết chúng tồn tại. Thêm họ mới = thêm một dòng.
"""

from .base import Prediction, SegmentationModel
from .build import build_model, model_param_names
from . import coco_predictions, mmdet  # noqa: F401

__all__ = ["Prediction", "SegmentationModel", "build_model", "model_param_names"]
