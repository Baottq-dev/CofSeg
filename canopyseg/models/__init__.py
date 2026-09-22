"""Thí sinh. Mỗi họ model một file, tất cả thoả cùng một hợp đồng.

Import ở đây để registry biết chúng tồn tại. Thêm họ mới = thêm một dòng.
"""

from .base import Prediction, SegmentationModel
from .build import build_model, model_param_names
from . import coco_predictions, compose, detectron2, maskrcnn, mmdet, sam_prompt, yolo_seg  # noqa: F401 - import để @register chạy

__all__ = ["Prediction", "SegmentationModel", "build_model", "model_param_names"]
