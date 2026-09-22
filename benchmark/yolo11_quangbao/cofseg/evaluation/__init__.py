"""Chấm model trên một split.

Hai đường, cùng một bộ dữ liệu:

- `evaluate_split` + `coco_eval`: đường CHUNG cho mọi model trong sổ đăng ký
  (YOLO, Mask R-CNN, SAM, model ghép, file dự đoán từ máy khác). Mask AP và
  Boundary AP chấm bằng pycocotools — bản tham chiếu mà detectron2, mmdet,
  torchvision và chính ultralytics (khi save_json trên COCO) gọi bên dưới —
  cộng các chỉ số biên từng vùng mà không bộ nào cung cấp.

- `validate` (native.py): `model.val()` của ultralytics, chỉ cho YOLO. Giữ để
  đối chiếu với mọi báo cáo YOLO khác; số của nó và số COCOeval chênh nhau vài
  phần nghìn do cách nội suy đường PR, không phải do model.
"""

from . import coco_eval
from .matching import Match, align_masks, match_instances
from .native import validate
from .report import write_coco_results, write_csv, write_predictions
from .runner import evaluate_split, summarize

__all__ = [
    "coco_eval",
    "evaluate_split",
    "summarize",
    "validate",
    "Match",
    "align_masks",
    "match_instances",
    "write_csv",
    "write_predictions",
    "write_coco_results",
]
