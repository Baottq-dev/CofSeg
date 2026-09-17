"""Chấm model trên một split.

Với model của ultralytics, phép chấm chuẩn là chính `model.val()` của nó —
xem canopyseg/evaluation/native.py. Không viết lại bộ đo đó: nó là thứ `yolo
val` chạy, nên số liệu trùng với mọi báo cáo YOLO khác.

Phần còn lại trong gói này là thứ ultralytics KHÔNG cung cấp: Boundary AP
(Cheng et al., CVPR 2021) và các chỉ số biên từng vùng. Chúng đọc lại chính
file predictions.json mà `val(save_json=True)` xuất ra, nên không thay thế mà
chỉ bổ sung cho phép chấm chuẩn.
"""

from .matching import Match, align_masks, match_instances
from .native import validate
from .report import write_csv, write_predictions

__all__ = [
    "validate",
    "Match",
    "align_masks",
    "match_instances",
    "write_csv",
    "write_predictions",
]
