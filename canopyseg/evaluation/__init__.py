"""Chạy một model bất kỳ trên một split và chấm điểm.

Tầng này không biết YOLO hay SAM tồn tại — nó chỉ làm việc với hợp đồng
SegmentationModel. Nhờ vậy hai họ model đi qua cùng một đường đo và các con số
so sánh được với nhau.
"""

from .matching import Match, align_masks, match_instances
from .runner import evaluate_split
from .report import format_report, write_csv

__all__ = [
    "Match",
    "align_masks",
    "match_instances",
    "evaluate_split",
    "format_report",
    "write_csv",
]
