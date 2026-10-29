"""Cải tiến YOLO26-seg cho CoffeeSeg: trò nhẹ, giám sát mặt nạ ngoài hộp, (sau này) chắt lọc.

Tách khỏi benchmark/: benchmark/ chỉ giữ các model đem so, chạy công thức chuẩn. Thư mục này dùng lại
phần train, bộ đọc dữ liệu và bộ chấm của benchmark/yolo/cofseg mà không sửa, không chép; benchmark/
không bao giờ import ngược lại đây.

Import gói này làm hai việc trước mọi thứ khác:

1. Đặt benchmark/yolo lên đầu sys.path, để `import cofseg` nạp đúng bản lõi của benchmark/yolo (mỗi thư
   mục model trong benchmark/ có một bản cofseg riêng).
2. Tắt tự cài gói của ultralytics (YOLO_AUTOINSTALL). Checkpoint của trò chứa lớp của gói này; nạp nó ở
   nơi không import được `distill` thì ultralytics coi `distill` là gói thiếu và tự `pip install distill`
   từ PyPI — một gói lạ, không phải code này. Biến môi trường phải đặt trước khi ultralytics được import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_YOLO = ROOT / "benchmark" / "yolo"

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
if str(BENCH_YOLO) not in sys.path:
    sys.path.insert(0, str(BENCH_YOLO))
