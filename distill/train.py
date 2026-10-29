"""Train một cấu hình của distill/ bằng đúng script train của benchmark/yolo.

Chạy từ GỐC REPO, như benchmark/yolo/train.py:

    python distill/train.py --config distill/configs/train/tro.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
    python distill/train.py --config distill/configs/train/yolo26s-ka.yaml --data data/export/field/f1 --print-config

Mọi cờ của benchmark/yolo/train.py giữ nguyên nghĩa (--seed, --epochs, --aug, --resume...). Script này chỉ
nạp gói distill (đăng ký trainer `distill_yolo`, đặt sys.path) rồi chạy benchmark/yolo/train.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Python đặt thư mục của script (distill/) lên đầu sys.path; bỏ nó đi để student.py, trainer.py... không
# thành module cấp cao nhất, rồi đặt gốc repo vào để `import distill` được.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import distill  # noqa: E402
import distill.trainer  # noqa: E402,F401 - đăng ký trainer distill_yolo

if __name__ == "__main__":
    runpy.run_path(str(distill.BENCH_YOLO / "train.py"), run_name="__main__")
