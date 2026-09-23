"""Gói kết quả trên máy Linux để mang về máy nhà.

    python scripts/pack_results.py                       # -> results_<ngày>.tar
    python scripts/pack_results.py --out ket_qua.tar
    python scripts/pack_results.py --list                # xem sẽ gói gì, không tạo file

Lấy: dự đoán test (preds/), file kết quả nhỏ mỗi thành viên đã chép sang
results/, và từ mỗi lần chạy trong benchmark/<model>/runs/ thì lấy trọng số
tốt nhất, config để dựng lại model, results.csv, số đo và log.

KHÔNG lấy: ảnh, checkpoint từng epoch, thư mục làm việc của framework — chúng
nặng và dựng lại được. `--weights` để bỏ luôn cả trọng số nếu chỉ cần số.
"""

from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: File lấy trong mỗi thư mục runs/train/<lần chạy>/
TRAIN_FILES = [
    "weights/best.pt", "weights/best.pth",          # detectron2, mmdet, torchvision
    "ultralytics/weights/best.pt",                  # ultralytics tự đặt chỗ này
    "weights/d2_config.yaml", "weights/mmdet_config.py",   # để dựng lại model ở nhà
    "results.csv", "config.yaml", "env.json", "dataset_check.json",
    "summary.json", "test_metrics.json", "predictions.json", "run.log",
]
#: File lấy trong mỗi thư mục runs/eval/<lần chấm>/
EVAL_FILES = ["metrics.json", "cocoeval.txt", "per_region.csv", "per_image.csv",
              "config.yaml"]
WEIGHT_SUFFIX = (".pt", ".pth")


def collect(weights: bool = True) -> list[Path]:
    out: list[Path] = []
    out += sorted((ROOT / "preds").glob("*.json"))
    out += sorted((ROOT / "runs").glob("remote_*.log"))
    for member in sorted((ROOT / "benchmark").iterdir()):
        if not member.is_dir():
            continue
        out += sorted(p for p in (member / "results").glob("*") if p.is_file())
        for run in sorted((member / "runs" / "train").glob("*/")):
            out += [run / f for f in TRAIN_FILES if (run / f).is_file()]
        for run in sorted((member / "runs" / "eval").glob("*/")):
            out += [run / f for f in EVAL_FILES if (run / f).is_file()]
    if not weights:
        out = [p for p in out if p.suffix not in WEIGHT_SUFFIX]
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="tên file tar; mặc định results_<ngày>.tar")
    ap.add_argument("--list", action="store_true", help="liệt kê, không tạo file")
    ap.add_argument("--no-weights", action="store_true", help="bỏ file trọng số")
    a = ap.parse_args(argv)

    files = collect(weights=not a.no_weights)
    if not files:
        print("Không có gì để gói: chưa chạy lần nào?")
        return 1
    total = sum(p.stat().st_size for p in files)
    if a.list:
        for p in files:
            print(f"  {p.stat().st_size / 2**20:8.1f} MB  {p.relative_to(ROOT).as_posix()}")
        print(f"{len(files)} file, {total / 2**20:.0f} MB")
        return 0

    out = Path(a.out or f"results_{time.strftime('%F_%H%M')}.tar")
    with tarfile.open(out, "w") as tf:
        for p in files:
            tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    print(f"{out}  ({out.stat().st_size / 2**20:.0f} MB, {len(files)} file)")
    print("Ở máy nhà: tar -xf " + out.name)
    print("           python scripts/summarize_folds.py --eval benchmark/*/runs/eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
