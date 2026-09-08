"""Huấn luyện một model bất kỳ. Model do config chỉ định, không phải script.

    python scripts/train.py --config configs/train/yolo26s.yaml
    python scripts/train.py --config configs/train/yolo26s.yaml --probe
    python scripts/train.py --config configs/train/yolo26s.yaml \
        --set train.imgsz=1536 --set train.batch=2

Script này KHÔNG biết YOLO tồn tại. Nó đọc khoá `trainer` trong config, tra
sổ đăng ký, rồi gọi ba phương thức của hợp đồng. Thêm Mask R-CNN hay model
tự viết = thêm một file trong canopyseg/training/, script giữ nguyên.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import artifacts  # noqa: E402
from canopyseg import config as cfgmod  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg import training  # noqa: F401,E402 - nạp để đăng ký trainer
from canopyseg.registry import available, resolve  # noqa: E402

console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè, vd --set train.imgsz=1536",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="chỉ ước lượng VRAM rồi dừng, không huấn luyện",
    )
    ap.add_argument("--runs", default="runs", help="thư mục gốc chứa kết quả")
    ap.add_argument("--name", default=None, help="tên lần chạy (mặc định lấy từ config)")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.overrides)
    if "trainer" not in cfg:
        raise SystemExit(
            f"config thiếu khoá 'trainer'. Hiện có: {available('trainer')}"
        )

    name = a.name or cfg.get("name") or Path(a.config).stem
    run_dir = artifacts.create_run_dir(a.runs, name)
    artifacts.write_env(run_dir)
    artifacts.snapshot_config(run_dir, cfg)
    print(f"Lần chạy: {run_dir}")

    trainer = resolve("trainer", cfg["trainer"])(cfg, run_dir)

    info = trainer.prepare()
    if info:
        print("Dữ liệu:", json.dumps(info, ensure_ascii=False)[:400])

    if a.probe:
        p = trainer.probe()
        print("\nDò VRAM:", json.dumps(p, indent=2, ensure_ascii=False))
        (run_dir / "probe.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    summary = trainer.fit()
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("\nXong. Trọng số:", json.dumps(summary.get("weights"), ensure_ascii=False))
    print("Kết quả:", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
