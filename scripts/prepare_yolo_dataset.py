"""Sinh nhãn YOLO-seg từ bộ COCO đã xuất, rồi tự kiểm lại.

    python scripts/prepare_yolo_dataset.py --config configs/dataset/dataset_v1.yaml

Chỉ GHI THÊM labels/ và data.yaml vào thư mục bộ dữ liệu; images/ và
annotations/ không bị đụng tới. Chạy lại nhiều lần cho kết quả y hệt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import config as cfgmod  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg.datasets import yolo as yolo_ds  # noqa: E402

console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="chỉ kiểm nhãn đang có, không ghi đè",
    )
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.overrides)
    root = Path(cfg["root"])
    splits = list(cfg["splits"])
    min_area = float(cfg.get("min_area", 0.0))

    if not a.check_only:
        stats = yolo_ds.write_labels(
            root, splits, class_index=int(cfg.get("class_index", 0)), min_area=min_area
        )
        print("Đã ghi nhãn:")
        for sp, s in stats["splits"].items():
            print(
                "  %-5s %3d ảnh  %4d vùng  (ảnh không có vùng: %d, bỏ: %d)"
                % (sp, s["images"], s["labels"], s["empty_images"], s["dropped"])
            )
        yaml_path = yolo_ds.write_data_yaml(
            root,
            {k: f"images/{v}" for k, v in cfg["yolo_splits"].items()},
            cfg["names"],
        )
        print("Đã ghi", yaml_path)

    # Kiểm luôn, không đợi ai nhớ chạy: nhãn sai ở đây thì mọi con số huấn
    # luyện về sau đều vô nghĩa mà không có gì báo trước.
    rep = yolo_ds.verify_roundtrip(root, splits, min_area=min_area)
    print("\nKiểm quay vòng COCO -> YOLO -> pixel:")
    for sp, r in rep["splits"].items():
        print(
            "  %-5s %4d vùng  sai số lớn nhất %.2e (chuẩn hoá)  %s"
            % (
                sp,
                r["regions_checked"],
                r["max_normalized_error"],
                "ĐẠT" if r["ok"] else "HỎNG: %d vấn đề" % r["n_problems"],
            )
        )
        for p in r["problems"]:
            print("        - " + p)
    print(json.dumps({"ok": rep["ok"]}, ensure_ascii=False))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
