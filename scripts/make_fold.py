"""Cắt bản xuất chưa chia thành fold leave-one-field-out.

    python scripts/make_fold.py --export data/export/all_v2 --fold f4
    python scripts/make_fold.py --export data/export/all_v2 --all

Fold khai trong configs/dataset/folds.yaml. Mỗi fold ra một thư mục cạnh bản
xuất (data/export/f4, ...) đúng bố cục mà train.py / evaluate.py đọc:
    --set data.root=data/export/f4          (Mask R-CNN, detectron2, evaluate)
    --set data.yaml=data/export/f4/data.yaml   (YOLO)

Ảnh được hardlink, không chép; id ảnh giữ nguyên từ bản xuất gốc. Thư mục
fold đã có thì script dừng — xoá tay rồi chạy lại, không ghi đè ngầm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.datasets import folds as foldmod  # noqa: E402

console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--export", required=True, help="thư mục bản xuất split_by=none")
    ap.add_argument("--folds", default="configs/dataset/folds.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", help="tên fold trong folds.yaml, vd f4")
    g.add_argument("--all", action="store_true", help="cắt mọi fold trong folds.yaml")
    ap.add_argument(
        "--out-root",
        help="thư mục chứa các fold; mặc định là thư mục cha của --export",
    )
    ap.add_argument(
        "--copy", action="store_true", help="chép ảnh thay vì hardlink (tốn đĩa)"
    )
    a = ap.parse_args()

    doc = foldmod.load_folds(a.folds)
    export = Path(a.export)
    out_root = Path(a.out_root) if a.out_root else export.parent
    names = sorted(doc["folds"]) if a.all else [a.fold]

    for name in names:
        fields = foldmod.fold_fields(doc, name)
        out = out_root / name
        s = foldmod.make_fold(export, name, fields, out, copy=a.copy)
        print(f"{name} -> {out}")
        for sp in foldmod.SPLITS:
            r = s["splits"][sp]
            print(
                "  %-5s %-28s %4d ảnh  %5d vùng  (ảnh nền: %d)"
                % (sp, ",".join(r["fields"]), r["images"], r["annotations"], r["empty_images"])
            )
        if s["images_copied"]:
            print(f"  ảnh chép thay vì hardlink: {s['images_copied']}")
        if s["images_skipped"]:
            print(f"  bỏ qua {len(s['images_skipped'])} ảnh không thuộc ruộng nào trong folds.yaml")
        if not s["labels"]:
            print("  bản xuất không có labels/ -> không ghi nhãn YOLO và data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
