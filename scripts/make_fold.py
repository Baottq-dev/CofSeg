"""Cắt bản xuất chưa chia thành sáu fold: mỗi ruộng làm test đúng một lần.

    python scripts/make_fold.py --export data/export/dataset_v1 \
        --val configs/dataset/val_block.yaml --all --out-root data/export/block

    python scripts/make_fold.py --export data/export/dataset_v1 \
        --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

    python scripts/make_fold.py --export data/export/dataset_v1 \
        --val configs/dataset/val_block.yaml --fold f4

Sáu lượt khai trong configs/dataset/folds.yaml (chỉ nói ruộng nào làm test).
Cách cắt val khai riêng trong --val, vì có hai cách và chúng cho hai bộ fold
khác nhau — nên để cạnh nhau mà so, đừng chồng lên nhau.

Mỗi fold ra một thư mục đúng bố cục mà train.py / evaluate.py đọc:
    --set data.root=data/export/block/f4          (detectron2, mmdet, evaluate)
    --set data.yaml=data/export/block/f4/data.yaml   (YOLO)

Ảnh được hardlink, không chép: 12 fold vẫn gần như không tốn thêm đĩa. id ảnh
giữ nguyên từ bản xuất gốc. Thư mục fold đã có thì script dừng — xoá tay rồi
chạy lại, không ghi đè ngầm.

fold.json của mỗi fold ghi lại cách chia, điểm cắt từng đường bay, danh sách
ảnh bị bỏ làm đệm, và số rò rỉ ĐO LẠI sau khi chia — để sau này đọc kết quả
còn biết fold đó được cắt ra thế nào.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.datasets import flightlog, folds as foldmod, valsplit  # noqa: E402

console.setup()


def pool_frames(export: Path, train_fields: list[str]) -> tuple[list, dict[str, int]]:
    """Frame của các ruộng train, kèm số vùng tán mỗi ảnh."""
    raw = json.loads((export / "annotations" / "instances.json").read_text(encoding="utf-8"))
    per_id: dict[int, int] = {}
    for a in raw["annotations"]:
        per_id[int(a["image_id"])] = per_id.get(int(a["image_id"]), 0) + 1
    regions, names = {}, []
    keep = set(train_fields)
    for im in raw["images"]:
        if foldmod.field_of(im["file_name"]) in keep:
            names.append(im["file_name"])
            regions[im["file_name"]] = per_id.get(int(im["id"]), 0)
    return flightlog.frames(names), regions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--export", required=True, help="thư mục bản xuất split_by=none")
    ap.add_argument("--folds", default="configs/dataset/folds.yaml")
    ap.add_argument("--val", default="configs/dataset/val_block.yaml",
                    help="cách cắt val: val_block.yaml hoặc val_flight.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", help="tên fold trong folds.yaml, vd f4")
    g.add_argument("--all", action="store_true", help="cắt mọi fold trong folds.yaml")
    ap.add_argument("--out-root",
                    help="thư mục chứa các fold; mặc định là thư mục cha của --export")
    ap.add_argument("--copy", action="store_true", help="chép ảnh thay vì hardlink (tốn đĩa)")
    ap.add_argument("--labels", default="auto", choices=foldmod.LABEL_MODES,
                    help="nhãn YOLO: auto = chép nếu bản xuất có labels/, không thì sinh từ COCO")
    a = ap.parse_args(argv)

    doc = foldmod.load_folds(a.folds)
    recipe = valsplit.load_recipe(a.val)
    graph = valsplit.load_graph(recipe)
    export = Path(a.export)
    out_root = Path(a.out_root) if a.out_root else export.parent
    order = sorted(doc["folds"])
    names = order if a.all else [a.fold]

    print(f"cách chia val: {recipe['method']} ({recipe['_path']})")
    print(f"bảng cạnh: {graph.n_edges} cạnh >= {graph.threshold:.0%}, scope={graph.scope}")
    if graph.scope == "none":
        print("  CHÚ Ý: không có bảng cạnh, mọi con số rò rỉ dưới đây là KHÔNG BIẾT, không phải 0")
    print()

    for name in names:
        fields = foldmod.fold_fields(doc, name)
        frames, regions = pool_frames(export, fields["train"])
        assign = valsplit.apply(recipe, frames, graph,
                                slot=order.index(name), n_slots=len(order))
        report = valsplit.audit(assign, frames, graph, regions)

        out = out_root / name
        s = foldmod.make_fold(export, name, fields, out,
                              copy=a.copy, labels=a.labels,
                              val_images=assign.val, drop_images=assign.drop,
                              val_split={**assign.why, "audit": report})
        print(f"{name} -> {out}")
        for sp in foldmod.SPLITS:
            r = s["splits"][sp]
            print("  %-5s %-34s %4d ảnh  %5d vùng  (ảnh nền: %d)"
                  % (sp, ",".join(r["fields"]), r["images"], r["annotations"],
                     r["empty_images"]))
        leak = report["leak"]
        print(f"  đệm   bỏ {s['dropped']['count']} ảnh; "
              f"val còn dính train: {leak['val_images_touching_train']} ảnh "
              f"({leak['percent_of_val']}%)")
        if s["images_copied"]:
            print(f"  ảnh chép thay vì hardlink: {s['images_copied']}")
        if s["images_skipped"]:
            print(f"  bỏ qua {len(s['images_skipped'])} ảnh không thuộc ruộng nào trong folds.yaml")
        if s["labels"] == "generate":
            gen = s.get("labels_generated") or {}
            n = sum(v["labels"] for v in gen.values())
            print(f"  nhãn YOLO: sinh từ COCO ({n} vùng) vì bản xuất không có labels/")
        else:
            print("  nhãn YOLO: chép từ bản xuất")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
