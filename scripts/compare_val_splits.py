"""So hai cách chia val trên cả sáu lượt, bằng chính mã dùng để cắt fold.

    python scripts/compare_val_splits.py
    python scripts/compare_val_splits.py --val configs/dataset/val_block.yaml
    python scripts/compare_val_splits.py --sweep-buffer
    python scripts/compare_val_splits.py --sweep-slack
    python scripts/compare_val_splits.py --json

Mọi con số trong báo cáo chia dữ liệu phải ra từ đây, không phải từ script
rời của ai đó. `make_fold.py` gọi đúng `valsplit.apply` và `valsplit.audit`
mà script này gọi, nên bảng dưới đây và `fold.json` không thể lệch nhau.

Bốn cột đáng đọc:

    train        ảnh còn lại để học sau khi trừ val và đệm
    val bẩn      ảnh val còn cạnh chồng lấn sang train — phải là 0
    vùng/ảnh     mật độ tán của val; lệch xa cả bộ nghĩa là val không đại diện
    chưa train   ảnh không vào tập train của LƯỢT NÀO trong sáu lượt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.datasets import flightlog, folds as foldmod, valsplit  # noqa: E402

console.setup()

SHIPPED = ["configs/dataset/val_block.yaml", "configs/dataset/val_flight.yaml",
           "configs/dataset/val_field.yaml"]

#: Khoảng cách khung hình ở hai chỗ nối mà val_flight.yaml chạm tới, đọc ra
#: từ scripts/inspect_flights.py: field_2/10/1->10/2 là 2, field_1/10/3->10/4 là 4.
SEAM_GAPS = (2, 4)


def read_export(export: Path) -> tuple[dict[str, str], dict[str, int]]:
    ann = export / "annotations" / "instances.json"
    if not ann.exists():
        raise SystemExit(f"Không thấy {ann}. Cần bản xuất chưa chia (split_by=none).")
    raw = json.loads(ann.read_text(encoding="utf-8"))
    per_id: dict[int, int] = Counter(int(a["image_id"]) for a in raw["annotations"])
    field, regions = {}, {}
    for im in raw["images"]:
        nm = im["file_name"]
        field[nm] = foldmod.field_of(nm)
        regions[nm] = per_id.get(int(im["id"]), 0)
    return field, regions


def evaluate(recipe: dict, doc: dict, field: dict[str, str], regions: dict[str, int],
             graph, *, override: dict | None = None, score_graph=None) -> dict:
    """Chạy một cách chia trên cả sáu lượt, trả về số liệu từng lượt và gộp.

    `score_graph` tách việc CHẤM khỏi việc CHIA. Bản "không đệm" chia bằng đồ
    thị rỗng, nhưng phải được chấm bằng đồ thị thật — chấm bằng đồ thị rỗng
    thì nó báo 0 ảnh bẩn vì không biết gì, và người đọc sẽ kết luận nhầm là
    khoảng đệm chẳng mua được gì.
    """
    recipe = {**recipe, **(override or {})}
    scorer = score_graph if score_graph is not None else graph
    order = sorted(doc["folds"])
    rows, trained = [], set()
    for k, name in enumerate(order):
        fields = foldmod.fold_fields(doc, name)
        keep = set(fields["train"])
        frames = flightlog.frames([n for n in field if field[n] in keep])
        a = valsplit.apply(recipe, frames, graph, slot=k, n_slots=len(order))
        rep = valsplit.audit(a, frames, scorer, regions)
        pool = {f.file_name for f in frames}
        trained |= pool - a.val - a.drop
        rows.append({"fold": name, "test": fields["test"][0], **rep})

    never = set(field) - trained
    n = len(rows)
    return {
        "recipe": recipe.get("_path", recipe["method"]),
        "method": recipe["method"],
        "folds": rows,
        "mean": {
            "train": sum(r["train"]["images"] for r in rows) / n,
            "val": sum(r["val"]["images"] for r in rows) / n,
            "dropped": sum(r["dropped"]["images"] for r in rows) / n,
            "val_regions_per_image": (sum(r["val"]["regions"] for r in rows)
                                      / max(sum(r["val"]["images"] for r in rows), 1)),
            "dirty_val_images": sum(r["leak"]["val_images_touching_train"] for r in rows) / n,
        },
        "never_trained": {"images": len(never),
                          "percent": round(100 * len(never) / len(field), 1),
                          "regions": sum(regions[x] for x in never)},
        "val_fields": dict(sorted(Counter(
            f for r in rows for f, c in r["val_fields"].items() for _ in range(c)).items())),
    }


def print_result(res: dict, field: dict[str, str], regions: dict[str, int]) -> None:
    print(f"\n=== {res['recipe']} ===")
    print(f"  {'lượt':5s} {'test':9s} {'train':>6s} {'val':>5s} {'đệm':>5s} "
          f"{'val bẩn':>8s} {'vùng/ảnh val':>13s}  ruộng trong val")
    for r in res["folds"]:
        comp = " ".join(f"{k.replace('field_', 'f')}:{v}" for k, v in r["val_fields"].items())
        print(f"  {r['fold']:5s} {r['test']:9s} {r['train']['images']:6d} "
              f"{r['val']['images']:5d} {r['dropped']['images']:5d} "
              f"{r['leak']['val_images_touching_train']:8d} "
              f"{r['val']['regions_per_image']:13.1f}  {comp}")
    m = res["mean"]
    print(f"  {'':5s} {'trung bình':9s} {m['train']:6.0f} {m['val']:5.0f} "
          f"{m['dropped']:5.0f} {m['dirty_val_images']:8.1f} "
          f"{m['val_regions_per_image']:13.1f}")
    nt = res["never_trained"]
    dens = sum(regions.values()) / len(field)
    print(f"  ảnh không vào train của lượt nào: {nt['images']} ({nt['percent']}% bộ dữ liệu, "
          f"{nt['regions']} vùng)")
    print(f"  mật độ cả bộ để so: {dens:.1f} vùng/ảnh")
    missing = sorted(set(field.values()) - set(res["val_fields"]))
    if missing:
        print(f"  ruộng KHÔNG BAO GIỜ có mặt trong val: {', '.join(missing)}")


def sweep_buffer(recipe, doc, field, regions, graph, values) -> None:
    print("\n=== đệm bao nhiêu ảnh ở chỗ nối thì đủ (cách flight) ===")
    print(f"  {'đệm':>5s} {'train':>6s} {'bỏ':>5s} {'val bẩn (đo)':>13s} "
          f"{'val bẩn (ước p(k))':>19s}")
    for b in values:
        res = evaluate(recipe, doc, field, regions, graph, override={"buffer": b})
        # Hai chỗ nối thật của bộ này: field_2 chênh 2 khung, field_1 chênh 4.
        est = sum(graph.expected_leak(gap, b) for gap in SEAM_GAPS)
        print(f"  {b:5d} {res['mean']['train']:6.0f} {res['mean']['dropped']:5.0f} "
              f"{res['mean']['dirty_val_images']:13.1f} {est:19.2f}")
    print("  Cột đo chỉ thấy cạnh trong cùng đường bay; chỗ nối hai thư mục KHÔNG có")
    print("  cạnh nào nên cột đó luôn 0 ở đó. Cột ước theo p(k) mới là thứ chọn đệm.")


def sweep_slack(recipe, doc, field, regions, graph, values) -> None:
    print("\n=== cho ranh giới trượt bao nhiêu thì lợi (cách block) ===")
    print(f"  {'slack':>6s} {'train':>6s} {'val':>5s} {'bỏ':>5s} {'val bẩn':>8s}")
    for s in values:
        res = evaluate(recipe, doc, field, regions, graph, override={"slack": s})
        print(f"  {s:6.2f} {res['mean']['train']:6.0f} {res['mean']['val']:5.0f} "
              f"{res['mean']['dropped']:5.0f} {res['mean']['dirty_val_images']:8.1f}")
    print("  slack=0 là cắt cứng ở đúng frac; tăng lên là cho ranh giới tìm chỗ mỏng.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/export/dataset_v1")
    ap.add_argument("--folds", default="configs/dataset/folds.yaml")
    ap.add_argument("--val", action="append", metavar="YAML",
                    help="cách chia cần chấm; lặp lại được. Mặc định: cả hai cách đang dùng")
    ap.add_argument("--sweep-buffer", action="store_true",
                    help="quét kích thước đệm cho cách flight")
    ap.add_argument("--sweep-slack", action="store_true",
                    help="quét độ trượt ranh giới cho cách block")
    ap.add_argument("--no-buffer", action="store_true",
                    help="thêm bản không đệm của mỗi cách, để thấy đệm mua được gì")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    export = Path(a.export)
    field, regions = read_export(export)
    doc = foldmod.load_folds(a.folds)
    paths = a.val or SHIPPED

    results = []
    for p in paths:
        recipe = valsplit.load_recipe(p)
        graph = valsplit.load_graph(recipe)
        results.append((recipe, graph, evaluate(recipe, doc, field, regions, graph)))
        if a.no_buffer:
            off = {"buffer": 0} if recipe["method"] == "flight" else {"edges": None}
            bare = valsplit.load_graph({**recipe, **off}) if "edges" in off else graph
            r = evaluate(recipe, doc, field, regions, bare, override=off,
                         score_graph=graph)
            r["recipe"] = f"{recipe['_path']}  (KHÔNG đệm, để so)"
            results.append((recipe, bare, r))

    if a.json:
        print(json.dumps([r for _, _, r in results], indent=2, ensure_ascii=False))
        return 0

    print(f"{export}: {len(field)} ảnh, {sum(regions.values())} vùng tán, "
          f"{len(doc['folds'])} lượt")
    for _, _, res in results:
        print_result(res, field, regions)

    if len(results) > 1:
        print("\n=== gộp lại ===")
        print(f"  {'cách':44s} {'train':>6s} {'val':>5s} {'bỏ':>5s} "
              f"{'val bẩn':>8s} {'chưa train':>11s}")
        for _, _, r in results:
            print(f"  {Path(r['recipe']).name if '/' in r['recipe'] else r['recipe']:44s} "
                  f"{r['mean']['train']:6.0f} {r['mean']['val']:5.0f} "
                  f"{r['mean']['dropped']:5.0f} {r['mean']['dirty_val_images']:8.1f} "
                  f"{r['never_trained']['percent']:10.1f}%")

    for recipe, graph, _ in results:
        if a.sweep_buffer and recipe["method"] == "flight":
            sweep_buffer(recipe, doc, field, regions, graph, [0, 5, 10, 15, 20, 30])
        if a.sweep_slack and recipe["method"] == "block":
            sweep_slack(recipe, doc, field, regions, graph, [0.0, 0.25, 0.5, 1.0])

    scopes = {g.scope for _, g, _ in results}
    if scopes == {"flight"}:
        print("\nPhạm vi đã đo: chồng lấn chỉ so trong cùng một đường bay. Chồng lấn giữa")
        print("hai đường bay của cùng một ruộng CHƯA đo, nên 'val bẩn = 0' nghĩa là 0")
        print("trong phạm vi đó. field_1 bay hai ngày là chỗ còn để ngỏ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
