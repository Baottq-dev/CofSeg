"""Tô hoa bằng model đã train, cho các tán của một vòng hoặc cho mọi tán.

    python scripts/flower_predict.py --run runs/flower/r2 --round r2
    python scripts/flower_predict.py --run runs/flower/r3 --all

Ghi vào thư mục lượt:
- pred/<ảnh>.png   xác suất hoa trong các tán được tô, 0..254 (254 = chắc là
                   hoa); 255 = ngoài các tán đó
- crowns.csv       mỗi tán một dòng: image, crown, field, cover (độ phủ hoa ở
                   ngưỡng 0,5), entropy (trung bình trong tán, bit — càng cao
                   càng kém chắc), tta_disagree (lệch giữa ảnh gốc và hai phép lật)

Vòng duyệt (kind=review) đọc crowns.csv để xếp tán kém chắc lên trước, và lấy
pred/ làm nền máy của mỗi tán.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import flower_data as fd  # noqa: E402
from app import flower_store as fs  # noqa: E402

CSV_COLS = ("image", "crown", "field", "cover", "entropy", "tta_disagree")


def targets(store, labels_dir, rnd=None, everything=False):
    """{ảnh phẳng: [tán, ...]} cần tô."""
    if everything:
        out = {}
        for p in sorted(Path(labels_dir).glob("*.json")):
            flat = p.name[:-5]
            polys, _ = fd.crown_polys(labels_dir, flat)
            if polys:
                out[flat] = list(range(len(polys)))
        return out
    out = {}
    for it in store.load_round(rnd)["items"]:
        out.setdefault(it["image"], []).append(int(it["crown"]))
    return out


def predict_image(model, rgb, polys, crowns, tile, device, tta=True):
    """(ảnh xác suất uint8 cả ảnh, [dòng thống kê theo tán])."""
    from app import flower_model as fm
    h, w = rgb.shape[:2]
    out = np.full((h, w), fs.IGNORE, np.uint8)
    rows = []
    for k in crowns:
        b = fs.crown_box(polys[k], w, h)
        inside = fs.crown_mask(polys[k], b)
        res = fm.predict_tiles(model, rgb[b[1]:b[3], b[0]:b[2]], tile=tile, device=device, tta=tta)
        prob, dis = res if tta else (res, np.zeros_like(res))
        dst = out[b[1]:b[3], b[0]:b[2]]
        dst[inside] = np.round(prob[inside] * 254).astype(np.uint8)
        rows.append(dict(crown=k, cover=float((prob[inside] >= 0.5).mean()) if inside.any() else 0.0,
                         entropy=float(fm.entropy(prob[inside]).mean()) if inside.any() else 0.0,
                         tta_disagree=float(dis[inside].mean()) if inside.any() else 0.0))
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="thư mục lượt train, có model.pt")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--round", help="tô các tán của vòng này")
    g.add_argument("--all", action="store_true", help="tô mọi tán của mọi ảnh có nhãn tán")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="chỉ tô ngần ấy ảnh (thử nhanh)")
    a = ap.parse_args(argv)

    import cv2
    import torch
    from app import flower_model as fm
    images_dir, labels_dir, flower_dir, _ = fd.default_dirs()
    store = fs.FlowerStore(flower_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    model, _ = fm.load_model(Path(a.run) / "model.pt", device)
    todo = targets(store, labels_dir, a.round, a.all)
    if a.limit:
        todo = dict(list(todo.items())[:a.limit])
    pred_dir = Path(a.run) / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (flat, crowns) in enumerate(sorted(todo.items()), 1):
        polys, _ = fd.crown_polys(labels_dir, flat)
        rgb = fd.imread_rgb(Path(images_dir) / fd.rel_of(flat))
        if rgb is None or not polys:
            print("  bỏ qua (không đọc được):", flat)
            continue
        prob, rs = predict_image(model, rgb, polys, [k for k in crowns if k < len(polys)], a.tile, device, not a.no_tta)
        ok, buf = cv2.imencode(".png", prob)
        buf.tofile(str(pred_dir / (flat.rsplit(".", 1)[0] + ".png")))
        for r in rs:
            rows.append(dict(image=flat, field=fd.field_of(flat), **r))
        if i % 20 == 0 or i == len(todo):
            print("  %d/%d ảnh, %d tán" % (i, len(todo), len(rows)))
    with open(Path(a.run) / "crowns.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: (round(r[c], 5) if isinstance(r[c], float) else r[c]) for c in CSV_COLS})
    print("xong: %d tán, ghi %s" % (len(rows), Path(a.run) / "crowns.csv"))
    return rows


if __name__ == "__main__":
    main()
