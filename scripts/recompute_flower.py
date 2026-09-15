"""Thêm khối `hsv` và `otsu` vào các file nhãn đã có trong data/masks/corrected.

    python scripts/recompute_flower.py --dry-run --limit 5
    python scripts/recompute_flower.py
    python scripts/recompute_flower.py --field field_3 --channel min2 --class-s-max 45

Việc làm với mỗi file định dạng mới (có `polygons`):

- Đọc ảnh theo `path` trong file. Không đọc được → bỏ qua file, ghi vào danh sách.
- Với từng polygon tính CẢ HAI phương pháp bằng app/flower.py — đúng thuật toán
  server đang dùng — và ghi vào `hsv` / `otsu` của polygon. Tham số HSV lấy từ
  chính file (`hsv.sat_max/val_min`, mặc định 50/180); tham số Otsu từ dòng lệnh.
- KHÔNG đổi các trường đầu đã có (flower_pixels, flower_ratio, flower_label,
  label_source…): đó là nhãn đã gán, giữ nguyên. Chỉ thêm `label_method = "hsv"`
  cho polygon chưa có (kể cả file trước 17/07 thiếu `label_source` — khi đó mọi
  mức đều do HSV gán; nhãn tay → None), `flower_method = "hsv"` cho file chưa
  có, và khối tham số `otsu` ở mức tài liệu.
- Ghi tạm rồi thay thế; file gốc được sao lưu vào data/masks/backup/<mốc>/
  trước khi ghi (tắt bằng --no-backup).

Lưu ý: hầu hết file được lưu trước 23/07/2026, khi HSV còn chạy trên cả ảnh
rồi mới đếm trong polygon; code hiện tại cắt polygon trước. Vì thế khối `hsv`
mới có thể lệch vài pixel so với `flower_pixels` cũ — script đếm và báo, không
sửa số cũ.

Cuối cùng in: số file/polygon, số lệch HSV, ma trận nhãn HSV × nhãn Otsu, và
phân vị độ tách η theo nhãn HSV — dữ liệu để chọn `sep_min` về sau.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import flower as fl  # noqa: E402


def imread(path: Path):
    # Chịu được đường dẫn có dấu (cv2.imread trả None) — như app/server._imread.
    try:
        buf = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    ap.add_argument("--labels", default=os.path.join(cfg["data"]["masks_dir"], "corrected"))
    ap.add_argument("--images", default=cfg["data"]["images_dir"])
    ap.add_argument("--field", default=None, help="chỉ làm file của một ruộng, vd field_3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="tính và in tổng kết, không ghi")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--channel", choices=fl.CHANNELS, default="white")
    ap.add_argument("--s-max", type=int, default=50, help="kênh white: cổng màu S < s_max")
    ap.add_argument("--class-s-max", type=float, default=0.0, help="cổng S trung bình của lớp chọn, 0 = tắt")
    ap.add_argument("--sep-min", type=float, default=0.0)
    ap.add_argument("--area-min", type=int, default=0)
    ap.add_argument("--area-max", type=int, default=0)
    ap.add_argument("--elong-max", type=float, default=0.0)
    ap.add_argument("--clip-rule", action="store_true")
    a = ap.parse_args()

    otsu_kw = dict(channel=a.channel, s_max=a.s_max, class_s_max=a.class_s_max, sep_min=a.sep_min,
                   area_min=a.area_min, area_max=a.area_max, elong_max=a.elong_max, clip_rule=a.clip_rule)
    labels = Path(a.labels)
    files = sorted(labels.glob("*.json"))
    if a.field:
        files = [f for f in files if f.name.startswith(a.field + "__")]
    if a.limit:
        files = files[: a.limit]
    backup = None
    if not a.dry_run and not a.no_backup:
        backup = labels.parent / "backup" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} file trong {labels}; Otsu {otsu_kw}; "
          f"{'CHỈ XEM, không ghi' if a.dry_run else 'ghi, sao lưu vào ' + str(backup) if backup else 'ghi, KHÔNG sao lưu'}")

    n_files = n_poly = n_mismatch = n_skipped_old = 0
    unreadable: list[str] = []
    confusion = Counter()              # (nhãn HSV cũ, nhãn Otsu) -> số polygon
    eta_by_hsv = defaultdict(list)     # nhãn HSV cũ -> [η]
    rejected = Counter()
    t0 = time.time()
    for k, f in enumerate(files, 1):
        doc = json.load(open(f, encoding="utf-8"))
        if not isinstance(doc.get("polygons"), list):
            n_skipped_old += 1           # định dạng COCO cũ, không có trường hoa
            continue
        img = imread(Path(a.images) / doc.get("path", ""))
        if img is None:
            unreadable.append(f.name)
            continue
        hp = doc.get("hsv") if isinstance(doc.get("hsv"), dict) else {}
        sat_max, val_min = int(hp.get("sat_max", 50)), int(hp.get("val_min", 180))
        for p in doc["polygons"]:
            poly = [float(v) for xy in (p.get("points") or []) for v in xy]
            if len(poly) < 6:
                continue
            fpx, tpx = fl.hsv_counts(img, poly, sat_max, val_min)
            r = round(fpx / tpx, 4) if tpx else 0.0
            p["hsv"] = dict(flower_pixels=int(fpx), total_pixels=int(tpx), ratio=r, label=fl.level(r)[0])
            o = fl.otsu_blob(img, poly, **otsu_kw)
            if o is None:
                o = dict(flower_pixels=0, ratio=0.0, label=0, channel=a.channel, threshold=None,
                         threshold1=None, separability=0.0, class_s=None, n_blobs=0, n_blobs_kept=0,
                         rejected="empty")
            p["otsu"] = dict(flower_pixels=o["flower_pixels"], total_pixels=int(tpx), ratio=o["ratio"],
                             label=o["label"], channel=o["channel"], threshold=o["threshold"],
                             threshold1=o["threshold1"], separability=o["separability"],
                             class_s=o["class_s"], n_blobs=o["n_blobs"], n_blobs_kept=o["n_blobs_kept"],
                             rejected=o["rejected"])
            if "label_method" not in p:
                # Trước 17/07/2026 chưa có cờ tay/máy (label_source thiếu): mọi mức
                # đều do HSV gán. Chỉ nhãn tay mới không có phương pháp máy.
                p["label_method"] = None if p.get("label_source") == "manual" else "hsv"
            n_poly += 1
            if fpx != int(p.get("flower_pixels") or 0):
                n_mismatch += 1
            old_lvl = p.get("flower_label")
            if isinstance(old_lvl, int):
                confusion[(old_lvl, o["label"])] += 1
                eta_by_hsv[old_lvl].append(o["separability"])
            if o["rejected"]:
                rejected[o["rejected"]] += 1
        doc.setdefault("flower_method", "hsv")
        doc["otsu"] = dict(**otsu_kw, thresholds=list(fl.THRESHOLDS))
        n_files += 1
        if not a.dry_run:
            if backup is not None:
                shutil.copy2(f, backup / f.name)
            tmp = f.with_suffix(".json.tmp")
            json.dump(doc, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, f)
        if k % 50 == 0 or k == len(files):
            print(f"  {k}/{len(files)} file, {n_poly} polygon ({time.time() - t0:.0f}s)")

    print(f"\nĐã xử lý {n_files} file, {n_poly} polygon; bỏ qua {n_skipped_old} file định dạng cũ, "
          f"{len(unreadable)} file không đọc được ảnh")
    for u in unreadable[:10]:
        print("   không đọc được:", u)
    print(f"HSV mới lệch với flower_pixels cũ: {n_mismatch}/{n_poly} polygon "
          f"(thuật toán cũ blur cả ảnh; số cũ giữ nguyên ở trường đầu)")
    if rejected:
        print("Otsu từ chối:", dict(rejected))
    print(f"\nMa trận nhãn HSV cũ (hàng) × nhãn Otsu {a.channel} (cột), số polygon:")
    print(f"{'':>6}" + "".join(f"{'O' + str(j):>8}" for j in range(4)) + f"{'tổng':>8}")
    for i in range(4):
        row = [confusion[(i, j)] for j in range(4)]
        print(f"{'H' + str(i):>6}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    agree = sum(confusion[(i, i)] for i in range(4))
    tot = sum(confusion.values())
    print(f"Trùng nhãn: {agree}/{tot} ({agree / tot:.0%})" if tot else "")
    print("\nPhân vị độ tách η theo nhãn HSV cũ (5/25/50/75/95%):")
    for i in range(4):
        v = eta_by_hsv.get(i)
        if v:
            q = np.percentile(v, [5, 25, 50, 75, 95])
            print(f"  H{i} (n={len(v):>5}): " + "  ".join(f"{x:.2f}" for x in q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
