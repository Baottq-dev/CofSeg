"""
07_data_inventory.py

Kiem ke + do chat luong ANH cho toan bo bo du lieu UAV, chay RIENG, KHONG dung
vao app. Dung de nam nhanh dac diem tung LAN BAY (moi folder anh = 1 lan bay
tren 1 ruong) truoc khi quyet dinh gan nhan / chia train-val-test.

Voi moi folder anh (la duyet de quy), in:
  - so anh, dung luong trung vi, kich thuoc
  - do sang V trung vi, ti le pixel bong toi (V < 60)
  - do net trung vi (phuong sai Laplacian) + liet ke anh MO (< 40% trung vi)
  - so thu tu DJI bi thieu (khung bi xoa/mat) + nhip chup (giay giua 2 khung)
  - PROXY NO HOA: ti le pixel "trang-sang" tren TOAN anh (S < sat_max & V > val_min)

  !! PROXY hoa CHi de dinh huong, KHONG phai % hoa that:
     (1) Dem nham ca vat trang khac hoa (mai ton, bao tai, da sang, dat chay nang)
         -> tri tuyet doi khong dang tin, chi co XU HUONG la tin hieu.
     (2) Lan bay TOI bi dem hut: nguong V > val_min loc ra it pixel hon khi anh
         toi, nen lan bay chieu/rop tan co the thap gia tao.
     Muon con so that thi phai gan nhan roi lay flower_ratio trong tung tan
     (xem app/server.py::_flower_counts, hoac scripts/05_viz_flower_hsv.py).

Vi du:
  python scripts/07_data_inventory.py
  python scripts/07_data_inventory.py --base data/iachim_dataset_export/data_compressed
  python scripts/07_data_inventory.py --sat-max 50 --val-min 180 --width 640

Yeu cau: opencv-python, numpy (khong can them gi).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import re

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------- IO Unicode
def imread_u(path: str, flags=cv2.IMREAD_COLOR):
    # cv2.imread tra None voi duong dan co dau (Windows tieng Viet).
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# ---------------------------------------------------------------- tien ich
def dji_meta(name: str):
    # Ten DJI: DJI_<14 so thoi gian>_<4 so thu tu>_D.jpg -> (datetime, seq).
    m = re.search(r"DJI_(\d{14})_(\d{4})_", name)
    if not m:
        return None, None
    return (datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S"),
            int(m.group(2)))


def med(a):
    return sorted(a)[len(a) // 2] if a else 0.0


def quart(a, p):
    return sorted(a)[int(p * (len(a) - 1))] if a else 0.0


def leaf_dirs(base: str):
    # Cac thu muc LA (chua truc tiep >=1 anh), sap theo duong dan.
    out = []
    for d, _, files in os.walk(base):
        if any(os.path.splitext(f)[1].lower() in EXTS for f in files):
            out.append(d)
    return sorted(out)


# ---------------------------------------------------------------- do 1 anh
def measure(path: str, width: int, sat_max: int, val_min: int):
    im = imread_u(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    small = cv2.resize(im, (width, max(1, int(width * h / w))),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return dict(
        w=w, h=h, mb=os.path.getsize(path) / 1048576.0,
        v_mean=float(V.mean()),
        shadow=float((V < 60).mean()),
        white=float(((S < sat_max) & (V > val_min)).mean()),
        sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Kiem ke + chat luong anh UAV")
    ap.add_argument("--base",
                    default="data/iachim_dataset_export/data_compressed",
                    help="thu muc goc chua cac field")
    ap.add_argument("--sat-max", type=int, default=50,
                    help="proxy hoa: pixel co S < sat_max (giong server.py)")
    ap.add_argument("--val-min", type=int, default=180,
                    help="proxy hoa: pixel co V > val_min (giong server.py)")
    ap.add_argument("--width", type=int, default=640,
                    help="be rong thu nho khi do (px)")
    args = ap.parse_args()

    if not os.path.isdir(args.base):
        raise SystemExit("Khong thay thu muc: " + args.base)

    dirs = leaf_dirs(args.base)
    if not dirs:
        raise SystemExit("Khong co anh nao trong: " + args.base)

    print("PROXY hoa = %% pixel (S < %d & V > %d) tren toan anh — chi de dinh huong,"
          " KHONG phai %% hoa that (xem docstring)." % (args.sat_max, args.val_min))
    print()
    print("%-26s | %4s | %5s | %-9s | %4s | %6s | %6s | %6s"
          % ("folder (lan bay)", "anh", "MB", "kich thuoc",
             "V tb", "bong%", "net tv", "hoa%tv"))
    print("-" * 96)

    sizes = set()
    blurry = []
    all_ms = {}
    for d in dirs:
        files = sorted(p for p in glob.glob(os.path.join(d, "*"))
                       if os.path.splitext(p)[1].lower() in EXTS)
        ms = []
        for p in files:
            r = measure(p, args.width, args.sat_max, args.val_min)
            if r is None:
                print("  LOI DOC:", p)
                continue
            r["path"] = p
            ms.append(r)
            sizes.add((r["w"], r["h"]))
        if not ms:
            continue
        all_ms[d] = ms
        sharp_med = med([x["sharp"] for x in ms])
        for x in ms:
            if x["sharp"] < 0.4 * sharp_med:
                blurry.append((d, os.path.basename(x["path"]),
                               x["sharp"], sharp_med))
        rel = os.path.relpath(d, args.base).replace("\\", "/")
        sz = "x".join(map(str, sorted(sizes)[-1]))
        print("%-26s | %4d | %5.1f | %-9s | %4.0f | %5.1f%% | %6.0f | %5.2f%%"
              % (rel, len(ms), med([x["mb"] for x in ms]), sz,
                 med([x["v_mean"] for x in ms]),
                 100 * med([x["shadow"] for x in ms]),
                 sharp_med, 100 * med([x["white"] for x in ms])))

    print("\nkich thuoc anh gap:", sorted(sizes))

    print("\n=== SO THU TU DJI THIEU (khung bi xoa/mat) ===")
    for d in dirs:
        ms = all_ms.get(d)
        if not ms:
            continue
        seqs = sorted(s for s in (dji_meta(os.path.basename(x["path"]))[1]
                                  for x in ms) if s is not None)
        if not seqs:
            continue
        gaps = ["%d..%d" % (a + 1, b - 1)
                for a, b in zip(seqs, seqs[1:]) if b - a > 1]
        rel = os.path.relpath(d, args.base).replace("\\", "/")
        print("  %-26s seq %04d-%04d  thieu %d khung%s"
              % (rel, seqs[0], seqs[-1], seqs[-1] - seqs[0] + 1 - len(seqs),
                 ("  (" + ", ".join(gaps[:6]) + ")") if gaps else ""))

    print("\n=== NHIP CHUP (giay giua 2 khung lien tiep) ===")
    for d in dirs:
        ms = all_ms.get(d)
        if not ms:
            continue
        ts = sorted(t for t in (dji_meta(os.path.basename(x["path"]))[0]
                                for x in ms) if t is not None)
        deltas = sorted((b - a).total_seconds() for a, b in zip(ts, ts[1:]))
        rel = os.path.relpath(d, args.base).replace("\\", "/")
        if deltas:
            print("  %-26s min %2.0fs  trung vi %2.0fs  max %3.0fs"
                  % (rel, deltas[0], med(deltas), deltas[-1]))

    print("\n=== ANH MO (do net < 40%% trung vi folder) ===")
    if not blurry:
        print("  (khong co)")
    for d, nm, s, m in blurry:
        rel = os.path.relpath(d, args.base).replace("\\", "/")
        print("  %-26s %-42s %6.0f (tv %5.0f)" % (rel, nm, s, m))


if __name__ == "__main__":
    main()
