# scripts/05_viz_flower_hsv.py
"""Truc quan hoa module dem hoa HSV (GIONG HET app/server.py) de KIEM CHUNG
no dem nham la / nhanh bi choi sang thanh "hoa" NGAY BEN TRONG tan cay.

Dung DUNG logic server.py: voi MOI polygon tan, dem pixel hoa
    blur 3x3 -> HSV -> (S < sat_max) & (V > val_min) -> opening
roi CHI lay pixel NAM TRONG polygon; ratio = pixel_hoa / dien_tich_polygon,
va phan muc 0..3 theo nguong (<2%%, <10%%, <30%%, >=30%%) -- y het server.py.

=> BAT BUOC anh phai da co file nhan (polygon). Anh chua co nhan se bi bo qua.

Moi anh -> 1 THU MUC rieng (ten = ten anh) chua 3 file tach roi:
    1_goc.jpg     : anh goc
    2_overlay.jpg : goc + to HONG pixel bi coi la "hoa" (chi trong tan) + vien
                    mau & "ratio%% Lx" tung tan -> thay ca vet choi la/nhanh
    3_mask.jpg    : chi pixel-hoa trong tan tren nen den

Vi du:
    python scripts/05_viz_flower_hsv.py --images data/iachim_dataset_export/data_compressed/field_1 --labels data/masks/corrected
    python scripts/05_viz_flower_hsv.py --images anh.jpg --labels data/masks/corrected --sat-max 50 --val-min 180
    python scripts/05_viz_flower_hsv.py --images data\iachim_dataset_export\data_compressed\field_1\10\1\DJI_20260301075335_0002_D.jpg --labels data/masks/corrected/field_1__10__1__DJI_20260301075335_0002_D.jpg.json --sat-max 50 --val-min 180
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# --- doc/ghi anh chiu duoc duong dan Unicode (Windows tieng Viet) ---
def imread_u(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# --- loi HSV (GIONG HET app/server.py) ---
def flower_mask(img_bgr, sat_max, val_min):
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flower = ((hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flower = cv2.morphologyEx(flower, cv2.MORPH_OPEN, kernel)
    return flower


# --- phan muc mat do (GIONG HET app/server.py) ---
FLOWER_THRESHOLDS = (0.02, 0.10, 0.30)
FLOWER_NAMES = ("no_flower", "few_flowers", "many_flowers", "very_many_flowers")


def flower_label(ratio):
    t1, t2, t3 = FLOWER_THRESHOLDS
    if ratio < t1:
        return 0
    if ratio < t2:
        return 1
    if ratio < t3:
        return 2
    return 3


# --- doc polygon tan tu file nhan (dinh dang moi hoac COCO cu) ---
def load_polys(jp):
    if not jp:
        return []
    try:
        data = json.load(open(jp, encoding="utf-8"))
    except Exception:
        return []
    polys = []
    if isinstance(data.get("polygons"), list):
        for p in data["polygons"]:
            arr = np.array(p.get("points") or [], dtype=np.float64).reshape(-1, 2)
            if len(arr) >= 3:
                polys.append(arr)
    else:
        for a in data.get("annotations", []):
            seg = a.get("segmentation") or []
            if seg:
                arr = np.array(seg[0], dtype=np.float64).reshape(-1, 2)
                if len(arr) >= 3:
                    polys.append(arr)
    return polys


def find_label(labels_dir, img_name):
    # Ten nhan = duong dan BASE-relative, '/' -> '__', + .json (giong server.py).
    if not labels_dir:
        return None
    # cho phep tro THANG toi 1 file .json (khong chi thu muc)
    if os.path.isfile(labels_dir):
        return labels_dir if labels_dir.lower().endswith(".json") else None
    base = os.path.basename(img_name)
    stem = base.rsplit(".", 1)[0]
    for jp in glob.glob(os.path.join(labels_dir, "*.json")):
        n = os.path.basename(jp)
        if n.endswith(base + ".json") or n == stem + ".json":
            return jp
    return None


# --- tao 3 anh RIENG, tinh THEO TUNG POLYGON (giong server.py) ---
# mau vien theo muc: 0 xam, 1 xanh la, 2 cam, 3 do (BGR).
LEVEL_COLORS = ((150, 150, 150), (0, 200, 60), (0, 170, 255), (0, 0, 235))


def poly_flower(img_bgr, arr, sat_max, val_min):
    # Theo README muc 6 cua bo du lieu (va app/server.py): TRICH XUAT vung
    # polygon roi moi dem -> blur/HSV/opening chay TRONG tung tan, khong phai
    # tren ca anh. Tra ve (cnt, area, y0, x0, flower_crop) - flower_crop chi
    # dung de ve overlay dung bang pixel da dem.
    h, w = img_bgr.shape[:2]
    pts = np.round(arr).astype(np.int32)
    # kep bbox vao trong anh: polygon ve tay co the vuot ra ngoai khung
    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0, None
    pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(pm, [pts - np.array([x0, y0], np.int32)], 1)
    area = int(pm.sum())
    if area == 0:
        return 0, 0, 0, 0, None
    crop = img_bgr[y0:y1, x0:x1]
    region = cv2.bitwise_and(crop, crop, mask=pm)   # chi giu pixel trong tan
    f = flower_mask(region, sat_max, val_min) & pm
    return int(f.sum()), area, y0, x0, f


def render_views(img_bgr, sat_max, val_min, polys):
    """Tra ve 3 anh RIENG (full res, khong ghep) + list ratio tung tan."""
    h, w = img_bgr.shape[:2]
    # Dem RIENG trong tung tan (README muc 6), roi gop lai chi de HIEN THI.
    # Ratio van lay tu tung tan, khong lay tu mask gop -> tan chong nhau khong
    # muon pixel cua nhau.
    in_flower = np.zeros((h, w), bool)
    ratios = []
    for arr in polys:
        cnt, area, y0, x0, f = poly_flower(img_bgr, arr, sat_max, val_min)
        ratios.append(cnt / area if area else 0.0)
        if f is not None:
            sub = in_flower[y0:y0 + f.shape[0], x0:x0 + f.shape[1]]
            sub |= f > 0

    a = 0.55
    PINK = np.array([180, 60, 255], np.float64)
    overlay = img_bgr.copy()
    overlay[in_flower] = (a * PINK + (1 - a) * overlay[in_flower]).astype(np.uint8)

    mask_vis = np.zeros_like(img_bgr)
    mask_vis[in_flower] = (255, 255, 255)

    for arr, ratio in zip(polys, ratios):
        lvl = flower_label(ratio)
        pts = np.round(arr).astype(np.int32)
        cv2.polylines(overlay, [pts], True, LEVEL_COLORS[lvl], 2)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(overlay, "%.0f%% L%d" % (ratio * 100, lvl), (cx - 34, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return {"1_goc": img_bgr, "2_overlay": overlay, "3_mask": mask_vis}, ratios


def iter_images(src):
    p = Path(src)
    if p.is_file():
        return [str(p)]
    return [str(q) for q in sorted(p.rglob("*")) if q.suffix.lower() in EXTS]


def main():
    ap = argparse.ArgumentParser(description="Truc quan hoa module dem hoa HSV")
    ap.add_argument("--images", required=True, help="anh hoac thu muc anh")
    ap.add_argument("--labels", required=True,
                    help="thu muc file nhan .json (BAT BUOC - anh phai co polygon)")
    ap.add_argument("--out", default="data/viz_flower", help="thu muc xuat")
    ap.add_argument("--sat-max", type=int, default=50)
    ap.add_argument("--val-min", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0, help="gioi han so anh (0=het)")
    args = ap.parse_args()

    files = iter_images(args.images)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("Khong tim thay anh trong:", args.images)
        return
    os.makedirs(args.out, exist_ok=True)
    print("Xu ly %d anh | S<%d & V>%d | nhan=%s"
          % (len(files), args.sat_max, args.val_min, args.labels))
    done = skip = 0
    for f in files:
        polys = load_polys(find_label(args.labels, f))
        if not polys:
            skip += 1
            continue
        img = imread_u(f)
        if img is None:
            print("  bo qua (khong doc duoc):", f)
            skip += 1
            continue
        views, ratios = render_views(img, args.sat_max, args.val_min, polys)
        stem = os.path.basename(f).rsplit(".", 1)[0]
        stem += '_HSV'
        d = os.path.join(args.out, stem)
        os.makedirs(d, exist_ok=True)
        for name, im in views.items():
            imwrite_u(os.path.join(d, name + ".jpg"), im)
        done += 1
        avg = 100.0 * sum(ratios) / len(ratios) if ratios else 0.0
        print("  [%d] %s -> %d tan, ratio TB %.1f%% -> %s"
              % (done, os.path.basename(f), len(ratios), avg, d))
    print("Xong: %d anh co nhan da ve, %d anh bo qua (chua co polygon)."
          % (done, skip))
    print("Anh o:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()