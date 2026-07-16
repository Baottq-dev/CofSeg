"""
06_superpixel_flower.py

Prototype: phat hien hoa bang SUPERPIXEL (SLIC) + dac trung vung, chay RIENG,
KHONG dung vao app. Dung de so sanh TRUC TIEP voi module HSV hien tai (server.py).

Y tuong (khong can nhan):
  1. HSV baseline: giong het server.py (S < sat_max & V > val_min), tinh trong
     tung polygon tan.
  2. SLIC: voi TUNG polygon tan, chay SLIC RIENG trong vung polygon do (crop
     bbox + mask), tinh dac trung VUNG cho tung superpixel (S, V, ExG, texture,
     ti le specular) roi quyet dinh hoa/khong hoa bang heuristic -> chong duoc
     dom choi le te va la non bac. Ratio = pixel hoa trong polygon / dien tich
     polygon (theo TUNG tan, y het server.py).
  3. Xuat anh so sanh + in ti le/muc 0-3 cua ca 2 phuong phap cho tung tan.

Xuat: data/viz_superpixel/<cac_folder_cha__ten_anh>/  (giong bo cuc 05: 3 anh)
  Ten folder ket qua gom ca thu muc cha (mac dinh 3 cap), vd:
  field_1__10__1__DJI_20260301075335_0002_D  -> tranh trung ten giua cac field/do cao.
  1_goc.jpg      anh goc
  2_overlay.jpg  overlay hoa (SLIC) mau XANH LA + text so sanh SP vs HSV
  3_mask.jpg     mask nhi phan hoa (SLIC), trang tren nen den

Vi du (nho du subfolder do cao/goc, vd 10/1):
  python scripts/06_superpixel_flower.py \
      --images data/iachim_dataset_export/data_compressed/field_1/10/1 \
      --labels data/masks/corrected --limit 10

Yeu cau: pip install scikit-image   (ngoai opencv-python, numpy)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.segmentation import slic
except ImportError:
    raise SystemExit("Can cai scikit-image: pip install scikit-image")

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# --- giong server.py ---
FLOWER_THRESHOLDS = (0.02, 0.10, 0.30)
FLOWER_NAMES = ("no_flower", "few_flowers", "many_flowers", "very_many_flowers")


def flower_label(ratio: float) -> int:
    t1, t2, t3 = FLOWER_THRESHOLDS
    if ratio < t1:
        return 0
    if ratio < t2:
        return 1
    if ratio < t3:
        return 2
    return 3


# ---------------------------------------------------------------- IO Unicode
def imread_u(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path: str, img) -> None:
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


# ---------------------------------------------------------------- helpers
def flower_mask_hsv(bgr, sat_max=50, val_min=180):
    """Y HET server.py: nguong HSV + blur 3x3 + MORPH_OPEN ellipse 3x3."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < sat_max) & (v > val_min)).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 127).astype(np.uint8)


def compute_exg(bgr):
    """Excess Green chuan hoa: cao = xanh la, ~0/am = trang/hoa."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    tot = b + g + r + 1e-6
    rn, gn, bn = r / tot, g / tot, b / tot
    return 2.0 * gn - rn - bn


def local_std(gray, k=5):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (k, k))
    mean_sq = cv2.boxFilter(g * g, -1, (k, k))
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(var)


def load_polys(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polys = []
    if isinstance(data, dict) and data.get("polygons"):
        for p in data["polygons"]:
            pts = p.get("points") or []
            if len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.int32))
    elif isinstance(data, dict) and data.get("annotations"):
        for a in data["annotations"]:
            seg = a.get("segmentation") or []
            if seg and isinstance(seg[0], list) and len(seg[0]) >= 6:
                arr = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
                polys.append(arr.astype(np.int32))
    return polys


def poly_mask(shape, poly):
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m


def ratio_of(mask, pm):
    area = int(pm.sum())
    if area == 0:
        return 0.0, 0
    inter = int(((mask > 0) & (pm > 0)).sum())
    return inter / area, area


def iter_images(root):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTS:
            yield p


def find_label(labels, stem):
    p = Path(labels)
    if p.is_file() and p.suffix.lower() == ".json":
        return str(p)
    for j in Path(labels).rglob("*.json"):
        if stem in j.name:
            return str(j)
    return None


# ---------------------------------------------------------------- SLIC flower
def flower_mask_slic(bgr, polys, args):
    """Chay SLIC RIENG cho TUNG polygon tan (crop bbox + mask) roi quyet dinh
    hoa/khong o cap SUPERPIXEL. Tra ve mask nhi phan hoa (chi trong cac tan)."""
    h_img, w_img = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    exg = compute_exg(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tex = local_std(gray, 5)
    spec = ((V > args.spec_val) & (S < args.spec_sat)).astype(np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    out = np.zeros((h_img, w_img), dtype=np.uint8)
    for poly in polys:
        pm = poly_mask((h_img, w_img), poly).astype(bool)
        area = int(pm.sum())
        if area < args.min_area:
            continue
        ys, xs = np.where(pm)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_pm = pm[y0:y1, x0:x1]
        sub_rgb = rgb[y0:y1, x0:x1]
        sub_s = S[y0:y1, x0:x1]
        sub_v = V[y0:y1, x0:x1]
        sub_exg = exg[y0:y1, x0:x1]
        sub_tex = tex[y0:y1, x0:x1]
        sub_spec = spec[y0:y1, x0:x1]

        # so superpixel ti le theo dien tich tan (moi superpixel ~ sp_size px)
        n_seg = max(20, area // max(50, args.sp_size))
        try:
            segments = slic(sub_rgb, n_segments=n_seg,
                            compactness=args.compactness,
                            mask=sub_pm, start_label=1)
        except TypeError:
            # skimage cu khong ho tro tham so mask
            segments = slic(sub_rgb, n_segments=n_seg,
                            compactness=args.compactness, start_label=1)

        sub_out = np.zeros(sub_pm.shape, dtype=np.uint8)
        for lbl in np.unique(segments):
            if lbl == 0:
                continue
            sp = (segments == lbl) & sub_pm
            n_in = int(sp.sum())
            if n_in < args.min_area:
                continue
            m_s = float(sub_s[sp].mean())
            m_v = float(sub_v[sp].mean())
            m_exg = float(sub_exg[sp].mean())
            m_tex = float(sub_tex[sp].mean())
            spec_frac = float(sub_spec[sp].mean())
            base = (m_s < args.sat_max) and (m_v > args.val_min)
            not_leaf = m_exg < args.exg_max
            # glare: nhieu pixel chay sang VA vung rat min/phang
            is_glare = (spec_frac > args.spec_frac) and (m_tex < args.tex_min)
            if base and not_leaf and not is_glare:
                sub_out[sp] = 1
        region = out[y0:y1, x0:x1]
        region[sub_out > 0] = 1
    return out


# ---------------------------------------------------------------- ve overlay
def tint(bgr, mask, color, alpha=0.5):
    out = bgr.copy()
    idx = mask > 0
    if idx.any():
        overlay = out.copy()
        overlay[idx] = color
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    return out


def draw_polys(img, polys, color=(0, 255, 255)):
    for poly in polys:
        cv2.polylines(img, [poly], True, color, 2)
    return img


def put_text(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# (da bo make_diff: khong con xuat anh diff/HSV rieng, chi 3 anh nhu 05)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="file anh hoac thu muc anh")
    ap.add_argument("--labels", required=True, help="file .json hoac thu muc chua .json")
    ap.add_argument("--out", default="data/viz_superpixel")
    ap.add_argument("--name-parents", type=int, default=3,
                    help="so thu muc cha gop vao ten folder ket qua (0 = chi ten anh)")
    # HSV baseline (giong server.py)
    ap.add_argument("--sat-max", type=int, default=50)
    ap.add_argument("--val-min", type=int, default=180)
    # SLIC
    ap.add_argument("--sp-size", type=int, default=400,
                    help="dien tich muc tieu moi superpixel (px), ~20x20")
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--min-area", type=int, default=80)
    # loc glare / la
    ap.add_argument("--exg-max", type=float, default=0.05)
    ap.add_argument("--tex-min", type=float, default=6.0)
    ap.add_argument("--spec-val", type=int, default=245)
    ap.add_argument("--spec-sat", type=int, default=25)
    ap.add_argument("--spec-frac", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    imgs = list(iter_images(args.images))
    if args.limit > 0:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"Khong tim thay anh trong: {args.images}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skip = 0
    for img_path in imgs:
        stem = img_path.stem
        label_path = find_label(args.labels, stem)
        if not label_path:
            print(f"[BO QUA] khong co nhan: {img_path.name}")
            n_skip += 1
            continue
        bgr = imread_u(str(img_path))
        if bgr is None:
            print(f"[BO QUA] khong doc duoc anh: {img_path}")
            n_skip += 1
            continue
        polys = load_polys(label_path)
        if not polys:
            print(f"[BO QUA] nhan khong co polygon: {img_path.name}")
            n_skip += 1
            continue

        canopy = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for poly in polys:
            cv2.fillPoly(canopy, [poly], 1)

        # HSV chi de IN so sanh ra console (khong xuat anh HSV rieng nua)
        hsv_m = flower_mask_hsv(bgr, args.sat_max, args.val_min) & canopy
        # SLIC tinh RIENG cho TUNG polygon tan
        slic_m = flower_mask_slic(bgr, polys, args)

        # --- xuat 3 anh: goc - overlay - mask (giong bo cuc 05) ---
        # ten folder gom ca thu muc cha (vd field_1__10__1__DJI_...) de khong trung
        parts = img_path.parts
        n_par = max(0, args.name_parents)
        parents = list(parts[-(n_par + 1):-1]) if n_par > 0 else []
        folder_name = "__".join(parents + [stem])
        folder = out_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        imwrite_u(str(folder / "1_goc.jpg"), bgr)

        overlay = tint(bgr, slic_m, (0, 255, 0))  # hoa (SLIC) = XANH LA
        draw_polys(overlay, polys)

        print(f"== {img_path.name} ({len(polys)} tan) ==")
        for i, poly in enumerate(polys):
            pm = poly_mask(bgr.shape, poly)
            hr, _ = ratio_of(hsv_m, pm)
            sr, _ = ratio_of(slic_m, pm)
            cx, cy = poly.mean(axis=0).astype(int)
            txt = (f"SP {sr * 100:.0f}% L{flower_label(sr)}"
                   f" (HSV {hr * 100:.0f}% L{flower_label(hr)})")
            put_text(overlay, txt, (int(cx) - 70, int(cy)))
            print(f"  tan {i:>2}: HSV {hr * 100:.1f}% L{flower_label(hr)}"
                  f" | SP {sr * 100:.1f}% L{flower_label(sr)}")
        imwrite_u(str(folder / "2_overlay.jpg"), overlay)

        imwrite_u(str(folder / "3_mask.jpg"), (slic_m * 255).astype(np.uint8))

        n_done += 1

    print(f"\nXong: {n_done} anh, bo qua {n_skip}. Ket qua o: {out_root}")


if __name__ == "__main__":
    main()