# app/flower.py — hai cách đếm pixel hoa trong một tán, dùng chung cho server
# và scripts/recompute_flower.py để không bao giờ có hai bản thuật toán lệch nhau.
#
#   M0  hsv_counts   ngưỡng HSV cố định: S < sat_max và V > val_min (README mục 6
#                    của bộ dữ liệu). Đúng từng dòng bản cũ trong server.py.
#   M1  otsu_blob    Otsu CỦA RIÊNG TỪNG TÁN → mask → blob → luật lọc. Chữa lỗi
#                    ngưỡng tuyệt đối (ảnh sáng/tối khác nhau tự nhận ngưỡng khác
#                    nhau); KHÔNG chữa chói (vẫn chỉ nhìn độ sáng).
#
# Kênh cho Otsu (tham số `channel`) — đo trên 207 tán ngày 15/09/2026:
#   v     Otsu một lần trên V, đúng đề cương §6. HỎNG trên dữ liệu này: histogram
#         V của mọi tán đều hai đỉnh (lá nắng / lá bóng), Otsu cắt ở T≈125–140 và
#         tán không hoa ra 40% "hoa"; độ tách η≈0,74 ở mọi tán nên không chặn được.
#   v2    Otsu hai tầng trên V: lần 1 tách nắng/bóng, lần 2 trên phần sáng.
#         Tán không hoa còn 16%.
#   min2  Otsu hai tầng trên min(R,G,B): pixel trắng có cả ba kênh cao, lá xanh
#         dù nắng vẫn thấp ở kênh xanh dương. Tán không hoa 6%, tán nhiều hoa 13%
#         — biến thể duy nhất cùng bậc với HSV, nên là mặc định.
#
# Cả hai chạy TRONG polygon (cắt bbox + mask), cùng blur 3×3 và opening 3×3, cùng
# ngưỡng mức 0,02 / 0,10 / 0,30 — để hai con số so được với nhau.
import cv2
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
THRESHOLDS = (0.02, 0.10, 0.30)   # tỉ lệ phủ hoa -> mức 0/1/2/3
NAMES = ("no_flower", "few_flowers", "many_flowers", "very_many_flowers")
CLIP_V = 250          # pixel coi là bão hoà (lõi chói)
CLIP_FRAC = 0.30      # blob có >= 30% pixel bão hoà thì luật clip_rule loại


def level(ratio):
    # ratio: tỉ lệ diện tích hoa / tán (0-1) -> (mức 0-3, tên mức).
    t1, t2, t3 = THRESHOLDS
    if ratio < t1:
        lvl = 0
    elif ratio < t2:
        lvl = 1
    elif ratio < t3:
        lvl = 2
    else:
        lvl = 3
    return lvl, NAMES[lvl]


def crop_polygon(img_bgr, poly):
    # Cắt bbox của polygon và dựng mask 0/1 của nó. Trả (crop, pm, (x0, y0), total)
    # hoặc None khi polygon nằm ngoài khung / rỗng. Polygon vẽ tay có thể vượt
    # ra ngoài ảnh nên bbox phải kẹp vào trong.
    h, w = img_bgr.shape[:2]
    pts = np.round(np.array(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(pm, [pts - np.array([x0, y0], np.int32)], 1)
    total = int(pm.sum())
    if total == 0:
        return None
    return img_bgr[y0:y1, x0:x1], pm, (x0, y0), total


def _hsv_in(crop, pm):
    # Tiền xử lý chung: chỉ giữ pixel trong tán, khử nhiễu, sang HSV.
    region = cv2.bitwise_and(crop, crop, mask=pm)
    blur = cv2.GaussianBlur(region, (3, 3), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


# ------------------------------------------------------------------ M0: HSV
def hsv_mask(crop, pm, sat_max, val_min):
    # Mask 0/1 cỡ crop: pixel hoa theo ngưỡng cố định, đã opening, chỉ trong tán.
    hsv = _hsv_in(crop, pm)
    flower = ((hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)).astype(np.uint8)
    flower = cv2.morphologyEx(flower, cv2.MORPH_OPEN, KERNEL)
    return ((flower > 0) & (pm > 0)).astype(np.uint8)


def hsv_counts(img_bgr, poly, sat_max, val_min):
    # (số pixel hoa, tổng pixel) của một tán. Hành vi y hệt server.py bản cũ.
    cp = crop_polygon(img_bgr, poly)
    if cp is None:
        return 0, 0
    crop, pm, _, total = cp
    return int(np.count_nonzero(hsv_mask(crop, pm, sat_max, val_min))), total


# ------------------------------------------------------------ M1: Otsu + blob
def separability(vals, thr):
    # Tiêu chuẩn của chính Otsu (1979): phương sai giữa hai nhóm / phương sai
    # tổng, trong [0, 1]. Đề cương định dùng nó để chặn tán không hoa (histogram
    # "một đỉnh" → η thấp). Thực tế đo được η≈0,7 ở mọi tán vì đỉnh nắng/bóng
    # luôn có; vẫn ghi ra để hiệu chỉnh `sep_min` sau khi có nhãn blob.
    vals = vals.astype(np.float64)
    tot = vals.var()
    if tot <= 0:
        return 0.0
    lo, hi = vals[vals <= thr], vals[vals > thr]
    if len(lo) == 0 or len(hi) == 0:
        return 0.0
    w0, w1 = len(lo) / len(vals), len(hi) / len(vals)
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / tot)


CHANNELS = ("v", "v2", "min2")
MIN_BRIGHT = 50       # tầng 2 cần ít nhất ngần này pixel sáng, không thì coi là không hoa


def _otsu(vals):
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def otsu_blob(img_bgr, poly, channel="min2", sep_min=0.0, area_min=0, area_max=0,
              elong_max=0.0, clip_rule=False, want_mask=False):
    # Trả dict: flower_pixels, total_pixels, ratio, label, channel, threshold
    # (ngưỡng cuối), threshold1 (ngưỡng tầng 1, None với "v"), separability (η của
    # tầng cuối), n_blobs, n_blobs_kept, rejected (None | "low_separability" |
    # "too_few_bright"); thêm mask (0/1 cỡ crop) và offset khi want_mask. None nếu
    # polygon rỗng.
    #
    # Mỗi luật lọc blob là một tham số, giá trị 0/False = tắt. Mặc định tắt hết:
    # theo đề cương, không luật nào được bật trước khi có nhãn blob (B3) chấm nó.
    if channel not in CHANNELS:
        raise ValueError(f"channel phải là một trong {CHANNELS}, không phải {channel!r}")
    cp = crop_polygon(img_bgr, poly)
    if cp is None:
        return None
    crop, pm, offset, total = cp
    hsv = _hsv_in(crop, pm)
    v = hsv[:, :, 2]
    if channel == "min2":
        chan = cv2.GaussianBlur(cv2.bitwise_and(crop, crop, mask=pm), (3, 3), 0).min(axis=2)
    else:
        chan = v
    vals = chan[pm > 0]
    # Otsu CHỈ trên pixel trong tán: đưa cả crop vào thì vùng đen ngoài polygon
    # thành một đỉnh giả ở 0 và kéo ngưỡng xuống.
    out = dict(flower_pixels=0, total_pixels=total, ratio=0.0, label=0,
               channel=channel, threshold=None, threshold1=None, separability=0.0,
               n_blobs=0, n_blobs_kept=0, rejected=None)

    def _reject(why):
        out["rejected"] = why
        if want_mask:
            out["mask"], out["offset"] = np.zeros_like(pm), offset
        return out

    if channel == "v":
        thr = _otsu(vals)
        eta = separability(vals, thr)
        out.update(threshold=thr, separability=round(eta, 4))
    else:
        t1 = _otsu(vals)
        bright = vals[vals > t1]
        out["threshold1"] = t1
        if len(bright) < MIN_BRIGHT:
            out["threshold"] = t1
            return _reject("too_few_bright")
        thr = _otsu(bright)
        eta = separability(bright, thr)
        out.update(threshold=thr, separability=round(eta, 4))
    if sep_min > 0 and eta < sep_min:
        return _reject("low_separability")

    m = ((chan > thr) & (pm > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
        if area_min > 0 and area < area_min:
            continue
        if area_max > 0 and area > area_max:
            continue
        if elong_max > 0 and max(bw, bh) / max(1, min(bw, bh)) > elong_max:
            continue
        if clip_rule:
            x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
            blob = labels[y:y + bh, x:x + bw] == k
            clipped = np.count_nonzero(v[y:y + bh, x:x + bw][blob] >= CLIP_V)
            if clipped >= CLIP_FRAC * area:
                continue
        keep[k] = True
    kept = keep[labels]
    fpx = int(np.count_nonzero(kept))
    ratio = round(fpx / total, 4) if total else 0.0
    out.update(flower_pixels=fpx, ratio=ratio, label=level(ratio)[0],
               n_blobs=int(n - 1), n_blobs_kept=int(keep.sum()))
    if want_mask:
        out["mask"], out["offset"] = kept.astype(np.uint8), offset
    return out


# ------------------------------------------------------------------ lớp phủ
def render_mask(img_bgr, polys, method, sat_max=50, val_min=180, **otsu_kw):
    # otsu_kw: channel, sep_min, area_min, area_max, elong_max, clip_rule.
    # Mask 0/1 cỡ cả ảnh: hợp pixel hoa của mọi polygon theo một phương pháp.
    h, w = img_bgr.shape[:2]
    full = np.zeros((h, w), np.uint8)
    for poly in polys:
        if len(poly) < 6:
            continue
        if method == "hsv":
            cp = crop_polygon(img_bgr, poly)
            if cp is None:
                continue
            crop, pm, (x0, y0), _ = cp
            m = hsv_mask(crop, pm, sat_max, val_min)
        else:
            r = otsu_blob(img_bgr, poly, want_mask=True, **otsu_kw)
            if r is None:
                continue
            m, (x0, y0) = r["mask"], r["offset"]
        mh, mw = m.shape
        full[y0:y0 + mh, x0:x0 + mw] |= m
    return full


def mask_png(mask, bgr, alpha=150):
    # PNG BGRA trong suốt: pixel mask nhận màu `bgr` với độ mờ alpha, còn lại 0.
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    on = mask > 0
    rgba[on, 0], rgba[on, 1], rgba[on, 2] = bgr
    rgba[on, 3] = alpha
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes() if ok else b""
