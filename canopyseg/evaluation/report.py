"""Gom kết quả từng vùng thành bảng đọc được.

Nguyên tắc trình bày: luôn tách nhóm "tìm được không" khỏi nhóm "vẽ biên có
đúng không". Gộp hai thứ vào một con số là cách mAP che mất lỗi đường biên
trên bộ này.
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

# Ngưỡng cỡ tán theo cạnh vuông tương đương, khớp các bậc của FPN.
SIZE_BINS = ((0, 128), (128, 256), (256, 512), (512, 10**9))

BOUNDARY_COLS = ("boundary_iou", "assd", "hd95", "nsd", "signed_median", "signed_std")
AREA_COLS = ("iou", "dice", "coverage", "excess")


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _nums(rows, col):
    out = []
    for r in rows:
        v = r.get(col, "")
        if v != "" and v is not None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:  # loại NaN
                out.append(f)
    return out


def _stat(rows, col) -> str:
    v = _nums(rows, col)
    if not v:
        return "     -"
    return f"{st.median(v):6.3f}"


def summarize(rows: list[dict], per_image: list[dict]) -> dict:
    gt = [r for r in rows if r["matched"] in (0, 1)]
    matched = [r for r in rows if r["matched"] == 1]
    spurious = [r for r in rows if r["matched"] == -1]
    n_gt, n_tp, n_fp = len(gt), len(matched), len(spurious)
    prec = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
    rec = n_tp / n_gt if n_gt else 0.0
    ce = [r["count_error"] for r in per_image]
    return {
        "n_images": len(per_image),
        "n_gt": n_gt,
        "n_pred": n_tp + n_fp,
        "true_positive": n_tp,
        "false_positive": n_fp,
        "false_negative": n_gt - n_tp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "count_error_mean": round(sum(ce) / len(ce), 2) if ce else 0.0,
        "count_error_abs_mean": round(sum(abs(c) for c in ce) / len(ce), 2) if ce else 0.0,
        **{f"median_{c}": round(st.median(_nums(matched, c)), 4)
           for c in AREA_COLS + BOUNDARY_COLS if _nums(matched, c)},
    }


def _ap_block(coco: dict | None) -> list[str]:
    """Bảng AP chuẩn: 12 con số của COCOeval, cạnh nhau cho Mask và Boundary."""
    if not coco or not coco.get("mask"):
        return ["  (không chấm được AP: " + str((coco or {}).get("error", "?")) + ")"]
    m, b = coco["mask"], coco["boundary"]
    W = 52
    L = [
        "  %-*s %9s %12s" % (W, "", "Mask AP", "Boundary AP"),
        "  " + "-" * (W + 23),
    ]
    # Nhãn giữ đúng cách viết của COCOeval.summarize() để đối chiếu được với
    # bảng của bất kỳ công bố nào.
    labels = (
        ("AP", "AP  @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"),
        ("AP50", "AP  @[ IoU=0.50      | area=   all | maxDets=100 ]"),
        ("AP75", "AP  @[ IoU=0.75      | area=   all | maxDets=100 ]"),
        ("AP_small", "AP  @[ IoU=0.50:0.95 | area= small | maxDets=100 ]"),
        ("AP_medium", "AP  @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]"),
        ("AP_large", "AP  @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"),
        ("AR_100", "AR  @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"),
        ("AR_large", "AR  @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"),
    )
    for key, label in labels:
        L.append("  %-*s %9.4f %12.4f" % (W, label, m[key], b[key]))
    L += [
        "",
        "  Boundary AP dùng cùng bộ máy COCOeval, chỉ đổi tiêu chí ghép cặp từ",
        "  Mask IoU sang Boundary IoU (Cheng et al., CVPR 2021), vành d = %.0f%%"
        % (coco["dilation_ratio"] * 100),
        "  đường chéo ảnh. Chênh lệch giữa hai cột chính là phần chất lượng",
        "  đường biên mà AP theo diện tích không nhìn thấy.",
    ]
    return L


def format_report(
    rows: list[dict], per_image: list[dict], title: str = "", coco: dict | None = None
) -> str:
    s = summarize(rows, per_image)
    matched = [r for r in rows if r["matched"] == 1]
    L: list[str] = []
    if title:
        L += [title, "=" * len(title), ""]

    L += ["1. ĐỘ CHÍNH XÁC TRUNG BÌNH (chuẩn COCO, pycocotools)", ""]
    L += _ap_block(coco)

    L += ["", "2. CHẤT LƯỢNG MẶT NẠ TRÊN CẶP ĐÃ GHÉP (trung vị, IoU ≥ 0.5)", ""]
    L.append("  IoU %s   Dice %s   độ phủ %s   phần lòi %s"
             % (_stat(matched, "iou"), _stat(matched, "dice"),
                _stat(matched, "coverage"), _stat(matched, "excess")))
    L.append("  ASSD %s px   HD95 %s px   NSD@2px %s"
             % (_stat(matched, "assd"), _stat(matched, "hd95"),
                _stat(matched, "nsd")))
    L.append("  Boundary IoU (vành theo cỡ tán, %s px trung vị) %s"
             % (_stat(matched, "gt_side"), _stat(matched, "boundary_iou")))
    L.append("  sai số biên có dấu: trung vị %s px   độ trải %s px"
             % (_stat(matched, "signed_median"), _stat(matched, "signed_std")))
    L.append("     âm = biên dự đoán nằm TRONG biên thật; độ trải lớn nghĩa là")
    L.append("     lệch không đều, tức nở bù một hằng số sẽ không sửa được")

    L += ["", "3. ĐẾM SỐ TÁN (chỉ số tầng ứng dụng, tại conf hiện tại)", ""]
    L.append(f"  {s['n_images']} ảnh · {s['n_gt']} vùng thật · {s['n_pred']} dự đoán")
    L.append(f"  đúng {s['true_positive']}   thừa {s['false_positive']}"
             f"   sót {s['false_negative']}")
    L.append(f"  precision {s['precision']:.4f}   recall {s['recall']:.4f}"
             f"   F1 {s['f1']:.4f}")
    L.append(f"  sai số đếm mỗi ảnh: trung bình {s['count_error_mean']:+.2f}"
             f"   trị tuyệt đối {s['count_error_abs_mean']:.2f}")
    L.append("     Ba số trên phụ thuộc ngưỡng conf nên KHÔNG so được giữa các")
    L.append("     công bố; AP ở mục 1 mới là con số để so sánh.")

    L += ["", "4. THEO CỠ TÁN (cạnh vuông tương đương)", ""]
    L.append("  %-14s %6s %7s %8s %8s %9s" %
             ("khoảng", "vùng", "recall", "IoU", "B-IoU", "ASSD px"))
    for lo, hi in SIZE_BINS:
        grp = [r for r in rows if r["matched"] in (0, 1)
               and r["gt_side"] != "" and lo <= float(r["gt_side"]) < hi]
        if not grp:
            continue
        ok = [r for r in grp if r["matched"] == 1]
        lbl = f"{lo}-{hi} px" if hi < 10**9 else f"{lo}+ px"
        L.append("  %-14s %6d %7.3f %8s %8s %9s" %
                 (lbl, len(grp), len(ok) / len(grp),
                  _stat(ok, "iou"), _stat(ok, "boundary_iou"), _stat(ok, "assd")))

    fields = sorted({r["field"] for r in rows if r.get("field")})
    if len(fields) > 1:
        L += ["", "5. THEO FIELD", ""]
        L.append("  %-12s %6s %7s %8s %8s" %
                 ("field", "vùng", "recall", "IoU", "B-IoU"))
        for f in fields:
            grp = [r for r in rows if r["field"] == f and r["matched"] in (0, 1)]
            if not grp:
                continue
            ok = [r for r in grp if r["matched"] == 1]
            L.append("  %-12s %6d %7.3f %8s %8s" %
                     (f, len(grp), len(ok) / len(grp),
                      _stat(ok, "iou"), _stat(ok, "boundary_iou")))
    return "\n".join(L)
