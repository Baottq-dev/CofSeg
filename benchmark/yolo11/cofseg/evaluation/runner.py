"""Chạy một model trên một split và đo từng vùng một.

Đây là đường chấm CHUNG: YOLO, Mask R-CNN, SAM oracle, hai giai đoạn, file dự
đoán từ máy thuê — tất cả đi qua đúng vòng lặp này, nên bảng benchmark so
sánh được. Đầu ra là:

  detections   danh sách COCO results -> pycocotools chấm Mask AP, Boundary AP
  rows         một hàng mỗi vùng thật (+ một hàng mỗi dự đoán thừa): IoU, Dice,
               Boundary IoU, ASSD, HD95, NSD, sai số biên có dấu, sai số diện
               tích — để cắt lát theo field, cỡ tán, ngưỡng conf mà không chạy
               lại GPU
  per_image    số vùng thật / dự đoán / khớp, sai số đếm, mili giây

Vì sao không chỉ dùng mAP: đo trên chính bộ này, mặt nạ co vào 5 px vẫn cho
AP75 = 0.953 trong khi Boundary IoU chỉ còn 0.233. Chỉ số theo diện tích không
nhìn thấy thứ dự án này quan tâm, nên ở đây chấm cả hai nhóm.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from ..datasets.coco import CocoDataset, ImageRecord
from ..datasets.instances import imread
from ..metrics import boundary as B
from ..metrics import mask as M
from ..models.base import Prediction, SegmentationModel
from .coco_eval import predictions_to_coco
from .matching import align_masks, match_instances

QUALITY_KEYS = ("iou", "dice", "coverage", "excess", "boundary_iou", "assd", "hd95",
                "nsd", "signed_median", "signed_std", "area_error_pct")


def _region_row(region, matched: Prediction | None, iou: float, band_ratio: float,
                nsd_tau: float) -> dict:
    row = {
        "image": region.image_file,
        "field": region.field,
        "flight": region.flight,
        "ann_id": region.ann_id,
        "gt_area": round(region.area, 1),
        "gt_side": round(region.equivalent_side, 1),
        "matched": int(matched is not None),
        "score": round(matched.score, 4) if matched is not None else "",
    }
    if matched is None:
        # Tán bị bỏ sót: không có mặt nạ để chấm, nhưng phải có mặt trong bảng
        # để recall và các lát cắt theo cỡ tán tính đúng.
        row.update({k: "" for k in QUALITY_KEYS})
        return row

    g, p = align_masks(region, matched)
    d = B.band_width(region.area, band_ratio)
    signed = B.signed_boundary_error(p, g)
    g_area = max(1, int(np.count_nonzero(g)))
    row.update({
        "iou": round(iou, 4),
        "dice": round(M.dice(p, g), 4),
        "coverage": round(M.coverage(p, g), 4),
        "excess": round(M.excess(p, g), 4),
        "boundary_iou": round(B.boundary_iou(p, g, d), 4),
        "assd": round(B.assd(p, g), 3),
        "hd95": round(B.hd95(p, g), 3),
        "nsd": round(B.normalized_surface_dice(p, g, nsd_tau), 4),
        # Trung vị nói lệch về phía nào, độ trải nói lệch có ĐỀU không. Hai số
        # này phân biệt "co vào đều" (nở bù là xong) với "lúc co lúc phình"
        # (bắt buộc phải huấn luyện thêm) — cùng một trung bình, hai cách chữa.
        "signed_median": round(float(np.median(signed)), 3) if signed.size else "",
        "signed_std": round(float(signed.std()), 3) if signed.size else "",
        # Diện tích tán là đầu vào của mật độ hoa; sai số này đi thẳng vào đó.
        "area_error_pct": round(100.0 * (int(np.count_nonzero(p)) - g_area) / g_area, 2),
    })
    return row


def _spurious_row(im: ImageRecord, p: Prediction) -> dict:
    return {
        "image": im.file_name,
        "field": im.file_name.split("__")[0],
        "flight": "/".join(im.file_name.split("__")[:-1]),
        "ann_id": -1, "gt_area": "", "gt_side": "",
        "matched": -1,  # -1 = dự đoán thừa, không có vùng thật tương ứng
        "score": round(p.score, 4),
        **{k: "" for k in QUALITY_KEYS},
    }


def evaluate_split(
    model: SegmentationModel,
    dataset: CocoDataset,
    iou_thr: float = 0.5,
    band_ratio: float = 0.02,
    nsd_tau: float = 2.0,
    limit: int | None = None,
    progress: bool = True,
) -> tuple[list[dict], dict]:
    """Chấm model trên toàn bộ split. Trả về (các hàng, thông tin cấp ảnh).

    Model cần gợi ý (needs_prompt) nhận box THẬT của ảnh — đó là định nghĩa
    của dòng oracle. Model có set_context() được báo ảnh nào sắp chấm — đó là
    cách file dự đoán từ máy khác tra đúng ảnh.
    """
    images = dataset.image_list()[:limit]
    rows: list[dict] = []
    per_image: list[dict] = []
    detections: list[dict] = []
    cat_id = int(dataset.categories[0]["id"]) if dataset.categories else 1
    set_context = getattr(model, "set_context", None)
    model.warmup()
    t0 = time.time()

    for i, im in enumerate(images, 1):
        img = imread(im.path)
        boxes = None
        if model.needs_prompt:
            boxes = np.array([r.bbox_xyxy for r in im.regions], np.float32).reshape(-1, 4)
        if set_context is not None:
            set_context(im)
        t1 = time.perf_counter()
        preds = model.predict(img, boxes)
        ms = (time.perf_counter() - t1) * 1000.0

        detections.extend(predictions_to_coco(im.image_id, preds, cat_id, size=img.shape[:2]))

        matches, missed, spurious = match_instances(im.regions, preds, iou_thr)
        by_region = {id(m.region): m for m in matches}
        for region in im.regions:
            m = by_region.get(id(region))
            rows.append(_region_row(region, m.pred if m else None, m.iou if m else 0.0,
                                    band_ratio, nsd_tau))
        rows.extend(_spurious_row(im, p) for p in spurious)
        per_image.append({
            "image": im.file_name,
            "field": im.file_name.split("__")[0],
            "n_gt": len(im.regions),
            "n_pred": len(preds),
            "n_matched": len(matches),
            "count_error": len(preds) - len(im.regions),
            "ms": round(ms, 1),
        })
        if progress and (i % 10 == 0 or i == len(images)):
            print(f"  {i}/{len(images)} ảnh  ({time.time() - t0:.0f}s)", flush=True)

    return rows, {
        "images": per_image,
        "detections": detections,
        "image_ids": [im.image_id for im in images],
        "total_seconds": round(time.time() - t0, 1),
    }


def _num(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) not in ("", None)]


def summarize(rows: list[dict], per_image: list[dict]) -> dict:
    """Gộp các hàng thành vài con số. Chất lượng chỉ tính trên cặp đã ghép;
    recall tính trên mọi vùng thật, kể cả vùng bỏ sót không có mặt nạ."""
    gt = [r for r in rows if r["matched"] in (0, 1)]
    tp = [r for r in gt if r["matched"] == 1]
    fp = [r for r in rows if r["matched"] == -1]
    n_gt, n_tp, n_fp = len(gt), len(tp), len(fp)
    out = {
        "images": len(per_image),
        "n_gt": n_gt,
        "true_positive": n_tp,
        "false_negative": n_gt - n_tp,
        "false_positive": n_fp,
        "precision": round(n_tp / (n_tp + n_fp), 4) if n_tp + n_fp else 0.0,
        "recall": round(n_tp / n_gt, 4) if n_gt else 0.0,
        "count_error_mean": round(statistics.fmean(p["count_error"] for p in per_image), 3)
        if per_image else 0.0,
        "ms_per_image": round(statistics.fmean(p["ms"] for p in per_image), 1)
        if per_image else 0.0,
    }
    for k in QUALITY_KEYS:
        v = _num(tp, k)
        if v:
            out[f"mean_{k}"] = round(statistics.fmean(v), 4)
            out[f"median_{k}"] = round(statistics.median(v), 4)
    fields = sorted({r["field"] for r in rows})
    out["by_field"] = {}
    for f in fields:
        fr = [r for r in rows if r["field"] == f]
        fgt = [r for r in fr if r["matched"] in (0, 1)]
        ftp = [r for r in fgt if r["matched"] == 1]
        b = _num(ftp, "boundary_iou")
        out["by_field"][f] = {
            "n_gt": len(fgt),
            "recall": round(len(ftp) / len(fgt), 4) if fgt else 0.0,
            "false_positive": sum(1 for r in fr if r["matched"] == -1),
            "mean_boundary_iou": round(statistics.fmean(b), 4) if b else None,
        }
    return out
