"""Chạy model trên một split và đo từng vùng một.

Đầu ra là một hàng cho mỗi vùng thật cộng một hàng cho mỗi dự đoán thừa, ghi
thẳng ra CSV. Mọi câu hỏi phát sinh về sau — "tệ hơn ở field nào", "tán to có
khác không", "ngưỡng conf nào tốt nhất" — đều cắt lại được từ file đó mà không
phải chạy lại GPU.

Vì sao không chỉ dùng mAP của ultralytics: đo trên chính bộ này, mặt nạ co vào
5 px vẫn cho AP75 = 0.953 trong khi Boundary IoU chỉ còn 0.233. Hai lần train
640 và 1024 chênh nhau đúng 0.006 mAP dù trần đường biên khác hẳn. Chỉ số theo
diện tích không nhìn thấy thứ dự án này quan tâm, nên ở đây chấm cả hai nhóm.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..datasets.coco import CocoDataset
from ..metrics import boundary as B
from ..metrics import mask as M
from ..models.base import SegmentationModel
from .coco_eval import predictions_to_coco
from .matching import align_masks, match_instances


def _imread(path: Path):
    import cv2

    # cv2.imread không đọc được đường dẫn có ký tự Unicode trên Windows.
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _region_row(region, matched, iou, band_ratio, nsd_tau) -> dict:
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
        row.update({k: "" for k in
                    ("iou", "dice", "coverage", "excess", "boundary_iou",
                     "assd", "hd95", "nsd", "signed_median", "signed_std")})
        return row

    g, p = align_masks(region, matched)
    d = B.band_width(region.area, band_ratio)
    signed = B.signed_boundary_error(p, g)
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
    })
    return row


def evaluate_split(
    model: SegmentationModel,
    dataset: CocoDataset,
    iou_thr: float = 0.5,
    band_ratio: float = 0.02,
    nsd_tau: float = 2.0,
    limit: int | None = None,
    progress: bool = True,
) -> tuple[list[dict], dict]:
    """Chấm model trên toàn bộ split. Trả về (các hàng, tóm tắt cấp ảnh)."""
    images = dataset.image_list()[:limit]
    rows: list[dict] = []
    per_image: list[dict] = []
    detections: list[dict] = []
    cat_id = int(dataset.categories[0]["id"]) if dataset.categories else 1
    model.warmup()
    t0 = time.time()

    for i, im in enumerate(images, 1):
        img = _imread(im.path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh {im.path}")
        t1 = time.time()
        preds = model.predict(img)
        dt = time.time() - t1

        # Nguồn chính để chấm theo chuẩn COCO. Các chỉ số từng vùng bên dưới là
        # phần bổ trợ, không thay thế AP.
        detections.extend(predictions_to_coco(im.image_id, preds, cat_id))

        matches, missed, spurious = match_instances(im.regions, preds, iou_thr)
        by_region = {id(m.region): m for m in matches}
        for region in im.regions:
            m = by_region.get(id(region))
            rows.append(
                _region_row(region, m.pred if m else None,
                            m.iou if m else 0.0, band_ratio, nsd_tau)
            )
        for p in spurious:
            rows.append({
                "image": im.file_name,
                "field": im.file_name.split("__")[0],
                "flight": "/".join(im.file_name.split("__")[:-1]),
                "ann_id": -1, "gt_area": "", "gt_side": "",
                "matched": -1,  # -1 = dự đoán thừa, không có vùng thật tương ứng
                "score": round(p.score, 4),
            })
        per_image.append({
            "image": im.file_name,
            "field": im.file_name.split("__")[0],
            "n_gt": len(im.regions),
            "n_pred": len(preds),
            "n_matched": len(matches),
            "count_error": len(preds) - len(im.regions),
            "seconds": round(dt, 3),
        })
        if progress and (i % 10 == 0 or i == len(images)):
            print(f"  {i}/{len(images)} ảnh  ({time.time() - t0:.0f}s)", flush=True)

    return rows, {
        "images": per_image,
        "detections": detections,
        "image_ids": [im.image_id for im in images],
        "total_seconds": round(time.time() - t0, 1),
    }
