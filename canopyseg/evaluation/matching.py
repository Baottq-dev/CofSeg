"""Ghép vùng thật với vùng dự đoán.

Ghép tham lam theo điểm tin cậy giảm dần, đúng lệ COCO: duyệt dự đoán từ chắc
chắn nhất, mỗi cái nhận vùng thật chưa ai lấy có IoU cao nhất và vượt ngưỡng.
Không dùng Hungarian vì nó tối ưu tổng IoU toàn ảnh, tức có thể hi sinh một
cặp rất khớp để cứu một cặp tồi — không phải cách một hệ thống thật hoạt động,
và cũng không phải cách COCO chấm.

Mọi phép so được làm trong KHUNG CỤC BỘ bao quanh cặp đang xét. Mặt nạ đầy đủ
2560x1440 cho từng vật thể là 3.7 MB, nhân 15 vật thể mỗi ảnh thì vừa chậm vừa
không thêm thông tin gì.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..datasets.region import Region
from ..models.base import Prediction

#: Bỏ qua sớm các cặp không thể khớp, để khỏi dựng mặt nạ cho từng cặp một.
BBOX_IOU_GATE = 0.05


@dataclass
class Match:
    region: Region
    pred: Prediction
    iou: float


def prediction_bbox(pred: Prediction) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) trong toạ độ ảnh gốc. None nếu mặt nạ rỗng."""
    return pred.bbox_xyxy


def _bbox_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def align_masks(
    region: Region, pred: Prediction, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Dựng mặt nạ thật và mặt nạ dự đoán trên CÙNG một khung cục bộ."""
    gb = region.bbox_xyxy
    pb = prediction_bbox(pred)
    if pb is None:
        pb = gb
    x0 = max(0, math.floor(min(gb[0], pb[0])) - pad)
    y0 = max(0, math.floor(min(gb[1], pb[1])) - pad)
    x1 = min(region.width, math.ceil(max(gb[2], pb[2])) + pad)
    y1 = min(region.height, math.ceil(max(gb[3], pb[3])) + pad)
    w, h = max(1, x1 - x0), max(1, y1 - y0)

    gm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(gm, [np.round(region.polygon - (x0, y0)).astype(np.int32)], 1)

    pm = np.zeros((h, w), bool)
    ox, oy = pred.origin
    # Giao giữa khung đang xét và vùng mà mặt nạ dự đoán thực sự phủ.
    sx0, sy0 = max(x0, ox), max(y0, oy)
    sx1 = min(x1, ox + pred.mask.shape[1])
    sy1 = min(y1, oy + pred.mask.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        pm[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = pred.mask[
            sy0 - oy : sy1 - oy, sx0 - ox : sx1 - ox
        ]
    return gm.astype(bool), pm


def _mask_iou(region: Region, pred: Prediction) -> float:
    g, p = align_masks(region, pred)
    u = np.count_nonzero(g | p)
    return 0.0 if u == 0 else np.count_nonzero(g & p) / u


def match_instances(
    regions: list[Region], preds: list[Prediction], iou_thr: float = 0.5
) -> tuple[list[Match], list[Region], list[Prediction]]:
    """Trả về (cặp đã ghép, vùng bị bỏ sót, dự đoán thừa)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    taken: set[int] = set()
    matches: list[Match] = []

    gt_boxes = [r.bbox_xyxy for r in regions]
    for pi in order:
        pred = preds[pi]
        pb = prediction_bbox(pred)
        if pb is None:
            continue
        best_j, best_iou = -1, 0.0
        for j, region in enumerate(regions):
            if j in taken or _bbox_iou(gt_boxes[j], pb) < BBOX_IOU_GATE:
                continue
            iou = _mask_iou(region, pred)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= iou_thr:
            taken.add(best_j)
            matches.append(Match(regions[best_j], pred, best_iou))

    matched_preds = {id(m.pred) for m in matches}
    missed = [r for j, r in enumerate(regions) if j not in taken]
    spurious = [p for p in preds if id(p) not in matched_preds]
    return matches, missed, spurious
