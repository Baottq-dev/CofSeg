# Metric segmentation: mask IoU, Boundary-F1, Panoptic Quality (PQ).
import json
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from skimage.segmentation import find_boundaries
from pycocotools import mask as mu


def mask_iou(a, b):
    return np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-6)


def boundary_f1(pred, gt, tol=2):
    pb = find_boundaries(pred, mode="inner")
    gb = find_boundaries(gt, mode="inner")
    dp, dg = edt(~pb), edt(~gb)
    prec = (dg[pb] <= tol).mean() if pb.any() else 0.0
    rec = (dp[gb] <= tol).mean() if gb.any() else 0.0
    return 2 * prec * rec / (prec + rec + 1e-6)


def match_instances(preds, gts, iou_thr=0.5):
    matched, tp, ious = set(), 0, []
    for p in preds:
        best, bi = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in matched:
                continue
            v = mask_iou(p, g)
            if v > best:
                best, bi = v, gi
        if best >= iou_thr:
            tp += 1
            matched.add(bi)
            ious.append(best)
    return tp, len(preds) - tp, len(gts) - tp, ious


def panoptic_quality(preds, gts, iou_thr=0.5):
    tp, fp, fn, ious = match_instances(preds, gts, iou_thr)
    sq = float(np.mean(ious)) if ious else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    return dict(PQ=sq * rq, SQ=sq, RQ=rq)


def _load(coco):
    coco = json.load(open(coco)) if isinstance(coco, str) else coco
    by_img = {}
    for a in coco["annotations"]:
        m = mu.decode(a["segmentation"]).astype(bool)
        by_img.setdefault(a["image_id"], []).append(m)
    return by_img


def evaluate_coco(pred_coco, gt_json, iou_thr=0.5):
    # So khớp theo image_id; trung bình metric trên toàn tập.
    P, G = _load(pred_coco), _load(gt_json)
    ious, bfs, pqs = [], [], []
    for img_id, gts in G.items():
        preds = P.get(img_id, [])
        _, _, _, matched = match_instances(preds, gts, iou_thr)
        ious += matched
        pqs.append(panoptic_quality(preds, gts, iou_thr)["PQ"])
        for p in preds:
            g = max(gts, key=lambda x: mask_iou(p, x))
            bfs.append(boundary_f1(p, g))
    return dict(mean_IoU=float(np.mean(ious or [0])),
                boundary_F1=float(np.mean(bfs or [0])),
                PQ=float(np.mean(pqs or [0])))