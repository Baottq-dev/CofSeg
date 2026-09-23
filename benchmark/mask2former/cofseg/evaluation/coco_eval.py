"""Đánh giá theo chuẩn COCO, và Boundary AP theo Cheng et al. (CVPR 2021).

Dùng thẳng pycocotools làm bộ máy chấm, không tự viết lại: đó là bản cài đặt
tham chiếu mà mọi công bố dùng, nên con số ra mới so được với văn liệu.

Boundary AP = đúng bộ máy đó nhưng đổi tiêu chí ghép cặp từ Mask IoU sang
Boundary IoU. Lý do cần nó, đo trên chính bộ này: mặt nạ co vào 5 px vẫn cho
AP75 = 0.953 trong khi Boundary IoU chỉ còn 0.233. Với tán lớn, AP theo diện
tích gần như mù với sai lệch đường biên.

Bề rộng vành d lấy đúng định nghĩa trong bài báo: 2% đường chéo ẢNH (không
phải 2% cạnh vật thể). Với ảnh 2560x1440 thì d ≈ 59 px.
"""

from __future__ import annotations

import contextlib
import io
import math

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#: Tỉ lệ trong bài báo Boundary IoU: d = 0.02 x đường chéo ảnh.
BOUNDARY_DILATION_RATIO = 0.02

#: 12 dòng chuẩn của COCOeval.summarize(), theo đúng thứ tự stats[0..11].
STAT_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def encode_mask(mask: np.ndarray) -> dict:
    """Mặt nạ nhị phân -> RLE mà pycocotools hiểu."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def full_frame(pred, size: tuple[int, int]) -> np.ndarray:
    """Mặt nạ của một Prediction trên khung ảnh đầy đủ (h, w).

    Prediction giữ mặt nạ trong khung cục bộ + origin để tiết kiệm bộ nhớ;
    pycocotools lại cần RLE của cả ảnh. Đây là chỗ duy nhất dán nó trở lại.
    """
    h, w = size
    ox, oy = pred.origin
    if (ox, oy) == (0, 0) and pred.mask.shape == (h, w):
        return pred.mask
    out = np.zeros((h, w), bool)
    mh, mw = pred.mask.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + mw), min(h, oy + mh)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = pred.mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return out


def predictions_to_coco(
    image_id: int, preds, category_id: int = 1, size: tuple[int, int] | None = None
) -> list[dict]:
    """Danh sách Prediction -> danh sách detection theo định dạng COCO.

    category_id mặc định 1 vì bộ COCO ở đây đánh số lớp từ 1 (đúng lệ COCO gốc),
    trong khi YOLO trả về class index 0. Lệch một là cố ý.

    `size` = (h, w) của ảnh; bắt buộc khi mặt nạ nằm trong khung cục bộ.
    """
    out = []
    for p in preds:
        if not p.mask.any():
            continue
        mask = p.mask if size is None else full_frame(p, size)
        out.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "segmentation": encode_mask(mask),
            "score": float(p.score),
        })
    return out


def _boundary_band(mask_u8: np.ndarray, d: float) -> np.ndarray:
    """Vành rộng d px phía trong biên.

    Dùng biến đổi khoảng cách thay cho erode: với d ≈ 59 px, nhân erode là hình
    tròn 119x119 và phải chạy cho từng mặt nạ của cả nghìn cặp. distanceTransform
    cho cùng kết quả trong thời gian tuyến tính.
    """
    if not mask_u8.any():
        return mask_u8.astype(bool)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (mask_u8 > 0) & (dist <= d)


class BoundaryCOCOeval(COCOeval):
    """COCOeval với Boundary IoU thay cho Mask IoU khi ghép cặp."""

    def __init__(self, cocoGt, cocoDt, dilation_ratio=BOUNDARY_DILATION_RATIO):
        super().__init__(cocoGt, cocoDt, iouType="segm")
        self.dilation_ratio = dilation_ratio
        self._band_cache: dict[int, np.ndarray] = {}

    def _band(self, rle, d: float) -> np.ndarray:
        key = id(rle)
        band = self._band_cache.get(key)
        if band is None:
            band = _boundary_band(mask_utils.decode(rle), d)
            self._band_cache[key] = band
        return band

    def computeIoU(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId] if p.useCats else [
            _ for cId in p.catIds for _ in self._gts[imgId, cId]
        ]
        dt = self._dts[imgId, catId] if p.useCats else [
            _ for cId in p.catIds for _ in self._dts[imgId, cId]
        ]
        if not gt or not dt:
            return []
        dt = sorted(dt, key=lambda x: -x["score"])
        if len(dt) > p.maxDets[-1]:
            dt = dt[: p.maxDets[-1]]

        h, w = self.cocoGt.imgs[imgId]["height"], self.cocoGt.imgs[imgId]["width"]
        d = self.dilation_ratio * math.sqrt(h * h + w * w)

        g_bands = [self._band(g["segmentation"], d) for g in gt]
        ious = np.zeros((len(dt), len(gt)))
        for i, det in enumerate(dt):
            db = self._band(det["segmentation"], d)
            for j, gb in enumerate(g_bands):
                union = np.count_nonzero(db | gb)
                ious[i, j] = 0.0 if union == 0 else np.count_nonzero(db & gb) / union
        return ious


def _run(evaluator, img_ids) -> tuple[dict, str]:
    evaluator.params.imgIds = sorted(img_ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = {n: float(v) for n, v in zip(STAT_NAMES, evaluator.stats)}
    return stats, buf.getvalue().rstrip()


def evaluate(
    gt_json: str,
    detections: list[dict],
    img_ids,
    dilation_ratio: float = BOUNDARY_DILATION_RATIO,
    boundary: bool = True,
) -> dict:
    """Chấm chuẩn COCO (Mask AP) và Boundary AP trên cùng bộ dự đoán.

    `boundary=False` bỏ Boundary AP — dùng khi chấm val giữa các epoch, nơi
    tốc độ quan trọng hơn và Boundary AP sẽ được chấm đầy đủ ở bước đánh giá.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        if not detections:
            return {"mask": None, "boundary": None, "error": "không có dự đoán nào"}
        dt = gt.loadRes(list(detections))

    mask_stats, mask_text = _run(COCOeval(gt, dt, iouType="segm"), img_ids)
    out = {
        "mask": mask_stats,
        "boundary": None,
        "mask_text": mask_text,
        "boundary_text": None,
        "dilation_ratio": dilation_ratio,
    }
    if boundary:
        out["boundary"], out["boundary_text"] = _run(
            BoundaryCOCOeval(gt, dt, dilation_ratio), img_ids
        )
    return out
