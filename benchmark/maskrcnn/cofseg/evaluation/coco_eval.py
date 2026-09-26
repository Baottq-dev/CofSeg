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
from typing import NamedTuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .. import progress

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


class _Band(NamedTuple):
    """Vành biên của một mặt nạ, giữ trong cửa sổ cắt sát nó.

    Vành trên cả khung 2560x1440 là một mảng bool 3,5 MiB. Tán trung vị chỉ
    rộng 324 px, nên hơn 95% mảng đó là số 0 — vừa tốn bộ nhớ vừa tốn phép
    đếm. Cửa sổ giữ nguyên toạ độ gốc ở (x0, y0) để hai vành khác cửa sổ vẫn
    giao nhau đúng chỗ.
    """

    m: np.ndarray
    x0: int
    y0: int
    area: int


def _band_of(rle, d: float, size: tuple[int, int]) -> _Band:
    """Vành của một RLE, tính trên cửa sổ bbox nới ra d px.

    Nới đúng d là đủ để kết quả TRÙNG với tính trên cả khung. distanceTransform
    đo tới điểm 0 gần nhất; điểm nào có khoảng cách <= d thì điểm 0 gần nhất
    của nó nằm trong bán kính d quanh nó, tức vẫn nằm trong cửa sổ. Ngoài bbox
    mặt nạ rỗng nên không có gì để mất.
    """
    h, w = size
    x, y, bw, bh = mask_utils.toBbox(rle)
    if bw <= 0 or bh <= 0:
        return _Band(np.zeros((0, 0), bool), 0, 0, 0)
    pad = int(math.ceil(d)) + 1
    x0, y0 = max(0, int(math.floor(x)) - pad), max(0, int(math.floor(y)) - pad)
    x1 = min(w, int(math.ceil(x + bw)) + pad)
    y1 = min(h, int(math.ceil(y + bh)) + pad)
    sub = np.ascontiguousarray(mask_utils.decode(rle)[y0:y1, x0:x1])
    band = _boundary_band(sub, d)
    return _Band(band, x0, y0, int(np.count_nonzero(band)))


def _band_iou(a: _Band, b: _Band) -> float:
    """IoU của hai vành nằm trên hai cửa sổ khác nhau.

    |a & b| chỉ cần tính trên phần giao của hai cửa sổ; hợp suy ra từ hai diện
    tích đã biết. Kết quả bằng đúng phép tính trên cả khung.
    """
    if a.area == 0 or b.area == 0:
        return 0.0
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1 = min(a.x0 + a.m.shape[1], b.x0 + b.m.shape[1])
    y1 = min(a.y0 + a.m.shape[0], b.y0 + b.m.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(np.count_nonzero(
        a.m[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
        & b.m[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]))
    return 0.0 if inter == 0 else inter / (a.area + b.area - inter)


class BoundaryCOCOeval(COCOeval):
    """COCOeval với Boundary IoU thay cho Mask IoU khi ghép cặp.

    KHÔNG có cache vành. Bản trước giữ lại vành full-frame của mọi vùng thật
    và mọi dự đoán tới hết lượt chấm: 280 ảnh x (14 vùng + tới 100 dự đoán) x
    3,5 MiB là hàng chục GB, và tiến trình bị kernel giết giữa chừng. Cache đó
    còn không bao giờ trúng — COCOeval.evaluate() gọi computeIoU đúng một lần
    cho mỗi cặp (ảnh, lớp), mà bộ này chỉ có một lớp, nên mỗi RLE được giải mã
    một lần dù có cache hay không.
    """

    def __init__(self, cocoGt, cocoDt, dilation_ratio=BOUNDARY_DILATION_RATIO,
                 on_image=None):
        super().__init__(cocoGt, cocoDt, iouType="segm")
        self.dilation_ratio = dilation_ratio
        #: Gọi một lần mỗi ảnh, để bước này có thanh tiến trình. Nó chạy vài
        #: phút và trước đây không in gì, nên nhìn từ ngoài không phân biệt
        #: được "đang chạy" với "đã treo".
        self.on_image = on_image

    def computeIoU(self, imgId, catId):
        if self.on_image is not None:
            self.on_image()
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

        g_bands = [_band_of(g["segmentation"], d, (h, w)) for g in gt]
        ious = np.zeros((len(dt), len(gt)))
        for i, det in enumerate(dt):
            db = _band_of(det["segmentation"], d, (h, w))
            for j, gb in enumerate(g_bands):
                ious[i, j] = _band_iou(db, gb)
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
    show_progress: bool = True,
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
        # Thanh phải dựng TRƯỚC _run: tqdm giữ lại sys.stdout lúc khởi tạo,
        # còn _run bọc cả lượt chấm trong redirect_stdout để nuốt bảng của
        # pycocotools. Dựng bên trong thì thanh vẽ vào StringIO.
        bar = progress.Bar(len(set(img_ids)), "boundary", enabled=show_progress)
        try:
            out["boundary"], out["boundary_text"] = _run(
                BoundaryCOCOeval(gt, dt, dilation_ratio, on_image=bar.advance),
                img_ids,
            )
        finally:
            bar.close()
    return out
