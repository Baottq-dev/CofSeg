"""Chỉ số chồng lấn theo diện tích.

Cảnh báo đã đo được trên chính bộ này: với tán lớn (cạnh tương đương trung vị
325 px), IoU bị phần ruột chi phối. Mặt nạ co vào 5 px vẫn cho IoU 0.908 và
COCO AP75 0.953, trong khi Boundary IoU chỉ còn 0.233. Đừng dùng riêng nhóm
chỉ số ở file này để kết luận chất lượng đường biên.
"""

from __future__ import annotations

import numpy as np


def _check(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"hai mặt nạ khác kích thước: {a.shape} vs {b.shape}")
    return a.astype(bool), b.astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Jaccard. Hai mặt nạ cùng rỗng quy ước bằng 1.0."""
    pred, gt = _check(pred, gt)
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt) / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _check(pred, gt)
    s = np.count_nonzero(pred) + np.count_nonzero(gt)
    if s == 0:
        return 1.0
    return float(2 * np.count_nonzero(pred & gt) / s)


def coverage(pred: np.ndarray, gt: np.ndarray) -> float:
    """Phần tán thật được phủ: |pred ∩ gt| / |gt|. Đây chính là con số
    "polygon phủ được bao nhiêu %" khi nhìn bằng mắt."""
    pred, gt = _check(pred, gt)
    n = np.count_nonzero(gt)
    if n == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt) / n)


def excess(pred: np.ndarray, gt: np.ndarray) -> float:
    """Phần lòi ra ngoài tán thật: |pred \\ gt| / |gt|.

    Đi kèm coverage để thấy CẢ HAI chiều. IoU trộn hai loại sai này vào một
    số duy nhất nên không phân biệt được thiếu với thừa.
    """
    pred, gt = _check(pred, gt)
    n = np.count_nonzero(gt)
    if n == 0:
        return 0.0 if not pred.any() else float("inf")
    return float(np.count_nonzero(pred & ~gt) / n)
