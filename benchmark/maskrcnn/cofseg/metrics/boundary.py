"""Chỉ số đường biên.

Đây là nhóm chỉ số CHÍNH của dự án. Lý do đo được trên chính bộ này: mặt nạ
co vào trong 5 px vẫn cho COCO AP@[.5:.95] = 0.889 và AP75 = 0.953, tức bảng
nghiệm thu báo "xuất sắc" đúng lúc đường biên hỏng nặng. Cùng những mặt nạ
đó, Boundary IoU chỉ còn 0.233. Tán ở đây quá lớn (cạnh tương đương trung vị
325 px) nên chỉ số theo diện tích gần như mù với sai lệch biên.

Quy ước dấu, dùng thống nhất toàn dự án:
    delta < 0  ->  biên dự đoán nằm TRONG biên thật (phủ thiếu)
    delta > 0  ->  biên dự đoán nằm NGOÀI biên thật (phủ thừa)

Khoảng cách đo giữa HAI ĐƯỜNG BIÊN (không phải từ biên tới vùng), nên không
dính sai lệch nửa pixel của phép biến đổi khoảng cách.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 là phụ thuộc bắt buộc, nhưng nêu rõ khi thiếu vẫn tốt hơn ImportError trần
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("cofseg.metrics.boundary cần opencv-python") from exc

_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _u8(m: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(m.astype(np.uint8))


def edge(mask: np.ndarray) -> np.ndarray:
    """Đường biên dày 1 px, nằm phía TRONG mặt nạ."""
    m = _u8(mask)
    return (m & ~cv2.erode(m, _K3)).astype(bool)


def boundary_band(mask: np.ndarray, d: float) -> np.ndarray:
    """Vành rộng d px chạy dọc biên, phía trong mặt nạ (Cheng et al., 2021)."""
    d = max(1, int(round(d)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    m = _u8(mask)
    return (m & ~cv2.erode(m, k)).astype(bool)


def band_width(area: float, ratio: float = 0.02) -> float:
    """Bề rộng vành theo cỡ tán: 2% cạnh hình vuông cùng diện tích.

    Vành cố định sẽ chấm tán 128 px khắt khe hơn hẳn tán 1000 px, trong khi
    bộ này trải từ 128 tới 1021 px. Cho vành co giãn theo cỡ thì mọi tán được
    chấm cùng một thang tương đối.
    """
    return max(1.0, ratio * float(np.sqrt(max(area, 0.0))))


def boundary_iou(pred: np.ndarray, gt: np.ndarray, d: float) -> float:
    """IoU của hai VÀNH biên. Bỏ hết phần ruột, chỉ chấm đường biên."""
    pb, gb = boundary_band(pred, d), boundary_band(gt, d)
    union = np.count_nonzero(pb | gb)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pb & gb) / union)


def _dist_to(edge_mask: np.ndarray) -> np.ndarray | None:
    """Trường khoảng cách tới đường biên gần nhất. None nếu biên rỗng."""
    if not edge_mask.any():
        return None
    return cv2.distanceTransform(
        _u8(~edge_mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def surface_distances(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """(d_gt->pred, d_pred->gt) tính bằng px. None nếu một bên không có biên."""
    pe, ge = edge(pred), edge(gt)
    dp, dg = _dist_to(pe), _dist_to(ge)
    if dp is None or dg is None:
        return None
    return dp[ge], dg[pe]


def signed_boundary_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Sai số biên CÓ DẤU tại từng điểm trên biên thật, tính bằng px.

    Đây là chỉ số quyết định của dự án: nó phân biệt được "co vào đều" (sửa
    bằng một hằng số nở) với "lúc co lúc phình" (bắt buộc phải huấn luyện).
    Trung bình của nó KHÔNG đủ — phải nhìn cả độ trải, và trải trong cùng một
    tán khác hẳn trải giữa các tán.
    """
    pe, ge = edge(pred), edge(gt)
    dp = _dist_to(pe)
    if dp is None or not ge.any():
        return np.array([], dtype=np.float64)
    dist = dp[ge].astype(np.float64)
    # Điểm trên biên thật mà nằm ngoài vùng dự đoán -> dự đoán đang co vào.
    inside_pred = pred.astype(bool)[ge]
    sign = np.where(inside_pred, 1.0, -1.0)
    return sign * dist


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Symmetric Surface Distance, px.

    Dễ diễn giải hơn mọi biến thể IoU khi viết báo cáo: "biên lệch trung bình
    2.3 px" ai cũng hiểu.
    """
    sd = surface_distances(pred, gt)
    if sd is None:
        return float("nan")
    a, b = sd
    n = a.size + b.size
    if n == 0:
        return float("nan")
    return float((a.sum() + b.sum()) / n)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Phân vị 95 của khoảng cách biên đối xứng, px. Bắt lỗi cục bộ tệ nhất
    mà ASSD làm mượt mất; bỏ 5% để không bị một điểm ngoại lai chi phối."""
    sd = surface_distances(pred, gt)
    if sd is None:
        return float("nan")
    both = np.concatenate(sd)
    if both.size == 0:
        return float("nan")
    return float(np.percentile(both, 95))


def normalized_surface_dice(pred: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """Tỉ lệ đường biên nằm trong dung sai tau px. Trả lời trực tiếp câu
    "bao nhiêu phần trăm đường biên đủ tốt để khỏi phải sửa tay"."""
    sd = surface_distances(pred, gt)
    if sd is None:
        return float("nan")
    a, b = sd
    n = a.size + b.size
    if n == 0:
        return float("nan")
    return float(((a <= tau).sum() + (b <= tau).sum()) / n)
