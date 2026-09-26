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

from typing import NamedTuple

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


class Surfaces(NamedTuple):
    """Khoảng cách biên của MỘT cặp (dự đoán, vùng thật), đã tính sẵn.

    Bốn chỉ số dưới đây — ASSD, HD95, NSD, sai số biên có dấu — đều chỉ đọc
    ba mảng này. Trước đây mỗi hàm tự gọi `surface_distances`, tức dựng lại
    đường biên và chạy lại `distanceTransform` bốn lượt cho cùng một cặp. Đo
    trên cửa sổ 340x340 (tán trung vị của bộ này): 6.44 ms mỗi vùng, trong
    đó 4.85 ms là ba lượt tính lặp. Dựng một lần rồi gọi các phương thức ở
    đây còn ~2.3 ms, cùng kết quả tới từng chữ số.
    """

    #: Khoảng cách tới biên DỰ ĐOÁN, đo tại từng điểm trên biên THẬT.
    gt_to_pred: np.ndarray
    #: Khoảng cách tới biên THẬT, đo tại từng điểm trên biên DỰ ĐOÁN.
    pred_to_gt: np.ndarray
    #: Điểm biên thật đó có nằm trong mặt nạ dự đoán không (để lấy dấu).
    gt_inside_pred: np.ndarray

    @property
    def n(self) -> int:
        return self.gt_to_pred.size + self.pred_to_gt.size

    def assd(self) -> float:
        """Average Symmetric Surface Distance, px.

        Dễ diễn giải hơn mọi biến thể IoU khi viết báo cáo: "biên lệch trung
        bình 2.3 px" ai cũng hiểu.
        """
        if self.n == 0:
            return float("nan")
        return float((self.gt_to_pred.sum() + self.pred_to_gt.sum()) / self.n)

    def hd95(self) -> float:
        """Phân vị 95 của khoảng cách biên đối xứng, px. Bắt lỗi cục bộ tệ
        nhất mà ASSD làm mượt mất; bỏ 5% để một điểm ngoại lai không chi phối."""
        both = np.concatenate((self.gt_to_pred, self.pred_to_gt))
        if both.size == 0:
            return float("nan")
        return float(np.percentile(both, 95))

    def nsd(self, tau: float) -> float:
        """Tỉ lệ đường biên nằm trong dung sai tau px. Trả lời trực tiếp câu
        "bao nhiêu phần trăm đường biên đủ tốt để khỏi phải sửa tay"."""
        if self.n == 0:
            return float("nan")
        return float(((self.gt_to_pred <= tau).sum()
                      + (self.pred_to_gt <= tau).sum()) / self.n)

    def signed(self) -> np.ndarray:
        """Sai số biên CÓ DẤU tại từng điểm trên biên thật, tính bằng px.

        Đây là chỉ số quyết định của dự án: nó phân biệt được "co vào đều"
        (sửa bằng một hằng số nở) với "lúc co lúc phình" (bắt buộc phải huấn
        luyện). Trung bình của nó KHÔNG đủ — phải nhìn cả độ trải, và trải
        trong cùng một tán khác hẳn trải giữa các tán.
        """
        # Điểm trên biên thật mà nằm ngoài vùng dự đoán -> dự đoán đang co vào.
        return (np.where(self.gt_inside_pred, 1.0, -1.0)
                * self.gt_to_pred.astype(np.float64))


def surfaces(pred: np.ndarray, gt: np.ndarray) -> Surfaces | None:
    """Dựng biên và trường khoảng cách MỘT lần cho cả bốn chỉ số biên.

    None nếu một trong hai bên không có đường biên (mặt nạ rỗng).
    """
    pe, ge = edge(pred), edge(gt)
    dp, dg = _dist_to(pe), _dist_to(ge)
    if dp is None or dg is None:
        return None
    return Surfaces(dp[ge], dg[pe], pred.astype(bool)[ge])


def surface_distances(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """(d_gt->pred, d_pred->gt) tính bằng px. None nếu một bên không có biên."""
    s = surfaces(pred, gt)
    return None if s is None else (s.gt_to_pred, s.pred_to_gt)


def signed_boundary_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Xem `Surfaces.signed`. Chấm cả một split thì dựng `surfaces()` một lần
    rồi gọi thẳng phương thức, đừng đi qua bốn hàm rời như thế này."""
    s = surfaces(pred, gt)
    return np.array([], dtype=np.float64) if s is None else s.signed()


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Xem `Surfaces.assd`."""
    s = surfaces(pred, gt)
    return float("nan") if s is None else s.assd()


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Xem `Surfaces.hd95`."""
    s = surfaces(pred, gt)
    return float("nan") if s is None else s.hd95()


def normalized_surface_dice(pred: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """Xem `Surfaces.nsd`."""
    s = surfaces(pred, gt)
    return float("nan") if s is None else s.nsd(tau)
