"""Đo chồng lấp giữa hai frame bằng khớp đặc trưng cục bộ và homography.

Câu hỏi cần trả lời: "hai ảnh này có nhìn thấy cùng một mảnh đất không, và
bao nhiêu phần?" Không có GPS thì chỉ còn cách hỏi chính nội dung ảnh.

Cách làm là bước đầu tiên của mọi phần mềm dựng ảnh (COLMAP, OpenDroneMap,
Metashape): tìm điểm đặc trưng (SIFT), ghép cặp, dùng RANSAC để tìm phép biến
đổi hình học giữa hai khung, rồi từ phép biến đổi đó suy ra phần khung này rơi
vào khung kia. Homography giả định mặt đất phẳng; ở 10 m so với tán cao 1–2 m
thì gần đúng, đủ để trả lời "có / không / bao nhiêu phần trăm", không đủ để
đo tới từng pixel — việc đó là của SfM.

Hai kết luận có thể rút ra từ một cặp:

- Nhiều inlier + homography lành → chồng lấp thật, và có con số.
- Ít inlier → KHÔNG kết luận được là không chồng lấp; chỉ là không tìm thấy
  quan hệ hình học. Tán cây lặp lại và bóng đổ khác chiều đều làm SIFT thất
  bại dù cùng chỗ. Vì vậy `status` được trả về cùng với con số.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PairResult:
    n_matches: int          # số cặp qua ratio test
    n_inliers: int          # số cặp khớp với homography
    overlap_ab: float       # phần khung a nằm trong khung b (0–1)
    overlap_ba: float       # phần khung b nằm trong khung a (0–1)
    shift_frac: float       # dịch chuyển tâm khung, tính theo bề rộng khung
    scale: float            # tỉ lệ phóng của homography (~1 nếu cùng độ cao)
    status: str             # ok | few_matches | no_homography | degenerate

    def as_dict(self) -> dict:
        return asdict(self)


NO_RELATION = dict(overlap_ab=0.0, overlap_ba=0.0, shift_frac=float("nan"), scale=float("nan"))


def load_gray(path, scale: float = 0.5) -> np.ndarray:
    """Ảnh xám đã co. Co 0,5 là đủ cho câu hỏi này và nhanh gấp 4."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im


def make_detector(method: str = "sift", n_features: int = 4000):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=n_features)
    if method == "orb":
        return cv2.ORB_create(nfeatures=n_features)
    raise ValueError(f"method phải là sift hoặc orb, không phải {method!r}")


def make_matcher(method: str = "sift"):
    # SIFT: descriptor float → FLANN kd-tree, nhanh hơn brute force nhiều khi
    # so mọi cặp. ORB: descriptor nhị phân → brute force Hamming.
    if method == "sift":
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def extract(detector, img: np.ndarray):
    kps, desc = detector.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    return pts, desc


def match(matcher, desc_a, desc_b, ratio: float = 0.75) -> np.ndarray:
    """Chỉ số cặp (i, j) qua ratio test của Lowe. Trả về mảng (n, 2)."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), np.int32)
    out = []
    for m in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            out.append((m[0].queryIdx, m[0].trainIdx))
    return np.array(out, np.int32).reshape(-1, 2)


def _quad(w: int, h: int) -> np.ndarray:
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def frame_overlap(H: np.ndarray, shape_a, shape_b) -> tuple[float, float, float, float]:
    """Từ homography a→b suy ra (overlap_ab, overlap_ba, shift_frac, scale).

    Chiếu bốn góc khung a sang toạ độ khung b, lấy giao với hình chữ nhật b.
    Cả hai đều lồi nên giao tính chính xác bằng intersectConvexConvex.
    """
    ha, wa = shape_a[:2]
    hb, wb = shape_b[:2]
    qa = cv2.perspectiveTransform(_quad(wa, ha).reshape(-1, 1, 2), H).reshape(-1, 2)
    area_a_in_b = abs(cv2.contourArea(qa))
    inter, _ = cv2.intersectConvexConvex(qa.reshape(-1, 1, 2), _quad(wb, hb).reshape(-1, 1, 2))
    area_b = float(wb * hb)
    overlap_ab = inter / area_a_in_b if area_a_in_b > 0 else 0.0
    overlap_ba = inter / area_b
    shift = float(np.linalg.norm(qa.mean(0) - [wb / 2, hb / 2]) / wb)
    scale = float(np.sqrt(area_a_in_b / (wa * ha)))
    return min(overlap_ab, 1.0), min(overlap_ba, 1.0), shift, scale


def pair_overlap(
    pts_a, desc_a, shape_a, pts_b, desc_b, shape_b, matcher,
    ratio: float = 0.75, ransac_px: float = 4.0, min_inliers: int = 30,
) -> PairResult:
    """Toàn bộ chuỗi cho một cặp: ghép → RANSAC → kiểm homography → diện tích."""
    idx = match(matcher, desc_a, desc_b, ratio)
    n = len(idx)
    if n < max(8, min_inliers // 2):
        return PairResult(n, 0, status="few_matches", **NO_RELATION)
    H, mask = cv2.findHomography(pts_a[idx[:, 0]], pts_b[idx[:, 1]], cv2.RANSAC, ransac_px)
    n_in = int(mask.sum()) if mask is not None else 0
    if H is None or n_in < min_inliers:
        return PairResult(n, n_in, status="no_homography", **NO_RELATION)
    ab, ba, shift, scale = frame_overlap(H, shape_a, shape_b)
    # Homography "lành": cùng độ cao thì tỉ lệ ~1; khung chiếu sang phải còn
    # là tứ giác lồi tử tế. Vi phạm là RANSAC bám vào cấu trúc lặp của tán.
    qa = cv2.perspectiveTransform(_quad(shape_a[1], shape_a[0]).reshape(-1, 1, 2), H).reshape(-1, 2)
    convex = cv2.isContourConvex(qa.reshape(-1, 1, 2).astype(np.float32))
    if not (0.5 <= scale <= 2.0) or not convex:
        return PairResult(n, n_in, ab, ba, shift, scale, status="degenerate")
    return PairResult(n, n_in, ab, ba, shift, scale, status="ok")


def self_check(detector, matcher, img: np.ndarray) -> PairResult:
    """Ảnh với chính nó phải ra ~100%. Không đạt thì đừng tin số nào bên dưới."""
    pts, desc = extract(detector, img)
    return pair_overlap(pts, desc, img.shape, pts, desc, img.shape, matcher)
