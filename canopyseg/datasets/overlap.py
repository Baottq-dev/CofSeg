"""Đo chồng lấp giữa hai frame bằng khớp đặc trưng cục bộ.

Câu hỏi cần trả lời: "hai ảnh này có nhìn thấy cùng một mảnh đất không, và
bao nhiêu phần?" Không có GPS thì chỉ còn cách hỏi chính nội dung ảnh.

Cách làm là bước đầu tiên của mọi phần mềm dựng ảnh (COLMAP, OpenDroneMap,
Metashape): tìm điểm đặc trưng (SIFT), ghép cặp, dùng RANSAC tìm quan hệ hình
học giữa hai khung. Hai mô hình hình học được dùng cho hai việc khác nhau:

- **Ma trận cơ bản F** để KIỂM có quan hệ hay không. F chấp nhận thị sai:
  tán cao 1–2 m ở độ cao 10 m làm điểm trên ngọn và điểm dưới đất dịch khác
  nhau, homography không tả được và loại oan phần lớn inlier. Bài học đo
  được trên field_6: cặp chồng 55% cho 9–12 inlier với homography nhưng
  49–66 với F, còn cặp không chồng lấp cho 9–10 với cả hai.
- **Homography H** chỉ để ƯỚC vết phủ (phần khung này rơi vào khung kia),
  khớp trên các inlier của F với ngưỡng lỏng. Con số là gần đúng vì mặt đất
  không phẳng; đủ để nói "có / không / cỡ bao nhiêu", không đủ đo tới pixel.

Đặc trưng: SIFT với ngưỡng tương phản hạ xuống 0,02 (mặc định OpenCV 0,04 bỏ
quá nhiều điểm trên tán lá), chuẩn hoá RootSIFT (căn bậc hai của L1) như
COLMAP làm — trên cùng cặp, RootSIFT nâng inlier từ 26 lên 31 với H.

Hai kết luận có thể rút ra từ một cặp:

- Đủ inlier F → chồng lấp thật; overlap (nếu H lành) cho biết bao nhiêu.
- Ít inlier → KHÔNG kết luận được là không chồng lấp; chỉ là không tìm thấy
  quan hệ. Tán lặp lại và bóng đổ khác chiều đều làm SIFT thất bại dù cùng
  chỗ. Vì vậy `status` được trả về cùng với con số. Muốn chắc hơn thì dùng
  SfM đầy đủ (sfm.py) hoặc GPS từ ảnh gốc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PairResult:
    n_matches: int          # số cặp qua ratio test
    n_inliers: int          # số cặp khớp với ma trận cơ bản F
    overlap_ab: float       # phần khung a nằm trong khung b (0–1); nan nếu H không lành
    overlap_ba: float       # phần khung b nằm trong khung a (0–1)
    shift_frac: float       # dịch chuyển tâm khung, tính theo bề rộng khung
    scale: float            # tỉ lệ phóng của H (~1 nếu cùng độ cao)
    status: str             # ok | verified | few_matches | no_geometry

    def as_dict(self) -> dict:
        return asdict(self)


_NAN = float("nan")
NO_RELATION = dict(overlap_ab=0.0, overlap_ba=0.0, shift_frac=_NAN, scale=_NAN)


def load_gray(path, scale: float = 0.5) -> np.ndarray:
    """Ảnh xám đã co. Co 0,5 là đủ cho câu hỏi này và nhanh gấp 4."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im


def make_detector(method: str = "sift", n_features: int = 8192, contrast: float = 0.02):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast)
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
    """Điểm + descriptor. SIFT được chuẩn hoá RootSIFT (Arandjelović & Zisserman 2012)."""
    kps, desc = detector.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    if desc is not None and desc.dtype == np.float32:
        desc = desc / (np.abs(desc).sum(1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc).astype(np.float32)
    return pts, desc


def match(matcher, desc_a, desc_b, ratio: float = 0.8) -> np.ndarray:
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


def sane_homography(H, shape_a, scale: float) -> bool:
    # Cùng độ cao thì tỉ lệ ~1; khung chiếu sang phải còn là tứ giác lồi.
    # Vi phạm là RANSAC bám vào cấu trúc lặp của tán.
    if not (0.5 <= scale <= 2.0):
        return False
    qa = cv2.perspectiveTransform(_quad(shape_a[1], shape_a[0]).reshape(-1, 1, 2), H).reshape(-1, 2)
    return bool(cv2.isContourConvex(qa.reshape(-1, 1, 2).astype(np.float32)))


def pair_overlap(
    pts_a, desc_a, shape_a, pts_b, desc_b, shape_b, matcher,
    ratio: float = 0.8, ransac_px: float = 3.0, min_inliers: int = 25,
) -> PairResult:
    """Toàn bộ chuỗi cho một cặp: ghép → F bằng RANSAC → H trên inlier của F → diện tích."""
    idx = match(matcher, desc_a, desc_b, ratio)
    n = len(idx)
    if n < max(8, min_inliers // 2):
        return PairResult(n, 0, status="few_matches", **NO_RELATION)
    p1, p2 = pts_a[idx[:, 0]], pts_b[idx[:, 1]]
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.999)
    n_in = int(mask.sum()) if mask is not None else 0
    if F is None or n_in < min_inliers:
        return PairResult(n, n_in, status="no_geometry", **NO_RELATION)
    sel = mask.ravel().astype(bool)
    # Vết phủ: H khớp lỏng (thị sai được phép) trên các cặp F đã chấp nhận.
    H, _ = cv2.findHomography(p1[sel], p2[sel], cv2.RANSAC, 8 * ransac_px)
    if H is None:
        return PairResult(n, n_in, _NAN, _NAN, _NAN, _NAN, status="verified")
    ab, ba, shift, scale = frame_overlap(H, shape_a, shape_b)
    if not sane_homography(H, shape_a, scale):
        return PairResult(n, n_in, _NAN, _NAN, shift, scale, status="verified")
    return PairResult(n, n_in, ab, ba, shift, scale, status="ok")


def self_check(detector, matcher, img: np.ndarray) -> PairResult:
    """Ảnh với chính nó phải ra ~100%. Không đạt thì đừng tin số nào bên dưới."""
    pts, desc = extract(detector, img)
    return pair_overlap(pts, desc, img.shape, pts, desc, img.shape, matcher)
