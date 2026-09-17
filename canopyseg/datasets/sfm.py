"""Đo chồng lấp bằng chuỗi khớp ảnh của COLMAP (pycolmap), dừng ở homography.

Khác với overlap.py (OpenCV trên ảnh đã co, kiểm quan hệ bằng H), đây dùng
phần khớp ảnh của COLMAP: SIFT ở độ phân giải gốc → ghép mọi cặp → kiểm hình
học hai khung bằng F/E/H và giữ tập inlier. Ta đọc tập inlier đó ra và tự
khớp homography để ước phần chồng. Hai tầng bằng chứng:

1. Cặp có `verified_inliers` ≥ ngưỡng — hai ảnh chia sẻ một quan hệ hình học.
   Đo trên sáu ruộng: trong ~33 000 cặp xa nhau (>10 khung), 99% có ĐÚNG 0
   inlier và chỉ 0,1–0,2% qua ngưỡng 15. Tầng này gần như không dương giả.
2. `homography_overlaps`: homography khớp ngay trên các inlier của COLMAP,
   chiếu khung này sang khung kia lấy phần giao → phần chồng hai chiều.

Bản trước còn dựng mô hình 3D (incremental mapping) và chiếu vết phủ xuống
mặt đất làm tầng 3–4. Đã bỏ sau khi kiểm chứng: trên 635 cặp có cả hai ước
lượng, homography lệch so với vết phủ trung vị 0,00, MAD 0,02; và mapper
không nối được chuỗi ở chồng lấp dọc ≈ 45% (hai khung liên tiếp chồng nhau
nhưng ba khung không còn điểm chung), nên tầng 3–4 chỉ có cho một phần nhỏ
cặp mà không thêm thông tin. Tiêu cự vì thế cũng không cần nữa: homography
không dùng tham số camera.

Cài: `pip install pycolmap` (bản PyPI chạy CPU; đủ cho vài trăm ảnh).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import overlap as ovl


def require_pycolmap():
    try:
        import pycolmap
    except ImportError as e:  # noqa: F841
        raise SystemExit(
            "Thiếu pycolmap. Cài bằng:  pip install pycolmap\n"
            "(bản PyPI chạy CPU, không cần CUDA; ~1,5 s/ảnh trích đặc trưng.)"
        ) from None
    return pycolmap


def extract_and_match(
    image_root: str | Path,
    image_names: list[str],
    database: str | Path,
    pairing: str = "exhaustive",
    seq_overlap: int = 3,
    max_features: int = 4096,
    max_image_size: int = -1,
    log=print,
) -> Path:
    """Trích đặc trưng → ghép cặp → kiểm hình học, ghi vào `database` (mới).

    `image_names` là đường dẫn tương đối so với `image_root`, dùng '/'.
    Camera SINGLE: mọi ảnh cùng một máy ảnh, đúng với một chuyến bay drone.
    Đây là phần tốn thời gian (~1,5 s/ảnh + ~0,1 s/cặp trên CPU); mọi thứ
    sau nó chỉ mất vài giây và có thể làm lại từ database này.
    """
    pycolmap = require_pycolmap()
    db = Path(database)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    t = time.time()
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = max_features
    eo.max_image_size = max_image_size
    pycolmap.extract_features(
        db, image_root, image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=eo,
        device=pycolmap.Device.cpu,
    )
    log(f"  COLMAP đặc trưng: {len(image_names)} ảnh, {time.time() - t:.0f}s")

    t = time.time()
    if pairing == "exhaustive":
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    else:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = seq_overlap
        po.quadratic_overlap = False
        po.loop_detection = False
        pycolmap.match_sequential(db, pairing_options=po, device=pycolmap.Device.cpu)
    log(f"  COLMAP ghép cặp ({pairing}): {time.time() - t:.0f}s")
    return db


def _config_name(pycolmap, cfg: int) -> str:
    try:
        return pycolmap.TwoViewGeometryConfiguration(cfg).name
    except Exception:  # noqa: BLE001 - chỉ là nhãn để đọc
        return str(cfg)


def read_pairs(database: str | Path) -> tuple[dict[int, str], dict[tuple[int, int], dict]]:
    """Từ database: tên ảnh theo id, và mỗi cặp đã ghép: số match thô, số
    inlier sau kiểm hình học, loại hình học tìm được."""
    pycolmap = require_pycolmap()
    d = pycolmap.Database.open(str(database))
    names = {im.image_id: im.name for im in d.read_all_images()}
    pairs: dict[tuple[int, int], dict] = {}
    for pid, m in zip(*d.read_all_matches()):
        pairs[pycolmap.pair_id_to_image_pair(pid)] = dict(raw_matches=int(len(m)), verified_inliers=0, geometry="")
    for pid, g in zip(*d.read_two_view_geometries()):
        key = pycolmap.pair_id_to_image_pair(pid)
        pairs.setdefault(key, dict(raw_matches=0))
        pairs[key].update(verified_inliers=int(len(g.inlier_matches)), geometry=_config_name(pycolmap, int(g.config)))
    d.close()
    return names, pairs


def homography_overlaps(
    database: str | Path, min_inliers: int = 15, ransac_px: float = 24.0,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Tầng 2: với mọi cặp có ≥ `min_inliers` inlier sau kiểm hình học, khớp
    homography trên chính các inlier đó (toạ độ keypoint gốc) và suy ra
    (phần a trong b, phần b trong a). Bỏ qua cặp mà H không lành.

    Ngưỡng 24 px là lỏng có chủ ý: inlier đã được F chấp nhận, H chỉ cần tả
    được mặt đất trung bình dù tán cao gây thị sai.
    """
    pycolmap = require_pycolmap()
    d = pycolmap.Database.open(str(database))
    shape = {c.camera_id: (c.height, c.width) for c in d.read_all_cameras()}
    cam_of = {im.image_id: im.camera_id for im in d.read_all_images()}
    kps: dict[int, np.ndarray] = {}

    def keypoints(i: int) -> np.ndarray:
        if i not in kps:
            kps[i] = np.asarray(d.read_keypoints(i))[:, :2].astype(np.float32)
        return kps[i]

    out = {}
    for pid, g in zip(*d.read_two_view_geometries()):
        inl = np.asarray(g.inlier_matches)
        if len(inl) < max(8, min_inliers):
            continue
        i, j = pycolmap.pair_id_to_image_pair(pid)
        p1, p2 = keypoints(i)[inl[:, 0]], keypoints(j)[inl[:, 1]]
        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_px)
        if H is None:
            continue
        sa, sb = shape[cam_of[i]], shape[cam_of[j]]
        ab, ba, _, scale = ovl.frame_overlap(H, sa, sb)
        if ovl.sane_homography(H, sa, scale):
            out[(i, j)] = (ab, ba)
    d.close()
    return out
