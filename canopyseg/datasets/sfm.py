"""Đo chồng lấp bằng SfM đầy đủ: COLMAP qua pycolmap.

Khác với overlap.py (một homography cho mỗi cặp, ảnh đã co), đây là toàn bộ
chuỗi dựng ảnh: SIFT ở độ phân giải gốc → ghép mọi cặp → kiểm hình học hai
khung (F/E/H) → dựng dần mô hình 3D chung. Bốn tầng bằng chứng, mỗi tầng khó
qua hơn tầng trước:

1. Cặp có `verified_inliers` — hai ảnh chia sẻ một quan hệ hình học nào đó.
   Đo trên sáu ruộng: trong ~33 000 cặp xa nhau (>10 khung), 99% có ĐÚNG 0
   inlier và chỉ 0,1–0,2% qua ngưỡng 15. Tầng này gần như không dương giả.
2. `homography_overlaps`: homography khớp ngay trên các inlier của COLMAP,
   chiếu khung này sang khung kia lấy phần giao. Cần vì mapper KHÔNG nối được
   chuỗi khi chồng lấp dọc ≈ 45%: hai khung liên tiếp chồng nhau nhưng ba
   khung liên tiếp không còn điểm chung, nên từng cặp khớp rất chắc (hàng
   trăm inlier) mà không ảnh nào vào mô hình. Kiểm chứng trên 635 cặp có cả
   hai ước lượng: lệch so với tầng 4 trung vị 0,00, MAD 0,02.
3. Hai ảnh cùng nằm trong một mô hình 3D — mapper đã tìm được tư thế camera
   nhất quán cho cả hai.
4. `footprint`: từ tư thế camera và mặt phẳng đất khớp vào đám mây điểm,
   chiếu bốn góc mỗi khung xuống đất rồi lấy giao — không cần GPS vì tỉ lệ
   diện tích không phụ thuộc thang đo.

Tiêu cự phải CỐ ĐỊNH. Ảnh đã bị lột EXIF nên COLMAP không có prior và tự ước
f; trên ảnh nadir chụp mặt gần phẳng thì f không quan sát được (đổi f và độ
cao cùng một hệ số cho ra cùng ảnh), và mapper đã trả về f từ 96 đến 14 506 px
cho cùng một máy ảnh. Cố định f (và tâm ảnh, méo = 0) cho cùng số ảnh đăng
ký, mô hình gọn hơn, và vết phủ ổn định. Xem `focal_px` trong `map_models`.

Với ảnh có overlap thật (mapping mission 70–80%), gần như mọi ảnh vào một
mô hình với hàng chục nghìn điểm. Với ảnh chụp rời, mapper ra vài mô hình tí
hon hoặc không ra gì; đó chính là câu trả lời.

Cài: `pip install pycolmap` (bản PyPI chạy CPU; đủ cho vài trăm ảnh).
"""

from __future__ import annotations

import itertools
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


def set_camera_prior(database: str | Path, focal_px: float) -> None:
    """Ghi tiêu cự đã biết vào mọi camera trong database: f, tâm ảnh, méo 0.
    Sau đó `map_models` giữ nguyên các tham số này khi tinh chỉnh."""
    pycolmap = require_pycolmap()
    d = pycolmap.Database.open(str(database))
    for cam in d.read_all_cameras():
        cam.params = [float(focal_px), cam.width / 2, cam.height / 2, 0.0]
        cam.has_prior_focal_length = True
        d.update_camera(cam)
    d.close()


def map_models(
    database: str | Path,
    image_root: str | Path,
    sparse: str | Path,
    focal_px: float | None = None,
    min_model_size: int = 3,
    log=print,
) -> dict:
    """Dựng mô hình 3D từ database đã ghép. Trả về {chỉ số mô hình: Reconstruction}
    và ghi mỗi mô hình ra sparse/<k>/ (text + points.ply).

    `focal_px`: tiêu cự đã biết (px). Có thì cố định f, tâm ảnh và méo trong
    bundle adjustment; None thì để COLMAP tự ước — chỉ nên khi ảnh còn EXIF.
    """
    pycolmap = require_pycolmap()
    sparse = Path(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    mo = pycolmap.IncrementalPipelineOptions()
    mo.min_model_size = min_model_size
    if focal_px:
        set_camera_prior(database, focal_px)
        mo.ba_refine_focal_length = False
        mo.ba_refine_principal_point = False
        mo.ba_refine_extra_params = False
    t = time.time()
    models = pycolmap.incremental_mapping(database, image_root, sparse, options=mo)
    log(f"  COLMAP dựng mô hình: {len(models)} mô hình, {time.time() - t:.0f}s"
        + (f" (f cố định {focal_px:.0f} px)" if focal_px else " (f tự do)"))
    for k, rec in models.items():
        rec.write_text(sparse / str(k))
        rec.export_PLY(sparse / str(k) / "points.ply")
    return models


def run_colmap(
    image_root: str | Path,
    image_names: list[str],
    work_dir: str | Path,
    pairing: str = "exhaustive",
    seq_overlap: int = 3,
    max_features: int = 4096,
    max_image_size: int = -1,
    focal_px: float | None = None,
    log=print,
) -> dict:
    """Cả chuỗi: trích → ghép → dựng. Trả về database, thư mục sparse, mô hình."""
    work = Path(work_dir)
    db = extract_and_match(image_root, image_names, work / "database.db", pairing,
                           seq_overlap, max_features, max_image_size, log)
    models = map_models(db, image_root, work / "sparse", focal_px, log=log)
    return dict(database=db, sparse=work / "sparse", models=models)


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


def fit_ground_plane(rec, max_error: float = 2.0, min_track: int = 3):
    """Mặt phẳng đất từ đám mây điểm thưa: SVD trên các điểm tin được.
    Trả về (điểm gốc P0, pháp tuyến n hướng về phía camera, hai vector cơ sở)."""
    pts = np.array([p.xyz for p in rec.points3D.values()
                    if p.error <= max_error and len(p.track.elements) >= min_track])
    if len(pts) < 20:
        pts = np.array([p.xyz for p in rec.points3D.values()])
    if len(pts) < 3:
        return None
    p0 = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - p0, full_matrices=False)
    n = vt[2]
    centers = np.array([rec.images[i].projection_center() for i in rec.reg_image_ids()])
    if np.dot(centers.mean(0) - p0, n) < 0:
        n = -n
    e1 = vt[0]
    e2 = np.cross(n, e1)
    return p0, n, e1, e2


def footprint(rec, image_id: int, plane) -> np.ndarray | None:
    """Bốn góc khung ảnh chiếu xuống mặt phẳng đất, toạ độ 2D trong mặt phẳng.
    None nếu một góc không cắt đất phía trước camera (ảnh nghiêng quá), hoặc
    không khử méo được góc đó (méo ước quá lớn — chỉ xảy ra khi f tự do)."""
    p0, n, e1, e2 = plane
    im = rec.images[image_id]
    cam = im.camera
    c = np.asarray(im.projection_center())
    r_wc = np.asarray(im.cam_from_world().rotation.matrix()).T   # world_from_cam
    w, h = cam.width, cam.height
    out = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        ray = cam.cam_ray_from_img([float(u), float(v)])
        if ray is None:
            return None
        ray = r_wc @ np.asarray(ray)
        denom = float(np.dot(n, ray))
        if abs(denom) < 1e-9:
            return None
        s = float(np.dot(n, p0 - c)) / denom
        if s <= 0:
            return None
        x = c + s * ray
        out.append([np.dot(x - p0, e1), np.dot(x - p0, e2)])
    return np.float32(out)


def quad_overlap(qa: np.ndarray, qb: np.ndarray) -> tuple[float, float]:
    """(phần a nằm trong b, phần b nằm trong a). Hai tứ giác lồi."""
    inter, _ = cv2.intersectConvexConvex(qa.reshape(-1, 1, 2), qb.reshape(-1, 1, 2))
    aa, ab = abs(cv2.contourArea(qa)), abs(cv2.contourArea(qb))
    return (min(inter / aa, 1.0) if aa > 0 else 0.0, min(inter / ab, 1.0) if ab > 0 else 0.0)


def model_pairs(rec) -> dict[tuple[int, int], dict]:
    """Với mọi cặp ảnh đã đăng ký trong một mô hình: số điểm 3D chung và
    chồng lấp vết phủ trên mặt đất."""
    ids = list(rec.reg_image_ids())
    seen = {i: {p.point3D_id for p in rec.images[i].points2D if p.has_point3D()} for i in ids}
    plane = fit_ground_plane(rec)
    quads = {i: footprint(rec, i, plane) for i in ids} if plane is not None else {}
    out = {}
    for a, b in itertools.combinations(sorted(ids), 2):
        shared = len(seen[a] & seen[b])
        row = dict(shared_points=shared,
                   shared_ratio=shared / max(1, min(len(seen[a]), len(seen[b]))))
        qa, qb = quads.get(a), quads.get(b)
        if qa is not None and qb is not None:
            row["footprint_ab"], row["footprint_ba"] = quad_overlap(qa, qb)
        out[(a, b)] = row
    return out


def model_summary(rec) -> dict:
    cam = next(iter(rec.cameras.values()))
    return dict(
        registered_images=int(rec.num_reg_images()),
        points3D=int(rec.num_points3D()),
        mean_track_length=float(rec.compute_mean_track_length()),
        mean_reprojection_error_px=float(rec.compute_mean_reprojection_error()),
        mean_observations_per_image=float(rec.compute_mean_observations_per_reg_image()),
        focal_px=float(cam.params[0]),
        camera_params=[float(x) for x in cam.params],
    )
