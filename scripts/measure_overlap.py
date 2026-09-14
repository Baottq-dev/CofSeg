"""Đo chồng lấp giữa các frame của một ruộng, để biết ảnh có dùng được cho
SfM / DSM / đa góc nhìn hay không, và để tìm ảnh chụp lặp cần khử trước khi
chia tập.

    python scripts/measure_overlap.py --field field_2
    python scripts/measure_overlap.py --field field_2 --pairs all
    python scripts/measure_overlap.py --field field_1 --pairs all --scope field
    python scripts/measure_overlap.py --field field_6 --limit 5 --viz 3
    python scripts/measure_overlap.py --field field_6 --method colmap
    python scripts/measure_overlap.py --field field_6 --method colmap --reuse runs/overlap/<lần-cũ>

Ba phương pháp:

    --method sift | orb   khớp đặc trưng trên ảnh đã co, kiểm bằng ma trận cơ
                          bản F, ước vết phủ bằng homography. Nhanh.
    --method colmap       SfM đầy đủ (pycolmap): SIFT độ phân giải gốc, kiểm
                          hình học F/E/H cho mọi cặp, rồi dựng mô hình 3D.
                          Chậm hơn (~1,5 s/ảnh + ~0,1 s/cặp trên CPU) nhưng
                          là cùng chuỗi mà Metashape/ODM chạy, và cho bốn tầng
                          bằng chứng thay vì một (xem canopyseg/datasets/sfm.py).
                          Tiêu cự cố định qua --focal-px vì ảnh không còn EXIF.
                          --reuse RUN_DIR lấy lại đặc trưng + match của lần
                          chạy cũ (phần chậm) và chỉ làm lại phần sau, vài giây.
                          Cài: pip install pycolmap

Hai chế độ cặp:

    --pairs consecutive   chỉ so frame i với i+1 (mặc định của sift/orb;
                          với colmap là ghép tuần tự i..i+3)
    --pairs all           so mọi cặp trong cùng lần bay (mặc định của colmap),
                          hoặc cả ruộng với --scope field — bắt được cả
                          overlap ngang giữa hai đường bay và ảnh chụp lặp

Mỗi lần chạy ghi vào runs/overlap/<thời-điểm>_<ruộng>_<chế-độ>/:

    pairs.csv      một dòng mỗi cặp: số match, số inlier, overlap hai chiều,
                   trạng thái — con số kèm bằng chứng
    summary.json   tổng hợp theo lần bay
    viz/           (sift/orb, với --viz K) ảnh ghép của K cặp chồng lấp nhất
    colmap/        (colmap) database.db và sparse/<k>/ cho từng mô hình 3D:
                   cameras/images/points3D.txt + points.ply (mở bằng MeshLab
                   hay CloudCompare để nhìn tận mắt)
    run.log        toàn bộ màn hình

Cách đọc kết quả sift/orb: cặp `ok` là chồng lấp thật kèm ước lượng phần
chồng; `verified` là chồng lấp thật nhưng không ước được phần chồng (H không
lành). `few_matches` / `no_geometry` là KHÔNG TÌM THẤY quan hệ — thường là
không chồng lấp, nhưng tán lặp và bóng đổ khác chiều cũng gây ra như vậy.

Cách đọc kết quả colmap: `status` xếp theo tầng bằng chứng, `overlap_from`
nói con số overlap đến từ đâu.

    footprint     cùng mô hình 3D, overlap từ tư thế camera + mặt đất
    registered    cùng mô hình 3D nhưng không chiếu được vết phủ
    homography    không vào mô hình; overlap từ homography trên inlier COLMAP
                  (đã kiểm: lệch so với footprint trung vị 0,00, MAD 0,02)
    verified      có quan hệ hình học nhưng H không lành, không ước được
    none          không tìm thấy quan hệ — trên dữ liệu này 99% cặp không
                  chồng lấp có đúng 0 inlier, nên `none` gần như là "không"

Bảng "cặp liền kề" là câu trả lời chính: chồng lấp dọc giữa hai lần bấm máy
liên tiếp. Ảnh có overlap mapping (70–80%) cho MỘT mô hình chứa gần hết ảnh;
overlap ~45% cho từng cặp khớp chắc nhưng mô hình rời rạc, vì ba khung liên
tiếp không còn điểm chung để mapper nối chuỗi.

Dù phương pháp nào, kết luận "không overlap" vẫn nên kèm một lập luận độc
lập (nhịp chụp × tốc độ bay so với bề rộng khung), hoặc GPS từ ảnh gốc.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import artifacts  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg import runlog  # noqa: E402
from canopyseg.datasets import overlap as ov  # noqa: E402
from canopyseg.datasets import sfm  # noqa: E402

console.setup()

# DJI_YYYYMMDDHHMMSS_NNNN_D.jpg → (thời điểm, số thứ tự)
_DJI = re.compile(r"DJI_(\d{14})_(\d{4})")


def frame_info(path: Path) -> tuple[datetime | None, int | None]:
    m = _DJI.search(path.name)
    if not m:
        return None, None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), int(m.group(2))


def list_flights(field_dir: Path, limit: int | None) -> dict[str, list[Path]]:
    """Mỗi thư mục lá chứa ảnh là một lần bay. Ảnh sắp theo số thứ tự DJI."""
    flights: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(field_dir.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            flights[p.parent.relative_to(field_dir).as_posix() or "."].append(p)
    out = {}
    for k, files in flights.items():
        files.sort(key=lambda p: (frame_info(p)[1] if frame_info(p)[1] is not None else 0, p.name))
        out[k] = files[:limit] if limit else files
    return out


def gap_seconds(a: Path, b: Path) -> float | None:
    ta, tb = frame_info(a)[0], frame_info(b)[0]
    return abs((tb - ta).total_seconds()) if ta and tb else None


def draw_pair(img_a, img_b, pts_a, pts_b, desc_a, desc_b, matcher, out_path: Path, title: str):
    idx = ov.match(matcher, desc_a, desc_b)
    kp_a = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts_a]
    kp_b = [cv2.KeyPoint(float(x), float(y), 1) for x, y in pts_b]
    dm = [cv2.DMatch(int(i), int(j), 0.0) for i, j in idx[:300]]
    vis = cv2.drawMatches(img_a, kp_a, img_b, kp_b, dm, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), vis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/iachim_dataset_export/data_compressed")
    ap.add_argument("--field", required=True, help="tên thư mục ruộng, vd field_2")
    ap.add_argument("--pairs", choices=["consecutive", "all"], default=None,
                    help="mặc định: consecutive với sift/orb, all với colmap")
    ap.add_argument("--scope", choices=["flight", "field"], default="flight",
                    help="với --pairs all: so trong từng lần bay, hay mọi cặp của cả ruộng")
    ap.add_argument("--method", choices=["sift", "orb", "colmap"], default="sift")
    ap.add_argument("--features", type=int, default=8192,
                    help="số đặc trưng mỗi ảnh (sift/orb: trên ảnh đã co; colmap: trên ảnh gốc)")
    ap.add_argument("--sfm-max-size", type=int, default=-1,
                    help="colmap: co ảnh về cạnh dài này trước khi trích (-1 = giữ nguyên)")
    ap.add_argument("--focal-px", type=float, default=1707.0,
                    help="colmap: tiêu cự cố định (px); 0 = để COLMAP tự ước (chỉ khi ảnh còn EXIF). "
                         "Mặc định 1707 = ống 24 mm tương đương trên khung rộng 2560 px")
    ap.add_argument("--sfm-min-inliers", type=int, default=15,
                    help="colmap: số inlier tối thiểu để coi một cặp là có quan hệ")
    ap.add_argument("--reuse", default=None, metavar="RUN_DIR",
                    help="colmap: lấy database.db (đặc trưng + match, phần tốn 20-40 phút) từ lần chạy cũ; "
                         "chỉ dựng lại mô hình và tính lại bảng cặp")
    ap.add_argument("--scale", type=float, default=0.5, help="co ảnh trước khi tìm đặc trưng")
    ap.add_argument("--ratio", type=float, default=0.8, help="ratio test của Lowe")
    ap.add_argument("--ransac-px", type=float, default=3.0, help="sift/orb: ngưỡng RANSAC cho ma trận cơ bản, px trên ảnh đã co")
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N ảnh đầu mỗi lần bay (chạy thử)")
    ap.add_argument("--viz", type=int, default=0, help="lưu ảnh ghép của K cặp chồng lấp nhất")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    if a.pairs is None:
        a.pairs = "all" if a.method == "colmap" else "consecutive"
    if a.method == "colmap":
        sfm.require_pycolmap()

    field_dir = Path(a.root) / a.field
    if not field_dir.is_dir():
        raise SystemExit(f"Không thấy {field_dir}")
    flights = list_flights(field_dir, a.limit)
    n_img = sum(len(v) for v in flights.values())
    if n_img < 2:
        raise SystemExit(f"{field_dir} có {n_img} ảnh, không có cặp nào để so.")

    tag = f"{a.pairs}-{a.scope}_{a.method}" if a.pairs == "all" else f"{a.pairs}_{a.method}"
    run_dir = artifacts.create_run_dir(a.runs, "overlap", a.field, tag)
    artifacts.write_env(run_dir)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Lần đo: {run_dir}")

    with runlog.capture(run_dir / "run.log"):
        if a.method == "colmap":
            return run_sfm(a, field_dir, flights, run_dir)
        return run(a, flights, run_dir)


def run_sfm(a, field_dir: Path, flights: dict[str, list[Path]], run_dir: Path) -> int:
    """SfM đầy đủ cho từng lần bay (hoặc cả ruộng), rồi quy về cùng bảng cặp."""
    all_files = [p for v in flights.values() for p in v]
    print(f"{a.field}: {len(flights)} lần bay, {len(all_files)} ảnh — COLMAP, "
          f"{'ghép mọi cặp' if a.pairs == 'all' else 'ghép tuần tự'}, scope={a.scope}, "
          f"f {'cố định %.0f px' % a.focal_px if a.focal_px else 'tự do'}")
    groups = {"field": all_files} if a.scope == "field" else flights
    reuse = Path(a.reuse) if a.reuse else None
    rows, summary = [], {}
    for g, files in groups.items():
        if len(files) < 2:
            continue
        names = [f.relative_to(field_dir).as_posix() for f in files]
        by_name = {n: f for n, f in zip(names, files)}
        print(f"[{g}] {len(files)} ảnh")
        work = run_dir / "colmap" / artifacts.slugify(g)
        work.mkdir(parents=True, exist_ok=True)
        db = work / "database.db"
        log = lambda s: print(s, flush=True)  # noqa: E731
        if reuse:
            src = reuse / "colmap" / artifacts.slugify(g) / "database.db"
            if not src.exists():
                raise SystemExit(f"--reuse: không thấy {src} — lần chạy cũ phải cùng --field, --scope, --limit")
            shutil.copy(src, db)
            print(f"  dùng lại đặc trưng + match từ {src}")
        else:
            sfm.extract_and_match(field_dir, names, db,
                                  pairing="exhaustive" if a.pairs == "all" else "sequential",
                                  max_features=a.features, max_image_size=a.sfm_max_size, log=log)
        id_name, pairs = sfm.read_pairs(db)
        if {Path(n).as_posix() for n in id_name.values()} != set(names):
            raise SystemExit(f"Ảnh trong {db} không khớp với {field_dir}/{g}; bỏ --reuse hoặc chọn đúng lần chạy.")
        models_by_k = sfm.map_models(db, field_dir, work / "sparse", focal_px=a.focal_px or None, log=log)
        hov = sfm.homography_overlaps(db, min_inliers=a.sfm_min_inliers)

        # Ảnh nào nằm trong mô hình nào, và các cặp đã đăng ký cùng mô hình.
        model_of: dict[int, int] = {}
        mpairs: dict[tuple[int, int], dict] = {}
        models = []
        for k, rec in sorted(models_by_k.items()):
            for i in rec.reg_image_ids():
                model_of[i] = k
            mpairs.update({key: dict(model=k, **v) for key, v in sfm.model_pairs(rec).items()})
            ms = sfm.model_summary(rec)
            ms["images"] = sorted(rec.images[i].name for i in rec.reg_image_ids())
            models.append(ms)
            print(f"  mô hình {k}: {ms['registered_images']} ảnh, {ms['points3D']} điểm, "
                  f"track {ms['mean_track_length']:.1f}, sai số chiếu {ms['mean_reprojection_error_px']:.2f} px, "
                  f"f {ms['focal_px']:.0f} px")

        ids = sorted(id_name)
        for i, j in itertools.combinations(ids, 2):
            p = pairs.get((i, j), {})
            m = mpairs.get((i, j), {})
            h = hov.get((i, j))
            inl = p.get("verified_inliers", 0)
            if "footprint_ab" in m:
                status, src, ab, ba = "footprint", "footprint", m["footprint_ab"], m["footprint_ba"]
            elif m:
                status, src, ab, ba = "registered", "", 0.0, 0.0
            elif h is not None:
                status, src, ab, ba = "homography", "homography", h[0], h[1]
            elif inl >= a.sfm_min_inliers:
                status, src, ab, ba = "verified", "", 0.0, 0.0
            else:
                status, src, ab, ba = "none", "", 0.0, 0.0
            if status == "registered" and h is not None:   # cùng mô hình nhưng thiếu vết phủ: lấy H
                src, ab, ba = "homography", h[0], h[1]
            fa, fb = by_name[id_name[i]], by_name[id_name[j]]
            rows.append(dict(
                flight=g, a=fa.name, b=fb.name, gap_s=gap_seconds(fa, fb),
                seq_gap=abs((frame_info(fa)[1] or 0) - (frame_info(fb)[1] or 0)),
                raw_matches=p.get("raw_matches", 0), verified_inliers=inl,
                geometry=p.get("geometry", ""), model=m.get("model", ""),
                shared_points=m.get("shared_points", 0),
                shared_ratio=round(m.get("shared_ratio", 0.0), 4),
                overlap_ab=round(ab, 4), overlap_ba=round(ba, 4), overlap_from=src,
                status=status,
            ))
        rs = [r for r in rows if r["flight"] == g]
        o = np.array([max(r["overlap_ab"], r["overlap_ba"]) for r in rs] or [0.0])
        consec = {(x.name, y.name) for x, y in zip(files[:-1], files[1:])}
        cr = [r for r in rs if (r["a"], r["b"]) in consec or (r["b"], r["a"]) in consec]
        co = np.array([max(r["overlap_ab"], r["overlap_ba"]) for r in cr] or [0.0])
        with_nb = {x for r in rs if max(r["overlap_ab"], r["overlap_ba"]) >= 0.3 for x in (r["a"], r["b"])}
        summary[g] = dict(
            images=len(files), pairs=len(rs),
            n_verified=sum(r["verified_inliers"] >= a.sfm_min_inliers for r in rs),
            n_homography=sum(r["overlap_from"] == "homography" for r in rs),
            n_registered_pairs=sum(r["status"] in ("registered", "footprint") for r in rs),
            n_over_30=int((o >= 0.3).sum()), n_over_50=int((o >= 0.5).sum()),
            max_overlap=float(o.max()),
            consecutive_pairs=len(cr),
            consecutive_over_30=int((co >= 0.3).sum()), consecutive_over_50=int((co >= 0.5).sum()),
            consecutive_median=float(np.median(co)),
            consecutive_median_overlapping=float(np.median(co[co >= 0.3])) if (co >= 0.3).any() else 0.0,
            images_with_neighbour_30=len(with_nb),
            images_in_any_model=len(model_of), models=models,
        )

    if not rows:
        print("Không có cặp nào.")
        return 1
    with open(run_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"{'lần bay':<14} {'ảnh':>5} {'cặp':>6} {'verified':>9} {'H-ước':>6} {'cùng mô hình':>13} "
          f"{'≥30%':>6} {'≥50%':>6} {'max':>6}  ảnh có tư thế / mô hình")
    for g, s in summary.items():
        print(f"{g:<14} {s['images']:>5} {s['pairs']:>6} {s['n_verified']:>9} {s['n_homography']:>6} "
              f"{s['n_registered_pairs']:>13} {s['n_over_30']:>6} {s['n_over_50']:>6} {s['max_overlap']:>6.0%}  "
              f"{s['images_in_any_model']}/{s['images']} trong {len(s['models'])} mô hình")

    print()
    print(f"Cặp liền kề theo thứ tự chụp (chồng lấp dọc):")
    print(f"{'lần bay':<14} {'cặp':>6} {'≥30%':>6} {'≥50%':>6} {'tv mọi cặp':>11} {'tv cặp ≥30%':>12}  ảnh có ảnh khác chồng ≥30%")
    for g, s in summary.items():
        print(f"{g:<14} {s['consecutive_pairs']:>6} {s['consecutive_over_30']:>6} {s['consecutive_over_50']:>6} "
              f"{s['consecutive_median']:>11.0%} {s['consecutive_median_overlapping']:>12.0%}  "
              f"{s['images_with_neighbour_30']}/{s['images']}")

    far = sorted((r for r in rows if r["seq_gap"] >= 4 and max(r["overlap_ab"], r["overlap_ba"]) >= 0.5),
                 key=lambda r: -max(r["overlap_ab"], r["overlap_ba"]))
    print(f"\nCặp cách nhau ≥4 khung mà chồng ≥50% (bay lại / đường bay kề): {len(far)}")
    for r in far[:20]:
        print(f"  {max(r['overlap_ab'], r['overlap_ba']):>4.0%}  {r['verified_inliers']:>5} inlier  "
              f"{r['gap_s']:>4.0f}s  {r['a']}  ↔  {r['b']}")
    if len(far) > 20:
        print(f"  ... và {len(far) - 20} cặp nữa trong pairs.csv")
    ver_only = [r for r in rows if r["status"] == "verified"]
    if ver_only:
        print(f"\nCặp qua kiểm hình học nhưng H không lành, không ước được phần chồng: {len(ver_only)} "
              f"(inlier trung vị {int(np.median([r['verified_inliers'] for r in ver_only]))}) — xem pairs.csv")

    (run_dir / "summary.json").write_text(
        json.dumps(dict(field=a.field, pairs_mode=a.pairs, scope=a.scope, method="colmap",
                        focal_px=a.focal_px, reuse=str(reuse) if reuse else None,
                        n_images=len(all_files), n_pairs=len(rows), flights=summary),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nKết quả: {run_dir}")
    return 0


def run(a, flights: dict[str, list[Path]], run_dir: Path) -> int:
    det, matcher = ov.make_detector(a.method, a.features), ov.make_matcher(a.method)
    all_files = [p for v in flights.values() for p in v]
    print(f"{a.field}: {len(flights)} lần bay, {len(all_files)} ảnh")
    for k, v in flights.items():
        print(f"  {k}: {len(v)} ảnh")

    # Tự kiểm: ảnh với chính nó. Không ra ~100% thì tham số đang hỏng.
    first = ov.load_gray(all_files[0], a.scale)
    sc = ov.self_check(det, matcher, first)
    print(f"Tự kiểm (ảnh với chính nó): {sc.n_inliers} inlier, overlap {sc.overlap_ab:.0%} → "
          f"{'OK' if sc.overlap_ab > 0.98 else 'HỎNG — dừng lại'}")
    if sc.overlap_ab <= 0.98:
        return 1

    # Đặc trưng tính một lần mỗi ảnh, cache trong RAM (4000 × 128 float32 ≈ 2 MB/ảnh).
    t0 = time.time()
    feats: dict[Path, tuple] = {}
    imgs: dict[Path, np.ndarray] = {}
    for i, p in enumerate(all_files, 1):
        im = ov.load_gray(p, a.scale)
        pts, desc = ov.extract(det, im)
        feats[p] = (pts, desc, im.shape)
        if a.viz:
            imgs[p] = im
        if i % 25 == 0 or i == len(all_files):
            print(f"  đặc trưng {i}/{len(all_files)} ({time.time() - t0:.0f}s)")

    # Danh sách cặp.
    if a.pairs == "consecutive":
        pairs = [(x, y, k) for k, v in flights.items() for x, y in zip(v[:-1], v[1:])]
    elif a.scope == "flight":
        pairs = [(x, y, k) for k, v in flights.items() for x, y in itertools.combinations(v, 2)]
    else:
        pairs = [(x, y, "field") for x, y in itertools.combinations(all_files, 2)]
    if not pairs:
        print("Không có cặp nào để so (mỗi lần bay chỉ có một ảnh?).")
        return 1
    print(f"So {len(pairs)} cặp ({a.pairs}, scope={a.scope}) ...")

    rows = []
    t0 = time.time()
    for i, (x, y, k) in enumerate(pairs, 1):
        pa, da, sa = feats[x]
        pb, db, sb = feats[y]
        r = ov.pair_overlap(pa, da, sa, pb, db, sb, matcher, a.ratio, a.ransac_px, a.min_inliers)
        rows.append(dict(flight=k, a=x.name, b=y.name, gap_s=gap_seconds(x, y), **r.as_dict()))
        if i % 200 == 0 or i == len(pairs):
            el = time.time() - t0
            print(f"  cặp {i}/{len(pairs)} ({el:.0f}s, còn ~{el / i * (len(pairs) - i):.0f}s)")

    with open(run_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Tổng hợp theo lần bay.
    summary = {}
    print()
    print(f"{'lần bay':<14} {'cặp':>6} {'inlier tv':>10} {'có quan hệ':>11} {'≥30%':>6} {'≥50%':>6} {'max':>6}  nhịp chụp tv")
    for k in sorted({r["flight"] for r in rows}):
        rs = [r for r in rows if r["flight"] == k]
        # `verified` có quan hệ nhưng không ước được phần chồng (nan) → tính là 0 ở đây.
        o = np.nan_to_num(np.array([max(r["overlap_ab"], r["overlap_ba"]) for r in rs], dtype=float))
        gaps = [r["gap_s"] for r in rs if r["gap_s"] is not None and a.pairs == "consecutive"]
        s = dict(
            pairs=len(rs),
            median_inliers=float(np.median([r["n_inliers"] for r in rs])),
            n_related=sum(r["status"] in ("ok", "verified") for r in rs),
            n_ok=sum(r["status"] == "ok" for r in rs),
            n_over_30=int((o >= 0.3).sum()),
            n_over_50=int((o >= 0.5).sum()),
            max_overlap=float(o.max()),
            median_gap_s=float(np.median(gaps)) if gaps else None,
        )
        summary[k] = s
        gap = f"{s['median_gap_s']:.0f}s" if s["median_gap_s"] is not None else "-"
        print(f"{k:<14} {s['pairs']:>6} {s['median_inliers']:>10.0f} {s['n_related']:>11} "
              f"{s['n_over_30']:>6} {s['n_over_50']:>6} {s['max_overlap']:>6.0%}  {gap}")

    dup = sorted((r for r in rows if r["status"] == "ok" and max(r["overlap_ab"], r["overlap_ba"]) >= 0.5),
                 key=lambda r: -max(r["overlap_ab"], r["overlap_ba"]))
    ver = [r for r in rows if r["status"] == "verified"]
    if ver:
        print(f"\nCặp có quan hệ hình học nhưng không ước được phần chồng: {len(ver)} — xem pairs.csv")
    print(f"\nCặp chồng lấp ≥ 50% (ứng viên ảnh chụp lặp): {len(dup)}")
    for r in dup[:30]:
        print(f"  {max(r['overlap_ab'], r['overlap_ba']):>4.0%}  {r['n_inliers']:>5} inlier  {r['a']}  ↔  {r['b']}")
    if len(dup) > 30:
        print(f"  ... và {len(dup) - 30} cặp nữa trong pairs.csv")

    (run_dir / "summary.json").write_text(
        json.dumps(dict(field=a.field, pairs_mode=a.pairs, scope=a.scope, method=a.method,
                        n_images=len(all_files), n_pairs=len(rows), flights=summary,
                        n_pairs_over_50=len(dup)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if a.viz:
        vdir = run_dir / "viz"
        vdir.mkdir()
        top = sorted((r for r in rows if r["status"] == "ok"),
                     key=lambda r: -max(r["overlap_ab"], r["overlap_ba"]))[: a.viz]
        # Kèm vài cặp KHÔNG khớp để mắt so được hai loại.
        none = [r for r in rows if r["status"] != "ok"][: min(3, a.viz)]
        by_name = {p.name: p for p in all_files}
        for j, r in enumerate(top + none):
            x, y = by_name[r["a"]], by_name[r["b"]]
            title = f"{r['status']} overlap={max(r['overlap_ab'], r['overlap_ba']):.0%} inliers={r['n_inliers']}"
            draw_pair(imgs[x], imgs[y], feats[x][0], feats[y][0], feats[x][1], feats[y][1],
                      matcher, vdir / f"{j:02d}_{r['status']}_{x.stem}__{y.stem}.jpg", title)
        print(f"Ảnh ghép: {vdir}")

    print(f"\nKết quả: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
