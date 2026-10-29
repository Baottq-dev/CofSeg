"""Đích mặt nạ mềm cho phần chắt lọc: ghép dự đoán của các thầy với tán nhãn, rồi lấy trung bình mặt nạ.

Chạy từ GỐC REPO sau teacher_infer.py (chỉ cần numpy, opencv, pycocotools; chạy được ở máy nhà):

    python distill/teacher_masks.py --data data/export/field/f1 --evals runs/distill_teachers/eval --out runs/distill_teachers/targets/mean3

Với mỗi ảnh train và mỗi thầy:

1. Tán nhãn là đa giác trong labels/train/<ảnh>.txt, đúng thứ ultralytics đọc lúc train, theo đúng thứ tự dòng.
2. IoU mặt nạ giữa dự đoán của thầy (predictions.json, độ phân giải gốc) và từng tán nhãn, bằng pycocotools.
3. Ghép một-một theo IoU giảm dần, chỉ nhận cặp IoU ≥ 0,5. Tán nhãn không chồng nhau nên với ngưỡng này
   mỗi dự đoán khớp nhiều nhất một tán, và phép ghép tham lam cho cùng kết quả với Hungarian.
4. Dự đoán không ghép được với tán nhãn nào bị bỏ: dạy theo chúng tương đương thêm nhãn.

Đích của tán j (`--mode mean`, mặc định) là trung bình mặt nạ của các thầy ghép được với j, mỗi mặt nạ thu về
cỡ ảnh train (cạnh dài `--imgsz`, như load_image của ultralytics) bằng INTER_AREA, nên giá trị ở mép là tỉ lệ
phủ. Thầy cho mặt nạ nhị phân (bộ chấm của benchmark), nên trong lòng tán, đích là tỉ lệ thầy đồng ý. Tán không
thầy nào ghép được thì không có đích thầy (lúc train chỉ học nhãn). `--mode best` lấy mặt nạ của thầy khớp nhãn
nhất; `--teachers yolo26s` cho bản một thầy (bảng ablation).

Đầu ra, ở <out>/<fold>/:

- maps/<ảnh>.png: ảnh 4 kênh uint8 cỡ ảnh train. Mỗi điểm giữ hai tán có đích lớn nhất ở điểm đó: (mã tán 1,
  giá trị 1, mã tán 2, giá trị 2), giá trị là round(255·đích). Mã tán là số dòng trong file nhãn; 255 là không
  có tán. Lúc train ảnh này đi qua đúng các phép co giãn, dịch, lật của ảnh (kd_data.py). Giữ hai tán là đủ:
  tán nhãn không chồng nhau, chỉ ở chỗ hai tán chạm nhau mới có hai đích khác 0.
- meta.json: fold, mã băm bản xuất, cỡ ảnh, các thầy và lượt chấm đã dùng, và cho từng ảnh: số tán, hộp của
  từng tán (để kd_data.py đối chiếu với nhãn ultralytics đã đọc), tán nào có đích thầy, IoU của từng thầy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.teachers import Teacher, find_eval, fold_info, parse_teachers  # noqa: E402

NONE = 255          # mã "không có tán" trong ảnh đích; khớp viền 255 mà ultralytics điền khi warp semantic_mask
MATCH_IOU = 0.5
MODES = ("mean", "best")
VERSION = 1


def work_shape(h0: int, w0: int, imgsz: int) -> tuple[int, int]:
    """(h, w) của ảnh sau load_image của ultralytics (rect_mode, cạnh dài về imgsz)."""
    r = imgsz / max(h0, w0)
    if r == 1:
        return h0, w0
    return min(math.ceil(h0 * r), imgsz), min(math.ceil(w0 * r), imgsz)


def read_label(path: Path) -> list[np.ndarray]:
    """Đa giác chuẩn hoá (k, 2) float32 của từng dòng, theo thứ tự dòng (như verify_image_label)."""
    if not path.exists():
        return []
    rows = [x.split() for x in path.read_text(encoding="utf-8").strip().splitlines() if len(x)]
    return [np.array(r[1:], dtype=np.float32).reshape(-1, 2) for r in rows]


def label_boxes(polys: list[np.ndarray]) -> np.ndarray:
    """Hộp xywh chuẩn hoá của từng đa giác, như segments2boxes của ultralytics."""
    if not polys:
        return np.zeros((0, 4), np.float32)
    b = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys], np.float32)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def gt_rles(polys: list[np.ndarray], h0: int, w0: int) -> list:
    from pycocotools import mask as mu

    out = []
    for p in polys:
        xy = (p * np.array([w0, h0], np.float32)).astype(np.float64).ravel().tolist()
        out.append(mu.merge(mu.frPyObjects([xy], h0, w0)) if len(xy) >= 6 else mu.encode(
            np.zeros((h0, w0), np.uint8, order="F")))
    return out


def match(iou: np.ndarray, thr: float = MATCH_IOU) -> dict[int, tuple[int, float]]:
    """Ghép một-một theo IoU giảm dần; iou có dạng (n_dự_đoán, n_nhãn). Trả {tán nhãn: (dự đoán, IoU)}."""
    if iou.size == 0:
        return {}
    d, g = np.nonzero(iou >= thr)
    order = np.argsort(-iou[d, g], kind="stable")
    used_d, out = set(), {}
    for k in order:
        di, gi = int(d[k]), int(g[k])
        if di in used_d or gi in out:
            continue
        used_d.add(di)
        out[gi] = (di, float(iou[di, gi]))
    return out


def coverage(rle, hw: tuple[int, int]) -> np.ndarray:
    """Mặt nạ thầy ở cỡ ảnh train, giá trị là tỉ lệ phủ trong [0, 1]."""
    from pycocotools import mask as mu

    m = mu.decode(rle).astype(np.float32)
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def top2_map(softs: dict[int, np.ndarray], hw: tuple[int, int]) -> np.ndarray:
    """Ảnh 4 kênh (mã 1, giá trị 1, mã 2, giá trị 2) từ {mã tán: đích (h, w)}."""
    h, w = hw
    v1 = np.zeros((h, w), np.float32)
    v2 = np.zeros((h, w), np.float32)
    i1 = np.full((h, w), NONE, np.uint8)
    i2 = np.full((h, w), NONE, np.uint8)
    for j, s in softs.items():
        ys, xs = np.nonzero(s > 0)
        if not len(ys):
            continue
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        sv, a1, a2, b1, b2 = s[sl], v1[sl], v2[sl], i1[sl], i2[sl]
        up1 = sv > a1
        up2 = ~up1 & (sv > a2)
        a2[up1], b2[up1] = a1[up1], b1[up1]
        a2[up2], b2[up2] = sv[up2], j
        a1[up1], b1[up1] = sv[up1], j
    q1 = np.rint(v1 * 255).astype(np.uint8)
    q2 = np.rint(v2 * 255).astype(np.uint8)
    i1[q1 == 0] = NONE
    i2[q2 == 0] = NONE
    return np.stack([i1, q1, i2, q2], -1)


def load_eval(d: Path, teacher: Teacher, fold: str, allow_partial: bool) -> tuple[dict, dict]:
    """(dự đoán theo image_id, metrics) của một lượt chấm train; dừng nếu lượt chấm không đúng thứ cần."""
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    data, model = m.get("data", {}), m.get("model", {})
    root = str(data.get("root", "")).replace("\\", "/").rstrip("/")
    w = str(model.get("weights", "")).replace("\\", "/")
    problems = []
    if data.get("split") != "train":
        problems.append(f"split {data.get('split')!r}")
    if Path(root).name != fold:
        problems.append(f"thư mục dữ liệu {root!r} không phải fold {fold}")
    if f"_{teacher.run_name}_field-{fold}_" not in w:
        problems.append(f"trọng số {w!r} không phải {teacher.run_name} fold {fold}")
    if data.get("limit") is not None and not allow_partial:
        problems.append(f"chỉ chấm {data.get('limit')} ảnh (--limit); thêm --allow-partial nếu đó là ý bạn")
    if problems:
        raise SystemExit(f"Lượt chấm {d} không dùng được làm đích thầy: " + "; ".join(problems))
    by_img: dict[int, list] = {}
    for p in json.loads((d / "predictions.json").read_text(encoding="utf-8")):
        by_img.setdefault(int(p["image_id"]), []).append(p)
    return by_img, m


def build(data: str, evals: str, out: str, teachers=None, mode: str = "mean", imgsz: int = 1024,
          allow_partial: bool = False, log=print) -> dict:
    if mode not in MODES:
        raise SystemExit(f"--mode phải là một trong {', '.join(MODES)}")
    from pycocotools import mask as mu

    t0 = time.time()
    root = Path(data)
    info = fold_info(root)
    fold = info["fold"]
    tlist = parse_teachers(teachers)
    preds, used = {}, {}
    for t in tlist:
        d = find_eval(evals, t, fold)
        if d is None:
            raise SystemExit(f"Không thấy lượt chấm train của {t.run_name} fold {fold} trong {evals}: "
                             "chạy distill/teacher_infer.py trước.")
        preds[t.key], _ = load_eval(d, t, fold, allow_partial)
        used[t.key] = d.as_posix()
    ann = json.loads((root / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    by_name = {im["file_name"]: im for im in ann["images"]}
    images = sorted(p for p in (root / "images" / "train").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    dst = Path(out) / fold
    (dst / "maps").mkdir(parents=True, exist_ok=True)

    meta_imgs: dict = {}
    n_gt = 0
    n_match = {t.key: 0 for t in tlist}
    iou_sum = {t.key: 0.0 for t in tlist}
    n_by_count = [0] * (len(tlist) + 1)
    for path in images:
        im = by_name.get(path.name)
        if im is None:
            raise SystemExit(f"{path.name} không có trong instances_train.json")
        h0, w0 = int(im["height"]), int(im["width"])
        hw = work_shape(h0, w0, imgsz)
        polys = read_label(root / "labels" / "train" / f"{path.stem}.txt")
        if len(polys) >= NONE:
            raise SystemExit(f"{path.name}: {len(polys)} tán, mã tán uint8 chỉ chứa được {NONE - 1}")
        gts = gt_rles(polys, h0, w0)
        matched: dict[int, list] = {j: [] for j in range(len(polys))}
        rec = {"n": len(polys), "boxes": np.round(label_boxes(polys), 6).tolist(), "iou": {}}
        for t in tlist:
            ps = preds[t.key].get(int(im["id"]), [])
            ious = np.zeros((len(ps), len(gts)))
            if ps and gts:
                ious = np.asarray(mu.iou([p["segmentation"] for p in ps], gts, [0] * len(gts)))
            pairs = match(ious)
            rec["iou"][t.key] = [round(pairs[j][1], 4) if j in pairs else -1 for j in range(len(polys))]
            for j, (di, v) in pairs.items():
                matched[j].append((v, ps[di]["segmentation"]))
                n_match[t.key] += 1
                iou_sum[t.key] += v
        softs = {}
        for j, ms in matched.items():
            n_by_count[len(ms)] += 1
            if not ms:
                continue
            if mode == "best":
                ms = [max(ms, key=lambda x: x[0])]
            softs[j] = sum(coverage(r, hw) for _, r in ms) / len(ms)
        rec["has"] = [int(j in softs) for j in range(len(polys))]
        n_gt += len(polys)
        meta_imgs[path.stem] = rec
        cv2.imwrite(str(dst / "maps" / f"{path.stem}.png"), top2_map(softs, hw))

    summary = {
        "images": len(images), "gt": n_gt,
        "match_rate": {k: round(n_match[k] / max(n_gt, 1), 4) for k in n_match},
        "mean_iou": {k: round(iou_sum[k] / max(n_match[k], 1), 4) for k in n_match},
        "gt_by_teacher_count": n_by_count,
        "seconds": round(time.time() - t0, 1),
    }
    meta = {"version": VERSION, **info, "imgsz": imgsz, "mode": mode, "teachers": [t.key for t in tlist],
            "match_iou": MATCH_IOU, "evals": used, "summary": summary, "images": meta_imgs}
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log(f"Đích thầy fold {fold}: {len(images)} ảnh, {n_gt} tán nhãn, chế độ {mode}, thầy {'+'.join(meta['teachers'])}")
    for k in n_match:
        log(f"  {k}: ghép {n_match[k]}/{n_gt} tán ({100 * summary['match_rate'][k]:.1f}%), IoU trung bình "
            f"{summary['mean_iou'][k]:.4f}")
    log("  số tán theo số thầy ghép được (0.." + str(len(tlist)) + "): " + ", ".join(map(str, n_by_count)))
    log(f"  ghi {dst} trong {summary['seconds']:.0f} s")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="thư mục fold, vd data/export/field/f1")
    ap.add_argument("--evals", required=True, help="thư mục chứa lượt chấm train của thầy (<runs>/eval)")
    ap.add_argument("--out", required=True, help="thư mục đích; ghi vào <out>/<fold>/")
    ap.add_argument("--teachers", default=None, help="tập con của yolo26s,solov2,m2f (mặc định cả ba)")
    ap.add_argument("--mode", default="mean", choices=MODES)
    ap.add_argument("--imgsz", type=int, default=1024, help="cạnh dài ảnh lúc train (phải trùng imgsz của trò)")
    ap.add_argument("--allow-partial", action="store_true", help="nhận lượt chấm có --limit (chỉ để thử)")
    a = ap.parse_args(argv)
    build(a.data, a.evals, a.out, a.teachers, a.mode, a.imgsz, a.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
