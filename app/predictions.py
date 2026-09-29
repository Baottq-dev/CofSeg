# app/predictions.py — Đọc dự đoán của các lượt chấm trong runs/eval/ để xem
# chồng lên ảnh trong annotator.
"""Lớp đọc dự đoán cho giao diện. CHỈ ĐỌC, và KHÔNG chép ra chỗ khác.

Vì sao đọc thẳng `runs/eval/` chứ không chép sang `data/`:

1. Không có bản sao thì không có bản sao đi cũ mà không ai biết. Đo trên bộ
   này, nạp + dựng chỉ mục một `predictions.json` mất 10-20 ms và trả dự đoán
   của một ảnh ở conf 0.5 mất 39-48 ms cho cả sáu model — rẻ tới mức chép
   trước chẳng đổi được gì.
2. `predictions.json` TRẦN không nói nó là của model nào, fold nào, trọng số
   nào, imgsz bao nhiêu. Những thứ đó nằm ở `config.yaml`, `env.json` và tên
   thư mục lượt chấm bên cạnh nó. Chép mỗi file json là vứt hết dấu vết, đúng
   loại mất mát mà `artifacts.check_imgsz` dựng riêng một hàng rào để chặn.
3. Repo này đã có một lần đầu ra model nằm trong `data/` rồi bị trích dẫn
   như thống kê của nhãn (`data/iachim_dataset_export/predictions/`). Để dự
   đoán ở `runs/` thì cái tên đã nói nó là kết quả của một lượt chạy.

Hai thứ `predictions.json` KHÔNG có, và đây là chỗ bù vào:

- Nó chỉ có `image_id` (số), không có tên ảnh. Bảng tra nằm trong
  `instances_<split>.json` của bộ xuất mà lượt chấm trỏ tới. May là số này
  toàn cục: đã đối chiếu cả ba bộ fold (field/block/flight), 850 ảnh, id chạy
  1..850, không id nào ứng với hai tên và không bộ nào lệch bộ nào.
- Nó là RLE của cả khung chứ không phải polygon. Chuyển mất 2,45 ms một dự
  đoán, trong đó 2,31 ms là `decode` — nên cắt cửa sổ bằng `toBbox` (0,03 ms)
  thay vì `np.nonzero` trên cả khung (9,21 ms) là đáng làm.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

#: Thư mục lượt chấm. Đổi được khi chạy annotator từ nơi khác.
RUNS_DIR = os.environ.get("EVAL_RUNS_DIR", "runs/eval")

#: Dung sai `approxPolyDP`, px ảnh. Đo trên f6: 1.0 giữ IoU 0.993-0.996 và
#: Boundary IoU 0.969-0.989 so với mặt nạ gốc, với 62-84 đỉnh mỗi tán — xấp xỉ
#: mật độ đỉnh của nhãn tay (trung vị 112). Để 2.0 (mặc định cũ của
#: `_mask_to_polygons` trong server.py) thì trung bình vẫn đẹp nhưng có tán
#: rơi xuống IoU 0.594; đây là chỗ hiển thị nên lấy bản trung thực hơn.
DEFAULT_EPS = 1.0

#: Cùng ngưỡng ghép mà bảng benchmark dùng (cofseg/evaluation/matching.py).
IOU_THR = 0.5

#: Số lượt chấm giữ sẵn trong RAM. Mỗi lượt tốn 1,5-6,1 MB nên bốn lượt là
#: ~25 MB, đủ để bấm qua lại giữa các model mà không phải nạp lại.
CACHE_RUNS = 4


@dataclass
class RunInfo:
    """Một lượt chấm trong runs/eval/, đọc từ config.yaml của chính nó."""

    run: str
    path: Path
    model: str
    dataset: str          # 'field' | 'block' | 'flight' | ''
    fold: str             # 'f6'
    split: str
    root: str             # thư mục bộ xuất đã chấm
    imgsz: int | None = None
    weights: str = ""
    conf: float | None = None

    def as_dict(self) -> dict:
        return {"run": self.run, "model": self.model, "dataset": self.dataset,
                "fold": self.fold, "split": self.split, "imgsz": self.imgsz,
                "weights": self.weights, "conf": self.conf}


@dataclass
class Index:
    runs: dict = field(default_factory=dict)        # run -> RunInfo
    by_image: dict = field(default_factory=dict)    # file_name -> [run, ...]
    size: dict = field(default_factory=dict)        # file_name -> (h, w)
    n_gt: dict = field(default_factory=dict)        # (run, file_name) -> số vùng lúc xuất
    stamp: tuple = ()
    error: str = ""


_index: Index | None = None
_preds: OrderedDict = OrderedDict()      # run -> {image_id: [det, ...]}

_CFG_ROOT = re.compile(r"^\s*root:\s*(\S+)\s*$", re.M)
_CFG_SPLIT = re.compile(r"^\s*split:\s*(\S+)\s*$", re.M)


def _stamp(base: Path) -> tuple:
    """Dấu vân tay của thư mục runs/eval: đủ để biết có lượt chấm mới hay không.

    Chỉ nhìn tên thư mục con — thêm/bớt một lượt là đổi, còn sửa bên trong một
    lượt đã có thì không. Dựng lại chỉ mục mất ~0,9 s nên không nên dò kỹ hơn
    ở mỗi lần bấm chuột.
    """
    if not base.is_dir():
        return ()
    return tuple(sorted(p.name for p in base.iterdir() if p.is_dir()))


def _fold_of(root: str) -> tuple[str, str]:
    """'data/export/field/f6' -> ('field', 'f6')."""
    parts = [x for x in Path(root).parts if x not in ("", ".", "..")]
    if not parts:
        return "", ""
    fold = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else ""
    return ("" if parent in ("", "export") else parent), fold


def _read_run(d: Path) -> RunInfo | None:
    cfg = d / "config.yaml"
    if not (cfg.is_file() and (d / "predictions.json").is_file()):
        return None
    try:
        txt = cfg.read_text(encoding="utf-8")
    except OSError:
        return None
    m_root, m_split = _CFG_ROOT.search(txt), _CFG_SPLIT.search(txt)
    if not m_root:
        return None
    root = m_root.group(1).strip().strip("'\"")
    split = (m_split.group(1).strip().strip("'\"") if m_split else "test")
    name = ""
    mn = re.search(r"^name:\s*(.+)$", txt, re.M)
    if mn:
        name = mn.group(1).strip().strip("'\"")
    ds, fold = _fold_of(root)
    info = RunInfo(run=d.name, path=d, model=name or d.name, dataset=ds,
                   fold=fold, split=split, root=root)
    # imgsz và trọng số nằm trong metrics.json (khối describe của model); thiếu
    # thì bỏ qua chứ không bỏ cả lượt chấm — chúng chỉ để hiện cho người xem.
    try:
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        mo = met.get("model") or {}
        info.imgsz = mo.get("imgsz")
        info.weights = str(mo.get("weights") or "")
        info.conf = mo.get("conf")
    except (OSError, ValueError):
        pass
    return info


def get_index(force: bool = False) -> Index:
    """Chỉ mục ảnh -> các lượt chấm phủ nó. Dựng một lần, ~0,9 s cho 37 lượt.

    Đọc `config.yaml` của từng lượt để biết nó chấm bộ xuất nào, rồi đọc
    `instances_<split>.json` của bộ đó lấy danh sách ảnh. Các lượt cùng trỏ
    vào một file GT thì chỉ đọc file đó một lần (6 file cho 37 lượt).
    """
    global _index
    base = Path(RUNS_DIR)
    st = _stamp(base)
    if _index is not None and not force and _index.stamp == st:
        return _index

    idx = Index(stamp=st)
    if not base.is_dir():
        idx.error = f"Không thấy {base.as_posix()} — chưa có lượt chấm nào."
        _index = idx
        return idx

    gt_cache: dict[tuple[str, str], tuple[list, dict]] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        info = _read_run(d)
        if info is None:
            continue
        key = (info.root, info.split)
        if key not in gt_cache:
            f = Path(info.root) / "annotations" / f"instances_{info.split}.json"
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                gt_cache[key] = ([], {})
            else:
                names = [(im["file_name"], int(im["height"]), int(im["width"]))
                         for im in doc.get("images", [])]
                dem: dict[int, int] = {}
                for a in doc.get("annotations", []):
                    dem[int(a["image_id"])] = dem.get(int(a["image_id"]), 0) + 1
                by_id = {int(im["id"]): im["file_name"] for im in doc.get("images", [])}
                gt_cache[key] = (names, {by_id[i]: n for i, n in dem.items() if i in by_id})
        names, dem = gt_cache[key]
        if not names:
            continue
        idx.runs[info.run] = info
        for fn, h, w in names:
            idx.by_image.setdefault(fn, []).append(info.run)
            idx.size.setdefault(fn, (h, w))
            idx.n_gt[(info.run, fn)] = dem.get(fn, 0)
    _index = idx
    return idx


def _load_preds(run: str) -> dict:
    """{image_id: [bản ghi COCO results, ...]} của một lượt chấm, có nhớ tạm."""
    if run in _preds:
        _preds.move_to_end(run)
        return _preds[run]
    idx = get_index()
    info = idx.runs.get(run)
    if info is None:
        return {}
    try:
        det = json.loads((info.path / "predictions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    by: dict[int, list] = {}
    for r in det:
        by.setdefault(int(r["image_id"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: -float(r.get("score", 0.0)))
    _preds[run] = by
    while len(_preds) > CACHE_RUNS:
        _preds.popitem(last=False)
    return by


_ids: dict[tuple[str, str], dict] = {}


def _image_ids(run: str) -> dict:
    """{tên ảnh: image_id} của bộ xuất mà lượt chấm này đã chấm.

    Nhớ theo (bộ xuất, split) chứ không theo lượt chấm: sáu model cùng chấm
    một fold thì dùng chung một bảng. Không nhớ thì mỗi lần bấm lại nạp
    `instances_test.json` 5,3 MB — 50-80 ms vứt đi cho mỗi lần đổi model.
    """
    info = get_index().runs.get(run)
    if info is None:
        return {}
    key = (info.root, info.split)
    if key in _ids:
        return _ids[key]
    f = Path(info.root) / "annotations" / f"instances_{info.split}.json"
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _ids[key] = {}
    else:
        _ids[key] = {im["file_name"]: int(im["id"]) for im in doc.get("images", [])}
    return _ids[key]


# ----------------------------------------------------------- RLE -> polygon
def _norm_rle(seg: dict) -> dict:
    """`counts` từ JSON là chuỗi; pycocotools muốn bytes ở vài lối vào."""
    c = seg.get("counts")
    if isinstance(c, str):
        return {"size": seg["size"], "counts": c.encode("ascii")}
    return seg


def rle_to_polygons(seg: dict, eps: float = DEFAULT_EPS) -> list[list[float]]:
    """RLE -> danh sách polygon phẳng [x1,y1,x2,y2,...] theo toạ độ ẢNH GỐC.

    Trả về NHIỀU vòng vì một dự đoán rời thành nhiều mảnh là chuyện thường:
    đo trên f6, tỉ lệ dự đoán có hơn một biên ngoài là 9,1% (Mask R-CNN),
    18,0% (SOLOv2), 26,7% (Mask2Former) và 42,7% (YOLOv8s). Giữ mỗi contour
    lớn nhất là bỏ mất gần một nửa hình của YOLOv8s.

    Lỗ bên trong bị LẤP (chỉ lấy biên ngoài): 6,7-19,7% dự đoán có lỗ nhưng
    tổng diện tích lỗ chỉ 0,05-0,14%, tức dưới 0,002 IoU. Đây là chỗ hiển thị
    nên đổi lấy hình đơn giản là hời; mọi con số chấm vẫn lấy từ RLE.
    """
    rle = _norm_rle(seg)
    m = mask_utils.decode(rle)
    x, y, w, h = mask_utils.toBbox(rle)
    if w <= 0 or h <= 0:
        return []
    x0, y0 = int(x), int(y)
    sub = np.ascontiguousarray(m[y0:int(np.ceil(y + h)), x0:int(np.ceil(x + w))])
    cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for k, c in enumerate(cnts):
        if hier is None or hier[0][k][3] >= 0:      # lỗ -> bỏ
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([round(float(v), 1)
                        for xy in a for v in (xy[0] + x0, xy[1] + y0)])
    return out


_gt_rle: OrderedDict = OrderedDict()     # (tên ảnh, vân tay nhãn) -> [RLE, ...]


def gt_rles(file_name: str, polys: list[list[float]], h: int, w: int) -> list[dict]:
    """RLE của nhãn đang hiển thị, có nhớ tạm theo nội dung nhãn.

    Tô một vùng mất ~4,4 ms, trong đó 3,4 ms là `np.asfortranarray` chép cả
    khung 2560x1440 — bắt buộc, vì `fillPoly` từ chối ghi vào mảng thứ tự cột
    còn `encode` thì đòi đúng thứ tự đó. (Tô vào mảng đã chuyển vị rồi encode
    thì nhanh 2,6 lần nhưng RLE lệch 199/200 vùng: thuật toán quét dòng của
    `fillPoly` không đối xứng qua chuyển vị, nên nó ra một tập pixel KHÁC — và
    IoU ở đây phải trùng với bảng benchmark thì mới nói được tán nào "khớp".)

    Nên chỗ chữa là nhớ tạm: ảnh 40 vùng tốn ~180 ms lần đầu, rồi sáu model
    dùng lại cùng bảng đó. Vân tay lấy từ chính toạ độ nên nhãn vừa sửa là
    khoá đổi và bảng được dựng lại — không có chuyện tô theo hình cũ.
    """
    van_tay = (len(polys), tuple(len(p) for p in polys),
               round(sum(sum(p) for p in polys), 3))
    key = (file_name, h, w, van_tay)
    if key in _gt_rle:
        _gt_rle.move_to_end(key)
        return _gt_rle[key]
    out = [_poly_rle(p, h, w) for p in polys]
    _gt_rle[key] = out
    while len(_gt_rle) > CACHE_RUNS:
        _gt_rle.popitem(last=False)
    return out


def _poly_rle(poly: list[float], h: int, w: int) -> dict:
    """Polygon phẳng -> RLE cả khung, tô bằng ĐÚNG phép mà bảng benchmark dùng.

    `cv2.fillPoly` trên mảng đã làm tròn, giống `matching.py:align_masks`.
    Không dùng `frPyObjects` của pycocotools: nó rasterise theo luật khác nên
    IoU ra lệch vài phần nghìn so với con số trong bảng, mà ở đây IoU chính là
    thứ quyết định một tán được tô "khớp" hay "sót".
    """
    # Tô trên mảng thứ tự HÀNG rồi mới đổi sang thứ tự cột: `fillPoly` từ chối
    # ghi vào mảng Fortran ("Layout of the output array img is incompatible
    # with cv::Mat"), còn `encode` thì bắt buộc thứ tự cột.
    m = np.zeros((h, w), np.uint8)
    pts = np.asarray(poly, np.float64).reshape(-1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return mask_utils.encode(np.asfortranarray(m))


def match_greedy(det_rles: list[dict], gt_rles: list[dict],
                 iou_thr: float = IOU_THR) -> tuple[list[int], list[float]]:
    """Ghép tham lam theo điểm giảm dần, đúng luật của bảng benchmark.

    `det_rles` phải đã sắp theo điểm giảm dần. Trả về `(gt_of_det, iou_of_det)`
    với -1 / 0.0 cho dự đoán không ghép được.

    Khác `cofseg/evaluation/matching.py` đúng một điểm: ở đó có cửa chặn
    `BBOX_IOU_GATE = 0.05` để khỏi dựng mặt nạ cho từng cặp. Ở đây ma trận IoU
    do pycocotools tính một lượt trong C nên cửa chặn không tiết kiệm được gì;
    bỏ nó KHÔNG đổi kết quả, vì hai hộp chồng nhau dưới 0,05 thì mặt nạ bên
    trong không thể đạt 0,5.
    """
    n, g = len(det_rles), len(gt_rles)
    gt_of = [-1] * n
    iou_of = [0.0] * n
    if not n or not g:
        return gt_of, iou_of
    M = np.asarray(mask_utils.iou([_norm_rle(d) for d in det_rles],
                                  [_norm_rle(x) for x in gt_rles],
                                  [0] * g), dtype=float).reshape(n, g)
    taken = np.zeros(g, bool)
    for i in range(n):                      # đã sắp theo điểm giảm dần
        row = np.where(taken, -1.0, M[i])
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            taken[j] = True
            gt_of[i], iou_of[i] = j, float(row[j])
    return gt_of, iou_of


# ------------------------------------------------------------------ lối vào
def runs_for(file_name: str) -> list[dict]:
    """Các lượt chấm có dự đoán cho ảnh này, sắp theo tên model."""
    idx = get_index()
    out = [idx.runs[r].as_dict() | {"n_gt_export": idx.n_gt.get((r, file_name))}
           for r in idx.by_image.get(file_name, []) if r in idx.runs]
    out.sort(key=lambda d: (d["model"], d["run"]))
    return out


def polygons_for(file_name: str, run: str, gt_polys: list[list[float]] | None = None,
                 conf: float = 0.5, eps: float = DEFAULT_EPS,
                 iou_thr: float = IOU_THR) -> dict:
    """Dự đoán của một ảnh, đã lọc theo điểm và đổi sang polygon.

    `gt_polys` là polygon nhãn ĐANG hiển thị (đọc từ masks/corrected/), không
    phải nhãn trong bộ xuất: ghép với chính thứ trên màn hình thì cái người
    dùng thấy và cái được tô "khớp"/"sót" luôn là một. Lệch với bộ xuất thì
    báo ở `warn` chứ không lặng lẽ dùng bản khác.
    """
    idx = get_index()
    info = idx.runs.get(run)
    if info is None:
        return {"ok": False, "error": f"Không thấy lượt chấm {run!r}."}
    iid = _image_ids(run).get(file_name)
    if iid is None:
        return {"ok": False,
                "error": f"Ảnh không nằm trong split {info.split} của {info.root}."}

    h, w = idx.size.get(file_name, (0, 0))
    tat_ca = _load_preds(run).get(iid, [])
    dets = [d for d in tat_ca if float(d.get("score", 0.0)) >= conf]

    gt_polys = [p for p in (gt_polys or []) if len(p) >= 6]
    rles = gt_rles(file_name, gt_polys, h, w)
    gt_of, iou_of = match_greedy([d["segmentation"] for d in dets], rles, iou_thr)

    polys = []
    for k, d in enumerate(dets):
        for ring in rle_to_polygons(d["segmentation"], eps):
            polys.append({"points": ring, "conf": round(float(d.get("score", 0.0)), 4),
                          "matched": gt_of[k] >= 0, "iou": round(iou_of[k], 4),
                          "gt": gt_of[k], "det": k})

    lay = {j: iou for j, iou in zip(gt_of, iou_of) if j >= 0}
    gt_out = [{"matched": j in lay, "iou": round(lay.get(j, 0.0), 4),
               "points": gt_polys[j]} for j in range(len(rles))]

    tp = sum(1 for j in gt_of if j >= 0)
    n_xuat = idx.n_gt.get((run, file_name))
    warn = ""
    if n_xuat is not None and n_xuat != len(rles):
        warn = (f"Nhãn hiện tại có {len(rles)} vùng, bộ xuất lúc chấm có "
                f"{n_xuat}. Nhãn đã sửa sau khi xuất, nên phần khớp/sót ở đây "
                f"không còn trùng với con số trong bảng benchmark.")

    return {
        "ok": True,
        "source": info.as_dict(),
        "conf": conf, "eps": eps, "iou_thr": iou_thr,
        "polygons": polys,
        "gt": gt_out,
        "summary": {
            "n_pred_all": len(tat_ca), "n_pred": len(dets), "n_gt": len(rles),
            "tp": tp, "fp": len(dets) - tp, "fn": len(rles) - tp,
            "mean_iou": round(float(np.mean([v for v in iou_of if v > 0])), 4)
            if tp else None,
        },
        "warn": warn,
    }
