# app/server.py — Backend FastAPI cho annotator tự xây (thay CVAT).
# Phase 1: annotate TRỰC TIẾP trên ảnh gốc (KHÔNG cắt tile).
# Tính năng: SAM Click, SAM Box, vẽ/sửa polygon tay, lưu COCO polygon (đa instance).
# Chạy:  python -m app.server   (mở http://localhost:8000)
# Chọn thư mục ảnh: $env:ANNOT_DIR="data/iachim_dataset_export/data_compressed/field_1"
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from app.annotator import SamAnnotator

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
# Mặc định annotate trên ảnh gốc; đặt ANNOT_DIR để trỏ tới 1 field.
BASE = CFG["data"]["images_dir"]   # thư mục gốc chứa các field
ROOT = os.environ.get("ANNOT_DIR", BASE)
OUT_DIR = os.path.join(CFG["data"]["masks_dir"], "corrected")
EXPORT_DIR = os.path.join(os.path.dirname(CFG["data"]["masks_dir"]) or ".", "export")
WEIGHTS_DIR = "weights"
_MODEL_EXTS = {".pt", ".pth"}
HERE = os.path.dirname(__file__)
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

app = FastAPI(title="Coffee Canopy Annotator")

_ann = None       # SamAnnotator (nạp lười khi dùng SAM lần đầu)
_current = None   # tên ảnh đang set_image

# --- Chuẩn hoá đường dẫn nhãn (theo BASE, gồm cả tên field) & mật độ hoa ---
DEFAULT_CONF_THR = 0.25   # placeholder: polygon vẽ tay không có ngưỡng detector
FLOWER_THRESHOLDS = (0.02, 0.10, 0.30)   # tỉ lệ phủ hoa -> mức 0/1/2/3
FLOWER_NAMES = ("no_flower", "few_flowers", "many_flowers", "very_many_flowers")


def _flower_label(ratio):
    # ratio: tỉ lệ diện tích hoa / tán (0-1) -> (mức 0-3, tên mức).
    t1, t2, t3 = FLOWER_THRESHOLDS
    if ratio < t1:
        lvl = 0
    elif ratio < t2:
        lvl = 1
    elif ratio < t3:
        lvl = 2
    else:
        lvl = 3
    return lvl, FLOWER_NAMES[lvl]


def _full_rel(name):
    # Đường dẫn ảnh tương đối so với BASE (gồm tên field) -> path/field/tên file.
    try:
        rel = os.path.relpath(os.path.join(ROOT, name), BASE).replace("\\", "/")
        if rel.startswith(".."):
            rel = name
    except Exception:
        rel = name
    return rel


def _field_of(rel):
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else ""


def _json_path(name):
    # Tên file nhãn: đường dẫn BASE-relative, '/' -> '__', GIỮ đuôi ảnh + .json.
    return os.path.join(OUT_DIR, _full_rel(name).replace("/", "__") + ".json")


def _flower_counts(img_bgr, poly, sat_max, val_min):
    # Trả về (số pixel hoa, tổng pixel) trong 1 polygon (HSV).
    h, w = img_bgr.shape[:2]
    pts = np.round(np.array(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    total = int(mask.sum())
    if total == 0:
        return 0, 0
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flower = ((hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flower = cv2.morphologyEx(flower, cv2.MORPH_OPEN, kernel)
    cnt = int(np.count_nonzero((flower > 0) & (mask > 0)))
    return cnt, total


def _poly_area(poly):
    xs, ys = poly[0::2], poly[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    return 0.5 * abs(sum(xs[j] * ys[(j + 1) % n] - xs[(j + 1) % n] * ys[j]
                         for j in range(n)))


def _read_records(jp):
    # Đọc 1 file nhãn (định dạng MỚI hoặc COCO cũ) -> (w, h, [ann...]).
    data = json.load(open(jp, encoding="utf-8"))
    anns = []
    if isinstance(data.get("polygons"), list):
        w = data.get("img_w", 0)
        h = data.get("img_h", 0)
        for p in data["polygons"]:
            pts = p.get("points") or []
            poly = [float(v) for xy in pts for v in xy]
            if len(poly) < 6:
                continue
            xs, ys = poly[0::2], poly[1::2]
            cid = p.get("flower_label", 0)
            anns.append(dict(poly=poly,
                             bbox_xywh=[min(xs), min(ys),
                                        max(xs) - min(xs), max(ys) - min(ys)],
                             area=_poly_area(poly), cat_id=cid,
                             cat_name=p.get("flower_label_name", str(cid))))
    else:
        im = (data.get("images") or [{}])[0]
        w = im.get("width", 0)
        h = im.get("height", 0)
        cmap = {c.get("id"): c.get("name", str(c.get("id")))
                for c in data.get("categories", [])}
        for a in data.get("annotations", []):
            seg = a.get("segmentation") or []
            if not seg:
                continue
            cid = a.get("category_id", 0)
            anns.append(dict(poly=seg[0], bbox_xywh=a.get("bbox", []),
                             area=float(a.get("area", 0.0)), cat_id=cid,
                             cat_name=cmap.get(cid, str(cid))))
    return w, h, anns


def _sam2_cfg_for(fn):
    f = fn.lower()
    if "hiera_l" in f or "large" in f:
        return "configs/sam2.1/sam2.1_hiera_l.yaml", "SAM 2.1 Large"
    if "b+" in f or "base_plus" in f or "hiera_b" in f:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml", "SAM 2.1 Base+"
    if "hiera_s" in f or "small" in f:
        return "configs/sam2.1/sam2.1_hiera_s.yaml", "SAM 2.1 Small"
    if "hiera_t" in f or "tiny" in f:
        return "configs/sam2.1/sam2.1_hiera_t.yaml", "SAM 2.1 Tiny"
    return CFG["sam"]["model_cfg"], "SAM 2.1"


def _model_meta(fn):
    f = fn.lower()
    if "sam3" in f:
        # SAM 3 / 3.1: config nội bộ package sam3 (chỉnh lại khi cài SAM3.1).
        return dict(id=fn, name="SAM 3.1 · " + fn, variant="sam3",
                    cfg="configs/sam3/sam3.yaml")
    if "finetuned" in f or "_p1" in f:
        return dict(id=fn, name="SAM 2.1 finetuned · " + fn, variant="sam2",
                    cfg="configs/sam2.1/sam2.1_hiera_l.yaml")
    cfg, lbl = _sam2_cfg_for(fn)
    return dict(id=fn, name=lbl, variant="sam2", cfg=cfg)


def _list_models():
    d = Path(WEIGHTS_DIR)
    if not d.exists():
        return []
    return [_model_meta(p.name) for p in sorted(d.iterdir())
            if p.suffix.lower() in _MODEL_EXTS]


def _default_selected():
    models = _list_models()
    if not models:
        return None
    want = os.path.basename(CFG["sam"].get("checkpoint", ""))
    for m in models:
        if m["id"] == want:
            return m
    return models[0]


SELECTED = _default_selected()


def _annotator():
    global _ann
    if _ann is None:
        if SELECTED is None:
            raise RuntimeError("Không có model nào trong thư mục weights/.")
        _ann = SamAnnotator(CFG["sam"],
                            checkpoint=os.path.join(WEIGHTS_DIR, SELECTED["id"]),
                            variant=SELECTED["variant"],
                            model_cfg=SELECTED["cfg"])
    return _ann


def _load_image(name):
    global _current
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, name)), cv2.COLOR_BGR2RGB)
    if _current != name:
        _annotator().set_image(img)   # encode 1 lần, các click sau dùng lại
        _current = name
    return img


def _mask_to_polygons(mask, min_area=200, epsilon=2.0):
    # Mask bool -> danh sách polygon COCO [[x1,y1,x2,y2,...], ...].
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([float(v) for xy in approx for v in xy])
    return polys


class ClickReq(BaseModel):
    name: str
    x: float
    y: float


class BoxReq(BaseModel):
    name: str
    box: list   # [x1, y1, x2, y2]


class SaveReq(BaseModel):
    name: str
    width: int
    height: int
    polygons: list           # mỗi polygon = [x1, y1, x2, y2, ...]
    classes: list = []       # category_id cho từng polygon (song song polygons)
    categories: list = []    # [{id, name, color}, ...] danh sách lớp của dự án
    confs: list = []         # confidence từng polygon (SAM/detector); mặc định 1.0
    sat_max: int = 50        # ngưỡng HSV S để tính % hoa lúc lưu
    val_min: int = 180       # ngưỡng HSV V để tính % hoa lúc lưu


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()


@app.get("/api/images")
def list_images():
    root = Path(ROOT)
    files = [p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.suffix.lower() in EXTS]
    return sorted(files)


@app.get("/api/image/{name:path}")
def get_image(name: str):
    return FileResponse(os.path.join(ROOT, name))


@app.get("/api/thumb/{name:path}")
def get_thumb(name: str, w: int = 96):
    # Thumbnail thu nhỏ để hiển thị danh sách ảnh nhanh.
    img = cv2.imread(os.path.join(ROOT, name))
    if img is None:
        return Response(status_code=404)
    h0, w0 = img.shape[:2]
    scale = w / float(w0)
    thumb = cv2.resize(img, (w, max(1, int(h0 * scale))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/labeled")
def labeled():
    # Danh sách ảnh đã có nhãn (đã lưu) — để thống kê & đánh dấu.
    root = Path(ROOT)
    names = [p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.suffix.lower() in EXTS]
    done = []
    for n in names:
        if os.path.exists(_json_path(n)):
            done.append(n)
    return {"labeled": done, "total": len(names)}


class FlowerReq(BaseModel):
    name: str
    polygons: list          # danh sách polygon [x1, y1, x2, y2, ...]
    sat_max: int = 50       # ngưỡng bão hoà: pixel hoa có S < sat_max
    val_min: int = 180      # ngưỡng độ sáng: pixel hoa có V > val_min


def _flower_ratio(img_bgr, poly, sat_max, val_min):
    # Đếm % pixel hoa (trắng/kem) trong 1 polygon theo phương pháp HSV.
    h, w = img_bgr.shape[:2]
    pts = np.round(np.array(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    area = int(mask.sum())
    if area == 0:
        return 0.0
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)            # khử nhiễu
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flower = ((hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flower = cv2.morphologyEx(flower, cv2.MORPH_OPEN, kernel)   # morphological opening
    cnt = int(np.count_nonzero((flower > 0) & (mask > 0)))
    return round(100.0 * cnt / area, 2)


@app.post("/api/flower")
def flower(req: FlowerReq):
    # Trả về % pixel hoa cho từng polygon -> phân lớp mật độ 0/1/2/3 ở client.
    img = cv2.imread(os.path.join(ROOT, req.name))
    if img is None:
        return {"ratios": []}
    ratios = [_flower_ratio(img, p, req.sat_max, req.val_min)
              for p in req.polygons]
    return {"ratios": ratios}


@app.get("/api/models")
def models_list():
    # Liệt kê model trong weights/ để giao diện chọn.
    return {"models": _list_models(),
            "current": SELECTED["id"] if SELECTED else None}


class ModelReq(BaseModel):
    id: str


@app.post("/api/model")
def set_model(req: ModelReq):
    # Đổi model đang dùng (nạp lười ở lần SAM kế tiếp, không cần restart).
    global SELECTED, _ann, _current
    for m in _list_models():
        if m["id"] == req.id:
            SELECTED = m
            _ann = None
            _current = None
            return {"ok": True, "current": m["id"], "variant": m["variant"]}
    return {"ok": False, "error": "Không tìm thấy model: " + req.id}


@app.post("/api/sam/click")
def sam_click(req: ClickReq):
    try:
        _load_image(req.name)
        mask = _annotator().predict_point(req.x, req.y, positive=True)
        return {"polygons": _mask_to_polygons(mask)}
    except Exception as e:
        return {"polygons": [], "error": str(e)}


@app.post("/api/sam/box")
def sam_box(req: BoxReq):
    try:
        _load_image(req.name)
        mask = _annotator().predict_box(req.box)
        return {"polygons": _mask_to_polygons(mask)}
    except Exception as e:
        return {"polygons": [], "error": str(e)}


@app.get("/api/load/{name:path}")
def load(name: str):
    # Trả về polygon đã lưu trước đó (nếu có) để khôi phục khi mở lại ảnh.
    fp = _json_path(name)
    if not os.path.exists(fp):
        return {"polygons": [], "classes": [], "categories": []}
    data = json.load(open(fp, encoding="utf-8"))
    polys, cls = [], []
    if isinstance(data.get("polygons"), list):
        # Định dạng mới: polygons=[{points:[[x,y],...], flower_label}]
        for p in data["polygons"]:
            pts = p.get("points") or []
            flat = [float(v) for xy in pts for v in xy]
            if len(flat) >= 6:
                polys.append(flat)
                cls.append(p.get("flower_label", 0))
    else:
        # Định dạng COCO cũ.
        for a in data.get("annotations", []):
            if a.get("segmentation"):
                polys.append(a["segmentation"][0])
                cls.append(a.get("category_id", 1))
    return {"polygons": polys, "classes": cls,
            "categories": data.get("categories", [])}


@app.post("/api/save")
def save(req: SaveReq):
    # Lưu theo ĐỊNH DẠNG MỚI (giống file mẫu): path/field/.../polygons/flower_stats.
    os.makedirs(OUT_DIR, exist_ok=True)
    out = _json_path(req.name)
    rel = _full_rel(req.name)
    polys, confs = [], []
    for i, poly in enumerate(req.polygons):
        if len(poly) >= 6:
            polys.append(poly)
            confs.append(req.confs[i] if i < len(req.confs) else 1.0)
    if not polys:
        # Không còn vùng nào -> xoá file cũ để ảnh không bị đánh dấu "đã nhãn".
        if os.path.exists(out):
            os.remove(out)
        return {"saved": None, "count": 0}
    img = cv2.imread(os.path.join(ROOT, req.name))
    stats = {nm: 0 for nm in FLOWER_NAMES}
    ratios, out_polys = [], []
    for poly, conf in zip(polys, confs):
        xs, ys = poly[0::2], poly[1::2]
        if img is not None:
            fpx, tpx = _flower_counts(img, poly, req.sat_max, req.val_min)
        else:
            fpx, tpx = 0, 0
        ratio = round(fpx / tpx, 4) if tpx else 0.0
        lvl, lname = _flower_label(ratio)
        stats[lname] += 1
        ratios.append(ratio)
        out_polys.append(dict(
            points=[[float(x), float(y)] for x, y in zip(xs, ys)],
            conf=round(float(conf), 4),
            bbox=[round(float(min(xs)), 1), round(float(min(ys)), 1),
                  round(float(max(xs)), 1), round(float(max(ys)), 1)],
            flower_pixels=int(fpx), total_pixels=int(tpx),
            flower_ratio=ratio, flower_label=lvl, flower_label_name=lname))
    doc = dict(path=rel, field=_field_of(rel),
               img_w=req.width, img_h=req.height,
               n_canopy=len(out_polys), conf_thr=DEFAULT_CONF_THR,
               polygons=out_polys,
               flower_stats=dict(
                   no_flower=stats["no_flower"], few_flowers=stats["few_flowers"],
                   many_flowers=stats["many_flowers"],
                   very_many_flowers=stats["very_many_flowers"],
                   avg_ratio=round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
                   max_ratio=round(max(ratios), 4) if ratios else 0.0))
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    return {"saved": out, "count": len(out_polys)}


class ExportReq(BaseModel):
    format: str = "coco"     # "coco" | "yolo"
    scope: str = ""          # lọc theo thư mục con của ROOT ("" = tất cả)
    out_dir: str = ""        # nơi lưu (rỗng = EXPORT_DIR mặc định)


def _labeled_json_path(name):
    return _json_path(name)


def _iter_labeled(scope):
    # (name, json_path) cho ảnh trong ROOT (lọc theo scope) đã có nhãn.
    root = Path(ROOT)
    scope = (scope or "").strip("/")
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        name = p.relative_to(root).as_posix()
        if scope and not (name == scope or name.startswith(scope + "/")):
            continue
        jp = _labeled_json_path(name)
        if os.path.exists(jp):
            yield name, jp


def _collect_categories(items):
    cats = {}
    for _, jp in items:
        _, _, anns = _read_records(jp)
        for a in anns:
            cid = a["cat_id"]
            if cid not in cats:
                cats[cid] = dict(id=cid, name=a["cat_name"], color="")
    if not cats:
        cats[0] = dict(id=0, name="0", color="")
    return [cats[i] for i in sorted(cats)]


def _export_coco(items, out_dir):
    cats = _collect_categories(items)
    images, anns = [], []
    iid = aid = 0
    for name, jp in items:
        w, h, recs = _read_records(jp)
        iid += 1
        images.append(dict(id=iid, file_name=name, width=w, height=h))
        for r in recs:
            aid += 1
            anns.append(dict(id=aid, image_id=iid,
                             category_id=r["cat_id"],
                             segmentation=[r["poly"]],
                             area=float(r["area"]),
                             bbox=r["bbox_xywh"], iscrowd=0))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "instances.json")
    json.dump(dict(images=images, annotations=anns, categories=cats),
              open(out, "w", encoding="utf-8"))
    return dict(file=out.replace("\\", "/"), image_root=ROOT.replace("\\", "/"),
                images=len(images), annotations=len(anns))


def _export_yolo(items, out_dir):
    cats = _collect_categories(items)
    idx = {c["id"]: k for k, c in enumerate(cats)}
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n_img = n_obj = 0
    for name, jp in items:
        w, h, recs = _read_records(jp)
        if not w or not h:
            m = cv2.imread(os.path.join(ROOT, name))
            if m is None:
                continue
            h, w = m.shape[:2]
        rows = []
        for r in recs:
            poly = r["poly"]
            ci = idx.get(r["cat_id"], 0)
            coords = " ".join("%.6f" % (poly[j] / w if j % 2 == 0 else poly[j] / h)
                              for j in range(len(poly)))
            rows.append(str(ci) + " " + coords)
            n_obj += 1
        if not rows:
            continue
        stem = name.replace("/", "__").rsplit(".", 1)[0]
        ext = os.path.splitext(name)[1] or ".jpg"
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        with open(os.path.join(ROOT, name), "rb") as fr:
            buf = fr.read()
        with open(os.path.join(img_dir, stem + ext), "wb") as fw:
            fw.write(buf)
        n_img += 1
    ylines = ["path: " + os.path.abspath(out_dir).replace("\\", "/"),
              "train: images", "val: images",
              "nc: " + str(len(cats)), "names:"]
    for k, c in enumerate(cats):
        ylines.append("  " + str(k) + ": " + str(c["name"]))
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(ylines) + "\n")
    return dict(dir=out_dir.replace("\\", "/"), images=n_img,
                objects=n_obj, classes=len(cats))


@app.post("/api/export")
def export(req: ExportReq):
    items = list(_iter_labeled(req.scope))
    if not items:
        return {"ok": False,
                "error": "Không có ảnh nào đã gán nhãn trong phạm vi này."}
    tag = (req.scope or "all").replace("/", "__") or "all"
    base = req.out_dir.strip() or EXPORT_DIR
    if req.format == "yolo":
        info = _export_yolo(items, os.path.join(base, "yolo_" + tag))
    else:
        info = _export_coco(items, os.path.join(base, "coco_" + tag))
    info["ok"] = True
    info["format"] = req.format
    return info


# ===================== XUẤT DATASET (train-ready) =====================
# Gộp nhiều field đã gán nhãn -> 1 bộ dataset có sẵn train/val/test,
# phục vụ CH CẢ promptable (SAM: COCO instance) LẮN task-specific
# (U-Net/DeepLab/SegFormer: mask PNG semantic). Kèm YOLO-seg tuỳ chọn.
class DatasetReq(BaseModel):
    scope: str = ""                       # lọc theo thư mục con của ROOT (""=tất cả)
    formats: list = ["coco", "yolo", "masks"]
    split_by: str = "none"               # "none" (không chia) | "field" | "ratio"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    val_fields: list = []                 # dùng khi split_by="field"
    test_fields: list = []
    seed: int = 42
    name: str = "dataset"                 # tên thư mục output dưới export/
    out_dir: str = ""                     # nơi lưu (rỗng = EXPORT_DIR mặc định)


def _field_key(name):
    return _field_of(_full_rel(name))


def _split_items(items, req):
    import random
    by_field = {}
    for it in items:
        by_field.setdefault(_field_key(it[0]), []).append(it)
    fields = sorted(by_field)
    mode = req.split_by
    if mode == "none":
        return {"all": list(items)}, fields
    splits = {"train": [], "val": [], "test": []}
    valf, testf = set(req.val_fields), set(req.test_fields)
    if mode == "field" and not valf and not testf:
        # Tự giữ field nếu đủ số field, ngược lại quay về chia theo ảnh.
        if len(fields) >= 3:
            testf = {fields[-1]}
            valf = {fields[-2]}
        else:
            mode = "ratio"
    if mode == "field":
        for f in fields:
            dst = "test" if f in testf else ("val" if f in valf else "train")
            splits[dst].extend(by_field[f])
    else:
        rng = random.Random(req.seed)
        for f in fields:
            lst = by_field[f][:]
            rng.shuffle(lst)
            n = len(lst)
            n_test = int(round(n * req.test_ratio))
            n_val = int(round(n * req.val_ratio))
            splits["test"].extend(lst[:n_test])
            splits["val"].extend(lst[n_test:n_test + n_val])
            splits["train"].extend(lst[n_test + n_val:])
    return splits, fields


def _semantic_mask(recs, w, h):
    # Mask semantic: 0=nền, 1..4 = mức hoa+1 (vẽ mức cao đè lên khi chồng lấn).
    mask = np.zeros((h, w), np.uint8)
    for r in sorted(recs, key=lambda x: x["cat_id"]):
        pts = np.round(np.array(r["poly"], dtype=np.float64)
                       .reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], int(r["cat_id"]) + 1)
    return mask


def _write_coco(items, out_file):
    cats = [dict(id=i, name=FLOWER_NAMES[i], color="") for i in range(4)]
    images, anns = [], []
    iid = aid = 0
    for name, jp in items:
        w, h, recs = _read_records(jp)
        iid += 1
        images.append(dict(id=iid, file_name=name.replace("/", "__"),
                           width=w, height=h))
        for r in recs:
            aid += 1
            anns.append(dict(id=aid, image_id=iid, category_id=int(r["cat_id"]),
                             segmentation=[r["poly"]], area=float(r["area"]),
                             bbox=r["bbox_xywh"], iscrowd=0))
    json.dump(dict(images=images, annotations=anns, categories=cats),
              open(out_file, "w", encoding="utf-8"))
    return len(images), len(anns)


def _copy_file(src, dst):
    if os.path.exists(src):
        with open(src, "rb") as fr, open(dst, "wb") as fw:
            fw.write(fr.read())


def _export_dataset(items, req):
    base_dir = req.out_dir.strip() or EXPORT_DIR
    root_out = os.path.join(base_dir, req.name)
    splits, fields = _split_items(items, req)
    formats = set(req.formats or [])
    no_split = (req.split_by == "none")
    summary = {}
    for sp, sp_items in splits.items():
        summary[sp] = {"images": len(sp_items), "annotations": 0}
        if not sp_items:
            continue
        sub = "" if no_split else sp
        img_out = os.path.join(root_out, "images", sub)
        os.makedirs(img_out, exist_ok=True)
        for name, _ in sp_items:
            _copy_file(os.path.join(ROOT, name),
                       os.path.join(img_out, name.replace("/", "__")))
        if "coco" in formats:
            cdir = os.path.join(root_out, "coco")
            os.makedirs(cdir, exist_ok=True)
            fn = ("instances" if no_split else sp) + ".json"
            _, na = _write_coco(sp_items, os.path.join(cdir, fn))
            summary[sp]["annotations"] = na
        if "masks" in formats:
            mdir = os.path.join(root_out, "masks", sub)
            os.makedirs(mdir, exist_ok=True)
            for name, jp in sp_items:
                w, h, recs = _read_records(jp)
                if not w or not h:
                    im = cv2.imread(os.path.join(ROOT, name))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                m = _semantic_mask(recs, w, h)
                flat = name.replace("/", "__").rsplit(".", 1)[0] + ".png"
                cv2.imwrite(os.path.join(mdir, flat), m)
        if "yolo" in formats:
            yi = os.path.join(root_out, "yolo", "images", sub)
            yl = os.path.join(root_out, "yolo", "labels", sub)
            os.makedirs(yi, exist_ok=True)
            os.makedirs(yl, exist_ok=True)
            for name, jp in sp_items:
                w, h, recs = _read_records(jp)
                if not w or not h:
                    im = cv2.imread(os.path.join(ROOT, name))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                rows = []
                for r in recs:
                    poly = r["poly"]
                    coords = " ".join(
                        "%.6f" % (poly[j] / w if j % 2 == 0 else poly[j] / h)
                        for j in range(len(poly)))
                    rows.append(str(int(r["cat_id"])) + " " + coords)
                flat = name.replace("/", "__")
                stem = flat.rsplit(".", 1)[0]
                with open(os.path.join(yl, stem + ".txt"), "w",
                          encoding="utf-8") as f:
                    f.write("\n".join(rows))
                _copy_file(os.path.join(ROOT, name), os.path.join(yi, flat))
    if "yolo" in formats:
        ydir = os.path.join(root_out, "yolo")
        os.makedirs(ydir, exist_ok=True)
        if no_split:
            ylines = ["path: " + os.path.abspath(ydir).replace("\\", "/"),
                      "train: images", "val: images", "nc: 4", "names:"]
        else:
            ylines = ["path: " + os.path.abspath(ydir).replace("\\", "/"),
                      "train: images/train", "val: images/val",
                      "test: images/test", "nc: 4", "names:"]
        for i in range(4):
            ylines.append("  %d: %s" % (i, FLOWER_NAMES[i]))
        with open(os.path.join(ydir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join(ylines) + "\n")
    os.makedirs(root_out, exist_ok=True)
    meta = dict(name=req.name, split_by=req.split_by, fields=fields,
                classes=list(FLOWER_NAMES), formats=sorted(formats),
                splits=summary,
                image_naming="flattened: field__<...>__file.ext")
    json.dump(meta, open(os.path.join(root_out, "meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    meta["dir"] = root_out.replace("\\", "/")
    meta["ok"] = True
    return meta


@app.post("/api/export_dataset")
def export_dataset(req: DatasetReq):
    items = list(_iter_labeled(req.scope))
    if not items:
        return {"ok": False,
                "error": "Không có ảnh nào đã gán nhãn trong phạm vi này."}
    return _export_dataset(items, req)


class RootReq(BaseModel):
    folder: str = ""


def _has_image(d):
    for p in d.rglob("*"):
        if p.suffix.lower() in EXTS:
            return True
    return False


@app.get("/api/folders")
def folders():
    # Liệt kê các thư mục con (field) có ảnh, để đổi nguồn dữ liệu từ giao diện.
    base = Path(BASE)
    subs = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and _has_image(d):
                subs.append(d.name)
    try:
        cur = Path(ROOT).relative_to(base).as_posix()
        cur = "" if cur == "." else cur
    except Exception:
        cur = ROOT
    return {"base": str(base), "folders": subs, "current": cur}


@app.post("/api/set_root")
def set_root(req: RootReq):
    # Đổi thư mục ảnh đang annotate (không cần restart server).
    global ROOT, _current
    folder = (req.folder or "").strip()
    if folder in ("", "."):
        new_root = BASE
    elif os.path.isabs(folder) or ":" in folder:
        new_root = folder
    else:
        new_root = os.path.join(BASE, folder)
    if not os.path.isdir(new_root):
        return {"ok": False, "error": "Không tìm thấy thư mục: " + new_root}
    ROOT = new_root
    _current = None
    return {"ok": True, "root": ROOT}


class PathsReq(BaseModel):
    out_dir: str = ""        # thư mục lưu file nhãn khi autosave
    export_dir: str = ""     # thư mục gốc khi xuất COCO/YOLO/dataset


@app.get("/api/paths")
def get_paths():
    # Trả về nơi lưu nhãn & nơi xuất hiện tại (để hiện lên giao diện).
    return {"out_dir": OUT_DIR.replace("\\", "/"),
            "export_dir": EXPORT_DIR.replace("\\", "/")}


@app.post("/api/set_paths")
def set_paths(req: PathsReq):
    # Đổi nơi lưu nhãn autosave (out_dir) & nơi xuất (export_dir); không cần restart.
    global OUT_DIR, EXPORT_DIR
    if req.out_dir.strip():
        OUT_DIR = req.out_dir.strip()
        os.makedirs(OUT_DIR, exist_ok=True)
    if req.export_dir.strip():
        EXPORT_DIR = req.export_dir.strip()
        os.makedirs(EXPORT_DIR, exist_ok=True)
    return {"ok": True, "out_dir": OUT_DIR.replace("\\", "/"),
            "export_dir": EXPORT_DIR.replace("\\", "/")}


@app.get("/api/browse")
def browse(path: str = ""):
    # Duyệt thư mục trên máy chạy server để chọn nơi lưu (cửa sổ browse).
    import string
    p = (path or "").strip()
    try:
        p = os.path.abspath(p) if p else os.path.abspath(".")
    except Exception:
        p = os.path.abspath(".")
    dirs, error = [], ""
    if os.path.isdir(p):
        try:
            for nm in sorted(os.listdir(p), key=str.lower):
                if os.path.isdir(os.path.join(p, nm)):
                    dirs.append(nm)
        except Exception as e:
            error = str(e)
    else:
        error = "Không truy cập được thư mục."
    parent = os.path.dirname(p.rstrip("\\/")) or p
    drives = []
    if os.name == "nt":
        for c in string.ascii_uppercase:
            d = c + ":\\"
            if os.path.exists(d):
                drives.append(d.replace("\\", "/"))
    return {"path": p.replace("\\", "/"), "parent": parent.replace("\\", "/"),
            "dirs": dirs, "drives": drives, "error": error}


class MkdirReq(BaseModel):
    path: str
    name: str = ""


@app.post("/api/mkdir")
def mkdir(req: MkdirReq):
    # Tạo thư mục con mới ngay trong cửa sổ browse.
    try:
        target = os.path.join(req.path, req.name) if req.name else req.path
        os.makedirs(target, exist_ok=True)
        return {"ok": True, "path": os.path.abspath(target).replace("\\", "/")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    import sys
    import uvicorn
    # python -m app.server -r  (hoặc --reload) -> tự reload backend khi sửa .py
    reload = ("-r" in sys.argv) or ("--reload" in sys.argv)
    uvicorn.run("app.server:app" if reload else app,
                host="127.0.0.1", port=8000, reload=reload)