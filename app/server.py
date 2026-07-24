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


# --- đọc/ghi ảnh chịu được đường dẫn Unicode (Windows tiếng Việt) ---
# cv2.imread trả None khi đường dẫn có dấu; máy để mã nguồn dưới
# C:\Users\Nguyễn Văn A\ sẽ hỏng toàn bộ nếu dùng thẳng cv2.imread.
def _imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


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


def _as_level(cid):
    # cid từ client -> mức hoa hợp lệ (0..3), hoặc None nếu không dùng được.
    try:
        lvl = int(cid)
    except (TypeError, ValueError):
        return None
    return lvl if 0 <= lvl < len(FLOWER_NAMES) else None


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


_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _flower_counts(img_bgr, poly, sat_max, val_min):
    # Theo README mục 6 của bộ dữ liệu: TRÍCH XUẤT vùng polygon rồi mới đếm ->
    # blur/HSV/opening chạy TRONG từng tán, không phải trên cả ảnh.
    # Trả về (số pixel hoa, tổng pixel) của 1 tán.
    h, w = img_bgr.shape[:2]
    pts = np.round(np.array(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
    # Kẹp bbox vào trong ảnh: polygon vẽ tay có thể vượt ra ngoài khung.
    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0
    pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(pm, [pts - np.array([x0, y0], np.int32)], 1)
    total = int(pm.sum())
    if total == 0:
        return 0, 0
    crop = img_bgr[y0:y1, x0:x1]
    region = cv2.bitwise_and(crop, crop, mask=pm)   # chỉ giữ pixel trong tán
    blur = cv2.GaussianBlur(region, (3, 3), 0)      # khử nhiễu
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flower = ((hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)).astype(np.uint8)
    flower = cv2.morphologyEx(flower, cv2.MORPH_OPEN, _KERNEL)  # morphological opening
    return int(np.count_nonzero((flower > 0) & (pm > 0))), total


def _poly_area(poly):
    xs, ys = poly[0::2], poly[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    return 0.5 * abs(sum(xs[j] * ys[(j + 1) % n] - xs[(j + 1) % n] * ys[j]
                         for j in range(n)))


def _prev_flower(jp):
    # Số liệu hoa đã lưu lần trước, theo đúng thứ tự polygon trong file.
    if not os.path.exists(jp):
        return []
    try:
        data = json.load(open(jp, encoding="utf-8"))
    except Exception:
        return []
    out = []
    for p in (data.get("polygons") or []):
        r = p.get("flower_ratio")
        out.append(dict(flower_pixels=int(p.get("flower_pixels") or 0),
                        total_pixels=int(p.get("total_pixels") or 0),
                        ratio=r if isinstance(r, (int, float)) else None))
    return out


def _read_records(jp):
    # Đọc 1 file nhãn (định dạng MỚI hoặc COCO cũ) -> (w, h, [ann...], bỏ qua).
    # "bỏ qua" = số vùng chưa xác định được mức hoa (lưu lúc không đọc được ảnh).
    data = json.load(open(jp, encoding="utf-8"))
    anns = []
    skipped = 0
    if isinstance(data.get("polygons"), list):
        w = data.get("img_w", 0)
        h = data.get("img_h", 0)
        for p in data["polygons"]:
            pts = p.get("points") or []
            poly = [float(v) for xy in pts for v in xy]
            if len(poly) < 6:
                continue
            cid = _as_level(p.get("flower_label"))
            if cid is None:
                skipped += 1     # thà bỏ còn hơn gán bừa mức 0
                continue
            xs, ys = poly[0::2], poly[1::2]
            anns.append(dict(poly=poly,
                             bbox_xywh=[min(xs), min(ys),
                                        max(xs) - min(xs), max(ys) - min(ys)],
                             area=_poly_area(poly), cat_id=cid,
                             cat_name=p.get("flower_label_name")
                             or FLOWER_NAMES[cid],
                             conf=p.get("conf"),
                             flower_ratio=p.get("flower_ratio"),
                             flower_pixels=p.get("flower_pixels"),
                             total_pixels=p.get("total_pixels"),
                             label_source=p.get("label_source")))
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
            cid = _as_level(a.get("category_id"))
            if cid is None:
                skipped += 1
                continue
            anns.append(dict(poly=seg[0], bbox_xywh=a.get("bbox", []),
                             area=float(a.get("area", 0.0)), cat_id=cid,
                             cat_name=cmap.get(cid) or FLOWER_NAMES[cid],
                             conf=None, flower_ratio=None, flower_pixels=None,
                             total_pixels=None, label_source=None))
    return w, h, anns, skipped


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
    raw = _imread(os.path.join(ROOT, name))
    if raw is None:
        raise RuntimeError("Không đọc được ảnh: " + name)
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
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
    manual: list = []        # True = mức hoa do người gán tay -> HSV không ghi đè
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
    img = _imread(os.path.join(ROOT, name))
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


@app.post("/api/flower")
def flower(req: FlowerReq):
    # Trả về % pixel hoa cho từng polygon -> phân lớp mật độ 0/1/2/3 ở client.
    img = _imread(os.path.join(ROOT, req.name))
    if img is None:
        return {"ratios": [], "error": "Không đọc được ảnh: " + req.name}
    ratios = []
    for p in req.polygons:
        cnt, total = _flower_counts(img, p, req.sat_max, req.val_min)
        ratios.append(round(100.0 * cnt / total, 2) if total else 0.0)
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
        return {"polygons": [], "classes": [], "ratios": [], "manual": [],
                "categories": []}
    data = json.load(open(fp, encoding="utf-8"))
    polys, cls, ratios, manual = [], [], [], []
    if isinstance(data.get("polygons"), list):
        # Định dạng mới: polygons=[{points:[[x,y],...], flower_label}]
        for p in data["polygons"]:
            pts = p.get("points") or []
            flat = [float(v) for xy in pts for v in xy]
            if len(flat) >= 6:
                polys.append(flat)
                cls.append(p.get("flower_label", 0))
                r = p.get("flower_ratio")
                # file lưu tỉ lệ 0-1, giao diện hiện %.
                ratios.append(round(100.0 * r, 2)
                              if isinstance(r, (int, float)) else None)
                manual.append(p.get("label_source") == "manual")
    else:
        # Định dạng COCO cũ.
        for a in data.get("annotations", []):
            if a.get("segmentation"):
                polys.append(a["segmentation"][0])
                cls.append(a.get("category_id", 1))
                ratios.append(None)
                manual.append(False)
    return {"polygons": polys, "classes": cls, "ratios": ratios,
            "manual": manual, "categories": data.get("categories", [])}


@app.post("/api/save")
def save(req: SaveReq):
    # Lưu theo ĐỊNH DẠNG MỚI (giống file mẫu): path/field/.../polygons/flower_stats.
    os.makedirs(OUT_DIR, exist_ok=True)
    out = _json_path(req.name)
    rel = _full_rel(req.name)
    polys, confs, cls, man = [], [], [], []
    for i, poly in enumerate(req.polygons):
        if len(poly) >= 6:
            polys.append(poly)
            confs.append(req.confs[i] if i < len(req.confs) else 1.0)
            cls.append(req.classes[i] if i < len(req.classes) else None)
            man.append(bool(req.manual[i]) if i < len(req.manual) else False)
    if not polys:
        # Không còn vùng nào -> xoá file cũ để ảnh không bị đánh dấu "đã nhãn".
        if os.path.exists(out):
            os.remove(out)
        return {"saved": None, "count": 0}
    img = _imread(os.path.join(ROOT, req.name))
    # Ảnh không đọc được (đường dẫn Unicode, ảnh bị di chuyển...) thì KHÔNG được
    # tính lại ra 0% rồi ghi đè -> giữ nguyên số cũ theo vị trí và báo lên UI.
    prev = _prev_flower(out) if img is None else []
    stats = {nm: 0 for nm in FLOWER_NAMES}
    ratios, out_polys = [], []
    for i, (poly, conf, cid, is_man) in enumerate(zip(polys, confs, cls, man)):
        xs, ys = poly[0::2], poly[1::2]
        old = prev[i] if i < len(prev) else None
        if img is not None:
            fpx, tpx = _flower_counts(img, poly, req.sat_max, req.val_min)
            ratio = round(fpx / tpx, 4) if tpx else 0.0
        elif old is not None:
            fpx, tpx, ratio = old["flower_pixels"], old["total_pixels"], old["ratio"]
        else:
            fpx, tpx, ratio = 0, 0, None    # chưa xác định được, không phải 0%
        # Nhãn người gán tay THẮNG HSV; chỉ suy từ ratio khi vùng đang ở chế độ auto.
        lvl = _as_level(cid) if is_man else None
        if lvl is not None:
            lname = FLOWER_NAMES[lvl]
        elif ratio is not None:
            is_man = False
            lvl, lname = _flower_label(ratio)
        else:
            # Không có ảnh và cũng không có số cũ -> để trống, export sẽ bỏ qua
            # vùng này thay vì gán bừa mức 0.
            is_man = False
            lvl, lname = None, None
        if lname:
            stats[lname] += 1
        if ratio is not None:
            ratios.append(ratio)
        out_polys.append(dict(
            points=[[float(x), float(y)] for x, y in zip(xs, ys)],
            conf=round(float(conf), 4),
            bbox=[round(float(min(xs)), 1), round(float(min(ys)), 1),
                  round(float(max(xs)), 1), round(float(max(ys)), 1)],
            flower_pixels=int(fpx), total_pixels=int(tpx),
            flower_ratio=ratio, flower_label=lvl, flower_label_name=lname,
            label_source=("manual" if is_man else
                          ("auto" if lvl is not None else "unknown"))))
    doc = dict(path=rel, field=_field_of(rel),
               img_w=req.width, img_h=req.height,
               n_canopy=len(out_polys), conf_thr=DEFAULT_CONF_THR,
               # Ghi lại ngưỡng đã dùng -> tỉ lệ hoa mới tái lập được về sau.
               hsv=dict(sat_max=int(req.sat_max), val_min=int(req.val_min),
                        thresholds=list(FLOWER_THRESHOLDS)),
               polygons=out_polys,
               flower_stats=dict(
                   no_flower=stats["no_flower"], few_flowers=stats["few_flowers"],
                   many_flowers=stats["many_flowers"],
                   very_many_flowers=stats["very_many_flowers"],
                   avg_ratio=round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
                   max_ratio=round(max(ratios), 4) if ratios else 0.0))
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    res = {"saved": out, "count": len(out_polys)}
    if img is None:
        res["warning"] = ("Không đọc được ảnh %s — giữ nguyên mức hoa cũ, "
                          "không tính lại." % req.name)
    return res


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


def _now():
    import datetime
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _hsv_used(items):
    # Ngưỡng HSV đã dùng khi gán nhãn. File cũ chưa có khối này -> trả mặc định
    # và nói rõ là suy đoán, để người đọc dataset không tưởng là đã xác nhận.
    seen = {}
    for _, jp in items:
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            continue
        h = d.get("hsv")
        if isinstance(h, dict) and "sat_max" in h:
            seen[(h.get("sat_max"), h.get("val_min"))] = h
    if not seen:
        return {"sat_max": 50, "val_min": 180, "assumed": True}
    if len(seen) == 1:
        return list(seen.values())[0]
    return {"mixed": [dict(sat_max=k[0], val_min=k[1]) for k in sorted(seen)]}


def _clip_poly(poly, w, h):
    # Kẹp polygon vào trong khung ảnh. CHỈ dùng lúc xuất — file nhãn giữ nguyên
    # đúng những gì người gán đã vẽ, kể cả phần trườn ra ngoài mép.
    out = []
    for j in range(0, len(poly) - 1, 2):
        out.append(min(max(float(poly[j]), 0.0), float(w)))
        out.append(min(max(float(poly[j + 1]), 0.0), float(h)))
    return out


def _size_of(name, w, h):
    # img_w/img_h thiếu trong file nhãn -> đọc từ chính ảnh.
    if w and h:
        return w, h
    im = _imread(os.path.join(ROOT, name))
    if im is None:
        return 0, 0
    return im.shape[1], im.shape[0]


# ===================== XUẤT DATASET (train-ready) =====================
# MỘT đường xuất duy nhất cho mọi định dạng. Trước đây có hai bộ song song
# (_export_coco/_export_yolo và _export_dataset) tự đánh số lớp khác nhau:
# cùng thư mục field_2/10/2 ra nc=3 ở bộ này và nc=4 ở bộ kia. Giờ FLOWER_NAMES
# là nguồn duy nhất: nc = 4, category_id = mức hoa, không đánh số lại bao giờ.
class DatasetReq(BaseModel):
    scope: str = ""                       # lọc theo thư mục con của ROOT (""=tất cả)
    formats: list = ["coco", "yolo"]      # coco | yolo | masks | masks_instance
    split_by: str = "none"                # "none" | "ratio" | "field"
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    val_fields: list = []                 # dùng khi split_by="field"
    test_fields: list = []
    seed: int = 42
    name: str = "dataset"                 # tên thư mục output dưới out_dir
    out_dir: str = ""                     # nơi lưu (rỗng = EXPORT_DIR mặc định)
    overwrite: bool = False               # ghi đè thư mục cùng tên đã có


SPLITS = ("train", "val", "test")


def _field_key(name):
    return _field_of(_full_rel(name))


def _group_key(name):
    # Nhóm = thư mục chứa ảnh, tức từng đường bay (field_2/10/1). Chia theo tỉ lệ
    # được thực hiện TRONG từng nhóm để mọi tập đều có mặt ở mọi đường bay.
    rel = _full_rel(name)
    return rel.rsplit("/", 1)[0] if "/" in rel else rel


def _split_items(items, req):
    import random
    fields = sorted({_field_key(it[0]) for it in items})
    if req.split_by == "none":
        return {"all": list(items)}, fields
    splits = {k: [] for k in SPLITS}
    if req.split_by == "field":
        valf, testf = set(req.val_fields), set(req.test_fields)
        for it in items:
            f = _field_key(it[0])
            splits["test" if f in testf else
                   ("val" if f in valf else "train")].append(it)
        return splits, fields
    # split_by == "ratio": xáo trong từng nhóm rồi cắt. An toàn vì đã đo được
    # các khung UAV liên tiếp KHÔNG chồng lấn nhau (tương quan 0.14, bằng đúng
    # mức của hai khung ngẫu nhiên không liên quan).
    by_group = {}
    for it in items:
        by_group.setdefault(_group_key(it[0]), []).append(it)
    rng = random.Random(req.seed)
    vr = max(0.0, min(1.0, req.val_ratio))
    tr = max(0.0, min(1.0 - vr, req.test_ratio))
    for g in sorted(by_group):
        lst = sorted(by_group[g])
        rng.shuffle(lst)
        n = len(lst)
        n_val = int(round(n * vr))
        n_test = int(round(n * tr))
        # Nhóm quá nhỏ: ưu tiên train có dữ liệu, đừng để train rỗng.
        if n - n_val - n_test < 1:
            n_val = min(n_val, max(0, n - 1))
            n_test = max(0, n - 1 - n_val)
        splits["val"].extend(lst[:n_val])
        splits["test"].extend(lst[n_val:n_val + n_test])
        splits["train"].extend(lst[n_val + n_test:])
    return splits, fields


def _semantic_mask(recs, w, h):
    # Mask semantic: 0=nền, 1..4 = mức hoa+1 (vẽ mức cao đè lên khi chồng lấn).
    # LƯU Ý: định dạng này chỉ ghi được LOẠI, không ghi được CÂY NÀO — hai tán
    # cùng mức nằm sát nhau sẽ dính thành một khối. Cần đếm cây thì dùng
    # masks_instance/ hoặc COCO.
    mask = np.zeros((h, w), np.uint8)
    for r in sorted(recs, key=lambda x: x["cat_id"]):
        pts = np.round(np.array(r["poly"], dtype=np.float64)
                       .reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], int(r["cat_id"]) + 1)
    return mask


def _instance_mask(recs, w, h):
    # Mask instance: 0=nền, 1..N = từng tán riêng, khớp thứ tự với COCO.
    # Vẽ tán LỚN TRƯỚC để tán nhỏ nằm đè lên: các tán chồng lấn nhau khá nhiều
    # (đo được 138 cặp dùng chung 107.591 pixel), nếu vẽ theo thứ tự gốc thì tán
    # nhỏ nằm lọt trong tán lớn sẽ bị phủ kín và mất hẳn id.
    mask = np.zeros((h, w), np.uint16)
    order = sorted(range(len(recs)),
                   key=lambda i: _poly_area(recs[i]["poly"]), reverse=True)
    for i in order:
        pts = np.round(np.array(recs[i]["poly"], dtype=np.float64)
                       .reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], int(i + 1))
    return mask


def _write_coco(per_image, out_file, hsv):
    cats = [dict(id=i, name=FLOWER_NAMES[i], supercategory="canopy")
            for i in range(len(FLOWER_NAMES))]
    images, anns = [], []
    iid = aid = 0
    for name, w, h, recs in per_image:
        iid += 1
        images.append(dict(id=iid, file_name=name.replace("/", "__"),
                           width=w, height=h, original_path=name))
        for r in recs:
            aid += 1
            xs, ys = r["poly"][0::2], r["poly"][1::2]
            anns.append(dict(
                id=aid, image_id=iid, category_id=int(r["cat_id"]),
                segmentation=[r["poly"]], area=float(_poly_area(r["poly"])),
                bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                iscrowd=0,
                # Giữ lại phần đo được: flower_ratio là số liên tục (dữ liệu gốc),
                # còn category_id chỉ là nó đã bị rời rạc thành 4 rổ.
                flower_ratio=r.get("flower_ratio"),
                flower_pixels=r.get("flower_pixels"),
                total_pixels=r.get("total_pixels"),
                label_source=r.get("label_source"),
                conf=r.get("conf")))
    info = dict(description="CoffeeSeg canopy + flower density",
                image_root=ROOT.replace("\\", "/"),
                date_created=_now(), flower_thresholds=list(FLOWER_THRESHOLDS),
                hsv=hsv)
    json.dump(dict(info=info, images=images, annotations=anns,
                   categories=cats),
              open(out_file, "w", encoding="utf-8"), ensure_ascii=False)
    return len(images), len(anns)


def _copy_file(src, dst):
    # Trả về True/False để người gọi ĐẾM được ảnh hỏng, thay vì bỏ qua im lặng
    # rồi sinh ra dataset có nhãn mà không có ảnh.
    try:
        with open(src, "rb") as fr, open(dst, "wb") as fw:
            fw.write(fr.read())
        return True
    except Exception:
        return False


def _write_yaml(root_out, no_split):
    # Bố cục chuẩn ultralytics: nó tìm nhãn bằng cách thay '/images/' cuối cùng
    # trong đường dẫn ảnh thành '/labels/'.
    lines = ["path: " + os.path.abspath(root_out).replace("\\", "/")]
    if no_split:
        # CHƯA chia thì KHÔNG ghi 'val:'. Trỏ val vào chính train sẽ cho ra mAP
        # đẹp mà vô nghĩa; thiếu khoá val thì ultralytics dừng ngay với
        # "'val:' key missing", hỏng to còn hơn hỏng ngầm.
        lines += ["train: images",
                  "# CHUA CHIA train/val. Bo comment dong duoi sau khi chia:",
                  "# val: images/val"]
    else:
        lines += ["train: images/train", "val: images/val", "test: images/test"]
    lines += ["nc: %d" % len(FLOWER_NAMES), "names:"]
    for i, nm in enumerate(FLOWER_NAMES):
        lines.append("  %d: %s" % (i, nm))
    with open(os.path.join(root_out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _export_dataset(items, req):
    import shutil
    base_dir = req.out_dir.strip() or EXPORT_DIR
    root_out = os.path.join(base_dir, req.name)
    if os.path.exists(root_out):
        if not req.overwrite:
            return {"ok": False, "exists": True,
                    "dir": root_out.replace("\\", "/"),
                    "error": "Thư mục '%s' đã tồn tại." % req.name}
        # Chỉ xoá đúng <out_dir>/<name>, không bao giờ đụng thư mục người dùng chọn.
        shutil.rmtree(root_out, ignore_errors=True)
    splits, fields = _split_items(items, req)
    formats = set(req.formats or [])
    no_split = (req.split_by == "none")
    hsv = _hsv_used(items)
    summary, problems = {}, []
    written = set()

    for sp in (["all"] if no_split else list(SPLITS)):
        sp_items = splits.get(sp) or []
        summary[sp] = {"images": 0, "annotations": 0, "skipped_regions": 0}
        if not sp_items:
            continue
        sub = "" if no_split else sp
        img_out = os.path.join(root_out, "images", sub)
        os.makedirs(img_out, exist_ok=True)

        per_image = []          # (name, w, h, recs đã kẹp) dùng chung mọi format
        for name, jp in sp_items:
            w, h, recs, skipped = _read_records(jp)
            w, h = _size_of(name, w, h)
            if not w or not h:
                problems.append("%s: không xác định được kích thước ảnh" % name)
                continue
            if not _copy_file(os.path.join(ROOT, name),
                              os.path.join(img_out, name.replace("/", "__"))):
                problems.append("%s: không chép được ảnh gốc" % name)
                continue
            for r in recs:
                r["poly"] = _clip_poly(r["poly"], w, h)
            per_image.append((name, w, h, recs))
            summary[sp]["images"] += 1
            summary[sp]["annotations"] += len(recs)
            summary[sp]["skipped_regions"] += skipped
        if not per_image:
            continue

        if "coco" in formats:
            cdir = os.path.join(root_out, "coco")
            os.makedirs(cdir, exist_ok=True)
            _write_coco(per_image, os.path.join(
                cdir, ("instances" if no_split else sp) + ".json"), hsv)
            written.add("coco")
        if "yolo" in formats:
            yl = os.path.join(root_out, "labels", sub)
            os.makedirs(yl, exist_ok=True)
            for name, w, h, recs in per_image:
                rows = []
                for r in recs:
                    p = r["poly"]
                    coords = " ".join(
                        "%.6f" % min(1.0, max(0.0, p[j] / (w if j % 2 == 0 else h)))
                        for j in range(len(p)))
                    rows.append("%d %s" % (int(r["cat_id"]), coords))
                stem = name.replace("/", "__").rsplit(".", 1)[0]
                with open(os.path.join(yl, stem + ".txt"), "w",
                          encoding="utf-8") as f:
                    f.write("\n".join(rows))
            written.add("yolo")
        for fmt, sub_dir, fn in (("masks", "masks", _semantic_mask),
                                 ("masks_instance", "masks_instance",
                                  _instance_mask)):
            if fmt not in formats:
                continue
            mdir = os.path.join(root_out, sub_dir, sub)
            os.makedirs(mdir, exist_ok=True)
            for name, w, h, recs in per_image:
                stem = name.replace("/", "__").rsplit(".", 1)[0]
                if not _imwrite(os.path.join(mdir, stem + ".png"),
                                fn(recs, w, h)):
                    problems.append("%s: không ghi được %s" % (name, sub_dir))
            written.add(fmt)

    os.makedirs(root_out, exist_ok=True)
    if "yolo" in written:
        _write_yaml(root_out, no_split)
    meta = dict(
        name=req.name, created=_now(),
        source_root=ROOT.replace("\\", "/"), scope=req.scope, fields=fields,
        formats=sorted(written),
        classes={i: nm for i, nm in enumerate(FLOWER_NAMES)},
        split=dict(mode=req.split_by, seed=req.seed,
                   val_ratio=req.val_ratio, test_ratio=req.test_ratio),
        splits=summary,
        image_naming="flattened: field__<...>__file.ext",
        # Quy ước mask LỆCH 1 so với category_id của COCO — phải ghi ra, nếu
        # không người train U-Net sẽ lệch một mức trên toàn bộ dataset.
        mask_values=("0=background, sau đó = flower_label + 1 "
                     "(1=no_flower ... 4=very_many_flowers); "
                     "COCO category_id = giá trị mask - 1"),
        instance_mask_values="0=background, 1..N = từng tán, khớp thứ tự COCO",
        flower=dict(thresholds=list(FLOWER_THRESHOLDS), hsv=hsv),
        problems=problems)
    json.dump(meta, open(os.path.join(root_out, "meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    out = dict(meta)
    out["dir"] = root_out.replace("\\", "/")
    out["ok"] = True
    return out


@app.post("/api/export_dataset")
def export_dataset(req: DatasetReq):
    items = list(_iter_labeled(req.scope))
    if not items:
        return {"ok": False,
                "error": "Không có ảnh nào đã gán nhãn trong phạm vi này."}
    return _export_dataset(items, req)


@app.post("/api/export")
def export(req: DatasetReq):
    # Giữ lại cho tương thích: cùng một đường xuất, chỉ khác điểm vào.
    return export_dataset(req)


@app.get("/api/export_preview")
def export_preview(scope: str = ""):
    # Cho hộp thoại xuất biết trước "sẽ xuất 50/51 ảnh · 446 vùng".
    root = Path(ROOT)
    sc = (scope or "").strip("/")
    total = 0
    labeled = 0
    regions = 0
    skipped = 0
    per_class = {i: 0 for i in range(len(FLOWER_NAMES))}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        name = p.relative_to(root).as_posix()
        if sc and not (name == sc or name.startswith(sc + "/")):
            continue
        total += 1
        jp = _labeled_json_path(name)
        if not os.path.exists(jp):
            continue
        labeled += 1
        try:
            _, _, recs, sk = _read_records(jp)
        except Exception:
            continue
        skipped += sk
        regions += len(recs)
        for r in recs:
            per_class[r["cat_id"]] = per_class.get(r["cat_id"], 0) + 1
    return {"scope": sc, "images_total": total, "images_labeled": labeled,
            "annotations": regions, "skipped_regions": skipped,
            "per_class": {FLOWER_NAMES[i]: per_class.get(i, 0)
                          for i in range(len(FLOWER_NAMES))}}


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


@app.get("/api/paths")
def get_paths():
    # Nơi autosave nhãn (cố định theo configs/config.yaml) và thư mục gợi ý khi
    # xuất. Không có endpoint đổi: bản cũ chỉ sửa biến toàn cục nên khởi động
    # lại server là mất, gây ra chuyện nhãn đi lạc chỗ mà không ai biết.
    return {"out_dir": OUT_DIR.replace("\\", "/"),
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