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

from app import flower as fl
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
FLOWER_THRESHOLDS = fl.THRESHOLDS   # tỉ lệ phủ hoa -> mức 0/1/2/3
FLOWER_NAMES = fl.NAMES


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
    return fl.level(ratio)


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


def _flower_counts(img_bgr, poly, sat_max, val_min):
    # Theo README mục 6 của bộ dữ liệu: TRÍCH XUẤT vùng polygon rồi mới đếm ->
    # blur/HSV/opening chạy TRONG từng tán, không phải trên cả ảnh. Thuật toán
    # nằm ở app/flower.py để scripts dùng chung. Trả về (số pixel hoa, tổng pixel).
    return fl.hsv_counts(img_bgr, poly, sat_max, val_min)


# Tham số của cả hai phương pháp đếm hoa, dùng chung cho /api/flower,
# /api/flower_mask và /api/save. `method` là phương pháp đang HIỂN THỊ: nó quyết
# định trường đầu của polygon khi lưu; cả hai kết quả đều được lưu bên cạnh.
class FlowerParams(BaseModel):
    sat_max: int = 50        # HSV: pixel hoa có S < sat_max
    val_min: int = 180       # HSV: pixel hoa có V > val_min
    method: str = "hsv"      # "hsv" | "otsu"
    channel: str = "white"   # Otsu: white | min2 | v2 | v (xem app/flower.py)
    s_max: int = 50          # Otsu kênh white: cổng màu theo pixel S < s_max
    class_s_max: float = 0.0 # Otsu: S trung bình của lớp chọn phải < ngưỡng, 0 = tắt
    sep_min: float = 0.0     # Otsu: độ tách tối thiểu, 0 = tắt
    area_min: int = 0        # Otsu: bỏ blob nhỏ hơn (px²), 0 = tắt
    area_max: int = 0        # Otsu: bỏ blob lớn hơn (px²), 0 = tắt
    elong_max: float = 0.0   # Otsu: bỏ blob dài/rộng vượt ngưỡng, 0 = tắt
    clip_rule: bool = False  # Otsu: bỏ blob có lõi bão hoà


def _check_params(req):
    if req.method not in ("hsv", "otsu"):
        return "method phải là hsv hoặc otsu"
    if req.channel not in fl.CHANNELS:
        return "channel phải là một trong " + ", ".join(fl.CHANNELS)
    return None


def _otsu_kwargs(req):
    return dict(channel=req.channel, s_max=req.s_max, class_s_max=req.class_s_max,
                sep_min=req.sep_min, area_min=req.area_min, area_max=req.area_max,
                elong_max=req.elong_max, clip_rule=req.clip_rule)


_OTSU_KEYS = ("flower_pixels", "ratio", "label", "channel", "threshold", "threshold1",
              "separability", "class_s", "n_blobs", "n_blobs_kept", "rejected")


def _both(img, poly, req):
    # Cả hai phương pháp cho một tán -> (khối hsv, khối otsu) đúng dạng lưu vào JSON.
    fpx, tpx = fl.hsv_counts(img, poly, req.sat_max, req.val_min)
    r = round(fpx / tpx, 4) if tpx else 0.0
    hsv = dict(flower_pixels=int(fpx), total_pixels=int(tpx), ratio=r, label=fl.level(r)[0])
    o = fl.otsu_blob(img, poly, **_otsu_kwargs(req))
    if o is None:
        o = dict(flower_pixels=0, ratio=0.0, label=0, channel=req.channel, threshold=None,
                 threshold1=None, separability=0.0, class_s=None, n_blobs=0, n_blobs_kept=0,
                 rejected="empty")
    otsu = {k: o.get(k) for k in _OTSU_KEYS}
    otsu["total_pixels"] = int(tpx)
    return hsv, otsu


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
                        ratio=r if isinstance(r, (int, float)) else None,
                        # Khối phụ của lần lưu trước (None với file cũ chưa có).
                        hsv=p.get("hsv") if isinstance(p.get("hsv"), dict) else None,
                        otsu=p.get("otsu") if isinstance(p.get("otsu"), dict) else None))
    return out


def _read_records(jp):
    # Đọc 1 file nhãn (định dạng MỚI hoặc COCO cũ) -> (w, h, [ann...]).
    # Bộ xuất chỉ có MỘT lớp "canopy": polygon là ranh giới tán, hết. Mức hoa
    # không còn là class index nữa mà chỉ là số đo đi kèm ra file phụ, nên không
    # vùng nào bị loại vì thiếu nó — mọi polygon đã vẽ đều được xuất.
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
            anns.append(dict(poly=poly, area=_poly_area(poly), cat_id=0,
                             flower_label=_as_level(p.get("flower_label")),
                             conf=p.get("conf"),
                             flower_ratio=p.get("flower_ratio"),
                             flower_pixels=p.get("flower_pixels"),
                             total_pixels=p.get("total_pixels"),
                             label_source=p.get("label_source")))
    else:
        im = (data.get("images") or [{}])[0]
        w = im.get("width", 0)
        h = im.get("height", 0)
        for a in data.get("annotations", []):
            seg = a.get("segmentation") or []
            if not seg:
                continue
            anns.append(dict(poly=seg[0], area=float(a.get("area", 0.0)),
                             cat_id=0,
                             flower_label=_as_level(a.get("category_id")),
                             conf=None, flower_ratio=None, flower_pixels=None,
                             total_pixels=None, label_source=None))
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


class SaveReq(FlowerParams):
    name: str
    width: int
    height: int
    polygons: list           # mỗi polygon = [x1, y1, x2, y2, ...]
    classes: list = []       # category_id cho từng polygon (song song polygons)
    manual: list = []        # True = mức hoa do người gán tay -> máy không ghi đè
    categories: list = []    # [{id, name, color}, ...] danh sách lớp của dự án
    confs: list = []         # confidence từng polygon (SAM/detector); mặc định 1.0
    # sat_max/val_min/method/tham số Otsu: từ FlowerParams — tính lại lúc lưu.


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


class FlowerReq(FlowerParams):
    name: str
    polygons: list          # danh sách polygon [x1, y1, x2, y2, ...]


def _otsu_brief(o):
    # Phần client cần để hiện tooltip; % thay vì tỉ lệ 0-1.
    return dict(ratio=round(100.0 * o["ratio"], 2), thr=o["threshold"], thr1=o["threshold1"],
                sep=o["separability"], class_s=o.get("class_s"), n_blobs=o["n_blobs"],
                n_kept=o["n_blobs_kept"], rejected=o["rejected"], channel=o["channel"])


@app.post("/api/flower")
def flower(req: FlowerReq):
    # Trả về % pixel hoa cho từng polygon theo CẢ HAI phương pháp; `ratios` là
    # của phương pháp đang chọn để client cũ vẫn chạy.
    err = _check_params(req)
    if err:
        return {"ratios": [], "error": err}
    img = _imread(os.path.join(ROOT, req.name))
    if img is None:
        return {"ratios": [], "error": "Không đọc được ảnh: " + req.name}
    hsv, otsu = [], []
    for p in req.polygons:
        h, o = _both(img, p, req)
        hsv.append(round(100.0 * h["ratio"], 2))
        otsu.append(_otsu_brief(o))
    ratios = hsv if req.method == "hsv" else [o["ratio"] for o in otsu]
    return {"ratios": ratios, "hsv": hsv, "otsu": otsu}


_MASK_COLOR = {"hsv": (0, 200, 255), "otsu": (255, 0, 255)}   # BGR: vàng / hồng


@app.post("/api/flower_mask")
def flower_mask(req: FlowerReq):
    # PNG trong suốt cỡ cả ảnh: pixel hoa của phương pháp đang chọn, trong mọi
    # polygon gửi lên. Client vẽ đè lên ảnh để NHÌN hai phương pháp bắt gì.
    err = _check_params(req)
    if err:
        return Response(json.dumps({"error": err}), status_code=400, media_type="application/json")
    img = _imread(os.path.join(ROOT, req.name))
    if img is None:
        return Response(json.dumps({"error": "Không đọc được ảnh: " + req.name}),
                        status_code=404, media_type="application/json")
    kw = _otsu_kwargs(req) if req.method == "otsu" else {}
    mask = fl.render_mask(img, req.polygons, req.method, req.sat_max, req.val_min, **kw)
    png = fl.mask_png(mask, _MASK_COLOR[req.method])
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


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
    hsv_r, otsu_r, otsu_i = [], [], []

    def pct(r):
        # file lưu tỉ lệ 0-1, giao diện hiện %.
        return round(100.0 * r, 2) if isinstance(r, (int, float)) else None

    if isinstance(data.get("polygons"), list):
        # Định dạng mới: polygons=[{points:[[x,y],...], flower_label}]
        for p in data["polygons"]:
            pts = p.get("points") or []
            flat = [float(v) for xy in pts for v in xy]
            if len(flat) >= 6:
                polys.append(flat)
                cls.append(p.get("flower_label", 0))
                ratios.append(pct(p.get("flower_ratio")))
                manual.append(p.get("label_source") == "manual")
                h, o = p.get("hsv"), p.get("otsu")
                # File cũ chưa có khối phụ: trường đầu chính là HSV.
                hsv_r.append(pct(h.get("ratio")) if isinstance(h, dict) else pct(p.get("flower_ratio")))
                if isinstance(o, dict):
                    otsu_r.append(pct(o.get("ratio")))
                    otsu_i.append(dict(thr=o.get("threshold"), thr1=o.get("threshold1"),
                                       sep=o.get("separability"), class_s=o.get("class_s"),
                                       n_blobs=o.get("n_blobs"), n_kept=o.get("n_blobs_kept"),
                                       rejected=o.get("rejected"), channel=o.get("channel")))
                else:
                    otsu_r.append(None)
                    otsu_i.append(None)
    else:
        # Định dạng COCO cũ.
        for a in data.get("annotations", []):
            if a.get("segmentation"):
                polys.append(a["segmentation"][0])
                cls.append(a.get("category_id", 1))
                ratios.append(None)
                manual.append(False)
                hsv_r.append(None)
                otsu_r.append(None)
                otsu_i.append(None)
    return {"polygons": polys, "classes": cls, "ratios": ratios,
            "manual": manual, "categories": data.get("categories", []),
            "hsv_ratios": hsv_r, "otsu_ratios": otsu_r, "otsu_info": otsu_i,
            "flower_method": data.get("flower_method", "hsv")}


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
    err = _check_params(req)
    if err:
        return {"saved": None, "count": 0, "error": err}
    img = _imread(os.path.join(ROOT, req.name))
    # Ảnh không đọc được (đường dẫn Unicode, ảnh bị di chuyển...) thì KHÔNG được
    # tính lại ra 0% rồi ghi đè -> giữ nguyên số cũ theo vị trí và báo lên UI.
    prev = _prev_flower(out) if img is None else []
    stats = {nm: 0 for nm in FLOWER_NAMES}
    ratios, out_polys = [], []
    for i, (poly, conf, cid, is_man) in enumerate(zip(polys, confs, cls, man)):
        xs, ys = poly[0::2], poly[1::2]
        old = prev[i] if i < len(prev) else None
        hsv_blk = otsu_blk = None
        if img is not None:
            hsv_blk, otsu_blk = _both(img, poly, req)
            sel = hsv_blk if req.method == "hsv" else otsu_blk
            fpx, tpx, ratio = sel["flower_pixels"], sel["total_pixels"], sel["ratio"]
        elif old is not None:
            fpx, tpx, ratio = old["flower_pixels"], old["total_pixels"], old["ratio"]
            hsv_blk, otsu_blk = old["hsv"], old["otsu"]
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
        source = "manual" if is_man else ("auto" if lvl is not None else "unknown")
        entry = dict(
            points=[[float(x), float(y)] for x, y in zip(xs, ys)],
            conf=round(float(conf), 4),
            bbox=[round(float(min(xs)), 1), round(float(min(ys)), 1),
                  round(float(max(xs)), 1), round(float(max(ys)), 1)],
            # Trường đầu = phương pháp đang chọn (req.method); cả hai khối bên dưới.
            flower_pixels=int(fpx), total_pixels=int(tpx),
            flower_ratio=ratio, flower_label=lvl, flower_label_name=lname,
            label_source=source,
            label_method=(req.method if source == "auto" else None))
        if hsv_blk is not None:
            entry["hsv"] = hsv_blk
        if otsu_blk is not None:
            entry["otsu"] = otsu_blk
        out_polys.append(entry)
    doc = dict(path=rel, field=_field_of(rel),
               img_w=req.width, img_h=req.height,
               n_canopy=len(out_polys), conf_thr=DEFAULT_CONF_THR,
               flower_method=req.method,
               # Ghi lại tham số đã dùng -> tỉ lệ hoa mới tái lập được về sau.
               hsv=dict(sat_max=int(req.sat_max), val_min=int(req.val_min),
                        thresholds=list(FLOWER_THRESHOLDS)),
               otsu=dict(**_otsu_kwargs(req), thresholds=list(FLOWER_THRESHOLDS)),
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
    split_by: str = "none"                # "none" | "ratio" | "field" | "folder"
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    val_fields: list = []                 # dùng khi split_by="field"
    test_fields: list = []
    val_folders: list = []                # dùng khi split_by="folder" (đường bay)
    test_folders: list = []
    seed: int = 42
    # Bộ xuất chỉ có MỘT lớp "canopy". Mức hoa là thứ để xem và để module mật độ
    # hoa đọc riêng, không phải lớp cần học: đẩy nó thành class index sẽ khiến
    # class-aware NMS coi cùng một tán ở hai mức là hai vật thể, một cây ra hai
    # instance. Dữ liệu ở đây là polygon cho bài toán phân đoạn tán, thế thôi.
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
    if req.split_by == "folder":
        # Gán NGUYÊN một đường bay vào một tập. Đây là cách chia duy nhất không
        # rò rỉ: trong mỗi lượt bay có những khung liền kề chụp trúng cùng mấy
        # cây (đo được 35/392 cặp, ở mọi đường bay), nên tách chúng ra hai tập
        # là rò rỉ cấp đối tượng. Để cả hai cùng một bên thì chồng lấn vô hại.
        # Khớp theo tiền tố: "field_1" ăn cả 4 folder, "field_2/10/2" ăn đúng 1.
        def _pick(g, names):
            return any(g == x or g.startswith(x.rstrip("/") + "/")
                       for x in names if x)
        for it in items:
            g = _group_key(it[0])
            splits["test" if _pick(g, req.test_folders) else
                   ("val" if _pick(g, req.val_folders) else "train")].append(it)
        return splits, fields
    # split_by == "ratio": xáo trong từng nhóm rồi cắt.
    # CẢNH BÁO: phép xáo này tách các khung liền kề của CÙNG một lượt bay ra hai
    # tập khác nhau. Comment cũ ở đây khẳng định là an toàn vì "khung liên tiếp
    # không chồng lấn (tương quan 0.14)" — số đó đo bằng matchTemplate với ngưỡng
    # hiệu chuẩn trên cặp KHÁC field, nên nó đo độ giống vân ảnh chứ không đo
    # cùng mảnh đất; các cặp cách nhau 30 khung (không thể chồng lấn) cũng vượt
    # ngưỡng 24-37%. Đo lại bằng SIFT+RANSAC: 35/392 cặp liền kề chồng lấn thật.
    # -> Dùng split_by="folder" nếu cần con số val/test đáng tin.
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
    # Mask nhị phân: 0=nền, 1=tán. Chỉ ghi được ĐÂU LÀ TÁN, không ghi được TÁN
    # NÀO — hai tán sát nhau dính thành một khối. Cần đếm cây thì dùng
    # masks_instance/ hoặc COCO.
    mask = np.zeros((h, w), np.uint8)
    for r in recs:
        pts = np.round(np.array(r["poly"], dtype=np.float64)
                       .reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
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


# COCO ở đây là COCO THUẦN: images chỉ có id/file_name/width/height, annotation
# chỉ có 7 trường của đặc tả, không kèm gì riêng của dự án.
# cat_id trong recs là class index 0-based (đúng thứ YOLO cần, ghi thẳng).
# COCO thì đánh số lớp TỪ 1: bộ COCO gốc dùng id 1..90 và chừa 0 cho nền, nên
# mọi công cụ đọc COCO đều làm `category_id - 1` để quy về class index. Ghi 0 ra
# COCO thì chúng cho ra lớp -1 mà không báo lỗi (thử với
# ultralytics.data.converter.convert_coco: ra dòng "-1 0.1 0.1 ..." kèm thông
# báo "converted successfully").
# Nên hai định dạng LỆCH NHAU 1 là cố ý, không phải lỗi: mỗi bên theo lệ của
# chính nó. Đừng "sửa" cho chúng giống nhau.
COCO_CAT_BASE = 1


def _write_coco(per_image, out_file):
    cats = [dict(id=COCO_CAT_BASE, name="canopy", supercategory="canopy")]
    images, anns = [], []
    iid = aid = 0
    for name, rel, w, h, recs in per_image:
        iid += 1
        images.append(dict(id=iid, file_name=rel.replace("/", "__"),
                           width=w, height=h))
        for r in recs:
            aid += 1
            xs, ys = r["poly"][0::2], r["poly"][1::2]
            anns.append(dict(
                id=aid, image_id=iid,
                category_id=int(r["cat_id"]) + COCO_CAT_BASE,
                segmentation=[r["poly"]], area=float(_poly_area(r["poly"])),
                bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                iscrowd=0))
    info = dict(description="CoffeeSeg coffee canopy segmentation",
                version="1.0", year=int(_now()[:4]), contributor="",
                date_created=_now())
    json.dump(dict(info=info, licenses=[], images=images, annotations=anns,
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


def _write_yaml(root_out, no_split, summary=None):
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
        # CHỈ khai split thực sự có ảnh. Khai một thư mục rỗng thì ultralytics
        # chết vì không tìm thấy đường dẫn — người đọc lỗi sẽ đi tìm nhầm chỗ.
        # Thiếu hẳn khoá val thì nó nói thẳng "'val:' key missing", đúng ý đồ ở
        # nhánh trên. Tệ hơn nữa là 'test' rỗng: train vẫn chạy ngon, lỗi chỉ nổ
        # ra lúc đo trên test, có khi vài tiếng sau.
        sm = summary or {}
        for k in SPLITS:
            if (sm.get(k) or {}).get("images"):
                lines.append("%s: images/%s" % (k, k))
            else:
                lines.append("# %s: images/%s  <- khong co anh nao" % (k, k))
    lines += ["nc: 1", "names:", "  0: canopy"]
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
    summary, problems = {}, []
    written = set()

    for sp in (["all"] if no_split else list(SPLITS)):
        sp_items = splits.get(sp) or []
        summary[sp] = {"images": 0, "annotations": 0}
        if not sp_items:
            continue
        sub = "" if no_split else sp
        img_out = os.path.join(root_out, "images", sub)
        os.makedirs(img_out, exist_ok=True)

        # (name, rel, w, h, recs đã kẹp). name để ĐỌC từ ROOT, rel để ĐẶT TÊN.
        # Tên đầu ra phải tính theo BASE: trước đây nó dùng name (tương đối so
        # với ROOT) nên chọn một field ở panel trái là mất luôn tên field trong
        # tên file — cùng một tấm ảnh xuất ra hai tên khác nhau tuỳ thiết lập
        # giao diện. Nhãn và khoá gom nhóm vốn đã theo BASE, giờ tên khớp nốt.
        per_image = []
        for name, jp in sp_items:
            w, h, recs = _read_records(jp)
            w, h = _size_of(name, w, h)
            if not w or not h:
                problems.append("%s: không xác định được kích thước ảnh" % name)
                continue
            rel = _full_rel(name)
            if not _copy_file(os.path.join(ROOT, name),
                              os.path.join(img_out, rel.replace("/", "__"))):
                problems.append("%s: không chép được ảnh gốc" % name)
                continue
            for r in recs:
                r["poly"] = _clip_poly(r["poly"], w, h)
            per_image.append((name, rel, w, h, recs))
            summary[sp]["images"] += 1
            summary[sp]["annotations"] += len(recs)
        if not per_image:
            continue

        if "coco" in formats:
            # Bố cục đúng lệ COCO: thư mục "annotations/", file mang tiền tố
            # "instances_" (bộ gốc là annotations/instances_train2017.json).
            # Tiền tố đó phân biệt với captions_/person_keypoints_ cùng nằm đó.
            cdir = os.path.join(root_out, "annotations")
            os.makedirs(cdir, exist_ok=True)
            _write_coco(per_image, os.path.join(
                cdir, "instances" + ("" if no_split else "_" + sp) + ".json"))
            written.add("coco")
        if "yolo" in formats:
            yl = os.path.join(root_out, "labels", sub)
            os.makedirs(yl, exist_ok=True)
            for name, rel, w, h, recs in per_image:
                rows = []
                for r in recs:
                    p = r["poly"]
                    coords = " ".join(
                        "%.6f" % min(1.0, max(0.0, p[j] / (w if j % 2 == 0 else h)))
                        for j in range(len(p)))
                    rows.append("%d %s" % (int(r["cat_id"]), coords))
                stem = rel.replace("/", "__").rsplit(".", 1)[0]
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
            for name, rel, w, h, recs in per_image:
                stem = rel.replace("/", "__").rsplit(".", 1)[0]
                if not _imwrite(os.path.join(mdir, stem + ".png"),
                                fn(recs, w, h)):
                    problems.append("%s: không ghi được %s" % (name, sub_dir))
            written.add(fmt)

    os.makedirs(root_out, exist_ok=True)
    if "yolo" in written:
        _write_yaml(root_out, no_split, summary)
    meta = dict(
        name=req.name, created=_now(),
        source_root=ROOT.replace("\\", "/"), scope=req.scope, fields=fields,
        formats=sorted(written),
        classes={COCO_CAT_BASE: "canopy"},
        split=dict(mode=req.split_by, seed=req.seed,
                   val_ratio=req.val_ratio, test_ratio=req.test_ratio),
        splits=summary,
        image_naming=("flattened from the path under source_base: "
                      "field__<...>__file.ext, '/' replaced by '__'"),
        source_base=BASE.replace("\\", "/"),
        coco_layout="annotations/instances_<split>.json",
        mask_values="0=background, 1=canopy",
        class_numbering=("COCO category_id = 1 (đúng lệ COCO: bộ gốc đánh 1..90, "
                         "chừa 0 cho nền). YOLO class index = 0 (đúng lệ YOLO). "
                         "Hai bên lệch nhau 1 là cố ý."),
        # Bộ xuất KHÔNG mang theo mật độ hoa: nó không phải nhãn của bài toán
        # phân đoạn tán. Số đo vẫn nằm nguyên trong file nhãn gốc dưới
        # masks/corrected/, lấy ra lúc nào cũng được.
        flower_note=("khong xuat; xem data/masks/corrected/*.json neu can"),
        instance_mask_values="0=background, 1..N = từng tán, khớp thứ tự COCO",
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
    folders = {}
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
            _, _, recs = _read_records(jp)
        except Exception:
            continue
        regions += len(recs)
        g = _group_key(name)
        folders[g] = folders.get(g, 0) + 1
    return {"scope": sc, "images_total": total, "images_labeled": labeled,
            "annotations": regions,
            "folders": [{"key": k, "images": folders[k]}
                        for k in sorted(folders)]}


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