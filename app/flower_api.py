# app/flower_api.py — Endpoint của thẻ Hoa (/api/fl/*): dựng vòng, hàng đợi, nét
# vẽ, Học & tô, chốt tán, đo đồng thuận.
#
# Polygon tán đọc từ file nhãn tán qua chính các hàm của server (chỉ đọc, không
# bao giờ ghi); nhãn hoa nằm riêng dưới <data.masks_dir>/<data_dir của vòng>/.
# Cấu hình vòng ở configs/flower_rounds.yaml, đọc lại mỗi lần cần.
#
# Gán đôi (khối `double` của vòng): người gán thứ hai chỉ thấy và chỉ sửa nét của
# chính mình, mask của họ nằm ở masks_<người>/, trạng thái ghi dưới vòng
# "<vòng>@<người>". Người gán chính không thấy nét của người thứ hai.
import json
import os
import statistics
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app import flower_rounds as fr
from app import flower_store as fs

router = APIRouter(prefix="/api/fl")
REPO = Path(__file__).resolve().parent.parent
# FLOWER_ROUNDS_CFG: chạy thử app với một bản cấu hình vòng khác (thư mục nháp).
CFG_PATH = Path(os.environ.get("FLOWER_ROUNDS_CFG") or REPO / "configs" / "flower_rounds.yaml")
REVIEWED = ("accepted", "fixed")
KINDS = ("scribble", "manual", "review")
CLF_DEFAULTS = dict(n_trees=100, max_depth=12, sigma_min=0.7, sigma_max=5.0,
                    num_sigma=4, max_samples=50000, unlabeled_ratio=4, unlabeled_weight=0.1)
SP_DEFAULTS = dict(area_px=50, compactness=10)
ROWS_PER_CROWN = 20000      # nét tô to cũng không để một tán lấn át cả vòng

_LOCK = threading.RLock()   # endpoint đồng bộ chạy trên nhiều luồng
_STORES = {}
_BANK = {}                  # (vòng, ảnh, tán) -> (chữ ký các op, X, y)
_IMG = OrderedDict()        # ảnh RGB gần nhất (giữ 3)
_SEG = OrderedDict()        # siêu điểm ảnh theo tán (giữ 32)
_CACHE = None               # FeatureCache, tạo lúc cần (sklearn/skimage nạp lười)


def _srv():
    # Nạp lúc gọi: server import module này lúc khởi động.
    from app import server
    return server


def _err(msg, code=400):
    return JSONResponse({"ok": False, "error": msg}, status_code=code)


def round_(v, n=4):
    return float(np.round(v, n))


# ------------------------------------------------------------------ cấu hình
def load_config(path=None):
    """(cfg, danh sách lỗi). Vòng thiếu id, trùng id, sai loại đều bị báo."""
    p = Path(path or CFG_PATH)
    errors = []
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        return {"rounds": [], "data_dir": "flower_v1"}, ["Không đọc được %s: %s" % (p.name, e)]
    seen, rounds = set(), []
    for r in cfg.get("rounds") or []:
        rid, kind = r.get("id"), r.get("kind")
        if not rid:
            errors.append("Có vòng không có id")
            continue
        if rid in seen:
            errors.append("Trùng id vòng: %s" % rid)
            continue
        if "@" in str(rid):
            errors.append("Vòng %s: id không được chứa @" % rid)
            continue
        if kind not in KINDS:
            errors.append("Vòng %s: kind phải là một trong %s" % (rid, ", ".join(KINDS)))
            continue
        if r.get("role") not in (None, "test"):
            errors.append("Vòng %s: role chỉ có thể là test" % rid)
            continue
        seen.add(rid)
        r = dict(r)
        r["classifier"] = dict(CLF_DEFAULTS, **(r.get("classifier") or {}))
        r["superpixel"] = dict(SP_DEFAULTS, **(r.get("superpixel") or {}))
        r["pick"] = dict(r.get("pick") or {})
        rounds.append(r)
    cfg["rounds"] = rounds
    cfg.setdefault("data_dir", "flower_v1")
    return cfg, errors


def _round(rid):
    cfg, _ = load_config()
    for r in cfg["rounds"]:
        if r["id"] == rid:
            return r
    return None


def _store():
    cfg, _ = load_config()
    root = os.path.join(os.path.dirname(_srv().OUT_DIR), cfg["data_dir"])
    st = _STORES.get(root)
    if st is None:
        st = _STORES[root] = fs.FlowerStore(root)
    return st


# ------------------------------------------------------------ người gán thứ hai
def _sec(rnd):
    """Người gán thứ hai của vòng (gán đôi), hoặc None."""
    return ((rnd or {}).get("double") or {}).get("annotator")


def _is_sec(rnd, who):
    s = _sec(rnd)
    return bool(s) and who == s


def _ops_view(ops, rnd, who):
    """Thao tác mà người `who` thấy: gán đôi thì mỗi bên chỉ thấy của mình."""
    s = _sec(rnd)
    if not s:
        return ops
    return [o for o in ops if (o.get("who") == s) == (who == s)]


def _status_round(rnd, who):
    return rnd["id"] + "@" + who if _is_sec(rnd, who) else rnd["id"]


def _mask_who(rnd, who):
    return who if _is_sec(rnd, who) else None


# ------------------------------------------------------------- ảnh và tán
def _crowns(name):
    """(danh sách polygon, (h, w)) từ file nhãn tán — chỉ đọc."""
    s = _srv()
    jp = s._json_path(name)
    if not os.path.exists(jp):
        return [], (0, 0)
    w, h, anns = s._read_records(jp)
    w, h = s._size_of(name, w, h)
    return [a["poly"] for a in anns], (int(h), int(w))


def _rgb(name):
    s = _srv()
    key = (s.ROOT, name)
    if key in _IMG:
        _IMG.move_to_end(key)
        return _IMG[key]
    bgr = s._imread(os.path.join(s.ROOT, name))
    if bgr is None:
        raise FileNotFoundError("Không đọc được ảnh: " + name)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _IMG[key] = rgb
    while len(_IMG) > 3:
        _IMG.popitem(last=False)
    return rgb


def _feats(name, flat, crown, box, ccfg):
    """Đặc trưng của crop tán, tính trên crop nới thêm lề để bộ lọc ở mép tán
    không bị cụt, rồi cắt về đúng khung tán."""
    global _CACHE
    from app import flower_classifier as fc
    if _CACHE is None:
        _CACHE = fc.FeatureCache()
    params = {k: ccfg[k] for k in ("sigma_min", "sigma_max", "num_sigma")}
    key = (flat, int(crown), tuple(box), tuple(sorted(params.items())))

    def make():
        rgb = _rgb(name)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = box
        pad = int(np.ceil(3 * params["sigma_max"])) + 1
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        f = fc.features(rgb[Y0:Y1, X0:X1], **params)
        return np.ascontiguousarray(f[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0])
    return _CACHE.get(key, make)


def _seg(name, flat, crown, box, rnd):
    """Siêu điểm ảnh SLIC trong tán (0 ngoài tán), khoảng `area_px` px một mảnh."""
    sp = (rnd or {}).get("superpixel") or SP_DEFAULTS
    key = (flat, int(crown), tuple(box), sp["area_px"], sp["compactness"])
    if key in _SEG:
        _SEG.move_to_end(key)
        return _SEG[key]
    from skimage.segmentation import slic
    polys, _ = _crowns(name)
    inside = fs.crown_mask(polys[crown], box)
    rgb = _rgb(name)[box[1]:box[3], box[0]:box[2]]
    # SLIC trên cả khung tán rồi xoá phần ngoài tán. Bản có `mask=` của skimage
    # chậm gấp ~30 lần (19 s so với 0,7 s trên một tán 230k px của Field 3);
    # mảnh nào vắt qua mép tán thì nét cũng bị cắt theo polygon sau đó.
    n = max(1, int(rgb.shape[0] * rgb.shape[1] / max(1, sp["area_px"])))
    seg = slic(rgb, n_segments=n, compactness=sp["compactness"], start_label=1,
               channel_axis=-1).astype(np.int32)
    seg[~inside] = 0
    _SEG[key] = seg
    while len(_SEG) > 32:
        _SEG.popitem(last=False)
    return seg


def _recompose(st, flat, name, rnd=None, who=None):
    """Dựng lại mask của ảnh (của người `who` nếu vòng gán đôi) và lưu."""
    polys, (h, w) = _crowns(name)
    doc = st.load_ops(flat)
    view = dict(doc, ops=_ops_view(doc["ops"], rnd, who))
    base = None
    if not _is_sec(rnd, who) and st.base_path(flat).exists():
        base = st.load_base(flat, (h, w))
    mask = fs.compose((h, w), polys, view, base,
                      seg_for=lambda k, b: _seg(name, flat, k, b, rnd))
    st.save_mask(flat, mask, _mask_who(rnd, who))
    return mask, polys, view


def _run_dir(rnd):
    p = Path((rnd or {}).get("model_run") or "")
    return p if p.is_absolute() else REPO / p


def _unc(rnd):
    """{(ảnh, tán): entropy} từ crowns.csv của lượt dự đoán; thiếu file thì rỗng."""
    import csv
    p = _run_dir(rnd) / "crowns.csv"
    out = {}
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out[(r["image"], int(r["crown"]))] = float(r["entropy"])
                except (KeyError, ValueError):
                    continue
    return out


def _ensure_pred_base(st, rnd, flat, name, crowns):
    """Vòng duyệt: tán nào chưa có nền máy thì lấy dự đoán của model (xác suất
    ≥ 0,5) làm nền. Trả True nếu có tán vừa được thêm nền."""
    pred = fs._read_png(_run_dir(rnd) / "pred" / (os.path.splitext(flat)[0] + ".png"))
    polys, (h, w) = _crowns(name)
    if pred is None or pred.shape != (h, w):
        return False
    base = st.load_base(flat, (h, w))
    changed = False
    for k in crowns:
        if k >= len(polys):
            continue
        b = fs.crown_box(polys[k], w, h)
        inside = fs.crown_mask(polys[k], b)
        sub = base[b[1]:b[3], b[0]:b[2]]
        if (sub[inside] != fs.IGNORE).any():
            continue
        p = pred[b[1]:b[3], b[0]:b[2]]
        sub[inside] = ((p[inside] != fs.IGNORE) & (p[inside] >= 127)).astype(np.uint8)
        changed = True
    if changed:
        st.save_base(flat, base)
    return changed


def _crown_ops(doc, crown):
    return [o for o in doc["ops"] if int(o["crown"]) == int(crown)]


def _state_of(rows, flat, crown, rnd=None):
    for (img, k, r), row in rows.items():
        if img == flat and k == int(crown) and (rnd is None or r == rnd):
            return row
    return None


def _name_of(flat):
    """Ảnh phẳng -> tên theo ROOT hiện tại, hoặc None nếu nằm ngoài ROOT."""
    s = _srv()
    rel = os.path.relpath(os.path.join(s.BASE, flat.replace("__", "/")), s.ROOT).replace("\\", "/")
    return None if rel.startswith("..") else rel


def _members(st, rnd, who):
    """Tập (ảnh, tán) mà người `who` được gán trong vòng; None = vòng chọn tay."""
    if rnd["pick"].get("mode") == "manual":
        return None
    items = st.load_round(rnd["id"])["items"]
    if _is_sec(rnd, who):
        items = fr.double_subset(items, int(rnd["double"].get("crowns", 10)))
    return {(it["image"], int(it["crown"])) for it in items}


def overview():
    """Số tán hoa theo ruộng × vòng × trạng thái (đọc status.csv), và danh sách
    tán có polygon đã đổi sau khi gán hoa (bbox lưu lúc gán khác bbox bây giờ)."""
    st = _store()
    counts, checks = {}, []
    for (flat, crown, rnd), row in sorted(st.read_status().items()):
        field = flat.split("__", 1)[0]
        c = counts.setdefault(field, {}).setdefault(rnd, {})
        state = row.get("state") or "todo"
        c[state] = c.get(state, 0) + 1
        name = _name_of(flat)
        if name is None or not row.get("bbox"):
            continue
        polys, (h, w) = _crowns(name)
        try:
            saved = [int(v) for v in json.loads(row["bbox"])]
        except Exception:  # noqa: BLE001
            continue
        if crown >= len(polys):
            checks.append(dict(image=name, crown=crown, round=rnd, problem="tán không còn trong nhãn tán"))
        elif list(fs.crown_box(polys[crown], w, h)) != saved:
            checks.append(dict(image=name, crown=crown, round=rnd, problem="polygon tán đã đổi sau khi gán hoa"))
    return counts, checks


# ---------------------------------------------------------------- endpoint
@router.get("/config")
def config():
    cfg, errors = load_config()
    st = _store()
    built = st.all_rounds()
    rounds = [dict(id=r["id"], label=r.get("label") or r["id"], kind=r["kind"], pick=r["pick"],
                   double=r.get("double"), built=len((built.get(r["id"]) or {}).get("items") or []))
              for r in cfg["rounds"]]
    return {"rounds": rounds, "errors": errors,
            "data_dir": str(st.root).replace("\\", "/")}


@router.get("/crowns")
def crowns(name: str, round: str = "", who: str = "A"):  # noqa: A002 - tên tham số của API
    with _LOCK:
        polys, (h, w) = _crowns(name)
        flat = _srv()._flat(name)
        if not polys:
            return {"ok": True, "image": flat, "crowns": [], "size": [h, w]}
        st = _store()
        rnd = _round(round) if round else None
        if rnd and rnd["kind"] == "review":
            mine = [int(it["crown"]) for it in st.load_round(rnd["id"])["items"] if it["image"] == flat]
            if mine and _ensure_pred_base(st, rnd, flat, name, mine):
                _recompose(st, flat, name, rnd, who)
        doc = st.load_ops(flat)
        ops = _ops_view(doc["ops"], rnd, who)
        mask = st.load_mask(flat, (h, w), _mask_who(rnd, who) if rnd else None)
        rows = st.read_status()
        skey = _status_round(rnd, who) if rnd else None
        members = _members(st, rnd, who) if rnd else None
        out = []
        for k, poly in enumerate(polys):
            b = fs.crown_box(poly, w, h)
            row = _state_of(rows, flat, k, skey)
            n_ops = len([o for o in ops if int(o["crown"]) == k])
            out.append(dict(crown=k, bbox=list(b), points=poly,
                            area=int(fs.crown_mask(poly, b).sum()),
                            state=(row or {}).get("state") or ("doing" if n_ops else "todo"),
                            seconds=float((row or {}).get("seconds") or 0),
                            n_ops=n_ops, cover=round_(fs.cover(mask, poly)),
                            in_round=(members is None or (flat, k) in members)))
        return {"ok": True, "image": flat, "round": doc.get("round"), "size": [h, w], "crowns": out}


# Màu lớp phủ (RGB): mask hoa vàng; nét Hoa xanh lá, Không hoa đỏ, Bỏ qua xám.
YELLOW, GREEN, RED, GREY = (250, 204, 21), (34, 197, 94), (239, 68, 68), (148, 163, 184)


def _paint(img, sel, rgb, alpha):
    img[sel, 0], img[sel, 1], img[sel, 2] = rgb[2], rgb[1], rgb[0]   # cv2 ghi BGRA
    img[sel, 3] = alpha


@router.get("/view")
def view(name: str, crown: int, layer: str = "mask", round: str = "", who: str = "A"):  # noqa: A002
    """PNG trong suốt cắt theo khung tán; header X-Crop cho biết đặt ở đâu.
    mask: hoa vàng, vùng người đánh Bỏ qua gạch chéo xám. human: các nét."""
    with _LOCK:
        polys, (h, w) = _crowns(name)
        if not (0 <= crown < len(polys)):
            return _err("Không có tán %s trong ảnh này." % crown, 404)
        st = _store()
        rnd = _round(round) if round else None
        flat = _srv()._flat(name)
        poly = polys[crown]
        b = fs.crown_box(poly, w, h)
        inside = fs.crown_mask(poly, b)
        ops = _crown_ops(dict(ops=_ops_view(st.load_ops(flat)["ops"], rnd, who)), crown)
        seg = _seg(name, flat, crown, b, rnd) if any(o.get("snap") for o in ops) else None
        lay = fs.human_layer(ops, poly, b, seg)
        img = np.zeros((b[3] - b[1], b[2] - b[0], 4), np.uint8)
        if layer == "human":
            _paint(img, lay == 1, GREEN, 230)
            _paint(img, lay == 0, RED, 230)
            _paint(img, lay == fs.IGNORE, GREY, 230)
        else:
            m = st.load_mask(flat, (h, w), _mask_who(rnd, who) if rnd else None)[b[1]:b[3], b[0]:b[2]]
            _paint(img, inside & (m == 1), YELLOW, 255)
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            hatch = ((xx + yy) // 4) % 2 == 0
            _paint(img, inside & (lay == fs.IGNORE) & hatch, GREY, 200)
            _paint(img, inside & (lay == fs.IGNORE) & ~hatch, GREY, 70)
        ok, buf = cv2.imencode(".png", img)
        return Response(buf.tobytes(), media_type="image/png",
                        headers={"X-Crop": "%d,%d,%d,%d" % b, "Cache-Control": "no-store",
                                 "Access-Control-Expose-Headers": "X-Crop"})


class RoundReq(BaseModel):
    round: str
    force: bool = False


class CrownReq(BaseModel):
    round: str
    name: str
    crown: int
    who: str = "A"


class StrokeReq(CrownReq):
    label: object            # 0 | 1 | 255 | "erase"
    r: float = 4.0           # bán kính cọ, px ảnh
    snap: bool = False       # bám siêu điểm ảnh (vòng vẽ tay)
    pts: list                # [[x, y], ...] toạ độ ảnh


class FinishReq(CrownReq):
    seconds: float = 0.0


def _too_close(st, rnd, flat):
    """Ảnh phía bên kia (bộ chấm nếu `rnd` là vòng học, và ngược lại) nằm cùng
    đường bay và cách `flat` ≤ buffer_frames khung: (ảnh, vòng, số khung) hoặc
    None. Khung liền kề chồng ~45%, tức chụp trùng cây."""
    from app import flower_data as fd
    cfg, _ = load_config()
    buffer = max([int(r["pick"].get("buffer_frames", 0)) for r in cfg["rounds"]] + [0])
    seq = fr.frame_no(flat, None)
    if not buffer or seq is None:
        return None
    tests = {r["id"] for r in cfg["rounds"] if fr.is_test(r)}
    mine = rnd["id"] in tests
    flight = fd.flight_of(flat)
    for rid, doc in st.all_rounds().items():
        if rid == rnd["id"] or (rid in tests) == mine:
            continue
        for img in sorted({it["image"] for it in doc.get("items") or []}):
            s = fr.frame_no(img, None)
            if s is not None and fd.flight_of(img) == flight and abs(s - seq) <= buffer:
                return img, rid, abs(s - seq)
    return None


def _close_err(hit):
    img, rid, k = hit
    return _err("Ảnh này cách ảnh %s của vòng %s %d khung trong cùng đường bay (chụp trùng cây): bộ test và "
                "bộ train không được chung cây. Chọn ảnh khác, hoặc đặt buffer_frames: 0 trong "
                "configs/flower_rounds.yaml nếu chấp nhận." % (_name_of(img) or img, rid, k), 409)


def _check(req):
    """(vòng, polygon, (h, w), ảnh phẳng) hoặc JSONResponse lỗi."""
    rnd = _round(req.round)
    if rnd is None:
        return _err("Không có vòng %s trong configs/flower_rounds.yaml." % req.round)
    polys, (h, w) = _crowns(req.name)
    if not polys:
        return _err("Ảnh này chưa có nhãn tán.")
    if not (0 <= req.crown < len(polys)):
        return _err("Không có tán %s trong ảnh này." % req.crown, 404)
    s = _srv()
    fields = rnd["pick"].get("fields")
    if fields and s._field_of(s._full_rel(req.name)) not in fields:
        return _err("Vòng %s chỉ nhận ảnh của %s." % (req.round, ", ".join(fields)))
    flat = s._flat(req.name)
    st = _store()
    doc = st.load_ops(flat)
    if doc.get("round") and doc["round"] != req.round:
        # Một ảnh chỉ thuộc một vòng: ảnh của vòng này mà lẫn vào vòng khác
        # là rò rỉ giữa bộ chấm và bộ học.
        return _err("Ảnh này đã thuộc vòng %s." % doc["round"], 409)
    if not doc.get("round") and rnd["pick"].get("mode") == "manual":
        hit = _too_close(st, rnd, flat)
        if hit:
            return _close_err(hit)
    members = _members(st, rnd, req.who)
    if members is not None:
        if not st.load_round(rnd["id"])["items"]:
            return _err("Vòng %s chưa dựng: bấm Dựng vòng trước." % req.round)
        if (flat, int(req.crown)) not in members:
            return _err("Tán này không thuộc phần việc của người %s trong vòng %s." % (req.who, req.round))
    return rnd, polys, (h, w), flat


def _result(mask, polys, doc, crown, **extra):
    return dict(ok=True, crown=crown, n_ops=len(_crown_ops(doc, crown)),
                cover=round_(fs.cover(mask, polys[crown])), **extra)


@router.post("/stroke")
def stroke(req: StrokeReq):
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        lab = req.label
        if lab != "erase":
            try:
                lab = int(lab)
            except (TypeError, ValueError):
                return _err("label phải là 0, 1, 255 hoặc erase")
            if lab not in fs.LABELS:
                return _err("label phải là 0, 1, 255 hoặc erase")
        pts = [[float(p[0]), float(p[1])] for p in (req.pts or []) if len(p) >= 2]
        if not pts:
            return _err("Nét không có điểm nào.")
        st = _store()
        if rnd["kind"] == "review":
            row = _state_of(st.read_status(), flat, req.crown, _status_round(rnd, req.who)) or {}
            if row.get("state") not in ("fixing", "fixed"):
                return _err("Vòng duyệt: bấm Sửa (R) trước rồi mới vẽ lên tán này.")
        st.add_op(flat, dict(round=req.round, crown=int(req.crown), who=req.who, kind="stroke",
                             label=lab, r=round_(max(0.5, req.r), 2),
                             snap=bool(req.snap and rnd["kind"] == "manual"), pts=pts))
        b = fs.crown_box(polys[req.crown], w, h)
        if rnd["pick"].get("mode") == "manual":
            ks = range(len(polys)) if rnd["pick"].get("whole_image") else [int(req.crown)]
            st.add_round_items(req.round, _image_items(req.name, flat, polys, (h, w), ks))
        mask, polys, doc = _recompose(st, flat, req.name, rnd, req.who)
        skey = _status_round(rnd, req.who)
        row = _state_of(st.read_status(), flat, req.crown, skey) or {}
        st.set_status(flat, req.crown, skey, who=req.who, n_ops=str(len(_crown_ops(doc, req.crown))),
                      state="done" if row.get("state") == "done" else "doing",
                      source=row.get("source") or "human", bbox=list(b))
        return _result(mask, polys, doc, req.crown)


@router.post("/undo")
def undo(req: CrownReq):
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        st = _store()
        s = _sec(rnd)
        keep = (lambda o: (o.get("who") == s) == (req.who == s)) if s else None
        op, _ = st.undo(flat, req.crown, keep)
        mask, polys, doc = _recompose(st, flat, req.name, rnd, req.who)
        if op is not None:
            st.set_status(flat, req.crown, _status_round(rnd, req.who),
                          n_ops=str(len(_crown_ops(doc, req.crown))))
        return _result(mask, polys, doc, req.crown, removed=(op or {}).get("id"))


def _rows(st, rnd, item, ccfg):
    """Mẫu học của một tán trong vòng: (X, y, w). Nhớ lại theo chữ ký các thao
    tác của tán, đổi nét thì tính lại.

    Ngoài pixel dưới nét (trọng số 1), lấy thêm ngẫu nhiên pixel CHƯA TÔ trong
    tán làm mẫu "không hoa" yếu: `unlabeled_ratio` lần số pixel nét Hoa, trọng số
    `unlabeled_weight`. Hoa chỉ chiếm phần nhỏ của tán nên phần lớn mẫu đó đúng
    là không hoa. Không có chúng, máy chỉ thấy vài chấm Không hoa người vẽ và đoán
    bừa phần còn lại: thử trên một tán Field 3 (31 nét Hoa, 5 chấm Không hoa), máy
    tô 93% tán là hoa; có chúng thì 10%, nằm trên các cụm trắng. Trọng số nhỏ (0,1)
    vì trong đó có cả hoa chưa tô: ở 0,3 máy bắt đầu bỏ viền các bông hoa. Nhờ vậy
    chỉ cần nét Hoa là học được; nét Không hoa dùng để sửa chỗ máy nhầm."""
    flat, name, crown = item["image"], item["name"], int(item["crown"])
    ops = _crown_ops(st.load_ops(flat), crown)
    sig = (tuple(o["id"] for o in ops), ccfg.get("unlabeled_ratio"), ccfg.get("unlabeled_weight"))
    key = (rnd, flat, crown)
    hit = _BANK.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1], hit[2], hit[3]
    polys, (h, w) = _crowns(name)
    X, y, wt = None, None, None
    if crown < len(polys) and ops:
        poly = polys[crown]
        b = fs.crown_box(poly, w, h)
        inside = fs.crown_mask(poly, b)
        lay = fs.human_layer(ops, poly, b)
        sel = (lay == 0) | (lay == 1)
        if sel.any():
            rng = np.random.default_rng(crown)
            F = _feats(name, flat, crown, b, ccfg)
            X, y = F[sel], lay[sel]
            if len(y) > ROWS_PER_CROWN:
                idx = rng.choice(len(y), ROWS_PER_CROWN, replace=False)
                X, y = X[idx], y[idx]
            wt = np.ones(len(y), np.float32)
            n_u = int(float(ccfg.get("unlabeled_ratio") or 0) * max(1, int((y == 1).sum())))
            free = np.flatnonzero((inside & (lay == fs.UNSET)).ravel())
            if n_u and len(free):
                pick = rng.choice(free, min(n_u, len(free), ROWS_PER_CROWN), replace=False)
                X = np.concatenate([X, F.reshape(-1, F.shape[-1])[pick]])
                y = np.concatenate([y, np.zeros(len(pick), y.dtype)])
                wt = np.concatenate([wt, np.full(len(pick), float(ccfg.get("unlabeled_weight", 0.1)), np.float32)])
    _BANK[key] = (sig, X, y, wt)
    return X, y, wt


def _load_clf(st, rid):
    import joblib
    p = st.root / "rounds" / (rid + ".joblib")
    return joblib.load(p) if p.exists() else None


@router.post("/classify")
def classify(req: CrownReq):
    """Học lại từ nét của mọi tán trong vòng, rồi tô tán đang mở."""
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        if rnd["kind"] != "scribble":
            return _err("Học & tô chỉ có ở vòng nét vẽ.")
        try:
            from app import flower_classifier as fc
        except ImportError as e:
            return _err("Thiếu thư viện cho bộ phân loại (%s). Cài scikit-learn vào môi trường app." % e)
        ccfg = rnd["classifier"]
        st = _store()
        t0 = time.perf_counter()
        Xs, ys, ws, n_crowns = [], [], [], 0
        for item in st.load_round(req.round)["items"]:
            X, y, wt = _rows(st, req.round, item, ccfg)
            if X is not None:
                Xs.append(X)
                ys.append(y)
                ws.append(wt)
                n_crowns += 1
        if not Xs:
            return _err("Chưa có nét nào trong vòng này.")
        t_feat = time.perf_counter() - t0
        try:
            clf = fc.fit(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), **ccfg)
        except ValueError as e:
            return _err(str(e))
        _save_clf(st, req.round, clf)
        t1 = time.perf_counter()
        poly = polys[req.crown]
        b = fs.crown_box(poly, w, h)
        inside = fs.crown_mask(poly, b)
        pred = fc.predict(clf, _feats(req.name, flat, req.crown, b, ccfg), inside)
        base = st.load_base(flat, (h, w))
        sub = base[b[1]:b[3], b[0]:b[2]]
        sub[inside] = pred[inside]
        st.save_base(flat, base)
        t_pred = time.perf_counter() - t1
        mask, polys, doc = _recompose(st, flat, req.name, rnd, req.who)
        # Tán chưa có nét nào cũng tô được (máy học từ nét của các tán khác), nên
        # có thể đây là dòng trạng thái đầu tiên của tán: ghi đủ người và trạng thái.
        skey = _status_round(rnd, req.who)
        row = _state_of(st.read_status(), flat, req.crown, skey) or {}
        st.set_status(flat, req.crown, skey, who=req.who, state=row.get("state") or "doing",
                      n_ops=str(len(_crown_ops(doc, req.crown))), source="classifier", bbox=list(b))
        return _result(mask, polys, doc, req.crown,
                       n_samples=clf.n_samples_, n_crowns=n_crowns, t_feat=round_(t_feat, 2),
                       t_fit=round_(clf.fit_seconds_, 2), t_pred=round_(t_pred, 2))


def _save_clf(st, rnd, clf):
    import joblib
    p = st.root / "rounds" / (rnd + ".joblib")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    joblib.dump(clf, tmp)
    os.replace(tmp, p)


@router.post("/finish")
def finish(req: FinishReq):
    """Chốt tán. Vòng vẽ tay (bộ vàng): phần trong tán chưa tô thành Không hoa."""
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        st = _store()
        if rnd["kind"] == "manual":
            st.add_op(flat, dict(round=req.round, crown=int(req.crown), who=req.who, kind="fill_rest"))
        mask, polys, doc = _recompose(st, flat, req.name, rnd, req.who)
        skey = _status_round(rnd, req.who)
        row = _state_of(st.read_status(), flat, req.crown, skey) or {}
        secs = float(row.get("seconds") or 0) + max(0.0, float(req.seconds))
        b = fs.crown_box(polys[req.crown], w, h)
        # Vòng duyệt: Xong sau khi Sửa = tán người đã sửa lại dự đoán của model.
        state, source = ("fixed", "human_fix") if rnd["kind"] == "review" else ("done", row.get("source") or "human")
        st.set_status(flat, req.crown, skey, state=state, who=req.who, seconds="%.1f" % secs,
                      n_ops=str(len(_crown_ops(doc, req.crown))), source=source, bbox=list(b))
        return _result(mask, polys, doc, req.crown, state=state, seconds=secs)


# ------------------------------------------------------------ vòng và hàng đợi
def _all_images():
    """Mọi ảnh có nhãn tán dưới ROOT: tên, ảnh phẳng, ruộng, đường bay, số khung,
    cỡ và polygon tán."""
    s = _srv()
    out = []
    for i, (name, jp) in enumerate(s._iter_labeled("")):
        try:
            w, h, anns = s._read_records(jp)
        except Exception:  # noqa: BLE001
            continue
        w, h = s._size_of(name, w, h)
        rel = s._full_rel(name)
        out.append(dict(name=name, flat=s._flat(name), field=s._field_of(rel),
                        flight=rel.rsplit("/", 1)[0] if "/" in rel else "",
                        seq=fr.frame_no(name, i), size=(int(h), int(w)),
                        crowns=[a["poly"] for a in anns]))
    return out


@router.post("/round/build")
def round_build(req: RoundReq):
    """Chọn tán cho vòng theo luật trong cấu hình và chốt vào rounds/<vòng>.json."""
    with _LOCK:
        rnd = _round(req.round)
        if rnd is None:
            return _err("Không có vòng %s trong configs/flower_rounds.yaml." % req.round)
        if rnd["pick"].get("mode") == "manual":
            return _err("Vòng %s chọn tán bằng tay, không dựng tự động." % req.round)
        cfg, _ = load_config()
        st = _store()
        built = st.all_rounds()
        old = built.get(req.round) or {}
        if old.get("items"):
            flats = {it["image"] for it in old["items"]}
            if any(o.get("round") == req.round for f in flats for o in st.load_ops(f)["ops"]):
                return _err("Vòng %s đã có nét vẽ, không dựng lại được." % req.round, 409)
            if not req.force:
                return _err("Vòng %s đã dựng (%d tán)." % (req.round, len(old["items"])), 409)
        test_ids = {r["id"] for r in cfg["rounds"] if fr.is_test(r)}
        if req.round not in test_ids and test_ids and not any(built.get(t, {}).get("items") for t in test_ids):
            return _err("Dựng vòng test (bộ chấm) trước: vòng học phải tránh các ảnh quanh ảnh test.")
        taken, test, train = set(), set(), set()
        for rid, doc in built.items():
            if rid == req.round:
                continue
            fl = {it["image"] for it in doc.get("items") or []}
            taken |= fl
            (test if rid in test_ids else train).update(fl)
        if req.round in test_ids:
            block = train
            buffer = max([int(r["pick"].get("buffer_frames", 0)) for r in cfg["rounds"]] + [0])
        else:
            block = test
            buffer = int(rnd["pick"].get("buffer_frames", 0))
        items = fr.build(rnd, _all_images(), taken, block, buffer)
        if not items:
            return _err("Không còn tán nào thoả luật của vòng %s." % req.round)
        doc = dict(id=req.round, kind=rnd["kind"], created=fs.now(), config=rnd, buffer=buffer, items=items)
        st.save_round(req.round, doc)
        by = {}
        for it in items:
            b = by.setdefault(it["field"], {"images": set(), "crowns": 0})
            b["images"].add(it["image"])
            b["crowns"] += 1
        return {"ok": True, "round": req.round, "n_items": len(items),
                "by_field": {f: {"images": len(v["images"]), "crowns": v["crowns"]} for f, v in sorted(by.items())}}


def _image_items(name, flat, polys, size, crowns):
    h, w = size
    field = _srv()._field_of(_srv()._full_rel(name))
    return [dict(image=flat, name=name, crown=int(k), field=field, bbox=list(fs.crown_box(polys[k], w, h)))
            for k in crowns]


class PickReq(BaseModel):
    round: str
    name: str
    action: str = "add"      # add | remove


@router.post("/round/image")
def round_image(req: PickReq):
    """Vòng chọn tay: thêm cả ảnh (mọi tán) vào vòng, hoặc bỏ một ảnh chưa có
    nét nào ra. Ảnh nào, bao nhiêu ảnh, ruộng nào là do người gán quyết định."""
    with _LOCK:
        rnd = _round(req.round)
        if rnd is None:
            return _err("Không có vòng %s trong configs/flower_rounds.yaml." % req.round)
        if rnd["pick"].get("mode") != "manual":
            return _err("Vòng %s dựng tự động, không thêm ảnh bằng tay." % req.round)
        polys, (h, w) = _crowns(req.name)
        if not polys:
            return _err("Ảnh này chưa có nhãn tán.")
        s = _srv()
        fields = rnd["pick"].get("fields")
        if fields and s._field_of(s._full_rel(req.name)) not in fields:
            return _err("Vòng %s chỉ nhận ảnh của %s." % (req.round, ", ".join(fields)))
        flat = s._flat(req.name)
        st = _store()
        doc = st.load_ops(flat)
        if req.action == "remove":
            if doc.get("round") not in (None, req.round):
                return _err("Ảnh này thuộc vòng %s, không thuộc vòng %s." % (doc["round"], req.round))
            if not st.drop_image(flat, req.round):
                return _err("Ảnh đã có nét: hoàn tác (Ctrl+Z) hết nét của mọi tán rồi mới bỏ khỏi vòng được.", 409)
        elif req.action == "add":
            if doc.get("round") and doc["round"] != req.round:
                return _err("Ảnh này đã thuộc vòng %s." % doc["round"], 409)
            if not doc.get("round"):
                hit = _too_close(st, rnd, flat)
                if hit:
                    return _close_err(hit)
                st.claim(flat, req.round)
            st.add_round_items(req.round, _image_items(req.name, flat, polys, (h, w), range(len(polys))))
        else:
            return _err("action phải là add hoặc remove")
        items = st.load_round(req.round)["items"]
        return {"ok": True, "round": req.round, "image": flat, "action": req.action,
                "in_round": any(it["image"] == flat for it in items),
                "images": len({it["image"] for it in items}), "crowns": len(items)}


def _queue(rid, who):
    rnd = _round(rid)
    if rnd is None:
        return None, "Không có vòng %s." % rid
    st = _store()
    items = st.load_round(rid)["items"]
    if _is_sec(rnd, who):
        items = fr.double_subset(items, int(rnd["double"].get("crowns", 10)))
    reasons = {}
    if rnd["kind"] == "review":
        rv = rnd.get("review") or {}
        order = fr.review_queue(items, _unc(rnd), float(rv.get("fix_top", 0.3)),
                                float(rv.get("check_random", 0.1)), rnd["pick"].get("seed", 0))
        items = [it for it, _, _ in order]
        reasons = {(it["image"], int(it["crown"])): (why, u) for it, why, u in order}
    rows = st.read_status()
    skey = _status_round(rnd, who)
    out, prog, secs = [], {}, []
    for it in items:
        row = _state_of(rows, it["image"], it["crown"], skey) or {}
        state = row.get("state") or "todo"
        name = _name_of(it["image"]) or it.get("name")
        why, u = reasons.get((it["image"], int(it["crown"])), (None, None))
        out.append(dict(name=name, image=it["image"], crown=int(it["crown"]), field=it["field"], state=state,
                        reason=why, unc=(None if u is None or u == float("inf") else round_(u, 3))))
        p = prog.setdefault(it["field"], {"done": 0, "total": 0, "images": set()})
        p["total"] += 1
        p["images"].add(it["image"])
        if state == "done":
            p["done"] += 1
            if row.get("seconds"):
                secs.append(float(row["seconds"]))
    if rnd["kind"] == "review":
        for p in prog.values():
            p["done"] = 0
        for it in out:
            if it["state"] in REVIEWED:
                prog[it["field"]]["done"] += 1
    for p in prog.values():
        p["images"] = len(p["images"])
    n_acc = sum(1 for it in out if it["state"] == "accepted")
    n_fix = sum(1 for it in out if it["state"] == "fixed")
    return dict(ok=True, round=rid, kind=rnd["kind"], who=who, second=_is_sec(rnd, who), items=out,
                progress=dict(sorted(prog.items())), done=sum(p["done"] for p in prog.values()),
                total=len(out), median_seconds=statistics.median(secs) if secs else None,
                accepted=n_acc, fixed=n_fix,
                reject_rate=(n_fix / (n_acc + n_fix)) if (n_acc + n_fix) else None), None


@router.get("/queue")
def queue(round: str, who: str = "A"):  # noqa: A002
    with _LOCK:
        q, err = _queue(round, who)
        return q if q else _err(err, 404)


@router.get("/progress")
def progress(round: str):  # noqa: A002
    """Tiến độ theo ruộng của người gán chính, và của người gán thứ hai nếu có."""
    with _LOCK:
        rnd = _round(round)
        if rnd is None:
            return _err("Không có vòng %s." % round, 404)
        main, _ = _queue(round, "")      # "" không bao giờ là người gán thứ hai
        out = {"ok": True, "round": round, "main": {k: main[k] for k in ("progress", "done", "total", "median_seconds")}}
        if _sec(rnd):
            sec, _ = _queue(round, _sec(rnd))
            out["second"] = {k: sec[k] for k in ("progress", "done", "total", "median_seconds")}
        return out


@router.get("/agreement")
def agreement(round: str, against: str = ""):  # noqa: A002
    """Đồng thuận trên các tán đã xong của vòng `round`, theo ruộng:
    - không có `against`: người gán chính so với người thứ hai (gán đôi);
    - `against=<vòng nét vẽ>`: bộ phân loại của vòng đó so với mask của vòng này
      (thường là bộ vàng). IoU và F1 của lớp hoa, bỏ pixel 255 ở bất kỳ bên nào."""
    with _LOCK:
        rnd = _round(round)
        if rnd is None:
            return _err("Không có vòng %s." % round, 404)
        st = _store()
        rows = st.read_status()
        clf, ccfg = None, None
        if against:
            other = _round(against)
            if other is None or other["kind"] != "scribble":
                return _err("against phải là một vòng nét vẽ.")
            clf = _load_clf(st, against)
            if clf is None:
                return _err("Vòng %s chưa có bộ phân loại (chưa bấm Học & tô lần nào)." % against)
            ccfg = other["classifier"]
            from app import flower_classifier as fc
            items = st.load_round(round)["items"]
        else:
            if not _sec(rnd):
                return _err("Vòng %s không có gán đôi." % round)
            items = fr.double_subset(st.load_round(round)["items"], int(rnd["double"].get("crowns", 10)))
        pairs, n = {}, {}
        for it in items:
            flat, k = it["image"], int(it["crown"])
            if (_state_of(rows, flat, k, round) or {}).get("state") != "done":
                continue
            name = _name_of(flat) or it.get("name")
            polys, (h, w) = _crowns(name)
            if k >= len(polys):
                continue
            b = fs.crown_box(polys[k], w, h)
            inside = fs.crown_mask(polys[k], b)
            a = st.load_mask(flat, (h, w))[b[1]:b[3], b[0]:b[2]][inside]
            if against:
                p = fc.predict(clf, _feats(name, flat, k, b, ccfg), inside)[inside]
            else:
                if (_state_of(rows, flat, k, _status_round(rnd, _sec(rnd))) or {}).get("state") != "done":
                    continue
                p = st.load_mask(flat, (h, w), _sec(rnd))[b[1]:b[3], b[0]:b[2]][inside]
            pairs.setdefault(it["field"], []).append((a, p))
            n[it["field"]] = n.get(it["field"], 0) + 1
        by_field = {f: dict(fr.agreement(v), crowns=n[f]) for f, v in sorted(pairs.items())}
        allp = [x for v in pairs.values() for x in v]
        return {"ok": True, "round": round, "against": against or None,
                "by_field": by_field, "all": dict(fr.agreement(allp), crowns=len(allp))}


class ReviewReq(CrownReq):
    action: str              # accept | reject


@router.post("/review")
def review(req: ReviewReq):
    """Vòng duyệt: Chấp nhận (A) nhận nguyên dự đoán của model; Sửa (R) mở tán
    để vẽ nét, Xong sau đó ghi là fixed."""
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        if rnd["kind"] != "review":
            return _err("Chấp nhận / Sửa chỉ có ở vòng duyệt.")
        if req.action not in ("accept", "reject"):
            return _err("action phải là accept hoặc reject")
        st = _store()
        if not _ensure_pred_base(st, rnd, flat, req.name, [req.crown]) and not st.base_path(flat).exists():
            return _err("Chưa có dự đoán của model cho ảnh này (chạy scripts/flower_predict.py --round %s)." % req.round)
        mask, polys, doc = _recompose(st, flat, req.name, rnd, req.who)
        run = "model:" + _run_dir(rnd).name
        b = fs.crown_box(polys[req.crown], w, h)
        state = "accepted" if req.action == "accept" else "fixing"
        st.set_status(flat, req.crown, _status_round(rnd, req.who), state=state, who=req.who,
                      source=run, bbox=list(b))
        return _result(mask, polys, doc, req.crown, state=state)


@router.post("/round/close")
def round_close(req: RoundReq):
    """Chốt vòng duyệt: mọi tán chưa được người nhận hay sửa thành `auto`, nền
    là dự đoán của model."""
    with _LOCK:
        rnd = _round(req.round)
        if rnd is None or rnd["kind"] != "review":
            return _err("Chỉ chốt được vòng duyệt.")
        st = _store()
        items = st.load_round(req.round)["items"]
        if not items:
            return _err("Vòng %s chưa dựng." % req.round)
        rows = st.read_status()
        run = "model:" + _run_dir(rnd).name
        n_auto, missing = 0, 0
        for flat, its in sorted({it["image"]: [] for it in items}.items()):
            mine = [int(it["crown"]) for it in items if it["image"] == flat]
            name = _name_of(flat)
            if name is None:
                continue
            todo = [k for k in mine if ((_state_of(rows, flat, k, req.round) or {}).get("state") not in REVIEWED + ("fixing",))]
            if not todo:
                continue
            if not _ensure_pred_base(st, rnd, flat, name, todo) and not st.base_path(flat).exists():
                missing += len(todo)
                continue
            _recompose(st, flat, name, rnd, "")
            polys, (h, w) = _crowns(name)
            for k in todo:
                st.set_status(flat, k, req.round, state="auto", source=run,
                              bbox=list(fs.crown_box(polys[k], w, h)))
                n_auto += 1
        return {"ok": True, "round": req.round, "auto": n_auto, "missing_prediction": missing}


class ExportReq(BaseModel):
    name: str = "flower_v1"
    out_dir: str = ""
    include_auto: bool = False
    overwrite: bool = False


@router.post("/export")
def export(req: ExportReq):
    """Xuất bộ nhãn hoa: images/ + masks/ (0 không hoa, 1 hoa, 255 bỏ qua),
    tách train/ và test/. Bộ chấm (vòng vẽ tay hoặc role: test) và ảnh đệm quanh
    nó không bao giờ vào train;
    meta.json ghi trạng thái và nguồn của từng tán."""
    import shutil
    from app import flower_data as fd
    with _LOCK:
        s = _srv()
        cfg, _ = load_config()
        st = _store()
        root = Path(req.out_dir.strip() or s.EXPORT_DIR) / req.name
        if root.exists():
            if not req.overwrite:
                return {"ok": False, "exists": True, "dir": str(root).replace("\\", "/"),
                        "error": "Thư mục '%s' đã tồn tại." % req.name}
            shutil.rmtree(root, ignore_errors=True)
        labels = s.OUT_DIR
        train = fd.collect(st, cfg["rounds"], labels, "train", req.include_auto)
        test = fd.collect(st, cfg["rounds"], labels, "test")
        crowns, problems, counts = [], [], {}
        for split, items in (("train", train), ("test", test)):
            img_dir, mask_dir = root / "images" / split, root / "masks" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            n_img = 0
            for flat, its in sorted(fd.group_by_image(items).items()):
                polys, (h, w) = fd.crown_polys(labels, flat)
                src = Path(s.BASE) / fd.rel_of(flat)
                if not polys or not src.exists():
                    problems.append("%s: thiếu ảnh hoặc nhãn tán" % flat)
                    continue
                if not h or not w:
                    h, w = cv2.imread(str(src)).shape[:2]
                shutil.copyfile(src, img_dir / flat)
                stem = os.path.splitext(flat)[0] + ".png"
                fs._atomic_bytes(mask_dir / stem, fs._png(fd.image_mask(st, flat, polys, [it["crown"] for it in its], (h, w))))
                n_img += 1
                for it in its:
                    crowns.append(dict(it, split=split))
            counts[split] = {"images": n_img, "crowns": sum(1 for c in crowns if c["split"] == split)}
        meta = dict(name=req.name, created=fs.now(), include_auto=req.include_auto,
                    mask_values="0=no flower, 1=flower, 255=ignore (outside the exported crowns, unsure, unlabelled)",
                    rule="test rounds (manual, or role: test) and images within the buffer frames of a test image never go to train",
                    splits=counts, crowns=crowns, problems=problems)
        (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"ok": True, "dir": str(root).replace("\\", "/"), "splits": counts, "problems": problems}
