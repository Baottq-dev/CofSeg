# app/flower_api.py — Endpoint của thẻ Hoa (/api/fl/*): nét vẽ, Học & tô, chốt tán.
#
# Polygon tán đọc từ file nhãn tán qua chính các hàm của server (chỉ đọc, không
# bao giờ ghi); nhãn hoa nằm riêng dưới <data.masks_dir>/<data_dir của vòng>/.
# Cấu hình vòng ở configs/flower_rounds.yaml, đọc lại mỗi lần cần.
import os
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

from app import flower_store as fs

router = APIRouter(prefix="/api/fl")
CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "flower_rounds.yaml"
KINDS = ("scribble", "manual", "review")
CLF_DEFAULTS = dict(n_trees=100, max_depth=12, sigma_min=0.7, sigma_max=5.0,
                    num_sigma=4, max_samples=50000)
ROWS_PER_CROWN = 20000      # nét tô to cũng không để một tán lấn át cả vòng

_LOCK = threading.RLock()   # endpoint đồng bộ chạy trên nhiều luồng
_STORES = {}
_BANK = {}                  # (vòng, ảnh, tán) -> (chữ ký các op, X, y)
_IMG = OrderedDict()        # ảnh RGB gần nhất (giữ 3)
_CACHE = None               # FeatureCache, tạo lúc cần (sklearn/skimage nạp lười)


def _srv():
    # Nạp lúc gọi: server import module này lúc khởi động.
    from app import server
    return server


def _err(msg, code=400):
    return JSONResponse({"ok": False, "error": msg}, status_code=code)


# ------------------------------------------------------------------ cấu hình
def load_config(path=None):
    """(cfg, danh sách lỗi). Vòng thiếu id, trùng id, sai loại đều bị báo."""
    p = Path(path or CFG_PATH)
    errors = []
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        return {"rounds": []}, ["Không đọc được %s: %s" % (p.name, e)]
    seen, rounds = set(), []
    for r in cfg.get("rounds") or []:
        rid, kind = r.get("id"), r.get("kind")
        if not rid:
            errors.append("Có vòng không có id")
            continue
        if rid in seen:
            errors.append("Trùng id vòng: %s" % rid)
            continue
        if kind not in KINDS:
            errors.append("Vòng %s: kind phải là một trong %s" % (rid, ", ".join(KINDS)))
            continue
        seen.add(rid)
        r = dict(r)
        r["classifier"] = dict(CLF_DEFAULTS, **(r.get("classifier") or {}))
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


def _recompose(st, flat, name):
    polys, (h, w) = _crowns(name)
    doc = st.load_ops(flat)
    base = st.load_base(flat, (h, w)) if st.base_path(flat).exists() else None
    mask = fs.compose((h, w), polys, doc, base)
    st.save_mask(flat, mask)
    return mask, polys, doc


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
            saved = [int(v) for v in json_loads(row["bbox"])]
        except Exception:  # noqa: BLE001
            continue
        if crown >= len(polys):
            checks.append(dict(image=name, crown=crown, round=rnd, problem="tán không còn trong nhãn tán"))
        elif list(fs.crown_box(polys[crown], w, h)) != saved:
            checks.append(dict(image=name, crown=crown, round=rnd, problem="polygon tán đã đổi sau khi gán hoa"))
    return counts, checks


def json_loads(s):
    import json
    return json.loads(s)


# ---------------------------------------------------------------- endpoint
@router.get("/config")
def config():
    cfg, errors = load_config()
    rounds = [dict(id=r["id"], label=r.get("label") or r["id"], kind=r["kind"], pick=r["pick"])
              for r in cfg["rounds"]]
    return {"rounds": rounds, "errors": errors,
            "data_dir": os.path.join(os.path.dirname(_srv().OUT_DIR), cfg["data_dir"]).replace("\\", "/")}


@router.get("/crowns")
def crowns(name: str, round: str = ""):  # noqa: A002 - tên tham số của API
    with _LOCK:
        polys, (h, w) = _crowns(name)
        if not polys:
            return {"ok": True, "image": _srv()._flat(name), "crowns": [], "size": [h, w]}
        st = _store()
        flat = _srv()._flat(name)
        doc = st.load_ops(flat)
        mask = st.load_mask(flat, (h, w))
        rows = st.read_status()
        out = []
        for k, poly in enumerate(polys):
            b = fs.crown_box(poly, w, h)
            row = _state_of(rows, flat, k, round or None)
            n_ops = len(_crown_ops(doc, k))
            out.append(dict(crown=k, bbox=list(b), points=poly,
                            area=int(fs.crown_mask(poly, b).sum()),
                            state=(row or {}).get("state") or ("doing" if n_ops else "todo"),
                            seconds=float((row or {}).get("seconds") or 0),
                            n_ops=n_ops, cover=round_(fs.cover(mask, poly))))
        return {"ok": True, "image": flat, "round": doc.get("round"), "size": [h, w], "crowns": out}


def round_(v, n=4):
    return float(np.round(v, n))


# Màu lớp phủ (RGB): mask hoa vàng; nét Hoa xanh lá, Không hoa đỏ, Bỏ qua xám.
YELLOW, GREEN, RED, GREY = (250, 204, 21), (34, 197, 94), (239, 68, 68), (148, 163, 184)


def _rgba(h, w):
    return np.zeros((h, w, 4), np.uint8)


def _paint(img, sel, rgb, alpha):
    img[sel, 0], img[sel, 1], img[sel, 2] = rgb[2], rgb[1], rgb[0]   # cv2 ghi BGRA
    img[sel, 3] = alpha


@router.get("/view")
def view(name: str, crown: int, layer: str = "mask"):
    """PNG trong suốt cắt theo khung tán; header X-Crop cho biết đặt ở đâu.
    mask: hoa vàng, vùng người đánh Bỏ qua gạch chéo xám. human: các nét."""
    with _LOCK:
        polys, (h, w) = _crowns(name)
        if not (0 <= crown < len(polys)):
            return _err("Không có tán %s trong ảnh này." % crown, 404)
        st = _store()
        flat = _srv()._flat(name)
        poly = polys[crown]
        b = fs.crown_box(poly, w, h)
        inside = fs.crown_mask(poly, b)
        lay = fs.human_layer(_crown_ops(st.load_ops(flat), crown), poly, b)
        img = _rgba(b[3] - b[1], b[2] - b[0])
        if layer == "human":
            _paint(img, lay == 1, GREEN, 230)
            _paint(img, lay == 0, RED, 230)
            _paint(img, lay == fs.IGNORE, GREY, 230)
        else:
            m = st.load_mask(flat, (h, w))[b[1]:b[3], b[0]:b[2]]
            _paint(img, inside & (m == 1), YELLOW, 255)
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            hatch = ((xx + yy) // 4) % 2 == 0
            _paint(img, inside & (lay == fs.IGNORE) & hatch, GREY, 200)
            _paint(img, inside & (lay == fs.IGNORE) & ~hatch, GREY, 70)
        ok, buf = cv2.imencode(".png", img)
        return Response(buf.tobytes(), media_type="image/png",
                        headers={"X-Crop": "%d,%d,%d,%d" % b, "Cache-Control": "no-store",
                                 "Access-Control-Expose-Headers": "X-Crop"})


class CrownReq(BaseModel):
    round: str
    name: str
    crown: int
    who: str = "A"


class StrokeReq(CrownReq):
    label: object            # 0 | 1 | 255 | "erase"
    r: float = 4.0           # bán kính cọ, px ảnh
    snap: bool = False       # bám siêu điểm ảnh (bộ vàng, chưa dùng ở vòng nét vẽ)
    pts: list                # [[x, y], ...] toạ độ ảnh


class FinishReq(CrownReq):
    seconds: float = 0.0


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
    doc = _store().load_ops(flat)
    if doc.get("round") and doc["round"] != req.round:
        # Một ảnh chỉ thuộc một vòng: ảnh của vòng này mà lẫn vào vòng khác
        # là rò rỉ giữa bộ chấm và bộ học.
        return _err("Ảnh này đã thuộc vòng %s." % doc["round"], 409)
    return rnd, polys, (h, w), flat


def _result(st, flat, name, crown, mask, polys, doc, **extra):
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
        if lab not in ("erase",):
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
        op, _ = st.add_op(flat, dict(round=req.round, crown=int(req.crown), who=req.who,
                                     kind="stroke", label=lab, r=round_(max(0.5, req.r), 2),
                                     snap=bool(req.snap), pts=pts))
        b = fs.crown_box(polys[req.crown], w, h)
        st.add_round_item(req.round, dict(image=flat, name=req.name, crown=int(req.crown),
                                          field=_srv()._field_of(_srv()._full_rel(req.name)), bbox=list(b)))
        mask, polys, doc = _recompose(st, flat, req.name)
        row = _state_of(st.read_status(), flat, req.crown, req.round)
        st.set_status(flat, req.crown, req.round, who=req.who, n_ops=str(len(_crown_ops(doc, req.crown))),
                      state=(row or {}).get("state") if (row or {}).get("state") == "done" else "doing",
                      source=(row or {}).get("source") or "human", bbox=list(b))
        return _result(st, flat, req.name, req.crown, mask, polys, doc, op=op["id"])


@router.post("/undo")
def undo(req: CrownReq):
    with _LOCK:
        chk = _check(req)
        if isinstance(chk, JSONResponse):
            return chk
        rnd, polys, (h, w), flat = chk
        st = _store()
        op, _ = st.undo(flat, req.crown)
        mask, polys, doc = _recompose(st, flat, req.name)
        if op is not None:
            st.set_status(flat, req.crown, req.round, n_ops=str(len(_crown_ops(doc, req.crown))))
        return _result(st, flat, req.name, req.crown, mask, polys, doc,
                       removed=(op or {}).get("id"))


def _rows(st, rnd, item, ccfg):
    """Pixel đã có nhãn 0/1 của một tán trong vòng: (X, y). Nhớ lại theo chữ ký
    các thao tác của tán, đổi nét thì tính lại."""
    flat, name, crown = item["image"], item["name"], int(item["crown"])
    ops = _crown_ops(st.load_ops(flat), crown)
    sig = tuple(o["id"] for o in ops)
    key = (rnd, flat, crown)
    hit = _BANK.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1], hit[2]
    polys, (h, w) = _crowns(name)
    if crown >= len(polys) or not ops:
        X, y = None, None
    else:
        poly = polys[crown]
        b = fs.crown_box(poly, w, h)
        lay = fs.human_layer(ops, poly, b)
        sel = (lay == 0) | (lay == 1)
        if not sel.any():
            X, y = None, None
        else:
            F = _feats(name, flat, crown, b, ccfg)
            X, y = F[sel], lay[sel]
            if len(y) > ROWS_PER_CROWN:
                idx = np.random.default_rng(crown).choice(len(y), ROWS_PER_CROWN, replace=False)
                X, y = X[idx], y[idx]
    _BANK[key] = (sig, X, y)
    return X, y


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
        Xs, ys, n_crowns = [], [], 0
        for item in st.load_round(req.round)["items"]:
            X, y = _rows(st, req.round, item, ccfg)
            if X is not None:
                Xs.append(X)
                ys.append(y)
                n_crowns += 1
        if not Xs:
            return _err("Chưa có nét nào trong vòng này.")
        t_feat = time.perf_counter() - t0
        try:
            clf = fc.fit(np.concatenate(Xs), np.concatenate(ys), **ccfg)
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
        mask, polys, doc = _recompose(st, flat, req.name)
        st.set_status(flat, req.crown, req.round, source="classifier", bbox=list(b))
        return _result(st, flat, req.name, req.crown, mask, polys, doc,
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
        mask, polys, doc = _recompose(st, flat, req.name)
        row = _state_of(st.read_status(), flat, req.crown, req.round) or {}
        secs = float(row.get("seconds") or 0) + max(0.0, float(req.seconds))
        b = fs.crown_box(polys[req.crown], w, h)
        st.set_status(flat, req.crown, req.round, state="done", who=req.who, seconds="%.1f" % secs,
                      n_ops=str(len(_crown_ops(doc, req.crown))), source=row.get("source") or "human",
                      bbox=list(b))
        return _result(st, flat, req.name, req.crown, mask, polys, doc, state="done", seconds=secs)
