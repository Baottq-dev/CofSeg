# app/flower_store.py — Lưu nhãn hoa của thẻ Hoa và dựng mask từ nét vẽ.
#
#     <data.masks_dir>/flower_v1/
#       ops/<ảnh phẳng>.json     mọi thao tác (nét, tẩy, "phần còn lại"), theo thứ tự
#       base/<ảnh phẳng>.png     nền máy tô (0/1); 255 = chưa có nền
#       masks/<ảnh phẳng>.png    mask cuối 0 = không hoa, 1 = hoa, 255 = bỏ qua
#       status.csv               mỗi (ảnh, tán, vòng) một dòng
#       rounds/<vòng>.json       danh sách tán của vòng; <vòng>.joblib bộ phân loại
#
# Một "tán" là một polygon trong file nhãn tán (đã chốt, chỉ đọc), đánh số theo
# thứ tự trong file. Nhãn hoa chỉ có nghĩa BÊN TRONG tán: nét vẽ bị cắt theo
# polygon, ngoài mọi tán mask luôn là 255. "Ảnh phẳng" là đường dẫn ảnh theo
# BASE với '/' thay bằng '__', đúng khoá bộ xuất COCO đang dùng.
#
# Mọi thứ trong mask dựng lại được từ ops + base: nét người luôn thắng nền máy.
import csv
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

UNSET = 254            # lớp người: chưa tô (để nền máy hiện ra)
IGNORE = 255           # "bỏ qua": mờ, không chắc — không tính vào loss
LABELS = (0, 1, IGNORE)
STATUS_COLS = ("image", "crown", "round", "state", "source", "who", "seconds",
               "n_ops", "updated", "bbox")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------- hình học
def crown_box(poly, w, h):
    """Khung bao (x0, y0, x1, y1) nguyên của polygon, kẹp trong ảnh."""
    xs, ys = poly[0::2], poly[1::2]
    x0 = max(0, int(np.floor(min(xs))))
    y0 = max(0, int(np.floor(min(ys))))
    x1 = min(w, int(np.ceil(max(xs))) + 1)
    y1 = min(h, int(np.ceil(max(ys))) + 1)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def crown_mask(poly, box):
    """Mask bool của polygon trên crop `box`."""
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round(np.array(poly, np.float64).reshape(-1, 2) - (x0, y0)).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def raster_stroke(op, box):
    """Pixel một nét đi qua, trên crop `box` (bool). Nét là đường gấp khúc dày
    2r + 1 px ảnh, đầu tròn."""
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round(np.array(op.get("pts") or [], np.float64).reshape(-1, 2) - (x0, y0)).astype(np.int32)
    if not len(pts):
        return m.astype(bool)
    r = max(1, int(round(float(op.get("r", 1)))))
    if len(pts) == 1:
        cv2.circle(m, tuple(int(v) for v in pts[0]), r, 1, -1)
    else:
        cv2.polylines(m, [pts], False, 1, thickness=2 * r + 1)
        for p in (pts[0], pts[-1]):
            cv2.circle(m, tuple(int(v) for v in p), r, 1, -1)
    return m.astype(bool)


def human_layer(ops, poly, box):
    """Lớp người của một tán trên crop: phát lại các thao tác theo thứ tự, thao
    tác sau đè thao tác trước. UNSET = chưa tô; ngoài polygon cũng UNSET."""
    inside = crown_mask(poly, box)
    lay = np.full(inside.shape, UNSET, np.uint8)
    for op in ops:
        kind = op.get("kind", "stroke")
        if kind == "fill_rest":
            lay[inside & (lay == UNSET)] = 0
            continue
        hit = raster_stroke(op, box) & inside
        lab = op.get("label")
        lay[hit] = UNSET if lab == "erase" else int(lab)
    return lay


def compose(shape, crowns, ops_doc, base):
    """Mask cả ảnh: trong mỗi tán, nét người nếu có, không thì nền máy, không
    thì 255. Tán chạm nhau thì vùng chung theo tán có thao tác gần nhất."""
    h, w = shape
    out = np.full((h, w), IGNORE, np.uint8)
    by_crown = {}
    for op in ops_doc.get("ops", []):
        by_crown.setdefault(int(op["crown"]), []).append(op)
    touched = set(by_crown)
    if base is not None:
        for k, poly in enumerate(crowns):
            if k in touched:
                continue
            b = crown_box(poly, w, h)
            sub = base[b[1]:b[3], b[0]:b[2]]
            if (sub[crown_mask(poly, b)] != IGNORE).any():
                touched.add(k)
    order = sorted((k for k in touched if 0 <= k < len(crowns)),
                   key=lambda k: max((o["id"] for o in by_crown.get(k, [])), default=0))
    for k in order:
        poly = crowns[k]
        b = crown_box(poly, w, h)
        inside = crown_mask(poly, b)
        lay = human_layer(by_crown.get(k, []), poly, b)
        dst = out[b[1]:b[3], b[0]:b[2]]
        val = np.full(inside.shape, IGNORE, np.uint8)
        if base is not None:
            val = base[b[1]:b[3], b[0]:b[2]].copy()
        set_ = lay != UNSET
        val[set_] = lay[set_]
        dst[inside] = val[inside]
    return out


def cover(mask, poly):
    """Độ phủ hoa của tán: pixel hoa / pixel tán."""
    h, w = mask.shape
    b = crown_box(poly, w, h)
    inside = crown_mask(poly, b)
    n = int(inside.sum())
    if not n:
        return 0.0
    return float((mask[b[1]:b[3], b[0]:b[2]][inside] == 1).sum()) / n


# ------------------------------------------------------------------ lưu trữ
def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 9:
                raise
            time.sleep(0.05)


def _read_png(path):
    try:
        buf = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None
    except Exception:  # noqa: BLE001
        return None


def _png(arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("không mã hoá được PNG")
    return buf.tobytes()


class FlowerStore:
    """Một thư mục nhãn hoa. Mọi lần ghi đi qua file tạm rồi thay thế, và qua
    một khoá: server chạy endpoint đồng bộ trên nhiều luồng."""

    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    # --- ops
    def ops_path(self, flat):
        return self.root / "ops" / (flat + ".json")

    def load_ops(self, flat):
        p = self.ops_path(flat)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"image": flat, "round": None, "next_id": 1, "ops": []}

    def save_ops(self, flat, doc):
        _atomic_bytes(self.ops_path(flat), json.dumps(doc, ensure_ascii=False).encode("utf-8"))

    def add_op(self, flat, op):
        with self.lock:
            doc = self.load_ops(flat)
            op = dict(op, id=int(doc.get("next_id", 1)), t=now())
            doc["next_id"] = op["id"] + 1
            doc["round"] = doc.get("round") or op.get("round")
            doc["ops"].append(op)
            self.save_ops(flat, doc)
            return op, doc

    def undo(self, flat, crown):
        """Bỏ thao tác cuối của đúng tán đó; trả về thao tác đã bỏ hoặc None."""
        with self.lock:
            doc = self.load_ops(flat)
            for i in range(len(doc["ops"]) - 1, -1, -1):
                if int(doc["ops"][i]["crown"]) == int(crown):
                    op = doc["ops"].pop(i)
                    self.save_ops(flat, doc)
                    return op, doc
            return None, doc

    # --- base / mask
    def base_path(self, flat):
        return self.root / "base" / (os.path.splitext(flat)[0] + ".png")

    def mask_path(self, flat):
        return self.root / "masks" / (os.path.splitext(flat)[0] + ".png")

    def load_base(self, flat, shape):
        b = _read_png(self.base_path(flat))
        if b is None or b.shape != tuple(shape):
            return np.full(shape, IGNORE, np.uint8)
        return b

    def save_base(self, flat, base):
        _atomic_bytes(self.base_path(flat), _png(base))

    def load_mask(self, flat, shape):
        m = _read_png(self.mask_path(flat))
        if m is None or m.shape != tuple(shape):
            return np.full(shape, IGNORE, np.uint8)
        return m

    def save_mask(self, flat, mask):
        _atomic_bytes(self.mask_path(flat), _png(mask))

    # --- trạng thái theo tán
    def status_path(self):
        return self.root / "status.csv"

    def read_status(self):
        p = self.status_path()
        if not p.exists():
            return {}
        rows = {}
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["image"], int(r["crown"]), r["round"])] = r
        return rows

    def set_status(self, image, crown, rnd, **fields):
        """Cập nhật (hoặc thêm) một dòng; giữ nguyên các cột không nói tới."""
        with self.lock:
            rows = self.read_status()
            key = (image, int(crown), rnd)
            r = rows.get(key) or {c: "" for c in STATUS_COLS}
            r.update(image=image, crown=str(int(crown)), round=rnd)
            for k, v in fields.items():
                r[k] = v if isinstance(v, str) else json.dumps(v)
            r["updated"] = now()
            rows[key] = r
            buf = io.StringIO()
            wr = csv.DictWriter(buf, fieldnames=STATUS_COLS, lineterminator="\n")
            wr.writeheader()
            for k in sorted(rows, key=lambda k: (k[2], k[0], k[1])):
                wr.writerow({c: rows[k].get(c, "") for c in STATUS_COLS})
            _atomic_bytes(self.status_path(), buf.getvalue().encode("utf-8"))
            return r

    # --- vòng
    def round_path(self, rnd):
        return self.root / "rounds" / (rnd + ".json")

    def load_round(self, rnd):
        p = self.round_path(rnd)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"id": rnd, "created": now(), "items": []}

    def add_round_item(self, rnd, item):
        """Thêm tán vào vòng chọn tay (lần đầu có nét); có rồi thì thôi."""
        with self.lock:
            doc = self.load_round(rnd)
            key = (item["image"], int(item["crown"]))
            if any((it["image"], int(it["crown"])) == key for it in doc["items"]):
                return doc
            doc["items"].append(item)
            _atomic_bytes(self.round_path(rnd), json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"))
            return doc
