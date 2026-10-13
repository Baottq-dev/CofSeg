# app/flower_data.py — Chọn nhãn hoa đã chốt cho train, predict và xuất bộ hoa.
#
# Luật dùng chung ở mọi nơi:
# - bộ chấm (vòng kind=manual hoặc role: test) chỉ để CHẤM, không bao giờ vào train;
# - ảnh cách ảnh test ≤ buffer khung trong cùng đường bay cũng không vào train
#   (khung liền kề chụp trùng cây);
# - một tán vào train khi đã có người chốt: done (vẽ/nét), fixed (sửa sau duyệt),
#   accepted (duyệt chấp nhận); `auto` (model tự nhận, chưa ai xem) chỉ khi
#   được yêu cầu rõ;
# - pixel của tán không được chọn, và mọi pixel ngoài tán, là 255.
import json
import os
from pathlib import Path

import numpy as np

from app import flower_rounds as fr
from app import flower_store as fs

TRAIN_STATES = {"done", "fixed", "accepted"}


def rel_of(flat):
    """Ảnh phẳng -> đường dẫn theo thư mục ảnh gốc (field/…/tên)."""
    return flat.replace("__", "/")


def field_of(flat):
    return flat.split("__", 1)[0]


def flight_of(flat):
    rel = rel_of(flat)
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


def crown_polys(labels_dir, flat):
    """(polygon tán theo thứ tự file, (h, w)) từ masks/corrected — chỉ đọc."""
    p = Path(labels_dir) / (flat + ".json")
    if not p.exists():
        return [], (0, 0)
    d = json.loads(p.read_text(encoding="utf-8"))
    polys = []
    for q in d.get("polygons") or []:
        pts = [float(v) for xy in (q.get("points") or []) for v in xy]
        if len(pts) >= 6:
            polys.append(pts)
    return polys, (int(d.get("img_h") or 0), int(d.get("img_w") or 0))


def kinds_of(rounds_cfg):
    return {r["id"]: r["kind"] for r in rounds_cfg}


def test_ids(rounds_cfg):
    return {r["id"] for r in rounds_cfg if fr.is_test(r)}


def test_and_buffer(store, rounds_cfg, labels_dir):
    """(ảnh test, ảnh đệm quanh ảnh test). Đệm = max buffer_frames của các vòng."""
    tests = test_ids(rounds_cfg)
    test_imgs = set()
    for rid, doc in store.all_rounds().items():
        if rid in tests:
            test_imgs |= {it["image"] for it in doc.get("items") or []}
    buf = max([int((r.get("pick") or {}).get("buffer_frames", 0)) for r in rounds_cfg] + [0])
    at = {}
    for f in test_imgs:
        at.setdefault(flight_of(f), []).append(fr.frame_no(f))
    near = set()
    if buf:
        for p in Path(labels_dir).glob("*.json"):
            flat = p.name[:-5]
            if flat in test_imgs:
                continue
            seq = fr.frame_no(flat, -10 ** 9)
            if any(abs(seq - s) <= buf for s in at.get(flight_of(flat), ())):
                near.add(flat)
    return test_imgs, near


def collect(store, rounds_cfg, labels_dir, split="train", include_auto=False, holdout=None):
    """Danh sách tán đã chốt: [{image, crown, round, state, source, field}].

    split="train": các vòng không phải bộ chấm, bỏ ảnh test và ảnh đệm, bỏ
    ruộng `holdout`. split="test": các tán xong của bộ chấm (người gán chính),
    chỉ ruộng `holdout` nếu có."""
    kinds = kinds_of(rounds_cfg)
    tests = test_ids(rounds_cfg)
    test_imgs, near = test_and_buffer(store, rounds_cfg, labels_dir)
    ok = TRAIN_STATES | ({"auto"} if include_auto else set())
    out = []
    for (flat, crown, rnd), row in sorted(store.read_status().items()):
        if "@" in rnd:                       # người gán thứ hai: chỉ để đo đồng thuận
            continue
        kind = kinds.get(rnd)
        if kind is None:
            continue
        state = row.get("state") or "todo"
        field = field_of(flat)
        if split == "test":
            if rnd not in tests or state != "done" or (holdout and field != holdout):
                continue
        else:
            if rnd in tests or flat in test_imgs or flat in near or state not in ok:
                continue
            if holdout and field == holdout:
                continue
        out.append(dict(image=flat, crown=int(crown), round=rnd, state=state,
                        source=row.get("source") or "", field=field))
    return out


def image_mask(store, flat, polys, crowns, shape, who=None):
    """Mask cả ảnh chỉ giữ pixel của các tán `crowns`; còn lại 255."""
    h, w = shape
    src = store.load_mask(flat, (h, w), who)
    out = np.full((h, w), fs.IGNORE, np.uint8)
    for k in crowns:
        if k >= len(polys):
            continue
        b = fs.crown_box(polys[k], w, h)
        inside = fs.crown_mask(polys[k], b)
        dst = out[b[1]:b[3], b[0]:b[2]]
        dst[inside] = src[b[1]:b[3], b[0]:b[2]][inside]
    return out


def group_by_image(items):
    by = {}
    for it in items:
        by.setdefault(it["image"], []).append(it)
    return by


def default_dirs(repo=None):
    """(thư mục ảnh, thư mục nhãn tán, thư mục nhãn hoa, cấu hình vòng) theo
    configs/config.yaml và configs/flower_rounds.yaml."""
    import yaml
    from app import flower_api
    repo = Path(repo or Path(__file__).resolve().parent.parent)
    cfg = yaml.safe_load((repo / "configs" / "config.yaml").read_text(encoding="utf-8"))
    rcfg, errors = flower_api.load_config(repo / "configs" / "flower_rounds.yaml")
    if errors:
        raise SystemExit("configs/flower_rounds.yaml: " + "; ".join(errors))
    masks = Path(cfg["data"]["masks_dir"])
    images = Path(cfg["data"]["images_dir"])
    if not masks.is_absolute():
        masks = repo / masks
    if not images.is_absolute():
        images = repo / images
    return images, masks / "corrected", masks / rcfg["data_dir"], rcfg["rounds"]


def imread_rgb(path):
    import cv2
    buf = np.fromfile(str(path), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def exists(path):
    return os.path.exists(path)
