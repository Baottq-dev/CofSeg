"""Khối `mask_head:` của config train: chọn đầu mặt nạ A1 (xem dyn_head.py).

Không có khối này thì YOLO-seg train với đầu gốc của ultralytics. A1 bật bằng
một FILE CONFIG RIÊNG (configs/train/yolo26s-dyn.yaml), không bằng cờ: đổi
kiến trúc là đổi model, và tên model đã nằm trong config. Lệnh train vẫn tự
nói nó chạy gì, qua tên file config.

File này chỉ kiểm tham số và đặt nhãn, không import torch hay ultralytics,
để --print-config chạy được tức thì.
"""

from __future__ import annotations

import difflib

#: Cấu hình đã chốt cho A1, kèm nhãn ngắn trong hậu tố tên khi khác mặc định.
DEFAULTS: dict = {
    "type": "dynamic",
    "dims": [8, 8],       # kênh các lớp ẩn của mạng động; [] = tuyến tính như gốc
    "coords": True,       # thêm toạ độ tương đối so với anchor
    "window": 1.5,        # cửa sổ giữ mặt nạ = box x window; 1.0 = cắt sát box như gốc
    "max_pos": 64,        # số tán tối đa mỗi ảnh được tính loss mặt nạ
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(block) -> dict | None:
    """Khối đã kiểm, đủ mọi khoá; None nếu config không có `mask_head`."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit(f"`mask_head` phải là khối lồng nhau, nhận được {block!r}")
    for k in block:
        if k not in DEFAULTS:
            near = difflib.get_close_matches(k, sorted(DEFAULTS), n=3, cutoff=0.5)
            hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
            raise SystemExit(f"mask_head không có khoá {k!r}.{hint} Có: {', '.join(DEFAULTS)}")
    out = {**DEFAULTS, **block}
    if out["type"] != "dynamic":
        raise SystemExit(f"mask_head.type chỉ có 'dynamic', nhận được {out['type']!r}")
    dims = out["dims"]
    if not isinstance(dims, list) or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0
                                             for d in dims):
        raise SystemExit(f"mask_head.dims phải là danh sách số nguyên dương (có thể rỗng), nhận được {dims!r}")
    if not isinstance(out["coords"], bool):
        raise SystemExit(f"mask_head.coords phải là true/false, nhận được {out['coords']!r}")
    if not _num(out["window"]) or float(out["window"]) < 1.0:
        raise SystemExit(f"mask_head.window phải là số >= 1, nhận được {out['window']!r}")
    mp = out["max_pos"]
    if not isinstance(mp, int) or isinstance(mp, bool) or mp < 1:
        raise SystemExit(f"mask_head.max_pos phải là số nguyên >= 1, nhận được {mp!r}")
    out["window"] = float(out["window"])
    return out


def tag(block: dict) -> str:
    """Hậu tố cho tên lượt chấm: `dyn`, kèm mọi tham số khác mặc định."""
    parts = ["dyn"]
    if block["dims"] != DEFAULTS["dims"]:
        parts.append("d" + ("x".join(map(str, block["dims"])) or "0"))
    if not block["coords"]:
        parts.append("nocoord")
    if block["window"] != DEFAULTS["window"]:
        parts.append(f"win{block['window']:g}")
    if block["max_pos"] != DEFAULTS["max_pos"]:
        parts.append(f"mp{block['max_pos']}")
    return "-".join(parts)


def describe_active(block: dict | None) -> str:
    """Một dòng cho run.log, in cả khi là đầu gốc."""
    if not block:
        return "Đầu mặt nạ: gốc của ultralytics (prototype x hệ số, cắt sát box)"
    return ("Đầu mặt nạ: A1 động "
            f"(dims {block['dims']}, coords {block['coords']}, window {block['window']:g}, "
            f"max_pos {block['max_pos']})")
