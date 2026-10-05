"""Khối `mask_head:` của config train: chọn kiến trúc nhánh mặt nạ.

Không có khối này thì YOLO-seg train với đầu gốc của ultralytics. Mỗi kiến
trúc bật bằng một FILE CONFIG RIÊNG, không bằng cờ: đổi kiến trúc là đổi model,
và tên model đã nằm trong config. Lệnh train vẫn tự nói nó chạy gì, qua tên
file config.

- `type: dynamic` — A1, đầu mặt nạ động không cắt sát box
  (configs/train/yolo26s-dyn.yaml, dyn_head.py).
- `type: gpr` — T1, lấy mẫu lại prototype có dẫn hướng (M1) cùng loss biên
  hiệu chỉnh theo nhiễu nhãn (L3) (configs/train/yolo26s-gpr.yaml,
  gpr_head.py, gpr_train.py).

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

#: Cấu hình đã chốt cho T1 (GPR + NCB).
GPR_DEFAULTS: dict = {
    "type": "gpr",
    "stages": [1, 2],     # 1 = GPR-1 thay ConvTranspose (s8->s4); 2 = GPR-2 đưa proto lên s2; [] = Proto gốc
    "detach_guide": True,  # bản đồ dẫn hướng không truyền gradient ngược vào backbone
    "groups": 4,          # số nhóm kênh của GPR-1, mỗi nhóm một trường độ lệch (GPR-2 luôn 1)
    "scope": 1.0,         # độ lệch tối đa, tính bằng ô lưới thô
    "sigma": 2.4,         # σ của nhãn mềm, px ảnh đầu vào (≈ 6 px ảnh gốc ở imgsz 1024); 0 = chỉ tỉ lệ phủ
    "max_pos": 64,        # số tán tối đa mỗi ảnh được tính loss mặt nạ
}

TYPES = {"dynamic": DEFAULTS, "gpr": GPR_DEFAULTS}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check(block) -> dict | None:
    """Khối đã kiểm, đủ mọi khoá; None nếu config không có `mask_head`."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit(f"`mask_head` phải là khối lồng nhau, nhận được {block!r}")
    kind = block.get("type", "dynamic")
    if kind not in TYPES:
        raise SystemExit(f"mask_head.type phải là một trong {', '.join(map(repr, TYPES))}, nhận được {kind!r}")
    defaults = TYPES[kind]
    for k in block:
        if k not in defaults:
            near = difflib.get_close_matches(k, sorted(defaults), n=3, cutoff=0.5)
            hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
            raise SystemExit(f"mask_head ({kind}) không có khoá {k!r}.{hint} Có: {', '.join(defaults)}")
    out = {**defaults, **block}
    mp = out["max_pos"]
    if not _int(mp) or mp < 1:
        raise SystemExit(f"mask_head.max_pos phải là số nguyên >= 1, nhận được {mp!r}")
    return _check_gpr(out) if kind == "gpr" else _check_dyn(out)


def _check_dyn(out: dict) -> dict:
    dims = out["dims"]
    if not isinstance(dims, list) or not all(_int(d) and d > 0 for d in dims):
        raise SystemExit(f"mask_head.dims phải là danh sách số nguyên dương (có thể rỗng), nhận được {dims!r}")
    if not isinstance(out["coords"], bool):
        raise SystemExit(f"mask_head.coords phải là true/false, nhận được {out['coords']!r}")
    if not _num(out["window"]) or float(out["window"]) < 1.0:
        raise SystemExit(f"mask_head.window phải là số >= 1, nhận được {out['window']!r}")
    out["window"] = float(out["window"])
    return out


def _check_gpr(out: dict) -> dict:
    st = out["stages"]
    if not isinstance(st, list) or not all(_int(s) and s in (1, 2) for s in st) or len(set(st)) != len(st):
        raise SystemExit(f"mask_head.stages phải là danh sách con của [1, 2], nhận được {st!r}")
    if not isinstance(out["detach_guide"], bool):
        raise SystemExit(f"mask_head.detach_guide phải là true/false, nhận được {out['detach_guide']!r}")
    if not _int(out["groups"]) or out["groups"] < 1:
        raise SystemExit(f"mask_head.groups phải là số nguyên >= 1, nhận được {out['groups']!r}")
    if not _num(out["scope"]) or float(out["scope"]) <= 0:
        raise SystemExit(f"mask_head.scope phải là số > 0, nhận được {out['scope']!r}")
    if not _num(out["sigma"]) or float(out["sigma"]) < 0:
        raise SystemExit(f"mask_head.sigma phải là số >= 0, nhận được {out['sigma']!r}")
    out["stages"] = sorted(st)
    out["scope"], out["sigma"] = float(out["scope"]), float(out["sigma"])
    return out


def check_train(block: dict | None, train: dict) -> None:
    """Ràng buộc giữa khối `mask_head` và khối `train` của cùng config."""
    if block and block["type"] == "gpr" and train.get("mask_ratio", 4) != 1:
        raise SystemExit(
            "Config GPR cần train.mask_ratio: 1: loss NCB tính tỉ lệ phủ từ nhãn ở đủ độ phân "
            f"giải đầu vào. Đang là {train.get('mask_ratio', 4)!r}."
        )


def tag(block: dict) -> str:
    """Hậu tố cho tên lượt chấm: `dyn` hoặc `gpr`, kèm mọi tham số khác mặc định."""
    if block["type"] == "gpr":
        d = GPR_DEFAULTS
        parts = ["gpr"]
        if block["stages"] != d["stages"]:
            parts.append("s" + ("".join(map(str, block["stages"])) or "0"))
        if not block["detach_guide"]:
            parts.append("grad")
        if block["groups"] != d["groups"]:
            parts.append(f"g{block['groups']}")
        if block["scope"] != d["scope"]:
            parts.append(f"sc{block['scope']:g}")
        if block["sigma"] != d["sigma"]:
            parts.append(f"sig{block['sigma']:g}")
        if block["max_pos"] != d["max_pos"]:
            parts.append(f"mp{block['max_pos']}")
        return "-".join(parts)
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
    if block["type"] == "gpr":
        return ("Đầu mặt nạ: GPR + loss NCB "
                f"(stages {block['stages']}, detach_guide {block['detach_guide']}, groups {block['groups']}, "
                f"scope {block['scope']:g}, sigma {block['sigma']:g} px, max_pos {block['max_pos']})")
    return ("Đầu mặt nạ: A1 động "
            f"(dims {block['dims']}, coords {block['coords']}, window {block['window']:g}, "
            f"max_pos {block['max_pos']})")
