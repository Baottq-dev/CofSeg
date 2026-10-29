"""Hai khối config của distill/: `student:` (trò nhẹ) và `outside_box:` (giám sát mặt nạ ngoài hộp).

Mỗi biến thể bật bằng FILE CONFIG RIÊNG (configs/train/*.yaml), như khối `mask_head:` của benchmark/yolo:
đổi kiến trúc hay đổi vùng giám sát là đổi model, và tên model nằm trong config. Tham số khác mặc định đi
vào tên thư mục run, để hai cấu hình khác nhau không bao giờ ra cùng một tên.

File này chỉ kiểm tham số và đặt nhãn, không import torch hay ultralytics, để --print-config chạy tức thì.
"""

from __future__ import annotations

import difflib

#: Trò 0,375: 5,76 triệu tham số,
#: 22,04 GFLOPs ở 1024×576 (đã fuse), −55,5% so với YOLO26s-seg. Trò 0,25 chạy song song ở đợt 1 (quyết định
#: 29/10) bằng configs/train/tro-025.yaml: 2,57 triệu tham số, 9,95 GFLOPs, −79,9%.
STUDENT_DEFAULTS: dict = {
    "width": 0.375,       # hệ số độ rộng; n = 0,25, s = 0,5
    "nm": 16,             # số prototype (gốc 32)
    "proto_s8": True,     # lớp 3×3 thứ hai của Proto chạy ở stride 8, trước bước phóng to (gốc: sau, ở stride 4)
    "drop_p3": True,      # bỏ nhánh hộp/điểm/hệ số ở P3; Proto vẫn nhận đặc trưng P3
    "init": "select",     # select = lấy trọng số từ checkpoint COCO ở khoá `model:`; scratch = khởi tạo ngẫu nhiên
}

#: Giám sát mặt nạ ngoài hộp (K-A): xem outside_box.py.
OUTSIDE_BOX_DEFAULTS: dict = {
    "train_expand": 1.25,  # loss mặt nạ tính trong hộp nhãn giãn ngần này lần quanh tâm (gốc 1,0)
    "infer_expand": 1.1,   # lúc chấm, cắt mặt nạ bằng hộp dự đoán giãn ngần này lần (crop_expand của bộ chấm)
}

INITS = ("select", "scratch")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _keys(name: str, block: dict, defaults: dict) -> None:
    for k in block:
        if k not in defaults:
            near = difflib.get_close_matches(k, sorted(defaults), n=3, cutoff=0.5)
            hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
            raise SystemExit(f"{name} không có khoá {k!r}.{hint} Có: {', '.join(defaults)}")


def check_student(block) -> dict | None:
    """Khối `student` đã kiểm, đủ mọi khoá; None nếu config không có."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit(f"`student` phải là khối lồng nhau, nhận được {block!r}")
    _keys("student", block, STUDENT_DEFAULTS)
    out = {**STUDENT_DEFAULTS, **block}
    w = out["width"]
    # Nguồn trọng số rộng nhất là YOLO26s (0,5); trò không được rộng hơn nguồn.
    if not _num(w) or not 0.0 < float(w) <= 0.5:
        raise SystemExit(f"student.width phải là số trong (0, 0.5], nhận được {w!r}")
    out["width"] = float(w)
    nm = out["nm"]
    if isinstance(nm, bool) or not isinstance(nm, int) or not 1 <= nm <= 32:
        raise SystemExit(f"student.nm phải là số nguyên trong [1, 32], nhận được {nm!r}")
    for k in ("proto_s8", "drop_p3"):
        if not isinstance(out[k], bool):
            raise SystemExit(f"student.{k} phải là true/false, nhận được {out[k]!r}")
    if out["init"] not in INITS:
        raise SystemExit(f"student.init phải là một trong {', '.join(INITS)}, nhận được {out['init']!r}")
    return out


def check_outside_box(block) -> dict | None:
    """Khối `outside_box` đã kiểm; None nếu config không có."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit(f"`outside_box` phải là khối lồng nhau, nhận được {block!r}")
    _keys("outside_box", block, OUTSIDE_BOX_DEFAULTS)
    out = {**OUTSIDE_BOX_DEFAULTS, **block}
    for k in ("train_expand", "infer_expand"):
        if not _num(out[k]) or float(out[k]) < 1.0:
            raise SystemExit(f"outside_box.{k} phải là số >= 1, nhận được {out[k]!r}")
        out[k] = float(out[k])
    return out


def student_tag(block: dict | None) -> str:
    """Hậu tố cho tên run: rỗng ở cấu hình đã chốt, chỉ ghi phần khác mặc định."""
    if not block:
        return ""
    d = STUDENT_DEFAULTS
    parts = []
    if block["width"] != d["width"]:
        parts.append(f"w{block['width']:g}")
    if block["nm"] != d["nm"]:
        parts.append(f"nm{block['nm']}")
    if block["proto_s8"] != d["proto_s8"]:
        parts.append("ps8" if block["proto_s8"] else "ps4")
    if block["drop_p3"] != d["drop_p3"]:
        parts.append("nop3" if block["drop_p3"] else "p3")
    if block["init"] != d["init"]:
        parts.append(block["init"])
    return "-".join(parts)


def outside_box_tag(block: dict | None) -> str:
    """Hậu tố cho tên run; tên `ka` nằm sẵn trong config, đây chỉ ghi hệ số khác mặc định."""
    if not block:
        return ""
    d = OUTSIDE_BOX_DEFAULTS
    parts = []
    if block["train_expand"] != d["train_expand"]:
        parts.append(f"tx{block['train_expand']:g}")
    if block["infer_expand"] != d["infer_expand"]:
        parts.append(f"ix{block['infer_expand']:g}")
    return "-".join(parts)


def describe_student(block: dict | None) -> str:
    """Một dòng cho run.log, in cả khi không bật."""
    if not block:
        return "Trò: không (kiến trúc của config)"
    return ("Trò: YOLO26-seg độ rộng {width:g}, {nm} prototype, Proto 3×3 ở stride {ps}, "
            "nhánh phát hiện {lv}, khởi tạo {init}").format(
        width=block["width"], nm=block["nm"], ps=8 if block["proto_s8"] else 4,
        lv="P4–P5" if block["drop_p3"] else "P3–P5",
        init="từ checkpoint COCO của khoá model" if block["init"] == "select" else "ngẫu nhiên")


def describe_outside_box(block: dict | None) -> str:
    if not block:
        return "Giám sát mặt nạ: trong hộp nhãn (gốc)"
    return (f"Giám sát mặt nạ ngoài hộp: loss trong hộp nhãn giãn ×{block['train_expand']:g}, "
            f"chia diện tích hộp gốc; lúc chấm cắt bằng hộp dự đoán giãn ×{block['infer_expand']:g}")
