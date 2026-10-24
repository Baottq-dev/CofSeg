"""Tăng cường thêm của YOLO-seg, bật bằng cờ `--aug` của train.py.

Không có `--aug` thì KHÔNG có gì ở đây chạy: dataset và chuỗi tăng cường là
của ultralytics, đúng như lượt baseline. Như --loss, phép thêm chỉ bật được
bằng cờ: khối `aug:` trong config hay `--set aug.use=...` đều bị chặn, để lệnh
train tự nói nó khác baseline ở đâu. Tham số của một phép đã bật thì chỉnh
bằng `--set aug.<tên>.<tham số>=<giá trị>`.

Ba phép, ghép bằng `+` (hoặc `,`), vd `--aug ag1+pa1`:

    ag1  chuyển vị (đổi trục x và y) với xác suất p, đặt sau hai phép lật có
         sẵn. Lật dọc 0,5 và lật ngang 0,5 của config chỉ cho 4 trong 8 phần tử
         của nhóm đối xứng hình vuông (không có xoay 90°); thêm chuyển vị 0,5
         thì đủ 8, mỗi phần tử xác suất 1/8.
    pa1  dán sau: dán tán cùng ruộng NẰM DƯỚI các tán có nhãn, chạm một tán
         đích với tỉ lệ tiếp xúc c* ~ U[c_lo, c_hi], phần bị che <= hid. Tán có
         sẵn giữ nguyên điểm ảnh và nhãn; nhãn của tán dán là phần nhìn thấy.
    ag4  đối chứng của pa1: cùng cách chọn tán đích và tán nguồn, cùng số tán
         mỗi ảnh, nhưng tán dán cách mọi tán >= gap px ảnh gốc, nên không chạm
         và không bị che. pa1 so với ag4 tách được riêng phần do tiếp xúc.

pa1 và ag4 loại trừ nhau: dán cả hai kiểu vào một ảnh thì không còn đối chứng.

File này chỉ chứa bảng tra và phần kiểm tham số, không import numpy, torch hay
ultralytics: `--list-augs` và `--print-config` phải chạy được tức thì. Phần
dán nằm ở aug_paste.py, phần nối vào dataloader ở aug_train.py.
"""

from __future__ import annotations

import difflib
import re


def _number(name: str, v, lo: float, hi: float | None = None) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) < lo or (hi is not None and float(v) > hi):
        rng = f"trong [{lo:g}, {hi:g}]" if hi is not None else f">= {lo:g}"
        raise SystemExit(f"aug.{name} phải là số {rng}, nhận được {v!r}")


def _integer(name: str, v, lo: int) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise SystemExit(f"aug.{name} phải là số nguyên >= {lo}, nhận được {v!r}")


def _check_k(name: str, p: dict) -> None:
    _integer(f"{name}.k_min", p["k_min"], 1)
    _integer(f"{name}.k_max", p["k_max"], 1)
    if p["k_max"] < p["k_min"]:
        raise SystemExit(f"aug.{name}.k_max ({p['k_max']}) phải >= aug.{name}.k_min ({p['k_min']})")


def _check_ag1(p: dict) -> None:
    _number("ag1.p", p["p"], 0.0, 1.0)


def _check_pa1(p: dict) -> None:
    _number("pa1.p", p["p"], 0.0, 1.0)
    _check_k("pa1", p)
    _number("pa1.c_lo", p["c_lo"], 0.0, 1.0)
    _number("pa1.c_hi", p["c_hi"], 0.0, 1.0)
    if p["c_hi"] < p["c_lo"]:
        raise SystemExit(f"aug.pa1.c_hi ({p['c_hi']}) phải >= aug.pa1.c_lo ({p['c_lo']})")
    _number("pa1.hid", p["hid"], 0.0, 0.95)


def _check_ag4(p: dict) -> None:
    _number("ag4.p", p["p"], 0.0, 1.0)
    _check_k("ag4", p)
    _number("ag4.gap", p["gap"], 1.0)


#: tên -> mô tả, tham số (mặc định, nhãn ngắn trong tên thư mục, giải thích), hàm kiểm.
#: Thứ tự khai ở đây là thứ tự trong tên thư mục: `--aug pa1+ag1` và
#: `--aug ag1+pa1` ra cùng một tên.
AUGS: dict = {
    "ag1": {
        "doc": "AG1: chuyển vị x<->y sau hai phép lật có sẵn, đủ 8 phần tử D4 (R2 chỉ có 4)",
        "params": {
            "p": (0.5, "p", "xác suất chuyển vị mỗi ảnh; 0,5 cùng lật dọc/ngang 0,5 cho D4 đều"),
        },
        "check": _check_ag1,
    },
    "pa1": {
        "doc": "PA1: dán sau tán cùng ruộng, chạm một tán đích với tiếp xúc c* ~ U[c_lo, c_hi], bị che <= hid",
        "params": {
            "p": (0.5, "p", "xác suất một ảnh được dán"),
            "k_min": (2, "k", "số tán dán ít nhất mỗi ảnh được chọn"),
            "k_max": (4, "K", "số tán dán nhiều nhất mỗi ảnh được chọn"),
            "c_lo": (0.2, "cl", "cận dưới của tiếp xúc đích c* (tỉ lệ biên tán đích trong 5 px gốc của tán dán)"),
            "c_hi": (0.5, "ch", "cận trên của c*"),
            "hid": (0.5, "hid", "phần tán dán bị các tán có sẵn che, tối đa"),
        },
        "check": _check_pa1,
    },
    "ag4": {
        "doc": "AG4: đối chứng của PA1, dán cùng kiểu tán nguồn cách mọi tán >= gap px gốc (không chạm)",
        "params": {
            "p": (0.5, "p", "xác suất một ảnh được dán"),
            "k_min": (2, "k", "số tán dán ít nhất mỗi ảnh được chọn"),
            "k_max": (4, "K", "số tán dán nhiều nhất mỗi ảnh được chọn"),
            "gap": (25.0, "g", "khoảng cách tối thiểu (px ảnh gốc) từ tán dán tới mọi tán, kể cả tán dán khác"),
        },
        "check": _check_ag4,
    },
}

#: Cặp không đi chung được, kèm lý do in ra cho người gõ lệnh.
EXCLUSIVE = {
    frozenset({"pa1", "ag4"}): "pa1 và ag4 là hai nhánh của cùng một phép so; dán cả hai kiểu vào một "
                               "ảnh thì không còn tách được phần do tiếp xúc.",
}

#: Họ dán, dùng chung bộ tán nguồn và chỗ cắm trong dataloader.
PASTE = ("pa1", "ag4")


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def apply(cfg: dict, spec: str | None) -> str:
    """Ghi phép thêm đã kiểm vào `cfg["aug"]`, trả về hậu tố cho tên run.

    Không có `spec` thì trả về "" và cfg không có khoá `aug`. Hậu tố gồm tên
    các phép theo thứ tự của AUGS và mọi tham số khác mặc định.
    """
    block = cfg.get("aug")
    if block is not None and not isinstance(block, dict):
        raise SystemExit(f"khoá `aug` phải là khối lồng nhau, nhận được {block!r}")
    if block and "use" in block:
        raise SystemExit(
            "Phép tăng cường thêm chỉ bật bằng cờ --aug, không ghi `aug.use` trong config "
            "hay --set: lệnh train phải tự nói nó có tăng cường khác gốc hay không."
        )
    if not spec:
        if block:
            raise SystemExit(
                f"Có `aug.{next(iter(block))}...` (từ config hoặc --set) nhưng không có --aug. "
                "Tham số tăng cường chỉ có tác dụng kèm cờ --aug <tên>."
            )
        cfg.pop("aug", None)
        return ""

    asked = [s.strip().lower() for s in re.split(r"[+,]", str(spec)) if s.strip()]
    for name in asked:
        if name not in AUGS:
            near = difflib.get_close_matches(name, sorted(AUGS), n=3, cutoff=0.5)
            hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
            raise SystemExit(f"--aug {name!r} không có.{hint} Xem --list-augs")
    use = [n for n in AUGS if n in asked]
    for pair, why in EXCLUSIVE.items():
        if pair <= set(use):
            raise SystemExit(f"--aug {'+'.join(sorted(pair))}: {why}")
    for extra in set(block or {}) - set(use):
        raise SystemExit(f"Có tham số cho `aug.{extra}` nhưng --aug không bật {extra!r}.")

    out: dict = {"use": use}
    parts = []
    for name in use:
        spec_ = AUGS[name]
        given = (block or {}).get(name) or {}
        if not isinstance(given, dict):
            raise SystemExit(f"`aug.{name}` phải là khối tham số, nhận được {given!r}")
        for k in given:
            if k not in spec_["params"]:
                near = difflib.get_close_matches(k, sorted(spec_["params"]), n=3, cutoff=0.5)
                hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
                raise SystemExit(f"aug {name!r} không có tham số {k!r}.{hint} Xem --list-augs")
        defaults = {k: v[0] for k, v in spec_["params"].items()}
        merged = {**defaults, **given}
        spec_["check"](merged)
        out[name] = merged
        diff = [f"{spec_['params'][k][1]}{_fmt(v)}" for k, v in merged.items() if v != defaults[k]]
        parts.append("-".join([name, *diff]))
    out["tag"] = "-".join(parts)
    cfg["aug"] = out
    return out["tag"]


def paste_mode(block: dict | None) -> str | None:
    """'pa1', 'ag4' hoặc None: phép dán nào đang bật (apply() bảo đảm tối đa một)."""
    return next((n for n in PASTE if block and n in block.get("use", ())), None)


def describe() -> str:
    """Bảng cho --list-augs."""
    lines = [f"{len(AUGS)} phép tăng cường thêm (bỏ trống --aug = chuỗi gốc của ultralytics):", ""]
    for name, spec_ in AUGS.items():
        lines.append(f"  {name}: {spec_['doc']}")
        for k, (default, _tag, doc) in spec_["params"].items():
            lines.append(f"      {name + '.' + k:<14} mặc định {default!r:<6} {doc}")
    lines += ["", "Không ghép được: " + "; ".join("+".join(sorted(p)) for p in EXCLUSIVE),
              "", "Bật:       --aug pa1            --aug ag1+pa1",
              "Chỉnh:     --aug pa1 --set aug.pa1.p=0.3"]
    return "\n".join(lines)


def describe_active(block: dict | None) -> str:
    """Một dòng cho run.log, in cả khi không bật gì."""
    if not block:
        return "Tăng cường thêm: không (chuỗi gốc của ultralytics)"
    return "Tăng cường thêm: " + "; ".join(
        f"{n} ({', '.join(f'{k} {_fmt(v)}' for k, v in block[n].items())})" for n in block["use"]
    )
