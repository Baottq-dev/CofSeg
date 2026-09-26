"""Đọc siêu tham số từ dòng lệnh, dùng chung cho mọi script.

Cùng một cú pháp ở train.py và evaluate.py, nên chỉ có một chỗ để đúng:

    --epochs 100        --imgsz=1024        --amp          (cờ trần = true)
    --cos-lr true       (gạch nối = gạch dưới)

Giá trị được ép kiểu theo YAML nên "0.5", "true", "[1,2]" đều thành đúng kiểu.
Tên lạ thì BÁO LỖI kèm gợi ý, thay vì im lặng bỏ qua rồi để người chạy chờ ba
tiếng mới biết tham số không vào. Khi không có danh sách tên hợp lệ (`valid`
là None) thì nhận mọi tên — đó là lựa chọn của trainer, không phải của đây.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

import yaml


def parse_overrides(
    tokens: list[str],
    valid: Iterable[str] | None,
    locked: Iterable[str] = (),
    what: str = "trainer",
) -> dict:
    """Danh sách đối số lạ -> dict ghi đè. `what` chỉ để thông báo lỗi dễ đọc."""
    valid_set = None if valid is None else set(valid)
    locked_set = set(locked)
    out: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(
                f"Không hiểu đối số {tok!r}. Siêu tham số phải có dạng --ten giatri."
            )
        body = tok[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            key, raw = body, tokens[i + 1]
            i += 1
        else:
            key, raw = body, "true"
        key = key.replace("-", "_")

        if key in locked_set:
            raise SystemExit(
                f"--{key} bị khoá: {what} tự đặt nó để kết quả rơi đúng thư mục run."
            )
        if valid_set is not None and key not in valid_set:
            near = difflib.get_close_matches(key, sorted(valid_set), n=3, cutoff=0.6)
            hint = f" Ý bạn là: {', '.join('--' + n for n in near)}?" if near else ""
            raise SystemExit(
                f"{what} không có tham số {key!r}.{hint}\n"
                f"Xem toàn bộ tham số: --list-params"
            )
        out[key] = _typed(raw)
        i += 1
    return out


def describe_params(defaults: dict, locked: Iterable[str] = ()) -> str:
    """Bảng tên/mặc định cho --list-params."""
    locked_set = set(locked)
    lines = []
    for k in sorted(defaults):
        mark = "  [khoá]" if k in locked_set else ""
        lines.append(f"  --{k:<24} mặc định {defaults[k]!r}{mark}")
    return "\n".join(lines)


def _typed(raw: str):
    """Ép kiểu giá trị dòng lệnh.

    yaml.safe_load lo phần lớn ("0.5", "true", "[1,2]"), nhưng PyYAML theo
    YAML 1.1 KHÔNG nhận ký hiệu khoa học thiếu dấu chấm: "5e-5" và "1e-4" ra
    chuỗi chứ không ra số. Mà "1e-4" đúng là cách viết tự nhiên nhất cho lr
    của Mask2Former, và chuỗi đó đi thẳng vào ultralytics thì hỏng.
    """
    v = yaml.safe_load(raw)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v
