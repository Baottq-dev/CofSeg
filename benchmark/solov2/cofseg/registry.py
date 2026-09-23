"""Sổ đăng ký: tên trong config -> lớp thực thi.

Đây là chỗ khiến benchmark/solov2/train.py không cần biết YOLO tồn tại. Thêm một họ
model mới (Mask R-CNN, model tự viết...) = thêm một file có @register, không
sửa dòng nào ở script hay ở vòng lặp đánh giá.

Hai "kind" đang dùng: "trainer" và "model".
"""

from __future__ import annotations

from typing import Callable, TypeVar

_REG: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


def register(kind: str, name: str) -> Callable[[T], T]:
    """Trang trí một lớp để config gọi được nó bằng tên."""

    def deco(cls: T) -> T:
        table = _REG.setdefault(kind, {})
        if name in table and table[name] is not cls:
            raise ValueError(f"{kind} '{name}' đã được đăng ký bởi {table[name]}")
        table[name] = cls
        return cls

    return deco


def resolve(kind: str, name: str) -> type:
    table = _REG.get(kind, {})
    if name not in table:
        raise KeyError(
            f"Không có {kind} tên '{name}'. Hiện có: {sorted(table) or '(chưa nạp)'}. "
            f"Nếu tên đúng thì module chứa nó chưa được import — xem cofseg/{kind}s/__init__.py."
        )
    return table[name]


def available(kind: str) -> list[str]:
    return sorted(_REG.get(kind, {}))
