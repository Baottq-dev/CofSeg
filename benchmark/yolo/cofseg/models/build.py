"""Dựng model từ một khối config.

    model:
      name: two_stage
      detector: {name: yolo_seg, weights: runs/train/.../best.pt, imgsz: 1024}
      prompter: {size: l}

`name` tra sổ đăng ký "model"; mọi khoá còn lại đưa vào hàm khởi tạo. Các
model ghép (two_stage, refine) nhận thêm khối con và tự gọi build_model cho
khối đó, nên config lồng sâu bao nhiêu cũng được mà hàm này không cần biết.
"""

from __future__ import annotations

import inspect

from ..registry import resolve
from .base import SegmentationModel


def build_model(spec: dict | SegmentationModel) -> SegmentationModel:
    if isinstance(spec, SegmentationModel):
        return spec
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"Khối model cần khoá 'name', nhận: {spec!r}")
    spec = dict(spec)
    cls = resolve("model", spec.pop("name"))
    return cls(**spec)


def model_param_names(name: str) -> set[str] | None:
    """Tên tham số hàm khởi tạo của model `name`, để kiểm --key ghi đè.
    None nếu lớp nhận **kwargs (không kiểm được)."""
    cls = resolve("model", name)
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {k for k in params if k != "self"}
