"""Đọc config YAML + ghi đè từ dòng lệnh.

Cho phép --set train.imgsz=1536 để quét tham số mà không đẻ ra một file YAML
mới cho mỗi lần thử. Giá trị được ép kiểu theo YAML nên "true"/"0.5"/"[1,2]"
đều hiểu đúng.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Nạp config; `overrides` là các chuỗi dạng "a.b.c=value"."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # base: <file khác> -> nạp file đó trước rồi phủ lên. Giữ cho các config
    # cùng họ không phải chép lại toàn bộ tham số.
    if "base" in cfg:
        parent = load(path.parent / cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khoa.muc=giatri, nhận được: {item!r}")
        key, raw = item.split("=", 1)
        cur: dict[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(f"--set {key}: '{p}' không phải một khoá lồng nhau")
        cur[parts[-1]] = yaml.safe_load(raw)
    return cfg


def dump(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
