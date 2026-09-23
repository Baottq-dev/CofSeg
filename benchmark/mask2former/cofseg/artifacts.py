"""Thư mục kết quả cho mỗi lần chạy, kèm dấu vết để tái lập.

Bố cục:

    runs/<viec>/<thoi-diem>_<ten>_<nhan-tham-so>/

    runs/train/2026-09-09_155401_yolo26s-seg_i640b4e100/
    runs/probe/2026-09-09_130145_yolo26s-seg_i1536b2e100/

Ba quyết định, mỗi cái sửa một khuyết điểm đã gặp thật:

1. THỜI ĐIỂM ĐỨNG TRƯỚC. Sắp xếp theo tên cũng là sắp theo thời gian, trong
   mọi trình duyệt file và mọi lệnh ls. Đặt tên trước thì các cấu hình khác
   nhau xen kẽ nhau và không lần ra được thứ tự đã chạy.

2. NHÃN THAM SỐ SINH TỪ CẤU HÌNH THẬT, không phải từ nhãn tĩnh trong config.
   Đây là lỗi đã xảy ra: thư mục tên "yolo26s_seg_1536" trong khi lần chạy
   thật dùng imgsz=640. Tên thư mục không được phép nói dối.

3. TÁCH THEO LOẠI VIỆC. Một lần dò VRAM 30 giây không nên nằm lẫn với một lần
   train 4 tiếng.

Trong mỗi thư mục luôn có config đã dùng và env.json (phiên bản thư viện, GPU,
commit git, repo sạch hay bẩn). Thiếu những thứ đó thì ba tháng sau không ai
nói được con số sinh ra từ đâu.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import config as cfgmod

# Gộp mọi ký tự không an toàn cho tên file thành một dấu gạch ngang. Windows
# cấm \ / : * ? " < > | và cả tên có dấu; giữ danh sách cho phép thay vì danh
# sách cấm để không sót.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    return _UNSAFE.sub("-", str(text)).strip("-_.") or "unnamed"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_dir(
    root: str | Path, kind: str, name: str, tag: str = "", when=None
) -> Path:
    """runs/<kind>/<thời-điểm>_<name>[_<tag>]/ — luôn là thư mục mới.

    `tag` mô tả cấu hình THẬT của lần chạy (vd "i640b4e100"); trainer sinh ra
    nó qua Trainer.run_tag() nên mỗi họ model tự quyết tham số nào đáng ghi.
    """
    ts = (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    stem = "_".join(p for p in (ts, slugify(name), slugify(tag) if tag else "") if p)
    parent = Path(root) / slugify(kind)
    d, n = parent / stem, 2
    # Hai lần chạy trong cùng một giây vẫn phải ra hai thư mục khác nhau, thay
    # vì ném lỗi và làm mất cả lần chạy.
    while d.exists():
        d = parent / f"{stem}~{n}"
        n += 1
    d.mkdir(parents=True)
    return d


def write_env(run_dir: str | Path) -> dict:
    """Ghi lại môi trường. Gọi TRƯỚC khi chạy để có dấu vết cả khi chạy hỏng."""
    env: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        # Repo bẩn nghĩa là commit ở trên KHÔNG mô tả đủ code đã chạy.
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": {},
    }
    for mod in ("torch", "torchvision", "ultralytics", "numpy", "cv2", "pycocotools"):
        try:
            m = __import__(mod)
            env["packages"][mod] = getattr(m, "__version__", "?")
        except ImportError:
            env["packages"][mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**20
                ),
                "cuda": torch.version.cuda,
            }
    except Exception:  # noqa: BLE001 - môi trường hỏng không được chặn việc chạy
        pass

    Path(run_dir, "env.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return env


def snapshot_config(run_dir: str | Path, cfg: dict) -> None:
    cfgmod.dump(cfg, Path(run_dir) / "config.yaml")
