"""Thư mục kết quả cho mỗi lần chạy, kèm dấu vết để tái lập.

Bố cục:

    runs/<viec>/<thoi-diem>_<ten>_<bo-fold>_<nhan-tham-so>/

    runs/train/2026-09-25_155401_maskrcnn-r50-d2_block-f4_i1024b4e50/
    runs/train/2026-09-25_181233_maskrcnn-r50-d2_flight-f4_i1024b4e50/
    runs/probe/2026-09-25_130145_maskrcnn-r50-d2_block-f4_i1024b4e50/

Bốn quyết định, mỗi cái sửa một khuyết điểm đã gặp thật:

1. THỜI ĐIỂM ĐỨNG TRƯỚC. Sắp xếp theo tên cũng là sắp theo thời gian, trong
   mọi trình duyệt file và mọi lệnh ls. Đặt tên trước thì các cấu hình khác
   nhau xen kẽ nhau và không lần ra được thứ tự đã chạy.

2. NHÃN THAM SỐ SINH TỪ CẤU HÌNH THẬT, không phải từ nhãn tĩnh trong config.
   Đây là lỗi đã xảy ra: thư mục tên "yolo26s_seg_1536" trong khi lần chạy
   thật dùng imgsz=640. Tên thư mục không được phép nói dối.

3. TÁCH THEO LOẠI VIỆC. Một lần dò VRAM 30 giây không nên nằm lẫn với một lần
   train 4 tiếng.

4. BỘ FOLD NẰM TRONG TÊN. Hai cách chia val cho hai bộ fold cùng đặt tên
   f1..f6, nên hai lần chạy khác hẳn nhau lại ra tên thư mục giống hệt. Nhìn
   `..._maskrcnn-r50-d2_i1024b4e50` thì không biết nó train trên bộ nào; nhìn
   `..._maskrcnn-r50-d2_block-f4_i1024b4e50` thì biết.

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


def write_env(run_dir: str | Path, cfg: dict | None = None) -> dict:
    """Ghi lại môi trường. Gọi TRƯỚC khi chạy để có dấu vết cả khi chạy hỏng.

    Có `cfg` thì ghi kèm bộ fold đã dùng (đọc từ fold.json của thư mục dữ
    liệu): ba tháng sau mở một thư mục kết quả là biết ngay nó train trên bộ
    nào, cắt val kiểu gì, bỏ bao nhiêu ảnh làm đệm.
    """
    env: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        # Repo bẩn nghĩa là commit ở trên KHÔNG mô tả đủ code đã chạy.
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": {},
    }
    # detectron2/mmdet/mmcv/mmengine là framework của chính ba model đang so;
    # thiếu chúng ở đây thì env.json không mô tả được thứ đã chạy, mà trên máy
    # thuê đó là bản ghi duy nhất còn lại về môi trường.
    for mod in ("torch", "torchvision", "ultralytics", "detectron2",
                "mmdet", "mmcv", "mmengine", "numpy", "cv2", "pycocotools"):
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

    if cfg is not None:
        env["dataset"] = fold_facts(cfg)

    Path(run_dir, "env.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return env


def snapshot_config(run_dir: str | Path, cfg: dict) -> None:
    cfgmod.dump(cfg, Path(run_dir) / "config.yaml")


# --------------------------------------------------------- bộ fold của lần chạy
def data_root(cfg: dict) -> Path | None:
    """Thư mục fold mà config trỏ vào, bất kể trainer nào.

    detectron2/mmdet đọc `data.root`; ultralytics đọc `data.yaml` nằm trong
    chính thư mục fold đó.
    """
    d = cfg.get("data") if isinstance(cfg, dict) else None
    if not isinstance(d, dict):
        return None
    if d.get("root"):
        return Path(str(d["root"]))
    if d.get("yaml"):
        return Path(str(d["yaml"])).parent
    return None


def dataset_tag(cfg: dict) -> str:
    """Nhãn ngắn nhận ra BỘ FOLD: 'block-f4', 'flight-f4'.

    Đi vào tên thư mục lần chạy, vào tên file dự đoán, và vào khoá của bảng
    tổng hợp. Thiếu nó thì `maskrcnn_f4` của hai bộ trông y hệt nhau và cái
    chạy sau đè cái trước.
    """
    p = data_root(cfg)
    if p is None:
        return ""
    parts = [x for x in p.parts if x not in ("", ".", "..")]
    if not parts:
        return ""
    fold = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else ""
    return f"{parent}-{fold}" if parent and parent != "export" else fold


def fold_facts(cfg: dict) -> dict | None:
    """Những gì fold.json của bộ dữ liệu nói về lần chạy này."""
    p = data_root(cfg)
    if p is None or not (p / "fold.json").is_file():
        return None
    try:
        doc = json.loads((p / "fold.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    vs = doc.get("val_split") or {}
    return {
        "root": p.as_posix(),
        "tag": dataset_tag(cfg),
        "fold": doc.get("fold"),
        "source_sha1": doc.get("source_sha1"),
        "val_method": vs.get("method"),
        "dropped": (doc.get("dropped") or {}).get("count"),
        "splits": {k: v.get("images") for k, v in (doc.get("splits") or {}).items()},
        "leak": (vs.get("audit") or {}).get("leak"),
    }
