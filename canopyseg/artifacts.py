"""Thư mục kết quả cho mỗi lần chạy, kèm dấu vết để tái lập.

Một lần chạy = runs/<ten>/<thoi-diem>/. Trong đó luôn có config đã dùng và
env.json (phiên bản thư viện, GPU, commit git, trạng thái sạch/bẩn của repo).
Thiếu những thứ này thì ba tháng sau không ai nói được con số sinh ra từ đâu.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import config as cfgmod


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_dir(root: str | Path, name: str) -> Path:
    d = Path(root) / name / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
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
