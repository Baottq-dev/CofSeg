"""Trọng số không vào git; configs/weights.yaml là bản kê: tên file trong
weights/, URL gốc, sha256.

Hai việc dùng nó:
- scripts/download_weights.py tải theo nhóm và kiểm sha, để máy lab dựng
  được y hệt máy nhà mà không phải chép tay.
- Trainer detectron2/mmdet hỏi `local_for_url(url)`: checkpoint COCO đã nằm
  trong weights/ thì dùng luôn, không tải lần hai về cache của framework.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import urllib.request
from pathlib import Path

import yaml

MANIFEST = Path("configs/weights.yaml")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: str | Path = MANIFEST) -> dict:
    """Đọc và kiểm bản kê: mọi file có url http(s) + sha256 hợp lệ, mọi nhóm
    chỉ trỏ tới file có khai. Hỏng thì báo ngay, không đợi tới lúc tải."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    files = doc.get("files") or {}
    if not files:
        raise ValueError(f"{path}: không có khối files")
    for name, spec in files.items():
        if "/" in name or "\\" in name:
            raise ValueError(f"{path}: tên {name!r} phải là tên file, không có thư mục")
        url = str((spec or {}).get("url", ""))
        if not url.startswith(("http://", "https://", "file://")):
            raise ValueError(f"{path}: {name} thiếu url http(s)")
        if not _SHA.match(str((spec or {}).get("sha256", ""))):
            raise ValueError(f"{path}: {name} thiếu sha256 (64 ký tự hex)")
    groups = doc.get("groups") or {}
    for g, members in groups.items():
        unknown = [m for m in (members or []) if m not in files]
        if unknown:
            raise ValueError(f"{path}: nhóm {g} trỏ tới file chưa khai: {unknown}")
    doc.setdefault("dir", "weights")
    return doc


def select(doc: dict, group: str | None = None, only: list[str] | None = None) -> list[str]:
    """Tên file cần tải: --only thắng --group; group 'all' là mọi file."""
    files = doc["files"]
    if only:
        bad = [n for n in only if n not in files]
        if bad:
            raise ValueError(f"không có trong bản kê: {bad}")
        return list(only)
    group = group or "benchmark"
    if group == "all":
        return list(files)
    groups = doc.get("groups") or {}
    if group not in groups:
        raise ValueError(f"nhóm {group!r} không có; có: {sorted(groups)} và all")
    return list(groups[group])


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check(path: str | Path, spec: dict) -> str:
    """'ok' | 'missing' | 'mismatch' cho một file so với bản kê."""
    p = Path(path)
    if not p.exists():
        return "missing"
    return "ok" if sha256(p) == spec["sha256"] else "mismatch"


def local_for_url(url: str, doc: dict | None = None, weights_dir: str | Path | None = None) -> Path | None:
    """Đường dẫn trong weights/ của checkpoint có URL này, nếu file đã tồn tại.

    Không kiểm sha ở đây: file 900 MB băm mất vài giây mỗi lần dựng model;
    sha kiểm lúc tải (download_weights.py) là đủ.
    """
    if doc is None:
        if not MANIFEST.exists():
            return None
        doc = load_manifest()
    root = Path(weights_dir or doc.get("dir", "weights"))
    for name, spec in doc["files"].items():
        if spec.get("url") == url and (root / name).exists():
            return root / name
    return None


def download(name: str, spec: dict, weights_dir: str | Path, force: bool = False,
             progress=None) -> dict:
    """Tải một file về weights_dir/name qua file tạm .part, kiểm sha rồi mới đổi tên.

    Trả {name, status, path}; status: ok (đã có, đúng sha) | downloaded |
    mismatch (file có sẵn sai sha, giữ nguyên trừ force) | bad_download (tải
    về sai sha, đã xoá). `progress(done_bytes, total_bytes)` nếu muốn in.
    """
    root = Path(weights_dir)
    root.mkdir(parents=True, exist_ok=True)
    dst = root / name
    if dst.exists() and not force:
        st = check(dst, spec)
        if st == "ok":
            return {"name": name, "status": "ok", "path": str(dst)}
        return {"name": name, "status": "mismatch", "path": str(dst)}
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(spec["url"]) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if progress:
                progress(done, total)
    if sha256(part) != spec["sha256"]:
        part.unlink(missing_ok=True)
        return {"name": name, "status": "bad_download", "path": str(dst)}
    shutil.move(str(part), str(dst))
    return {"name": name, "status": "downloaded", "path": str(dst)}
