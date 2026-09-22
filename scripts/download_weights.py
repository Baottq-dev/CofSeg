"""Tải trọng số theo configs/weights.yaml về weights/, kiểm sha256.

    python scripts/download_weights.py                     # nhóm benchmark
    python scripts/download_weights.py --group all
    python scripts/download_weights.py --only solov2_r50_fpn_3x_coco.pth,yolo11s-seg.pt
    python scripts/download_weights.py --check             # chỉ kiểm, không tải

Thoát khác 0 khi có file sai sha hoặc tải hỏng, để setup.sh dừng đúng chỗ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canopyseg import weights as W  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(W.MANIFEST))
    ap.add_argument("--group", default="benchmark", help="benchmark | annotator | sam_all | backup | all")
    ap.add_argument("--only", default="", help="tên file, phẩy ngăn; thắng --group")
    ap.add_argument("--dir", default=None, help="mặc định theo `dir` trong bản kê (weights/)")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm sha các file đã có")
    ap.add_argument("--force", action="store_true", help="tải đè file sai sha")
    a = ap.parse_args(argv)

    doc = W.load_manifest(a.manifest)
    root = Path(a.dir or doc["dir"])
    names = W.select(doc, a.group, [s for s in a.only.split(",") if s.strip()] or None)
    bad = 0
    for name in names:
        spec = doc["files"][name]
        if a.check:
            st = W.check(root / name, spec)
        else:
            def bar(done, total, _n=name):
                mb = done / 2**20
                pct = f"{100 * done / total:5.1f}%" if total else "     "
                print(f"\r  {_n}: {mb:7.1f} MB {pct}", end="", flush=True)

            st = W.download(name, spec, root, force=a.force, progress=bar)["status"]
            if st == "downloaded":
                print()
        mark = {"ok": "✓", "downloaded": "↓", "missing": "-", "mismatch": "✗", "bad_download": "✗"}[st]
        print(f"{mark} {name:42s} {st:12s} {spec.get('size_mb', '?')} MB  {spec.get('for', '')}")
        bad += st in ("mismatch", "bad_download")
    if bad:
        print(f"{bad} file sai sha256: xoá file rồi chạy lại, hoặc --force để tải đè.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
