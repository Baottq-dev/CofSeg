"""Đối chiếu phần CHẤM ĐIỂM giữa các bản sao trong benchmark/*/cofseg/.

Mỗi thành viên có một bản sao lõi riêng nên sửa được tuỳ ý — đó là chủ đích.
Cái giá: hai người sửa khác nhau thì mAP / Boundary AP của họ đo bằng hai
thước, và bảng "Δ% mAP so với Mask R-CNN" không còn nghĩa.

Script này băm từng file của phần chấm điểm rồi so với bản gốc canopyseg/.
Giống hệt -> số so được với nhau. Lệch -> in ra lệch ở đâu để người đọc bảng
biết mà ghi chú, hoặc để đồng bộ lại.

    python benchmark/check_copies.py            # bảng tóm tắt
    python benchmark/check_copies.py --diff     # kèm diff từng dòng
    python benchmark/check_copies.py --sync     # chép bản gốc đè lên chỗ lệch

Thoát khác 0 khi có lệch, để dùng được trong CI hoặc trước lúc dựng bảng.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "canopyseg"
MEMBERS = ROOT / "benchmark"

#: Những file quyết định CON SỐ. Đây là phần phải giống nhau để so sánh được;
#: trainer và model wrapper của từng người thì cố ý khác nhau, không kiểm.
SCORING = [
    "metrics/mask.py",
    "metrics/boundary.py",
    "evaluation/coco_eval.py",
    "evaluation/matching.py",
    "evaluation/runner.py",
    "evaluation/report.py",
    "datasets/coco.py",
    "datasets/instances.py",
    "datasets/region.py",
]


def normalise(text: str) -> str:
    """Bỏ khác biệt duy nhất do nhân bản: tên gói canopyseg -> cofseg."""
    return text.replace("canopyseg", "cofseg")


def digest(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:12]


def members() -> list[Path]:
    return sorted(d for d in MEMBERS.iterdir() if d.is_dir() and (d / "cofseg").is_dir())


def compare(show_diff: bool = False, sync: bool = False) -> int:
    rows, bad = [], 0
    for member in members():
        for rel in SCORING:
            ref_p, copy_p = REFERENCE / rel, member / "cofseg" / rel
            if not copy_p.exists():
                rows.append((member.name, rel, "THIẾU"))
                bad += 1
                continue
            ref, cur = ref_p.read_text(encoding="utf-8"), copy_p.read_text(encoding="utf-8")
            if digest(ref) == digest(cur):
                continue
            rows.append((member.name, rel, f"LỆCH {digest(ref)} != {digest(cur)}"))
            bad += 1
            if show_diff:
                print(f"\n--- canopyseg/{rel}\n+++ {member.name}/cofseg/{rel}")
                print("".join(difflib.unified_diff(
                    normalise(ref).splitlines(True), normalise(cur).splitlines(True), n=1))[:4000])
            if sync:
                copy_p.write_text(normalise(ref), encoding="utf-8", newline="\n")
                print(f"đồng bộ lại: {member.name}/cofseg/{rel}")

    print(f"{len(members())} bản sao x {len(SCORING)} file chấm điểm")
    if not rows:
        print("Tất cả giống bản gốc -> số của bốn model so được với nhau.")
        return 0
    for name, rel, what in rows:
        print(f"  {name:22s} {rel:28s} {what}")
    if sync:
        print("Đã đồng bộ. Chạy lại để xác nhận.")
        return 0
    print("\nCÓ LỆCH: bốn model không còn đo bằng cùng một thước. Hoặc ghi rõ")
    print("khác biệt khi trình bày bảng, hoặc chạy --sync rồi chấm lại.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true", help="in diff từng dòng chỗ lệch")
    ap.add_argument("--sync", action="store_true", help="chép bản gốc đè lên chỗ lệch")
    a = ap.parse_args(argv)
    return compare(a.diff, a.sync)


if __name__ == "__main__":
    sys.exit(main())
