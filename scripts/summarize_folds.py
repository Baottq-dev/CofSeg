"""Gộp các lần chấm leave-one-field-out thành bảng model x ruộng.

    python scripts/summarize_folds.py --eval benchmark/*/runs/eval
    python scripts/summarize_folds.py --eval runs/eval --out docs/reports/bang_ket_qua_2026-09-30.md

Đọc <thư mục eval>/<...>/{config.yaml,metrics.json} của các lần chấm tên
"<model>_<fold>" (score_remote.py đặt tên này), ánh xạ fold -> ruộng test
theo configs/dataset/folds.yaml, và in bảng: mAP, Δ% mAP so với mốc
(maskrcnn) trên cùng ruộng, AP50/75, Boundary AP, Boundary IoU, sai số diện
tích, recall/precision, ms/ảnh; thêm trung bình nhóm nội suy / ngoại suy.
Không có --out thì in ra màn hình; có thì ghi .md và .csv cạnh nhau.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.evaluation import folds as rep  # noqa: E402

console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="+", default=["runs/eval"],
                    help="một hay nhiều thư mục runs/eval; mỗi thành viên có một")
    ap.add_argument("--folds", default="configs/dataset/folds.yaml")
    ap.add_argument("--reference", default=rep.REFERENCE, help="model mốc cho cột Δ%")
    ap.add_argument("--out", default=None, help="file .md; .csv ghi cùng tên")
    a = ap.parse_args()

    rows, groups, md = rep.build_report(a.eval, a.folds, a.reference)
    if not rows:
        print(f"Không thấy lần chấm nào tên <model>_<fold> trong {' '.join(map(str, a.eval))}")
        return 1
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        rep.write_csv(rows + groups, out.with_suffix(".csv"))
        print(f"Đã ghi {out} và {out.with_suffix('.csv')} ({len(rows)} hàng, {len(groups)} hàng trung bình)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
