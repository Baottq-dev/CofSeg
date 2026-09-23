"""Chấm mọi file dự đoán mang về từ máy thuê qua đúng vòng chấm của repo.

    python scripts/score_remote.py --preds preds --export data/export
    python scripts/score_remote.py --preds preds --export data/export --folds f4,f2 --limit 20

Mỗi preds/<model>_<fold>.json (COCO results của split test, do
benchmark/run.py gom) được chấm bằng scripts/evaluate.py với
configs/eval/coco_predictions.yaml trên data/export/<fold> split test, tên
lần chấm "<model>_<fold>" để summarize_folds.py nhận ra. Chạy từng file trong
tiến trình con nên một file hỏng không kéo cả loạt; lỗi in ra cuối.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg.evaluation.folds import parse_run_name  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="preds", help="thư mục chứa <model>_<fold>.json")
    ap.add_argument("--export", default="data/export", help="thư mục chứa các fold f1..f6")
    ap.add_argument("--folds", default="", help="chỉ chấm các fold này, vd f4,f2")
    ap.add_argument("--models", default="", help="chỉ chấm các model này, vd maskrcnn,solov2")
    ap.add_argument("--config", default="configs/eval/coco_predictions.yaml")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chấm N ảnh đầu (thử)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in lệnh")
    a = ap.parse_args()

    folds = {f for f in a.folds.split(",") if f}
    models = {m for m in a.models.split(",") if m}
    files = sorted(Path(a.preds).glob("*.json"))
    jobs = []
    for f in files:
        parsed = parse_run_name(f.stem)
        if not parsed:
            print(f"bỏ qua {f.name}: tên không phải <model>_<fold>")
            continue
        model, fold = parsed
        if (folds and fold not in folds) or (models and model not in models):
            continue
        root = Path(a.export) / fold
        if not (root / "annotations" / "instances_test.json").exists():
            print(f"bỏ qua {f.name}: không thấy {root} (chạy make_fold.py --fold {fold})")
            continue
        jobs.append((model, fold, f, root))
    if not jobs:
        print("Không có file nào để chấm.")
        return 1

    failed = []
    for model, fold, f, root in jobs:
        cmd = [sys.executable, "scripts/evaluate.py", "--config", a.config,
               "--file", str(f), "--set", f"data.root={root}", "--split", "test",
               "--name", f"{model}_{fold}", "--runs", a.runs]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print("\n== " + " ".join(cmd), flush=True)
        if a.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc:
            failed.append((model, fold, rc))
    if failed:
        print("\nLỗi:", ", ".join(f"{m}_{f} (mã {rc})" for m, f, rc in failed))
        return 1
    print(f"\nĐã chấm {len(jobs)} file. Tổng hợp: python scripts/summarize_folds.py --eval {a.runs}/eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
