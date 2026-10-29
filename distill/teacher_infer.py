"""Pha 0 của chắt lọc: ba thầy suy luận trên ảnh TRAIN của một fold, bằng đúng bộ chấm của benchmark.

Chạy từ GỐC REPO, trong môi trường có đủ ultralytics, mmdetection và detectron2 (molab):

    python distill/teacher_infer.py --data data/export/field/f1 --weights-from runs/train_2 --runs runs/distill_teachers
    python distill/teacher_infer.py --data data/export/field/f1 --weights-from runs/train_2 --runs runs/distill_teachers --dry-run

Mỗi thầy là một lệnh `benchmark/<model>/evaluate.py --split train` với config chấm của benchmark (conf 0,05,
max_det 100, mặt nạ ở độ phân giải gốc), nên mặt nạ thầy dùng để dạy đúng là mặt nạ đã chấm trong benchmark.
Kết quả: <runs>/eval/<thời điểm>_<tên run>_field-<fold>_train_i1024/predictions.json. Bước sau
(teacher_masks.py) ghép các dự đoán đó với tán nhãn.

Các lệnh chạy lần lượt, không song song. Thầy đã có lượt chấm train xong (có predictions.json và
metrics.json) thì bỏ qua, trừ khi gõ --force; nhờ vậy chạy lại sau khi bị ngắt chỉ làm phần còn thiếu.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill import ROOT  # noqa: E402
from distill.teachers import find_eval, find_weights, fold_info, parse_teachers  # noqa: E402


def commands(data: str, weights_from: str, runs: str, teachers=None, force: bool = False) -> list[tuple]:
    """(thầy, lệnh hoặc None nếu đã có, lượt chấm đã có) cho từng thầy."""
    fold = fold_info(data)["fold"]
    out = []
    for t in parse_teachers(teachers):
        done = find_eval(Path(runs) / "eval", t, fold)
        if done is not None and not force:
            out.append((t, None, done))
            continue
        w = find_weights(weights_from, t, fold)
        bench = Path("benchmark") / t.bench
        cmd = [sys.executable, (bench / "evaluate.py").as_posix(),
               "--config", (bench / "configs" / "eval" / t.config).as_posix(),
               "--weights", w.as_posix(), "--data", str(data).replace("\\", "/").rstrip("/"),
               "--split", "train", "--runs", runs]
        out.append((t, cmd, None))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="thư mục fold, vd data/export/field/f1")
    ap.add_argument("--weights-from", required=True, help="thư mục runs chứa lượt train seed 0 của thầy")
    ap.add_argument("--runs", required=True, help="thư mục kết quả; lượt chấm ghi vào <runs>/eval")
    ap.add_argument("--teachers", default=None, help="tập con của yolo26s,solov2,m2f (mặc định cả ba)")
    ap.add_argument("--force", action="store_true", help="chấm lại cả thầy đã có lượt chấm train")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in lệnh")
    a = ap.parse_args(argv)

    plan = commands(a.data, a.weights_from, a.runs, a.teachers, a.force)
    for t, cmd, done in plan:
        if cmd is None:
            print(f"[bỏ qua] {t.key}: đã có {done}", flush=True)
            continue
        print(f"[{t.key}] {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[LỖI] {t.key}: lệnh chấm trả về {r.returncode}; dừng.", flush=True)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
