"""Chạy một hay nhiều model của bảng benchmark trên một hay nhiều fold.

    python benchmark/run.py f4                        # cả bốn model
    python benchmark/run.py f4 --only solov2
    python benchmark/run.py f4 --smoke                # vài iteration, kiểm đường chạy
    python benchmark/run.py f4 f2 --only yolo11       # nhiều fold
    python benchmark/run.py f4 --epochs 30 --batch 2  # tham số đi thẳng vào train
    python benchmark/run.py --list

Script này KHÔNG biết model nào chạy thế nào. Với mỗi model nó gọi
`benchmark/<model>/train.py` rồi `benchmark/<model>/evaluate.py` — hai file
thuộc về người phụ trách model đó, dùng bản sao lõi trong
`benchmark/<model>/cofseg/`. Sửa cách một model chạy thì sửa trong thư mục
của model đó, file này không đổi.

Mỗi model xong để lại:
    preds/<model>_<fold>.json                         COCO results của split TEST
    benchmark/<model>/runs/                           lần chạy đầy đủ
    benchmark/<model>/results/<model>_<fold>_*.json   file nhỏ để commit

Chạy từ gốc repo: data/ và weights/ dùng chung cho cả nhóm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "preds"

#: Model -> cách chạy. `score` quyết định bước chấm:
#:   "predictions" — trainer đã ghi predictions.json (detectron2, mmdet), chấm
#:                   lại file đó qua configs/eval/_coco.yaml
#:   "weights"     — chấm bằng trọng số tốt nhất (ultralytics)
MODELS: dict[str, dict] = {
    "maskrcnn": dict(
        dir="maskrcnn", train="configs/train/maskrcnn_r50_d2.yaml",
        eval="configs/eval/_coco.yaml", score="predictions",
        data="root", smoke=["--set", "data.limit=16", "--epochs", "1"],
        about="Mask R-CNN R50-FPN (mốc số 0)"),
    "solov2": dict(
        dir="solov2", train="configs/train/solov2_r50_mm.yaml",
        eval="configs/eval/_coco.yaml", score="predictions",
        data="root", smoke=["--set", "data.limit=16", "--epochs", "1"],
        about="SOLOv2 R50-FPN (box-free)"),
    "yolo11": dict(
        dir="yolo11", train="configs/train/yolo11s.yaml",
        eval="configs/eval/yolo11s.yaml", score="weights",
        data="yaml", smoke=["--epochs", "1", "--fraction", "0.05"],
        about="YOLOv11-Seg (một giai đoạn)"),
    "mask2former": dict(
        dir="mask2former", train="configs/train/mask2former_r50_d2.yaml",
        eval="configs/eval/_coco.yaml", score="predictions",
        data="root", smoke=["--set", "data.limit=16", "--epochs", "1"],
        about="Mask2Former R50 (query)"),
}
#: Thứ tự chạy: mốc 0 trước để có số so sánh sớm, Mask2Former cuối vì lâu nhất.
ORDER = ["maskrcnn", "solov2", "yolo11", "mask2former"]


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def latest(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Không thấy lần chạy nào khớp {pattern}")
    return hits[-1]          # tên bắt đầu bằng thời điểm nên sắp chuỗi là đủ


def fold_root(fold: str) -> Path:
    root = ROOT / "data" / "export" / fold
    if not (root / "annotations" / "instances_test.json").exists():
        raise SystemExit(f"Không thấy {root.relative_to(ROOT)}: cắt fold trước\n"
                         f"    python scripts/make_fold.py --export data/export/<bản xuất> --all")
    return root


def run_one(name: str, fold: str, smoke: bool, extra: list[str]) -> dict:
    spec = MODELS[name]
    folder = ROOT / "benchmark" / spec["dir"]
    root = fold_root(fold)
    runs = folder / "runs"
    results = folder / "results"
    for d in (runs, results, PREDS):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"{name}-{fold}"
    rel = folder.relative_to(ROOT).as_posix()

    data_arg = (["--set", f"data.yaml={(root / 'data.yaml').as_posix()}"] if spec["data"] == "yaml"
                else ["--set", f"data.root={root.as_posix()}"])
    t0 = time.time()
    run([sys.executable, f"{rel}/train.py", "--config", f"{rel}/{spec['train']}",
         *data_arg, "--runs", f"{rel}/runs", "--name", tag,
         *extra, *(spec["smoke"] if smoke else [])])
    train_run = latest(f"{rel}/runs/train/*_{tag}_*")

    pred = PREDS / f"{name}_{fold}.json"
    if spec["score"] == "predictions":
        src = train_run / "predictions.json"
        if not src.exists():
            raise SystemExit(f"{train_run.relative_to(ROOT)} không có predictions.json")
        shutil.copy2(src, pred)
        eval_args = ["--file", pred.relative_to(ROOT).as_posix(),
                     "--set", f"data.root={root.as_posix()}"]
    else:
        weights = train_run / "ultralytics" / "weights" / "best.pt"
        if not weights.exists():
            weights = weights.with_name("last.pt")
        eval_args = ["--set", f"model.weights={weights.relative_to(ROOT).as_posix()}",
                     "--set", f"data.root={root.as_posix()}"]

    run([sys.executable, f"{rel}/evaluate.py", "--config", f"{rel}/{spec['eval']}",
         *eval_args, "--split", "test", "--runs", f"{rel}/runs", "--name", f"{name}_{fold}"])
    eval_run = latest(f"{rel}/runs/eval/*_{name}_{fold}_*")
    if spec["score"] == "weights":
        shutil.copy2(eval_run / "predictions.json", pred)

    # File nhỏ để commit; runs/ không vào git.
    for src_name, dst_name in (("metrics.json", f"{name}_{fold}_metrics.json"),
                               ("per_region.csv", f"{name}_{fold}_per_region.csv")):
        src = eval_run / src_name
        if src.exists():
            shutil.copy2(src, results / dst_name)
    src = train_run / "results.csv"
    if src.exists():
        shutil.copy2(src, results / f"{name}_{fold}_train.csv")

    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    mask = (metrics.get("coco") or {}).get("mask") or {}
    return {"model": name, "fold": fold, "minutes": round((time.time() - t0) / 60, 1),
            "mAP": mask.get("AP"), "AP50": mask.get("AP50"),
            "BAP": ((metrics.get("coco") or {}).get("boundary") or {}).get("AP"),
            "predictions": pred.relative_to(ROOT).as_posix()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folds", nargs="*", help="tên fold, vd f4 f2")
    ap.add_argument("--only", default="", help="chỉ chạy các model này, phẩy ngăn")
    ap.add_argument("--smoke", action="store_true", help="vài iteration mỗi model")
    ap.add_argument("--list", action="store_true", help="liệt kê model rồi thoát")
    a, extra = ap.parse_known_args(argv)

    if a.list:
        for name in ORDER:
            print(f"  {name:12s} {MODELS[name]['about']:36s} benchmark/{MODELS[name]['dir']}/")
        return 0
    if not a.folds:
        ap.error("cần ít nhất một fold, vd: python benchmark/run.py f4")

    names = [n.strip() for n in a.only.split(",") if n.strip()] or ORDER
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"không có model: {unknown}; có: {ORDER}")
    names = [n for n in ORDER if n in names]

    done, failed = [], []
    for fold in a.folds:
        for name in names:
            print(f"\n== {time.strftime('%F %T')}  {name}  {fold}"
                  f"{'  (khói)' if a.smoke else ''}", flush=True)
            try:
                done.append(run_one(name, fold, a.smoke, extra))
            except subprocess.CalledProcessError as e:
                failed.append((name, fold, f"lệnh thoát {e.returncode}"))
                print(f"!! {name} {fold} lỗi, bỏ qua và chạy tiếp", flush=True)
            except (SystemExit, FileNotFoundError) as e:
                failed.append((name, fold, str(e)))
                print(f"!! {name} {fold}: {e}", flush=True)

    print(f"\n== xong {len(done)}/{len(done) + len(failed)}")
    for r in done:
        print(f"  {r['model']:12s} {r['fold']:4s} mAP {r['mAP']}  BAP {r['BAP']}  "
              f"{r['minutes']} phút  -> {r['predictions']}")
    for name, fold, why in failed:
        print(f"  LỖI {name:10s} {fold:4s} {why}")
    if done:
        print("\nBảng model x ruộng: python scripts/summarize_folds.py "
              "--eval benchmark/*/runs/eval")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
