"""Chấm một model đã huấn luyện trên một split, bằng chỉ số đường biên.

    python scripts/evaluate.py --weights runs/train/<...>/ultralytics/weights/best.pt
    python scripts/evaluate.py --weights <...>/best.pt --split test --imgsz 1024
    python scripts/evaluate.py --weights <...>/best.pt --limit 10   # chạy thử

Vì sao tồn tại thay vì dùng thẳng `yolo val`: đo trên chính bộ này, mặt nạ co
vào 5 px vẫn cho AP75 = 0.953 trong khi Boundary IoU chỉ còn 0.233; và hai lần
train ở 640 với 1024 chênh nhau đúng 0.006 mAP dù trần đường biên khác hẳn.
mAP không nhìn thấy thứ dự án này quan tâm.

Script không biết YOLO tồn tại: nó tra sổ đăng ký model theo tên trong --model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import artifacts  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg import models  # noqa: F401,E402 - nạp để đăng ký model
from canopyseg import runlog  # noqa: E402
from canopyseg.datasets import CocoDataset  # noqa: E402
from canopyseg.evaluation import coco_eval  # noqa: E402
from canopyseg.evaluation import evaluate_split, format_report, write_csv  # noqa: E402
from canopyseg.evaluation.report import summarize  # noqa: E402
from canopyseg.registry import available, resolve  # noqa: E402

console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", required=True, help="đường dẫn tới best.pt / last.pt")
    ap.add_argument("--model", default="yolo_seg", help=f"loại model {available('model')}")
    ap.add_argument("--data", default="data/export/dataset_v1")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="ngưỡng NMS của model")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="ngưỡng IoU để coi một dự đoán là khớp với vùng thật")
    ap.add_argument("--band-ratio", type=float, default=0.02,
                    help="bề rộng vành biên, theo tỉ lệ cạnh tán")
    ap.add_argument("--nsd-tau", type=float, default=2.0,
                    help="dung sai NSD tính bằng px")
    ap.add_argument("--dilation-ratio", type=float, default=0.02,
                    help="vành Boundary AP, theo tỉ lệ đường chéo ảnh "
                         "(0.02 là giá trị trong bài báo)")
    ap.add_argument("--min-area", type=float, default=50.0)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chấm N ảnh đầu")
    ap.add_argument("--device", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    weights = Path(a.weights)
    if not weights.exists():
        raise SystemExit(f"Không thấy trọng số: {weights}")

    name = a.name or f"{weights.parent.parent.parent.name}_{a.split}"
    run_dir = artifacts.create_run_dir(a.runs, "eval", name, f"i{a.imgsz}c{a.conf}")
    artifacts.write_env(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(vars(a), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Lần chấm: {run_dir}")

    with runlog.capture(run_dir / "run.log"):
        try:
            ds = CocoDataset(a.data, a.split, min_area=a.min_area)
            print("Bộ dữ liệu:", json.dumps(ds.summary(), ensure_ascii=False))

            model = resolve("model", a.model)(
                str(weights), imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device
            )
            print("Model:", json.dumps(model.describe, ensure_ascii=False, default=str))
            print()

            rows, info = evaluate_split(
                model, ds,
                iou_thr=a.match_iou, band_ratio=a.band_ratio,
                nsd_tau=a.nsd_tau, limit=a.limit,
            )

            print("\nChấm theo chuẩn COCO (Mask AP + Boundary AP)...", flush=True)
            coco = coco_eval.evaluate(
                str(ds.ann_file), info["detections"], info["image_ids"],
                dilation_ratio=a.dilation_ratio,
            )

            csv_path = write_csv(rows, run_dir / "per_region.csv")
            write_csv(info["images"], run_dir / "per_image.csv")
            summary = summarize(rows, info["images"])
            summary["coco"] = {k: coco.get(k) for k in ("mask", "boundary", "dilation_ratio")}
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Giữ nguyên văn 12 dòng của COCOeval.summarize() để trích dẫn được.
            if coco.get("mask_text"):
                (run_dir / "cocoeval.txt").write_text(
                    "== Mask AP ==\n" + coco["mask_text"]
                    + "\n\n== Boundary AP ==\n" + coco["boundary_text"] + "\n",
                    encoding="utf-8",
                )

            report = format_report(
                rows, info["images"], coco=coco,
                title=f"{weights.name} · split {a.split} · imgsz {a.imgsz} · conf {a.conf}",
            )
            (run_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
            print()
            print(report)
            print()
            print(f"Thời gian suy luận: {info['total_seconds']}s")
            print("Từng vùng:", csv_path)
            print("Kết quả:", run_dir)
        except SystemExit:
            raise
        except BaseException:
            import traceback

            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
