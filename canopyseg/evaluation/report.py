"""Ghi kết quả ra đĩa. Không diễn giải, không xếp hạng, không bình luận.

Đầu ra là các định dạng chuẩn để công cụ khác đọc lại được:

  predictions.json  định dạng COCO results — nạp được bằng COCO.loadRes(),
                    nên chấm lại bằng bất kỳ công cụ nào cũng ra cùng số
  cocoeval.txt      nguyên văn 12 dòng của COCOeval.summarize()
  metrics.json      12 con số đó ở dạng máy đọc được
  per_region.csv    một hàng mỗi vùng, để tự phân tích

Con số duy nhất được in ra màn hình là đầu ra gốc của pycocotools.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def write_predictions(detections: list[dict], path: str | Path) -> Path:
    """Định dạng COCO results, đúng thứ COCO.loadRes() nhận."""
    path = Path(path)
    path.write_text(json.dumps(detections), encoding="utf-8")
    return path


def write_coco_results(coco: dict, run_dir: str | Path) -> tuple[Path, Path]:
    """cocoeval.txt (nguyên văn) và metrics.json (máy đọc)."""
    run_dir = Path(run_dir)
    txt = run_dir / "cocoeval.txt"
    txt.write_text(
        "Mask AP\n" + (coco.get("mask_text") or "")
        + "\n\nBoundary AP (dilation_ratio="
        + str(coco.get("dilation_ratio")) + ")\n"
        + (coco.get("boundary_text") or "") + "\n",
        encoding="utf-8",
    )
    js = run_dir / "metrics.json"
    js.write_text(
        json.dumps(
            {"mask": coco.get("mask"), "boundary": coco.get("boundary"),
             "dilation_ratio": coco.get("dilation_ratio")},
            indent=2,
        ),
        encoding="utf-8",
    )
    return txt, js
