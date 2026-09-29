"""Ghi kết quả ra đĩa. Không diễn giải, không xếp hạng, không bình luận.

Đầu ra là các định dạng chuẩn để công cụ khác đọc lại được:

  predictions.json  định dạng COCO results — nạp được bằng COCO.loadRes(),
                    nên chấm lại bằng bất kỳ công cụ nào cũng ra cùng số
  cocoeval.txt      nguyên văn 12 dòng của COCOeval.summarize(), cộng một khối
                    TÁCH RIÊNG cho AP ở ngưỡng IoU cao (0.90, 0.95)
  coco_metrics.json những con số đó ở dạng máy đọc được, kèm AP_by_iou (cả mười
                    ngưỡng 0.50:0.05:0.95)
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


def nguong_cao(stats: dict | None) -> str:
    """Khối AP ở ngưỡng IoU cao — thứ `summarize()` của pycocotools không in.

    Đặt thành khối RIÊNG phía dưới, không chèn vào giữa: mười hai dòng kia phải
    giữ đúng từng chữ pycocotools in ra, vì đó mới là thứ đối chiếu được với
    văn liệu. Ghi rõ nguồn ngay trên đầu khối để ba tháng sau không ai tưởng
    pycocotools in ra hai dòng này.

    Thiếu khối này thì `evaluate.py` in bảng ra màn hình mà KHÔNG có AP90:
    người chạy không thấy con số mình vừa chấm, dù nó đã nằm trong
    coco_metrics.json.
    """
    if not stats:
        return ""
    co = [(k, stats[k]) for k in ("AP90", "AP95")
          if isinstance(stats.get(k), (int, float))]
    if not co:
        return ""
    dong = [f" AP @[ IoU={int(k[2:]) / 100:0.2f} | area=   all | maxDets=100 ] = {v:0.3f}"
            for k, v in co]
    return ("\n  ngưỡng cao (cofseg đọc từ eval['precision']; summarize() không in)\n"
            + "\n".join(dong))


def write_coco_results(coco: dict, run_dir: str | Path) -> tuple[Path, Path]:
    """cocoeval.txt (nguyên văn) và coco_metrics.json (máy đọc)."""
    run_dir = Path(run_dir)
    txt = run_dir / "cocoeval.txt"
    body = ("Mask AP\n" + (coco.get("mask_text") or coco.get("error") or "")
            + nguong_cao(coco.get("mask")))
    if coco.get("box_text"):
        body += ("\n\nBox AP (hộp bao quanh mặt nạ dự đoán)\n" + coco["box_text"]
                 + nguong_cao(coco.get("box")))
    if coco.get("boundary_text"):
        body += ("\n\nBoundary AP (dilation_ratio=" + str(coco.get("dilation_ratio")) + ")\n"
                 + coco["boundary_text"] + nguong_cao(coco.get("boundary")))
    txt.write_text(body + "\n", encoding="utf-8")
    js = run_dir / "coco_metrics.json"
    js.write_text(
        json.dumps(
            {"mask": coco.get("mask"), "box": coco.get("box"),
             "boundary": coco.get("boundary"),
             "dilation_ratio": coco.get("dilation_ratio"), "error": coco.get("error")},
            indent=2,
        ),
        encoding="utf-8",
    )
    return txt, js
