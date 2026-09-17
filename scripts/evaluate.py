"""Chấm model đã huấn luyện trên một split, bằng chính bộ đánh giá của ultralytics.

    python scripts/evaluate.py --weights runs/train/<...>/weights/best.pt
    python scripts/evaluate.py --weights <...>/best.pt --split test --imgsz 1024
    python scripts/evaluate.py --weights <...>/best.pt --split val

Đây là `model.val()` — đúng thứ `yolo val` chạy — nên bảng số liệu in ra trùng
với mọi báo cáo YOLO khác và không thể trôi lệch khi ultralytics đổi cách tính.
Script chỉ thêm: neo kết quả vào bố cục runs/ của dự án, chép nguyên văn màn
hình vào run.log, và ghi lại môi trường để tái lập.

Mặc định `save_json` bật, nên predictions.json (định dạng COCO results) luôn có
để chấm lại bằng công cụ khác.

Mọi tham số của val đều truyền thẳng được:

    python scripts/evaluate.py --weights <...>/best.pt --conf 0.001 --rect
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import artifacts  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg import runlog  # noqa: E402
from canopyseg.evaluation import validate  # noqa: E402

console.setup()

# Script tự đặt bốn khoá này để kết quả rơi đúng thư mục run.
LOCKED = {"data", "project", "name", "exist_ok"}


def parse_extra(tokens: list[str]) -> dict:
    """Đối số lạ -> tham số cho val(), kiểm tên theo danh sách sống của ultralytics."""
    from ultralytics.cfg import get_cfg

    valid = set(vars(get_cfg()))
    out: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(f"Không hiểu đối số {tok!r}.")
        body = tok[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            key, raw = body, tokens[i + 1]
            i += 1
        else:
            key, raw = body, "true"
        key = key.replace("-", "_")
        if key in LOCKED:
            raise SystemExit(f"--{key} bị khoá: script tự đặt để kết quả vào đúng run dir.")
        if key not in valid:
            near = difflib.get_close_matches(key, sorted(valid), n=3, cutoff=0.6)
            hint = f" Ý bạn là: {', '.join('--' + n for n in near)}?" if near else ""
            raise SystemExit(f"val() không có tham số {key!r}.{hint}")
        out[key] = yaml.safe_load(raw)
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", required=True, help="đường dẫn tới best.pt / last.pt")
    ap.add_argument("--data", default="data/export/dataset_v1/data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    # max_det=300 (mặc định của ultralytics) làm tràn VRAM ở khâu val: nó phóng
    # TẤT CẢ mặt nạ về đúng độ phân giải gốc 2560x1440 trước khi chấm, tức
    # 300 x 2560 x 1440 x 4 byte ~ 4.4 GB cho một phép nội suy. Con số này
    # không phụ thuộc batch, nên giảm batch không cứu được.
    # Ảnh dày nhất của bộ này có 48 vùng, nên 100 đã dư gấp đôi.
    ap.add_argument("--max-det", type=int, default=100,
                    help="số vật thể tối đa mỗi ảnh (mặc định ultralytics 300 gây OOM)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    a, extra = ap.parse_known_args()

    weights = Path(a.weights)
    if not weights.exists():
        raise SystemExit(f"Không thấy trọng số: {weights}")
    data = Path(a.data)
    if not data.exists():
        raise SystemExit(
            f"Không thấy {data}. Chạy scripts/prepare_yolo_dataset.py trước."
        )

    kw = parse_extra(extra)
    # Tên run nói rõ chấm trọng số nào, trên split nào, ở độ phân giải nào.
    src = weights.parent.parent.parent.name
    run_dir = artifacts.create_run_dir(
        a.runs, "eval", a.name or src, f"{a.split}_i{a.imgsz}"
    )
    artifacts.write_env(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps({**vars(a), **kw}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Lần chấm: {run_dir}")

    with runlog.capture(run_dir / "run.log"):
        try:
            if kw:
                print("Ghi đè từ dòng lệnh:", json.dumps(kw, ensure_ascii=False))
            out = validate(
                weights, data, split=a.split, imgsz=a.imgsz, batch=a.batch,
                max_det=a.max_det, device=a.device, run_dir=run_dir, **kw,
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("\nKết quả:", run_dir)
            if out.get("predictions_json"):
                print("Dự đoán (COCO results):", out["predictions_json"])
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
