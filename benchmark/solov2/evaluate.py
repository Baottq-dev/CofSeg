# Bản của benchmark/solov2: chạy hoàn toàn trong thư mục này (cofseg/ là bản sao
# lõi của riêng thư mục). Chạy từ GỐC REPO để data/ và weights/ dùng chung:
#
#     python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml
"""Chấm một model trên một split, qua một đường chung cho mọi model.

    python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml --weights <best.pt>
    python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml --weights <best.pt> --data data/export/field/f1 --split test
    python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml --size t --limit 5
    python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/_coco.yaml --file preds/cascade.json

Config có ba khối: `model:` (name trong sổ đăng ký + tham số khởi tạo, lồng
nhau được), `data:` (root, split, min_area), `eval:` (ngưỡng ghép, vành biên).
Ghi đè bằng cờ thật: `--weights` (trọng số của lần train), `--data` (thư mục
fold), `--split`, `--limit`, và `--k v` cho tham số khởi tạo model (kiểm tên
theo chữ ký hàm khởi tạo, gõ sai thì báo lỗi). `--set a.b.c=v` vẫn còn làm lối
thoát cho khoá lồng nhau chưa có cờ riêng, nhưng lệnh thường ngày không cần.

Điểm chấm: Mask AP và Boundary AP bằng pycocotools (bản tham chiếu mà
detectron2, mmdet, torchvision và ultralytics đều gọi bên dưới), cộng chỉ số
biên từng vùng (Boundary IoU, ASSD, HD95, NSD, sai số biên có dấu, sai số diện
tích) mà không bộ chấm nào cung cấp. Kết quả nằm trong runs/eval/<...>/:

    predictions.json   COCO results — nạp lại bằng COCO.loadRes() hay coco_predictions
    cocoeval.txt       nguyên văn COCOeval.summarize() cho Mask AP và Boundary AP
    metrics.json       mọi con số ở dạng máy đọc, kèm describe của model
    per_region.csv     một hàng mỗi vùng thật (+ mỗi dự đoán thừa)
    per_image.csv      số vùng, số dự đoán, sai số đếm, mili giây mỗi ảnh

Đường riêng cho YOLO — `model.val()` của ultralytics, đúng thứ `yolo val` chạy:

    python benchmark/solov2/evaluate.py --native --weights <best.pt> --split test --imgsz 1024

Giữ để đối chiếu với mọi báo cáo YOLO khác; số của nó và số COCOeval chênh nhau
vài phần nghìn do cách nội suy đường PR, không phải do model.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # thư mục của model này

from cofseg import artifacts  # noqa: E402
from cofseg import cli  # noqa: E402
from cofseg import config as cfgmod  # noqa: E402
from cofseg import console  # noqa: E402
from cofseg import runlog  # noqa: E402
from cofseg.datasets import CocoDataset  # noqa: E402
from cofseg.evaluation import coco_eval, evaluate_split, summarize, validate  # noqa: E402
from cofseg.evaluation.report import write_coco_results, write_csv, write_predictions  # noqa: E402
from cofseg.models import build_model, model_param_names  # noqa: E402
from cofseg.registry import available  # noqa: E402


console.setup()

EVAL_DEFAULTS = {"iou_thr": 0.5, "band_ratio": 0.02, "nsd_tau": 2.0, "dilation_ratio": 0.02}


# ------------------------------------------------------------------ đường chung
def run_config(a, extra: list[str]) -> int:
    # --data đứng trước --set: --set cụ thể hơn nên thắng nếu dùng cả hai.
    overrides = list(a.overrides)
    if a.data:
        overrides.insert(0, "data.root=" + str(a.data).replace("\\", "/").rstrip("/"))
    cfg = cfgmod.load(a.config, overrides)
    if "model" not in cfg or "name" not in cfg["model"]:
        raise SystemExit(f"config cần khối model: {{name: ...}}. Model có: {available('model')}")
    # Trọng số là thứ đổi mỗi lần chấm, nên nó là cờ chứ không phải --set.
    if a.weights:
        cfg["model"]["weights"] = str(a.weights).replace("\\", "/")
    name = cfg["model"]["name"]
    hp = cli.parse_overrides(extra, model_param_names(name), what=f"model {name!r}")
    # Bốn cờ dùng chung với đường --native: kiểm theo chữ ký của chính lớp
    # model rồi mới nhận, chứ không im lặng bỏ qua. `--imgsz` với detectron2
    # là ví dụ: nó không có tham số đó, độ phân giải nằm trong d2_config.yaml
    # cạnh trọng số.
    nhan = model_param_names(name)
    for k, v in (("imgsz", a.imgsz), ("batch", a.batch),
                 ("max_det", a.max_det), ("device", a.device)):
        if v is None:
            continue
        if nhan is not None and k not in nhan:
            raise SystemExit(f"--{k.replace('_', '-')} không phải đối số của model "
                             f"{name!r}; nó chỉ dùng cho --native.")
        hp[k] = v
    cfg["model"].update(hp)

    data = {"root": "data/export/block/f4", "split": "test", "min_area": 50.0,
            **(cfg.get("data") or {})}
    if a.split:
        data["split"] = a.split
    ev = {**EVAL_DEFAULTS, **(cfg.get("eval") or {})}
    # Bốn ngưỡng chấm là cờ thật: đổi chúng là đổi ý nghĩa con số, nên chúng
    # phải hiện ra trong --help chứ không nấp trong --set.
    for k in ("iou_thr", "band_ratio", "nsd_tau", "dilation_ratio"):
        v = getattr(a, k, None)
        if v is not None:
            ev[k] = float(v)
    limit = a.limit if a.limit is not None else ev.get("limit")

    run_name = a.name or cfg.get("name") or Path(a.config).stem
    # config.yaml của lần chấm phải ghi đúng tên đã dùng (--name thắng file),
    # vì summarize_folds.py đọc tên "<model>_<fold>" từ đó.
    cfg["name"] = run_name
    tag = data["split"] + (f"_i{cfg['model']['imgsz']}" if "imgsz" in cfg["model"] else "")
    ds = artifacts.dataset_tag(cfg)
    run_dir = artifacts.create_run_dir(a.runs, "eval", run_name,
                                       "_".join(p for p in (ds, tag) if p))
    artifacts.write_env(run_dir, cfg)
    artifacts.snapshot_config(run_dir, cfg)
    print(f"Lần chấm: {run_dir}")

    with runlog.capture(run_dir / "run.log"):
        try:
            if hp:
                print("Ghi đè từ dòng lệnh:", json.dumps(hp, ensure_ascii=False))
            ds = CocoDataset(data["root"], data["split"], min_area=float(data["min_area"]))
            print("Bộ dữ liệu:", json.dumps(ds.summary(), ensure_ascii=False))
            model = build_model(cfg["model"])
            print("Model:", json.dumps(model.describe, ensure_ascii=False, default=str))
            if model.needs_prompt:
                print("Model cần gợi ý: dùng box THẬT của nhãn (dòng oracle, cận trên).")
            print()

            rows, info = evaluate_split(
                model, ds, iou_thr=float(ev["iou_thr"]), band_ratio=float(ev["band_ratio"]),
                nsd_tau=float(ev["nsd_tau"]), limit=limit,
            )
            write_predictions(info["detections"], run_dir / "predictions.json")
            write_csv(rows, run_dir / "per_region.csv")
            write_csv(info["images"], run_dir / "per_image.csv")

            print("\nChấm theo chuẩn COCO (Mask AP + Boundary AP)...", flush=True)
            coco = coco_eval.evaluate(
                str(ds.ann_file), info["detections"], info["image_ids"],
                dilation_ratio=float(ev["dilation_ratio"]),
            )
            txt, _ = write_coco_results(coco, run_dir)
            summary = summarize(rows, info["images"])
            metrics = {
                "model": model.describe,
                "data": {**data, "images_scored": len(info["image_ids"]), "limit": limit},
                "eval": ev,
                "coco": {k: coco.get(k) for k in ("mask", "boundary", "dilation_ratio", "error")},
                "summary": summary,
                "total_seconds": info["total_seconds"],
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )

            print()
            print(txt.read_text(encoding="utf-8"))
            keys = ("precision", "recall", "mean_boundary_iou", "median_boundary_iou",
                    "median_signed_median", "median_area_error_pct", "ms_per_image")
            print("Từng vùng (cặp đã ghép ở IoU >= %.2f):" % ev["iou_thr"])
            for k in keys:
                if k in summary:
                    print(f"  {k:<26} {summary[k]}")
            print("\nKết quả:", run_dir)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            return 1
    return 0


# ------------------------------------------------------------------ đường native
NATIVE_LOCKED = {"data", "project", "name", "exist_ok"}


def run_native(a, extra: list[str]) -> int:
    from ultralytics.cfg import get_cfg

    if not a.weights:
        raise SystemExit("--native cần --weights <best.pt>")
    weights = Path(a.weights)
    if not weights.exists():
        raise SystemExit(f"Không thấy trọng số: {weights}")
    data = Path(a.native_data)
    if not data.exists():
        raise SystemExit(f"Không thấy {data}. Cắt fold trước: scripts/make_fold.py --export <bản xuất> --all")
    kw = cli.parse_overrides(extra, set(vars(get_cfg())), NATIVE_LOCKED, what="val()")
    split = a.split or "test"
    # Mặc định của riêng đường native; cờ để None để đường config phân biệt
    # được "không gõ" với "gõ đúng bằng mặc định".
    imgsz, batch, max_det = a.imgsz or 1024, a.batch or 4, a.max_det or 100

    src = weights.parent.parent.parent.name
    run_dir = artifacts.create_run_dir(a.runs, "eval", a.name or src, f"native_{split}_i{imgsz}")
    artifacts.write_env(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps({**vars(a), **kw}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Lần chấm: {run_dir}")

    with runlog.capture(run_dir / "run.log"):
        try:
            if kw:
                print("Ghi đè từ dòng lệnh:", json.dumps(kw, ensure_ascii=False))
            out = validate(
                weights, data, split=split, imgsz=imgsz, batch=batch,
                max_det=max_det, device=a.device, run_dir=run_dir, **kw,
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        # Không cho viết tắt. Mặc định của argparse nhận mọi tiền tố không
        # nhập nhằng, nên `--conf 0.05` bị nuốt thành `--config 0.05` và lỗi
        # hiện ra ở tận chỗ mở file. Siêu tham số của model phải rơi xuống
        # parse_known_args để đi đúng đường kiểm tên.
        allow_abbrev=False,
    )
    ap.add_argument("--config", help="config đánh giá (đường chung)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="lối thoát cho khoá lồng nhau chưa có cờ riêng, "
                         "vd --set eval.limit=5")
    ap.add_argument("--split", default=None, help="train | val | test (mặc định theo config)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chấm N ảnh đầu")
    ap.add_argument("--native", action="store_true",
                    help="chấm bằng model.val() của ultralytics (chỉ YOLO)")
    ap.add_argument("--weights", default=None, metavar="ĐƯỜNG_DẪN",
                    help="trọng số của lần train, vd <run>/weights/best.pth")
    ap.add_argument("--data", default=None, metavar="THƯ_MỤC_FOLD",
                    help="thư mục fold, vd data/export/block/f1 — viết thẳng ra, "
                         "nhìn lệnh là biết đang chấm fold nào")
    ap.add_argument("--native-data", dest="native_data",
                    default="data/export/block/f4/data.yaml", help="[native] data.yaml")
    # Bốn cờ dùng chung hai đường chấm. Mặc định để None: trên đường config
    # chúng đi vào khối `model:` và chỉ khi người chạy THỰC SỰ gõ ra, nếu
    # không thì giá trị trong config mới là thứ quyết định.
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    # max_det=300 mặc định của ultralytics làm tràn VRAM ở khâu val: nó phóng
    # TẤT CẢ mặt nạ về 2560x1440 trước khi chấm (~4.4 GB một phép nội suy).
    # Ảnh dày nhất của bộ này có 48 vùng, nên 100 đã dư gấp đôi.
    ap.add_argument("--max-det", dest="max_det", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou-thr", dest="iou_thr", type=float, default=None,
                    help="ngưỡng IoU mặt nạ để ghép dự đoán với vùng thật (bảng từng vùng)")
    ap.add_argument("--band-ratio", dest="band_ratio", type=float, default=None,
                    help="vành Boundary IoU từng vùng, theo cạnh hình vuông cùng diện tích")
    ap.add_argument("--nsd-tau", dest="nsd_tau", type=float, default=None,
                    help="dung sai NSD, px")
    ap.add_argument("--dilation-ratio", dest="dilation_ratio", type=float, default=None,
                    help="vành Boundary AP, theo đường chéo ảnh")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    a, extra = ap.parse_known_args()

    if a.native:
        return run_native(a, extra)
    if not a.config:
        raise SystemExit("Cần --config <benchmark/solov2/configs/eval/...yaml>, hoặc --native --weights <best.pt>.")
    return run_config(a, extra)


if __name__ == "__main__":
    raise SystemExit(main())
