"""Gộp các lần chấm leave-one-field-out thành một bảng model x ruộng.

Mỗi thư mục runs/eval/<...>/ là một (model, fold) chấm trên split test của
fold — tức đúng một ruộng model chưa thấy. Gộp 6 fold lại thì mỗi ruộng có
một ô cho mỗi model, và cột "Δ% mAP" so với Mask R-CNN (mốc 0) trên CÙNG
ruộng, vì so chéo ruộng là so hai bài toán khác nhau.

Tên model và fold lấy từ `name` trong config.yaml của lần chấm
(score_remote.py đặt "<model>_<fold>"); fold -> ruộng test lấy từ folds.yaml.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

METRICS = [
    # (khoá cột, đường trong metrics.json, số chữ số, nhân)
    ("mAP", ("coco", "mask", "AP"), 3, 1),
    ("AP50", ("coco", "mask", "AP50"), 3, 1),
    ("AP75", ("coco", "mask", "AP75"), 3, 1),
    ("BAP", ("coco", "boundary", "AP"), 3, 1),
    ("BIoU", ("summary", "mean_boundary_iou"), 3, 1),
    ("area_err_pct", ("summary", "mean_area_error_pct"), 2, 1),
    ("recall", ("summary", "recall"), 3, 1),
    ("precision", ("summary", "precision"), 3, 1),
    ("ms_img", ("summary", "ms_per_image"), 0, 1),
]
REFERENCE = "maskrcnn"
_RUN_NAME = re.compile(r"^(?P<model>.+)_(?P<fold>f\d+)$")
# Thư mục lần chấm: <YYYY-MM-DD_HHMMSS>_<tên>_<split>[_i<imgsz>][~n]
_RUN_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_(?P<name>.+?)_(?:train|val|test)(?:_i\d+)?(?:~\d+)?$")


def parse_run_name(name: str) -> tuple[str, str] | None:
    m = _RUN_NAME.match(name.strip())
    return (m["model"], m["fold"]) if m else None


def _dig(d: dict, path: tuple) -> float | None:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return None if cur is None else float(cur)


def collect(eval_root: str | Path, folds_yaml: str | Path) -> list[dict]:
    """Một hàng cho mỗi lần chấm nhận diện được: model, fold, ruộng, các số đo.

    Cùng (model, fold) chấm nhiều lần thì lấy lần mới nhất (tên thư mục bắt
    đầu bằng thời điểm nên sắp xếp chuỗi là đủ).
    """
    doc = yaml.safe_load(Path(folds_yaml).read_text(encoding="utf-8"))
    test_field = {name: spec["test"][0] for name, spec in doc["folds"].items()}
    rows: dict[tuple[str, str], dict] = {}
    for run in sorted(Path(eval_root).glob("*/")):
        cfg_p, met_p = run / "config.yaml", run / "metrics.json"
        if not (cfg_p.exists() and met_p.exists()):
            continue
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
        parsed = parse_run_name(str(cfg.get("name", "")))
        if not parsed:
            # Lần chấm cũ: config.yaml còn giữ tên của file config, lấy từ tên thư mục.
            m = _RUN_DIR.match(run.name)
            parsed = parse_run_name(m["name"]) if m else None
        if not parsed or parsed[1] not in test_field:
            continue
        model, fold = parsed
        met = json.loads(met_p.read_text(encoding="utf-8"))
        if (met.get("data") or {}).get("split", "test") != "test":
            continue
        n_img = _dig(met, ("data", "images_scored"))
        row = {"model": model, "fold": fold, "field": test_field[fold], "run": run.name,
               "images": None if n_img is None else int(n_img)}
        for col, path, nd, mul in METRICS:
            v = _dig(met, path)
            row[col] = None if v is None else round(v * mul, nd)
        rows[(model, fold)] = row            # lần sau đè lần trước
    return sorted(rows.values(), key=lambda r: (r["field"], r["model"]))


def add_reference_delta(rows: list[dict], reference: str = REFERENCE) -> list[dict]:
    """Δ% mAP so với model mốc trên cùng ruộng; None khi mốc thiếu ở ruộng đó."""
    ref = {r["field"]: r["mAP"] for r in rows if r["model"] == reference and r["mAP"] is not None}
    for r in rows:
        base = ref.get(r["field"])
        r["dmAP_pct"] = (None if base in (None, 0) or r["mAP"] is None
                         else round(100.0 * (r["mAP"] - base) / base, 1))
    return rows


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def group_means(rows: list[dict], fields: list[str], label: str) -> list[dict]:
    """Trung bình theo model trên một nhóm ruộng (nội suy / ngoại suy), chỉ khi
    model có đủ mọi ruộng của nhóm — thiếu một ruộng là số không so được."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model and r["field"] in fields]
        if {r["field"] for r in mine} != set(fields):
            continue
        row = {"model": model, "fold": "", "field": label, "run": f"{len(fields)} ruộng",
               "images": sum(r["images"] or 0 for r in mine)}
        for col, *_ in METRICS:
            row[col] = _mean([r[col] for r in mine])
        row["dmAP_pct"] = _mean([r["dmAP_pct"] for r in mine])
        out.append(row)
    return out


def fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and nd == 0:
        return f"{v:.0f}"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def to_markdown(rows: list[dict], title: str = "") -> str:
    cols = ["field", "model", "images", "mAP", "dmAP_pct", "AP50", "AP75", "BAP", "BIoU",
            "area_err_pct", "recall", "precision", "ms_img", "run"]
    head = ["ruộng", "model", "ảnh", "mAP", "Δ% vs " + REFERENCE, "AP50", "AP75",
            "Boundary AP", "Boundary IoU", "sai số DT %", "recall", "precision", "ms/ảnh", "lần chấm"]
    nd = {c: n for c, _, n, _ in METRICS}
    nd.update({"dmAP_pct": 1})
    lines = [f"# {title}", ""] if title else []
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c), nd.get(c, 3)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    cols = ["field", "fold", "model", "images", "mAP", "dmAP_pct", "AP50", "AP75", "BAP",
            "BIoU", "area_err_pct", "recall", "precision", "ms_img", "run"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def build_report(eval_root: str | Path, folds_yaml: str | Path, reference: str = REFERENCE):
    """(hàng theo ruộng, hàng trung bình nhóm, markdown)."""
    doc = yaml.safe_load(Path(folds_yaml).read_text(encoding="utf-8"))
    rows = add_reference_delta(collect(eval_root, folds_yaml), reference)
    groups = []
    for key, label in (("interpolation", "TB nội suy"), ("extrapolation", "TB ngoại suy")):
        if doc.get(key):
            groups += group_means(rows, list(doc[key]), label)
    md = to_markdown(rows, "Kết quả leave-one-field-out — mỗi ruộng là split test của fold tương ứng")
    if groups:
        md += "\n" + to_markdown(groups, "Trung bình theo nhóm ruộng (chỉ model có đủ mọi ruộng của nhóm)")
    md += (f"\nΔ% mAP tính trên cùng ruộng so với model mốc `{reference}`. Boundary IoU và sai số "
           "diện tích tính trên cặp đã ghép (IoU ≥ 0.5); hàng trung bình là trung bình các ruộng.\n")
    return rows, groups, md
