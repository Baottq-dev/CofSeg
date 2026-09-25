"""Gộp các lần chấm leave-one-field-out thành một bảng model x ruộng.

Mỗi thư mục runs/eval/<...>/ là một (model, fold) chấm trên split test của
fold — tức đúng một ruộng model chưa thấy. Gộp 6 fold lại thì mỗi ruộng có
một ô cho mỗi model, và cột "Δ% mAP" so với Mask R-CNN (mốc 0) trên CÙNG
ruộng, vì so chéo ruộng là so hai bài toán khác nhau.

Tên model và fold lấy từ `name` trong config.yaml của lần chấm
(người chạy đặt bằng --name, xem benchmark/README.md); fold -> ruộng test
lấy từ folds.yaml.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

from .. import artifacts

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
#: Tên lần chạy có thể còn đuôi fold theo quy ước cũ ("maskrcnn_f4"); đuôi đó
#: là tuỳ chọn vì fold nay lấy từ đường dẫn dữ liệu.
_RUN_NAME = re.compile(r"^(?P<model>.+?)[_-](?P<fold>f\d+)$")
#: Nhãn bộ dữ liệu do artifacts.dataset_tag sinh: "block-f4", hoặc chỉ "f4".
_DATA_TAG = re.compile(r"^(?:(?P<dataset>.+)-)?(?P<fold>f\d+)$")
# Thư mục lần chấm: <YYYY-MM-DD_HHMMSS>_<tên>_<split>[_i<imgsz>][~n]
_RUN_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_(?P<name>.+?)_(?:train|val|test)(?:_i\d+)?(?:~\d+)?$")


def parse_run_name(name: str) -> tuple[str, str | None]:
    """'maskrcnn_f4' -> ('maskrcnn', 'f4'); 'maskrcnn' -> ('maskrcnn', None).

    Đuôi fold là tuỳ chọn. Quy ước mới chỉ đặt tên model, vì fold đọc được từ
    đường dẫn dữ liệu đã chấm — chỗ đó không gõ nhầm được.
    """
    name = name.strip()
    m = _RUN_NAME.match(name)
    return (m["model"], m["fold"]) if m else (name, None)


def parse_data_tag(tag: str) -> tuple[str, str | None]:
    """'block-f4' -> ('block', 'f4'); 'f4' -> ('', 'f4'); khác -> ('', None)."""
    m = _DATA_TAG.match((tag or "").strip())
    return ((m["dataset"] or "", m["fold"]) if m else ("", None))


def _dig(d: dict, path: tuple) -> float | None:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return None if cur is None else float(cur)


def collect(eval_root: str | Path | list, folds_yaml: str | Path,
            dataset: str | None = None, warn=print) -> list[dict]:
    """Một hàng cho mỗi lần chấm nhận diện được: model, bộ fold, fold, ruộng, số đo.

    `eval_root` là một thư mục hoặc nhiều — mỗi thành viên ghi kết quả vào
    runs/eval của thư mục mình, nên bảng chung phải quét được cả bốn.

    Khoá của một hàng là (model, BỘ FOLD, fold). Bộ fold phải nằm trong khoá:
    hai cách chia val cho hai bộ fold cùng đặt tên f1..f6, nên chấm
    `maskrcnn_f4` trên cả hai bộ sẽ ra hai lần chấm trông y hệt nhau. Trước
    đây khoá chỉ có (model, fold) và lần sau lặng lẽ đè lần trước — chạy xong
    48 lượt mới phát hiện mất một nửa bảng.

    Cùng một khoá mà chấm nhiều lần thì vẫn lấy lần mới nhất, nhưng BÁO RA
    (`warn`) chứ không im lặng. `dataset` lọc theo một bộ fold.
    """
    doc = yaml.safe_load(Path(folds_yaml).read_text(encoding="utf-8"))
    test_field = {name: spec["test"][0] for name, spec in doc["folds"].items()}
    roots = [eval_root] if isinstance(eval_root, (str, Path)) else list(eval_root)
    rows: dict[tuple[str, str, str], dict] = {}
    for run in sorted((r for root in roots for r in Path(root).glob("*/")), key=lambda p: p.name):
        cfg_p, met_p = run / "config.yaml", run / "metrics.json"
        if not (cfg_p.exists() and met_p.exists()):
            continue
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
        # Bộ fold VÀ fold đều suy từ đường dẫn dữ liệu đã chấm, không từ tên
        # người đặt: gõ --name maskrcnn_f4 mà trỏ vào f5 thì tên nói dối, còn
        # đường dẫn thì không. Tên chỉ còn dùng để biết model nào.
        ds, fold_path = parse_data_tag(artifacts.dataset_tag(cfg))
        model, fold_named = parse_run_name(str(cfg.get("name", "")))
        if fold_path is None and fold_named is None:
            # Lần chấm cũ: config.yaml giữ tên của FILE config chứ không phải
            # tên lần chạy, và chưa ghi data.root. Tên thư mục còn giữ cả hai.
            m = _RUN_DIR.match(run.name)
            if m:
                from_dir = parse_run_name(m["name"])
                if from_dir[1]:
                    model, fold_named = from_dir
        if fold_path and fold_named and fold_path != fold_named and warn:
            warn(f"  {run.name}: tên ghi {fold_named} nhưng dữ liệu là {fold_path}; "
                 f"lấy theo dữ liệu")
        fold = fold_path or fold_named
        if not model or fold not in test_field:
            continue
        if dataset is not None and ds != dataset:
            continue
        met = json.loads(met_p.read_text(encoding="utf-8"))
        if (met.get("data") or {}).get("split", "test") != "test":
            continue
        n_img = _dig(met, ("data", "images_scored"))
        row = {"model": model, "dataset": ds, "fold": fold, "field": test_field[fold],
               "run": run.name, "images": None if n_img is None else int(n_img)}
        for col, path, nd, mul in METRICS:
            v = _dig(met, path)
            row[col] = None if v is None else round(v * mul, nd)
        key = (model, ds, fold)
        if key in rows and warn:
            warn(f"  hai lần chấm cùng ({model}, {ds or 'không rõ bộ'}, {fold}): "
                 f"giữ {run.name}, bỏ {rows[key]['run']}")
        rows[key] = row            # lần sau đè lần trước
    return sorted(rows.values(), key=lambda r: (r["dataset"], r["field"], r["model"]))


def add_reference_delta(rows: list[dict], reference: str = REFERENCE) -> list[dict]:
    """Δ% mAP so với model mốc trên cùng ruộng; None khi mốc thiếu ở ruộng đó."""
    ref = {(r.get("dataset", ""), r["field"]): r["mAP"]
           for r in rows if r["model"] == reference and r["mAP"] is not None}
    for r in rows:
        base = ref.get((r.get("dataset", ""), r["field"]))
        r["dmAP_pct"] = (None if base in (None, 0) or r["mAP"] is None
                         else round(100.0 * (r["mAP"] - base) / base, 1))
    return rows


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def group_means(rows: list[dict], fields: list[str], label: str) -> list[dict]:
    """Trung bình theo model trên một nhóm ruộng (nội suy / ngoại suy), chỉ khi
    model có đủ mọi ruộng của nhóm — thiếu một ruộng là số không so được.

    Gom trong phạm vi MỘT bộ fold: trung bình trộn số của hai cách chia val là
    trộn hai thí nghiệm khác nhau."""
    out = []
    for model, ds in sorted({(r["model"], r.get("dataset", "")) for r in rows}):
        mine = [r for r in rows
                if r["model"] == model and r.get("dataset", "") == ds
                and r["field"] in fields]
        if {r["field"] for r in mine} != set(fields):
            continue
        row = {"model": model, "dataset": ds, "fold": "", "field": label,
               "run": f"{len(fields)} ruộng",
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
    cols = ["dataset", "field", "model", "images", "mAP", "dmAP_pct", "AP50", "AP75", "BAP",
            "BIoU", "area_err_pct", "recall", "precision", "ms_img", "run"]
    head = ["bộ fold", "ruộng", "model", "ảnh", "mAP", "Δ% vs " + REFERENCE, "AP50", "AP75",
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
    cols = ["dataset", "field", "fold", "model", "images", "mAP", "dmAP_pct", "AP50", "AP75",
            "BAP", "BIoU", "area_err_pct", "recall", "precision", "ms_img", "run"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def build_report(eval_root: str | Path | list, folds_yaml: str | Path,
                 reference: str = REFERENCE, dataset: str | None = None, warn=print):
    """(hàng theo ruộng, hàng trung bình nhóm, markdown)."""
    doc = yaml.safe_load(Path(folds_yaml).read_text(encoding="utf-8"))
    rows = add_reference_delta(collect(eval_root, folds_yaml, dataset, warn), reference)
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
