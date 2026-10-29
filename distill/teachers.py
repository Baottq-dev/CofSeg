"""Ba thầy của phần chắt lọc: mô hình gốc YOLO26s, SOLOv2 và Mask2Former, seed 0, cùng fold với trò.

Thầy là các lượt train seed 0 của benchmark (thư mục runs của nó, vd runs/train_2), không train thêm. Trò của
fold nào chỉ học từ thầy của fold đó, tức các model chưa từng thấy ruộng thử của fold.

File này chỉ giữ bảng thầy và phép tìm file, không import torch hay ultralytics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Teacher:
    key: str        # tên ngắn trong lệnh và trong meta.json
    bench: str      # thư mục model trong benchmark/
    config: str     # config chấm trong benchmark/<bench>/configs/eval/
    run_name: str   # tên run của lượt train và lượt chấm
    weights: str    # file trọng số trong <run>/weights/


TEACHERS: dict[str, Teacher] = {
    "yolo26s": Teacher("yolo26s", "yolo", "yolo26s.yaml", "yolo26s-seg", "best.pt"),
    "solov2": Teacher("solov2", "solov2", "solov2_r50_mm.yaml", "solov2-r50-mm", "best.pth"),
    "m2f": Teacher("m2f", "mask2former", "mask2former_r50_d2.yaml", "mask2former-r50-d2", "best.pth"),
}


def parse_teachers(spec: str | list[str] | None) -> list[Teacher]:
    """`yolo26s,solov2,m2f` (mặc định cả ba) -> danh sách thầy, giữ thứ tự của bảng."""
    if spec is None:
        return list(TEACHERS.values())
    keys = [k.strip() for k in (spec.split(",") if isinstance(spec, str) else spec) if k.strip()]
    bad = [k for k in keys if k not in TEACHERS]
    if bad or not keys:
        raise SystemExit(f"Thầy không có: {', '.join(bad) or '(rỗng)'}. Có: {', '.join(TEACHERS)}")
    return [t for k, t in TEACHERS.items() if k in keys]


def fold_info(data_root: str | Path) -> dict:
    """fold.json của thư mục fold: tên fold và mã băm của bản xuất."""
    p = Path(data_root) / "fold.json"
    if not p.exists():
        raise SystemExit(f"Không thấy {p}: --data phải là thư mục fold (vd data/export/field/f1)")
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"fold": d["fold"], "source_sha1": d.get("source_sha1")}


def find_weights(runs_root: str | Path, teacher: Teacher, fold: str) -> Path:
    """Trọng số của thầy cho fold: đúng một thư mục <runs_root>/*_<run_name>_field-<fold>_*/weights/<file>."""
    root = Path(runs_root)
    hits = sorted(root.glob(f"*_{teacher.run_name}_field-{fold}_*/weights/{teacher.weights}"))
    if len(hits) != 1:
        found = "\n  ".join(str(h) for h in hits) or "(không có)"
        raise SystemExit(
            f"Cần đúng một trọng số {teacher.weights} của {teacher.run_name} fold {fold} trong {root}, thấy:\n  {found}")
    return hits[0]


def find_eval(evals_root: str | Path, teacher: Teacher, fold: str, split: str = "train") -> Path | None:
    """Lượt chấm mới nhất (theo tên) của thầy trên split, đã ghi xong predictions.json và metrics.json."""
    root = Path(evals_root)
    done = [d for d in sorted(root.glob(f"*_{teacher.run_name}_field-{fold}_{split}_*"))
            if (d / "predictions.json").exists() and (d / "metrics.json").exists()]
    return done[-1] if done else None
