"""Cắt một bản xuất CHƯA CHIA thành fold train/val/test theo ruộng.

Đầu vào là thư mục app/ xuất với split_by=none:
    <export>/annotations/instances.json
    <export>/images/<field__flight__file.jpg>
    <export>/labels/<stem>.txt          (chỉ khi xuất kèm YOLO)

Đầu ra là thư mục đúng bố cục CocoDataset và ultralytics mong đợi:
    <out>/annotations/instances_<split>.json
    <out>/images/<split>/...   <out>/labels/<split>/...   <out>/data.yaml   <out>/fold.json

Hai quyết định đáng ghi:
- Ảnh được HARDLINK chứ không chép: 6 fold × 1 GB ảnh mà chép là 6 GB cho
  cùng một byte. Hardlink chỉ là tên thứ hai của cùng file, không tốn đĩa, và
  ai đọc cũng thấy là file thường. Khác ổ đĩa thì tự chuyển sang chép.
- id ảnh và id annotation GIỮ NGUYÊN từ bản xuất gốc. Máy thuê chấm test rồi
  trả về file dự đoán theo image_id; nếu đánh số lại thì file đó không khớp
  với bản chấm ở nhà mà không có lỗi nào báo.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

from .yolo import write_data_yaml

SPLITS = ("train", "val", "test")


def field_of(file_name: str) -> str:
    """Ruộng của một ảnh đã làm phẳng: 'field_2__10__1__DJI_x.jpg' -> 'field_2'."""
    return file_name.split("__")[0]


# ------------------------------------------------------------------ cấu hình fold
def load_folds(path: str | Path) -> dict:
    """Đọc folds.yaml và kiểm ngay: một fold sai thì hỏng cả bảng, tốt hơn là
    hỏng trước khi cắt bất kỳ thứ gì."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = list(doc.get("fields") or [])
    folds = doc.get("folds") or {}
    if not fields or not folds:
        raise ValueError(f"{path}: cần cả 'fields' lẫn 'folds'")
    for name, spec in folds.items():
        test, val = list(spec.get("test") or []), list(spec.get("val") or [])
        unknown = [f for f in test + val if f not in fields]
        if unknown:
            raise ValueError(f"fold {name}: ruộng không có trong 'fields': {unknown}")
        if not test or not val:
            raise ValueError(f"fold {name}: test và val đều phải có ít nhất một ruộng")
        if set(test) & set(val):
            raise ValueError(f"fold {name}: một ruộng vừa test vừa val: {set(test) & set(val)}")
        if not [f for f in fields if f not in test and f not in val]:
            raise ValueError(f"fold {name}: không còn ruộng nào cho train")
    return doc


def fold_fields(doc: dict, name: str) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'test': [...]} cho một fold."""
    if name not in doc["folds"]:
        raise KeyError(f"không có fold {name!r}; có: {sorted(doc['folds'])}")
    spec = doc["folds"][name]
    test, val = list(spec["test"]), list(spec["val"])
    train = [f for f in doc["fields"] if f not in test and f not in val]
    return {"train": train, "val": val, "test": test}


# ------------------------------------------------------------------- cắt fold
def _place(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            # Khác ổ đĩa, hệ file không hỗ trợ, hoặc hết số link: chép là đúng
            # và chỉ tốn đĩa, không đổi kết quả.
            pass
    shutil.copy2(src, dst)
    return "copy"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_fold(
    export: str | Path,
    name: str,
    fields: dict[str, list[str]],
    out: str | Path,
    copy: bool = False,
) -> dict:
    """Cắt <export> thành <out> theo `fields` ({split: [ruộng]}). Trả về tóm tắt.

    <out> phải chưa tồn tại — thư mục fold là thứ sinh ra được, xoá đi làm lại
    rẻ hơn là đoán xem bên trong còn gì của lần trước.
    """
    export, out = Path(export), Path(out)
    ann_file = export / "annotations" / "instances.json"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Không thấy {ann_file}. Cần bản xuất CHƯA chia (split_by=none); "
            "bản đã chia có instances_train.json thì không cắt lại được."
        )
    if out.exists():
        raise FileExistsError(f"{out} đã tồn tại; xoá đi rồi chạy lại")

    raw = json.loads(ann_file.read_text(encoding="utf-8"))
    field_to_split = {f: sp for sp, fs in fields.items() for f in fs}
    known = set(field_to_split)

    by_split: dict[str, list[dict]] = {sp: [] for sp in SPLITS}
    skipped: list[str] = []
    for im in raw["images"]:
        fld = field_of(im["file_name"])
        if fld not in known:
            skipped.append(im["file_name"])
            continue
        by_split[field_to_split[fld]].append(im)
    empty = [sp for sp in SPLITS if not by_split[sp]]
    if empty:
        raise ValueError(
            f"fold {name}: tập {empty} không có ảnh nào — ruộng "
            f"{[fields[sp] for sp in empty]} chưa có trong bản xuất?"
        )

    anns_by_image: dict[int, list[dict]] = {}
    for a in raw["annotations"]:
        anns_by_image.setdefault(int(a["image_id"]), []).append(a)

    has_labels = (export / "labels").is_dir()
    summary: dict = {
        "fold": name,
        "fields": fields,
        "source": str(export).replace("\\", "/"),
        "source_sha1": _sha1(ann_file),
        "images_skipped": skipped,
        "splits": {},
        "labels": has_labels,
    }
    how = {"link": 0, "copy": 0}
    (out / "annotations").mkdir(parents=True)
    for sp in SPLITS:
        images = by_split[sp]
        anns = [a for im in images for a in anns_by_image.get(int(im["id"]), [])]
        doc = {
            **{k: v for k, v in raw.items() if k not in ("images", "annotations")},
            "images": images,
            "annotations": anns,
        }
        doc.setdefault("info", {})
        doc["info"] = {**doc["info"], "fold": name, "split": sp, "fields": fields[sp]}
        (out / "annotations" / f"instances_{sp}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
        n_bg = 0
        for im in images:
            src = export / "images" / im["file_name"]
            if not src.exists():
                raise FileNotFoundError(f"{im['file_name']}: có trong COCO nhưng thiếu ảnh")
            how[_place(src, out / "images" / sp / im["file_name"], copy)] += 1
            if not anns_by_image.get(int(im["id"])):
                n_bg += 1
            if has_labels:
                stem = Path(im["file_name"]).stem
                lab = export / "labels" / (stem + ".txt")
                dst = out / "labels" / sp / (stem + ".txt")
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Ảnh nền có file nhãn rỗng trong bản xuất; nếu bản xuất thiếu
                # thì tạo rỗng, vì thiếu hẳn file là "chưa gán" với ultralytics.
                if lab.exists():
                    shutil.copy2(lab, dst)
                else:
                    dst.write_text("", encoding="utf-8")
        summary["splits"][sp] = {
            "fields": fields[sp],
            "images": len(images),
            "annotations": len(anns),
            "empty_images": n_bg,
        }
    summary["images_linked"] = how["link"]
    summary["images_copied"] = how["copy"]

    if has_labels:
        names = {int(c["id"]) - 1: c["name"] for c in raw.get("categories", [])}
        write_data_yaml(out, {sp: f"images/{sp}" for sp in SPLITS}, names or {0: "canopy"})
    (out / "fold.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
