"""Cắt một bản xuất CHƯA CHIA thành fold train/val/test theo ruộng.

Đầu vào là thư mục app/ xuất với split_by=none:
    <export>/annotations/instances.json
    <export>/images/<field__flight__file.jpg>
    <export>/labels/<stem>.txt          (nếu bản xuất có kèm YOLO)

Đầu ra là thư mục đúng bố cục CocoDataset và ultralytics mong đợi:
    <out>/annotations/instances_<split>.json
    <out>/images/<split>/...   <out>/labels/<split>/...   <out>/data.yaml   <out>/fold.json

Hai quyết định đáng ghi:
- Ảnh được HARDLINK chứ không chép: 6 fold × 1 GB ảnh mà chép là 6 GB cho
  cùng một byte. Hardlink chỉ là tên thứ hai của cùng file, không tốn đĩa, và
  ai đọc cũng thấy là file thường. Khác ổ đĩa thì tự chuyển sang chép.
- Nhãn YOLO: bản xuất có labels/ thì CHÉP; không có thì SINH từ chính file
  COCO của fold. Trước đây thiếu labels/ là fold ra file .txt rỗng và YOLO
  train trên không có gì mà chẳng ai báo — một bộ xuất quên tick ô YOLO là mất
  một đêm máy. Sinh ra thì nhãn luôn khớp COCO của cùng fold đó.
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
from typing import Iterable

import yaml

from .yolo import write_data_yaml, write_labels

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


LABEL_MODES = ("auto", "copy", "generate")


def make_fold(
    export: str | Path,
    name: str,
    fields: dict[str, list[str]],
    out: str | Path,
    copy: bool = False,
    labels: str = "auto",
    val_images: Iterable[str] | None = None,
    drop_images: Iterable[str] | None = None,
    val_split: dict | None = None,
) -> dict:
    """Cắt <export> thành <out> theo `fields` ({split: [ruộng]}). Trả về tóm tắt.

    <out> phải chưa tồn tại — thư mục fold là thứ sinh ra được, xoá đi làm lại
    rẻ hơn là đoán xem bên trong còn gì của lần trước.

    `labels`: "auto" (chép nếu bản xuất có labels/, không thì sinh từ COCO),
    "copy" (bắt buộc chép, lỗi nếu bản xuất không có), "generate" (luôn sinh —
    dùng khi nghi labels/ của bản xuất đã cũ so với instances.json).

    `val_images` chọn val ở mức ẢNH thay vì mức ruộng: ảnh có tên trong đó
    sang val, phần còn lại của cùng ruộng vẫn ở train. Cần thiết vì yêu cầu
    của đồ án là train đủ 5 ruộng, mà val lấy trọn một ruộng thì chỉ còn 4.

    `drop_images` là khoảng đệm: những ảnh KHÔNG vào tập nào, vì chúng chồng
    lấn với val. Bỏ hẳn chứ không đẩy sang train — để ở train là val nhìn thấy
    chúng, mà đó đúng là thứ khoảng đệm sinh ra để chặn.

    `val_split` là phần ghi chép của bên gọi (cách chia, tham số, điểm cắt,
    rò rỉ đo lại) để `fold.json` giải thích được vì sao fold này trông như vậy.
    """
    if labels not in LABEL_MODES:
        raise ValueError(f"labels={labels!r} không hợp lệ; có: {LABEL_MODES}")
    val_set = set(val_images or ())
    drop_set = set(drop_images or ())
    both = val_set & drop_set
    if both:
        raise ValueError(f"fold {name}: {len(both)} ảnh vừa là val vừa bị bỏ: {sorted(both)[:3]}")
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
    dropped: list[str] = []
    for im in raw["images"]:
        nm = im["file_name"]
        fld = field_of(nm)
        if fld not in known:
            skipped.append(nm)
            continue
        sp = field_to_split[fld]
        if sp != "test" and nm in drop_set:
            dropped.append(nm)
        elif sp != "test" and nm in val_set:
            by_split["val"].append(im)
        else:
            by_split[sp].append(im)

    stray = val_set - {im["file_name"] for im in by_split["val"]}
    if stray:
        raise ValueError(
            f"fold {name}: {len(stray)} ảnh val không nằm trong ruộng train của fold này "
            f"(hoặc không có trong bản xuất): {sorted(stray)[:3]}")
    empty = [sp for sp in SPLITS if not by_split[sp]]
    if empty:
        raise ValueError(
            f"fold {name}: tập {empty} không có ảnh nào — ruộng "
            f"{[fields.get(sp, []) for sp in empty]} chưa có trong bản xuất?"
        )

    anns_by_image: dict[int, list[dict]] = {}
    for a in raw["annotations"]:
        anns_by_image.setdefault(int(a["image_id"]), []).append(a)

    export_has_labels = (export / "labels").is_dir()
    if labels == "copy" and not export_has_labels:
        raise FileNotFoundError(
            f"labels='copy' nhưng {export / 'labels'} không có. Xuất lại kèm format "
            "yolo, hoặc dùng labels='generate' để sinh từ COCO.")
    label_mode = "copy" if (labels == "copy" or (labels == "auto" and export_has_labels)) else "generate"
    has_labels = label_mode == "copy"
    summary: dict = {
        "fold": name,
        "fields": fields,
        "source": str(export).replace("\\", "/"),
        "source_sha1": _sha1(ann_file),
        "images_skipped": skipped,
        "splits": {},
        "labels": label_mode,
        "val_split": val_split,
        "dropped": {"count": len(dropped), "files": sorted(dropped)},
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
        doc["info"] = {**doc["info"], "fold": name, "split": sp,
                       "fields": sorted({field_of(im["file_name"]) for im in images})}
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
            "fields": sorted({field_of(im["file_name"]) for im in images}),
            "images": len(images),
            "annotations": len(anns),
            "empty_images": n_bg,
        }
    summary["images_linked"] = how["link"]
    summary["images_copied"] = how["copy"]

    cats = raw.get("categories") or []
    names = {int(c["id"]) - 1: c["name"] for c in cats} or {0: "canopy"}
    if label_mode == "generate":
        # min_area=0: model COCO (detectron2, mmdet) đọc thẳng instances_*.json
        # nên train trên MỌI annotation; lọc bớt ở nhãn YOLO là cho YOLO một bộ
        # nhãn khác các model kia, tức so sánh không còn sạch.
        if len(cats) > 1:
            raise ValueError(
                f"Sinh nhãn YOLO chỉ làm được với bộ một lớp; bản xuất có {len(cats)} lớp. "
                "Xuất lại kèm format yolo rồi dùng labels='copy'.")
        summary["labels_generated"] = write_labels(
            out, list(SPLITS), class_index=min(names), min_area=0.0)["splits"]
    write_data_yaml(out, {sp: f"images/{sp}" for sp in SPLITS}, names)
    (out / "fold.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
