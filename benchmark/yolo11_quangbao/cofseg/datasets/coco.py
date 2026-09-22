"""Đọc bộ COCO do app/ xuất ra.

Bố cục mong đợi (đúng lệ COCO):
    <root>/annotations/instances_<split>.json
    <root>/images/<split>/<ten anh da lam phang>.jpg

Lưu ý về đánh số lớp: COCO ở đây đánh category_id TỪ 1 (đúng lệ COCO gốc,
chừa 0 cho nền), còn YOLO đánh class index TỪ 0. Hai bên lệch nhau 1 là cố ý.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .region import Region


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    path: Path
    regions: list[Region] = field(default_factory=list)


class CocoDataset:
    """Một split của bộ dữ liệu."""

    def __init__(self, root: str | Path, split: str, min_area: float = 0.0):
        self.root = Path(root)
        self.split = split
        self.ann_file = self.root / "annotations" / f"instances_{split}.json"
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Không thấy {self.ann_file}")
        raw = json.loads(self.ann_file.read_text(encoding="utf-8"))

        self.categories = raw.get("categories", [])
        img_dir = self.root / "images" / split
        self.images: dict[int, ImageRecord] = {
            int(im["id"]): ImageRecord(
                image_id=int(im["id"]),
                file_name=im["file_name"],
                width=int(im["width"]),
                height=int(im["height"]),
                path=img_dir / im["file_name"],
            )
            for im in raw["images"]
        }
        self.dropped = 0
        for ann in raw["annotations"]:
            seg = ann.get("segmentation") or []
            # Cần >= 3 đỉnh mới thành đa giác; số toạ độ phải chẵn.
            if not seg or len(seg[0]) < 6 or len(seg[0]) % 2:
                self.dropped += 1
                continue
            if float(ann.get("area", 0.0)) < min_area:
                self.dropped += 1
                continue
            im = self.images[int(ann["image_id"])]
            im.regions.append(Region.from_coco(ann, im))

    # ------------------------------------------------------------------ truy cập
    def __len__(self) -> int:
        return len(self.images)

    def image_list(self) -> list[ImageRecord]:
        return sorted(self.images.values(), key=lambda i: i.file_name)

    def regions(self) -> list[Region]:
        return [r for im in self.image_list() for r in im.regions]

    @property
    def n_regions(self) -> int:
        return sum(len(im.regions) for im in self.images.values())

    def summary(self) -> dict:
        flights: dict[str, int] = {}
        fields: dict[str, int] = {}
        for im in self.images.values():
            key = "/".join(im.file_name.split("__")[:-1])
            flights[key] = flights.get(key, 0) + 1
            fld = im.file_name.split("__")[0]
            fields[fld] = fields.get(fld, 0) + 1
        return {
            "split": self.split,
            "images": len(self.images),
            "regions": self.n_regions,
            "dropped": self.dropped,
            "fields": dict(sorted(fields.items())),
            "flights": dict(sorted(flights.items())),
        }


def load_splits(
    root: str | Path, splits: list[str], min_area: float = 0.0
) -> dict[str, CocoDataset]:
    return {s: CocoDataset(root, s, min_area=min_area) for s in splits}
