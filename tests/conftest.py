"""Fixture dùng chung cho cả bộ test.

Nguyên tắc: không test nào được phụ thuộc vào dữ liệu thật trong data/ hay
trọng số trong weights/. Cả hai đều nằm ngoài git, nên test dựa vào chúng sẽ
hỏng trên máy người khác. Mọi thứ ở đây được dựng tại chỗ.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil

import numpy as np
import pytest

IMG_W, IMG_H = 400, 300


# --------------------------------------------------------- tên ảnh chuyến bay
def _flight_name(field: str, flight: str, when: str, counter: int) -> str:
    """'field_1', '10/1', '20260301075324', 1 -> tên ảnh đã làm phẳng."""
    return f"{field}__{flight.replace('/', '__')}__DJI_{when}_{counter:04d}_D.jpg"


@pytest.fixture
def flight_name():
    return _flight_name


@pytest.fixture
def flight_run():
    """n ảnh liên tiếp của một đường bay: cách nhau `step` giây, bộ đếm của
    máy bay chạy từ `first`.

    Là fixture chứ không phải hàm import được: pytest chạy với
    --import-mode=importlib nên file test không import lẫn nhau được, còn
    fixture thì dùng chung được.
    """

    def make(field: str, flight: str, day: str, start_hhmmss: int,
             first: int, n: int, step: int = 10) -> list[str]:
        t0 = dt.datetime.strptime(day + f"{start_hhmmss:06d}", "%Y%m%d%H%M%S")
        return [_flight_name(field, flight,
                             (t0 + dt.timedelta(seconds=step * i)).strftime("%Y%m%d%H%M%S"),
                             first + i)
                for i in range(n)]

    return make


@pytest.fixture
def disk_mask():
    """Đĩa tròn — hình duy nhất có đáp án giải tích cho mọi chỉ số ở đây."""

    def make(r: float, size: int = 300, cx: float | None = None, cy: float | None = None):
        cx = size / 2 if cx is None else cx
        cy = size / 2 if cy is None else cy
        y, x = np.ogrid[:size, :size]
        return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r

    return make


@pytest.fixture
def square_polygon():
    def make(x0: float, y0: float, size: float) -> np.ndarray:
        return np.array(
            [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
            dtype=np.float64,
        )

    return make


def _coco_doc(images, annotations, cat_id=1):
    return {
        "info": {"description": "bộ giả lập cho test"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_id, "name": "canopy", "supercategory": "canopy"}],
    }


@pytest.fixture
def tiny_dataset(tmp_path, square_polygon):
    """Bộ COCO nhỏ trên đĩa, đúng bố cục mà CocoDataset mong đợi.

    Ba ảnh có chủ đích: một ảnh hai vùng bình thường, một ảnh có vùng hỏng
    (thiếu đỉnh) để kiểm việc loại bỏ, một ảnh không có vùng nào — trường hợp
    ultralytics coi là "ảnh nền" và dễ bị quên.
    """
    root = tmp_path / "dataset"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)

    names = [
        "field_1__10__1__A.jpg",   # 2 vùng
        "field_2__10__B.jpg",      # 1 vùng tốt + 1 vùng hỏng
        "field_2__10__C.jpg",      # 0 vùng
    ]
    images = [
        {"id": i + 1, "file_name": n, "width": IMG_W, "height": IMG_H}
        for i, n in enumerate(names)
    ]
    # Ảnh thật (nhiễu ngẫu nhiên) chứ không phải file rỗng: bộ đọc cho model
    # detection phải giải mã được chúng. Cỡ nhỏ để test chạy trong vài mili giây.
    import cv2

    rng = np.random.default_rng(0)
    for n in names:
        img = rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / "train" / n), img)

    def ann(aid, iid, x, y, s):
        poly = square_polygon(x, y, s)
        return {
            "id": aid, "image_id": iid, "category_id": 1,
            "segmentation": [poly.reshape(-1).tolist()],
            "area": float(s * s), "bbox": [x, y, s, s], "iscrowd": 0,
        }

    anns = [
        ann(1, 1, 20, 20, 60),
        ann(2, 1, 150, 100, 80),
        ann(3, 2, 40, 40, 100),
        # Hai đỉnh: không đủ để thành đa giác, phải bị loại.
        {"id": 4, "image_id": 2, "category_id": 1,
         "segmentation": [[10.0, 10.0, 20.0, 20.0]], "area": 5.0,
         "bbox": [10, 10, 10, 10], "iscrowd": 0},
    ]
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps(_coco_doc(images, anns)), encoding="utf-8"
    )
    return root


@pytest.fixture
def three_splits(tiny_dataset):
    """tiny_dataset chỉ có train; nhân thành val/test để prepare() của các
    trainer đọc đủ ba tập."""
    ann = tiny_dataset / "annotations"
    for sp in ("val", "test"):
        shutil.copy2(ann / "instances_train.json", ann / f"instances_{sp}.json")
        shutil.copytree(tiny_dataset / "images" / "train", tiny_dataset / "images" / sp)
    return tiny_dataset
