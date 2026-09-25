"""Thư mục kết quả và dấu vết của một lần chạy.

Điều phải giữ: tên thư mục nói được lần chạy dùng BỘ FOLD nào. Hai cách chia
val cùng đặt tên f1..f6, nên thiếu chiều đó thì hai lần chạy khác hẳn nhau ra
tên giống hệt và không ai phân biệt được lúc đọc kết quả.
"""

from __future__ import annotations

import json

from cofseg import artifacts


def test_dataset_tag_names_the_fold_set():
    tag = artifacts.dataset_tag
    assert tag({"data": {"root": "data/export/block/f4"}}) == "block-f4"
    assert tag({"data": {"root": "data/export/flight/f4"}}) == "flight-f4"
    # Ultralytics trỏ vào data.yaml nằm trong chính thư mục fold.
    assert tag({"data": {"yaml": "data/export/flight/f2/data.yaml"}}) == "flight-f2"


def test_dataset_tag_survives_a_layout_without_a_fold_set():
    tag = artifacts.dataset_tag
    assert tag({"data": {"root": "data/export/f4"}}) == "f4"
    assert tag({"data": {}}) == ""
    assert tag({}) == ""


def test_two_fold_sets_never_share_a_run_directory(tmp_path):
    a = artifacts.create_run_dir(tmp_path, "train", "m-f4", "block-f4_i1024b4e50")
    b = artifacts.create_run_dir(tmp_path, "train", "m-f4", "flight-f4_i1024b4e50")
    assert a != b and "block-f4" in a.name and "flight-f4" in b.name


def test_the_same_run_twice_still_gets_two_directories(tmp_path):
    a = artifacts.create_run_dir(tmp_path, "train", "m-f4", "block-f4_i1024b4e50")
    b = artifacts.create_run_dir(tmp_path, "train", "m-f4", "block-f4_i1024b4e50")
    assert a != b and b.name.endswith("~2")


def test_env_records_the_frameworks_being_compared(tmp_path):
    """Ba trong bốn model chạy trên detectron2/mmdet. Thiếu chúng ở đây thì
    env.json không mô tả được thứ đã chạy, mà trên máy thuê đó là bản ghi duy
    nhất còn lại."""
    env = artifacts.write_env(tmp_path)
    for mod in ("detectron2", "mmdet", "mmcv", "mmengine", "torch", "ultralytics"):
        assert mod in env["packages"], mod


def test_env_writes_down_which_fold_set_was_used(tmp_path):
    fold = tmp_path / "block" / "f4"
    (fold / "annotations").mkdir(parents=True)
    (fold / "fold.json").write_text(json.dumps({
        "fold": "f4", "source_sha1": "abc",
        "splits": {"train": {"images": 590}, "val": {"images": 110}, "test": {"images": 106}},
        "dropped": {"count": 44},
        "val_split": {"method": "block", "audit": {"leak": {"val_images_touching_train": 0}}},
    }), encoding="utf-8")

    run = tmp_path / "run"
    run.mkdir()
    env = artifacts.write_env(run, {"data": {"root": str(fold)}})
    ds = env["dataset"]
    assert ds["tag"].endswith("block-f4") and ds["fold"] == "f4"
    assert ds["val_method"] == "block" and ds["dropped"] == 44
    assert ds["splits"] == {"train": 590, "val": 110, "test": 106}
    assert ds["leak"]["val_images_touching_train"] == 0
    # Đọc lại từ đĩa: đây là thứ còn lại sau khi máy thuê trả về.
    on_disk = json.loads((run / "env.json").read_text(encoding="utf-8"))
    assert on_disk["dataset"]["val_method"] == "block"


def test_a_missing_fold_json_is_not_fatal(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    env = artifacts.write_env(run, {"data": {"root": str(tmp_path / "khong-co")}})
    assert env["dataset"] is None
