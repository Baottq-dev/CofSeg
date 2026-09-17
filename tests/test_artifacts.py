"""Thư mục kết quả: đặt tên và ghi dấu vết tái lập."""

import datetime as dt
import json

import pytest

from canopyseg import artifacts


T0 = dt.datetime(2026, 9, 9, 16, 30, 0)


# ------------------------------------------------------------------ slugify
@pytest.mark.parametrize(
    "raw,expected",
    [("yolo26s-seg", "yolo26s-seg"),
     ("yolo26s seg/1536:v2 *", "yolo26s-seg-1536-v2"),
     ("a\\b", "a-b"),
     ("__", "unnamed"),
     ("", "unnamed")],
)
def test_slugify_keeps_only_names_windows_accepts(raw, expected):
    assert artifacts.slugify(raw) == expected


# ------------------------------------------------------------------ tên run
def test_directory_name_puts_time_first(tmp_path):
    d = artifacts.create_run_dir(tmp_path, "train", "yolo26s-seg", "i640b4e100", when=T0)
    assert d.parent.name == "train"
    assert d.name == "2026-09-09_163000_yolo26s-seg_i640b4e100"


def test_tag_is_optional(tmp_path):
    d = artifacts.create_run_dir(tmp_path, "probe", "m", when=T0)
    assert d.name == "2026-09-09_163000_m"


def test_kinds_live_in_separate_trees(tmp_path):
    """Một lần dò VRAM 30 giây không được nằm lẫn với lần train 4 tiếng."""
    a = artifacts.create_run_dir(tmp_path, "train", "m", when=T0)
    b = artifacts.create_run_dir(tmp_path, "probe", "m", when=T0)
    assert a.parent != b.parent and a.exists() and b.exists()


def test_same_second_does_not_lose_a_run(tmp_path):
    a = artifacts.create_run_dir(tmp_path, "train", "m", "t", when=T0)
    b = artifacts.create_run_dir(tmp_path, "train", "m", "t", when=T0)
    assert a != b and a.exists() and b.exists()
    assert b.name.endswith("~2")


def test_names_sort_chronologically(tmp_path):
    """Sắp theo tên phải là sắp theo thời gian, kể cả khi cấu hình khác nhau."""
    later = artifacts.create_run_dir(
        tmp_path, "train", "zzz", "i640", when=T0 + dt.timedelta(hours=1))
    earlier = artifacts.create_run_dir(tmp_path, "train", "aaa", "i1536", when=T0)
    assert sorted([later.name, earlier.name]) == [earlier.name, later.name]


def test_created_directory_is_empty_and_new(tmp_path):
    d = artifacts.create_run_dir(tmp_path, "train", "m", when=T0)
    assert d.is_dir() and not list(d.iterdir())


# --------------------------------------------------------------- dấu vết chạy
def test_env_records_what_is_needed_to_reproduce(tmp_path):
    env = artifacts.write_env(tmp_path)
    for key in ("time", "python", "platform", "git_commit", "git_dirty", "packages"):
        assert key in env
    # git_dirty là thứ phân biệt "commit này mô tả đúng code đã chạy" với
    # "commit này chỉ là điểm gần nhất".
    assert isinstance(env["git_dirty"], bool)
    assert "torch" in env["packages"]
    on_disk = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
    assert on_disk["python"] == env["python"]


def test_missing_library_is_recorded_as_null_not_omitted(tmp_path):
    env = artifacts.write_env(tmp_path)
    assert all(k in env["packages"] for k in ("torch", "numpy", "cv2"))


def test_config_snapshot_round_trips(tmp_path):
    from canopyseg import config as cfgmod

    cfg = {"name": "x", "train": {"imgsz": 1536, "ghi_chu": "đường biên"}}
    artifacts.snapshot_config(tmp_path, cfg)
    assert cfgmod.load(tmp_path / "config.yaml") == cfg
