"""Nạp config và ghi đè từ dòng lệnh."""

import pytest

from canopyseg import config as cfgmod


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_plain_yaml(tmp_path):
    p = write(tmp_path, "a.yaml", "name: x\ntrain:\n  epochs: 10\n")
    cfg = cfgmod.load(p)
    assert cfg == {"name": "x", "train": {"epochs": 10}}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cfgmod.load(tmp_path / "khong-co.yaml")


def test_empty_file_gives_empty_dict(tmp_path):
    assert cfgmod.load(write(tmp_path, "e.yaml", "")) == {}


# ------------------------------------------------------------------ kế thừa
def test_base_include_merges_deeply(tmp_path):
    write(tmp_path, "base.yaml", "name: base\ntrain:\n  epochs: 100\n  imgsz: 640\n")
    child = write(tmp_path, "c.yaml", "base: base.yaml\nname: con\ntrain:\n  imgsz: 1280\n")
    cfg = cfgmod.load(child)
    assert cfg["name"] == "con"
    assert cfg["train"] == {"epochs": 100, "imgsz": 1280}
    assert "base" not in cfg


def test_base_is_resolved_relative_to_the_child(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    write(sub, "base.yaml", "a: 1\n")
    cfg = cfgmod.load(write(sub, "c.yaml", "base: base.yaml\nb: 2\n"))
    assert cfg == {"a": 1, "b": 2}


# ------------------------------------------------------------------- ghi đè
@pytest.mark.parametrize(
    "raw,expected",
    [("10", 10), ("0.5", 0.5), ("true", True), ("false", False),
     ("[1, 2]", [1, 2]), ("chuoi", "chuoi")],
)
def test_override_values_keep_their_yaml_type(tmp_path, raw, expected):
    """Ghi đè phải ra đúng kiểu; "10" thành chuỗi sẽ làm ultralytics nổ sau đó."""
    p = write(tmp_path, "a.yaml", "train:\n  x: 0\n")
    assert cfgmod.load(p, [f"train.x={raw}"])["train"]["x"] == expected


def test_override_creates_missing_nesting(tmp_path):
    p = write(tmp_path, "a.yaml", "name: x\n")
    assert cfgmod.load(p, ["a.b.c=1"])["a"]["b"]["c"] == 1


def test_override_without_equals_is_rejected(tmp_path):
    p = write(tmp_path, "a.yaml", "name: x\n")
    with pytest.raises(ValueError, match="--set"):
        cfgmod.load(p, ["train.epochs"])


def test_override_through_a_scalar_is_rejected(tmp_path):
    """train là số nên train.epochs không có nghĩa; phải báo thay vì ghi đè bừa."""
    p = write(tmp_path, "a.yaml", "train: 5\n")
    with pytest.raises(ValueError):
        cfgmod.load(p, ["train.epochs=10"])


def test_later_override_wins(tmp_path):
    p = write(tmp_path, "a.yaml", "train:\n  epochs: 1\n")
    assert cfgmod.load(p, ["train.epochs=2", "train.epochs=3"])["train"]["epochs"] == 3


def test_loading_does_not_mutate_the_base_file(tmp_path):
    base = write(tmp_path, "base.yaml", "train:\n  epochs: 100\n")
    child = write(tmp_path, "c.yaml", "base: base.yaml\ntrain:\n  epochs: 5\n")
    cfgmod.load(child)
    assert cfgmod.load(base)["train"]["epochs"] == 100


# --------------------------------------------------------------------- ghi ra
def test_dump_round_trip_keeps_unicode_readable(tmp_path):
    out = tmp_path / "out.yaml"
    cfgmod.dump({"ghi_chu": "đường biên tán cà phê", "n": 1}, out)
    assert "đường biên" in out.read_text(encoding="utf-8")
    assert cfgmod.load(out)["ghi_chu"] == "đường biên tán cà phê"
