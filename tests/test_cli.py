"""Đọc siêu tham số từ dòng lệnh.

Đây là cửa ngõ của mọi lần chạy: nếu nó nuốt lỗi gõ sai, người chạy chỉ biết
sau ba tiếng huấn luyện rằng tham số không hề vào. Nên phần "phải báo lỗi"
được test kỹ ngang phần "phải nhận đúng".
"""

import pytest

from canopyseg import cli

VALID = {"imgsz", "batch", "epochs", "cos_lr", "amp", "lr0"}
LOCKED = {"data", "project"}


def parse(tokens, valid=VALID, locked=LOCKED):
    return cli.parse_overrides(tokens, valid, locked, what="trainer")


# ------------------------------------------------------------------ nhận đúng
def test_space_and_equals_forms_are_equivalent():
    assert parse(["--imgsz", "1024"]) == parse(["--imgsz=1024"]) == {"imgsz": 1024}


def test_values_are_yaml_typed():
    out = parse(["--batch", "2", "--lr0", "0.01", "--cos_lr", "false"])
    assert out == {"batch": 2, "lr0": 0.01, "cos_lr": False}
    assert isinstance(out["batch"], int) and isinstance(out["lr0"], float)


def test_bare_flag_means_true():
    assert parse(["--amp"]) == {"amp": True}
    # Cờ trần đứng trước một cờ khác vẫn là true, không nuốt cờ sau làm giá trị.
    assert parse(["--amp", "--imgsz", "640"]) == {"amp": True, "imgsz": 640}


def test_dashes_become_underscores():
    assert parse(["--cos-lr", "true"]) == {"cos_lr": True}


def test_no_valid_list_accepts_anything():
    assert parse(["--anything", "1"], valid=None) == {"anything": 1}


def test_empty_tokens_give_empty_dict():
    assert parse([]) == {}


# ------------------------------------------------------------------ báo lỗi
def test_unknown_key_fails_with_a_hint():
    with pytest.raises(SystemExit, match="imgz.*--imgsz"):
        parse(["--imgz", "1024"])


def test_unknown_key_without_close_match_still_fails():
    with pytest.raises(SystemExit, match="không có tham số"):
        parse(["--zzz", "1"])


def test_locked_key_is_refused_even_if_valid():
    with pytest.raises(SystemExit, match="bị khoá"):
        parse(["--project", "x"], valid=VALID | LOCKED)


def test_positional_token_is_refused():
    with pytest.raises(SystemExit, match="Không hiểu đối số"):
        parse(["1024"])


# ------------------------------------------------------------------ liệt kê
def test_describe_params_marks_locked_and_sorts():
    text = cli.describe_params({"b": 1, "a": "x", "data": None}, locked={"data"})
    lines = text.splitlines()
    assert lines[0].startswith("  --a") and lines[1].startswith("  --b")
    assert "[khoá]" in lines[2] and "[khoá]" not in lines[0]
