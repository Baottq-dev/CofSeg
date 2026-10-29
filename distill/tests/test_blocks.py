"""Khối `student:` và `outside_box:`: mặc định, nhãn tên run, giá trị sai bị chặn."""

from __future__ import annotations

import pytest

from distill import blocks


def test_mac_dinh_trung_thiet_ke_da_chot():
    st = blocks.check_student({})
    assert st == {"width": 0.375, "nm": 16, "proto_s8": True, "drop_p3": True, "init": "select"}
    assert blocks.student_tag(st) == ""
    ob = blocks.check_outside_box({})
    assert ob == {"train_expand": 1.25, "infer_expand": 1.1}
    assert blocks.outside_box_tag(ob) == ""
    assert blocks.check_student(None) is None and blocks.check_outside_box(None) is None


@pytest.mark.parametrize("block,tag", [
    ({"nm": 32, "proto_s8": False, "drop_p3": False}, "nm32-ps4-p3"),   # đầu gốc ở độ rộng 0,375
    ({"width": 0.25}, "w0.25"),
    ({"init": "scratch"}, "scratch"),
])
def test_nhan_chi_ghi_phan_khac_mac_dinh(block, tag):
    assert blocks.student_tag(blocks.check_student(block)) == tag


def test_nhan_ngoai_hop():
    assert blocks.outside_box_tag(blocks.check_outside_box({"train_expand": 1.5})) == "tx1.5"
    assert blocks.outside_box_tag(blocks.check_outside_box({"infer_expand": 1.0})) == "ix1"


@pytest.mark.parametrize("block,match", [
    ({"widht": 0.375}, "width"),          # gõ sai khoá: gợi ý
    ({"width": 0.6}, r"\(0, 0.5\]"),      # rộng hơn thầy: không chọn trọng số được
    ({"width": 0}, r"\(0, 0.5\]"),
    ({"nm": 0}, r"\[1, 32\]"),
    ({"nm": 16.0}, r"\[1, 32\]"),
    ({"proto_s8": 1}, "true/false"),
    ({"init": "coco"}, "select, scratch"),
    ([], "khối lồng nhau"),
])
def test_tro_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        blocks.check_student(block)


@pytest.mark.parametrize("block,match", [
    ({"train_expand": 0.9}, ">= 1"),
    ({"infer_expand": True}, ">= 1"),
    ({"expand": 1.25}, "train_expand"),
])
def test_ngoai_hop_gia_tri_sai_bi_chan(block, match):
    with pytest.raises(SystemExit, match=match):
        blocks.check_outside_box(block)


def test_mo_ta_cho_run_log():
    assert "0.375" in blocks.describe_student(blocks.check_student({}))
    assert "P4–P5" in blocks.describe_student(blocks.check_student({}))
    assert blocks.describe_student(None).startswith("Trò: không")
    assert "×1.25" in blocks.describe_outside_box(blocks.check_outside_box({}))
    assert "gốc" in blocks.describe_outside_box(None)
