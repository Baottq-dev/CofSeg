"""Đồ thị chồng lấn: các phép hỏi mà việc chia val dựa vào."""

from __future__ import annotations

import json

import pytest

from canopyseg.datasets import overlap_graph as og


def chain(n: int, overlap: float = 0.5, prefix: str = "i") -> og.Graph:
    """n ảnh xếp thành chuỗi, ảnh nào cũng chồng ảnh liền sau."""
    g = og.Graph()
    for k in range(n - 1):
        g.add(f"{prefix}{k:02d}", f"{prefix}{k + 1:02d}", overlap, gap=1)
    return g


# ------------------------------------------------------------------ cạnh
def test_edges_are_symmetric_and_keep_the_strongest_overlap():
    g = og.Graph()
    g.add("a", "b", 0.40)
    g.add("b", "a", 0.70)
    assert g.neighbours("a") == {"b": 0.70}
    assert g.neighbours("b") == {"a": 0.70}
    assert g.n_edges == 1


def test_an_overlap_below_the_threshold_is_not_an_edge():
    g = og.Graph(threshold=0.30)
    g.add("a", "b", 0.29)
    assert g.n_edges == 0
    assert g.neighbours("a") == {}


def test_neighbours_can_be_asked_at_a_stricter_threshold():
    g = og.Graph(threshold=0.30)
    g.add("a", "b", 0.35)
    g.add("a", "c", 0.80)
    assert set(g.neighbours("a")) == {"b", "c"}
    assert set(g.neighbours("a", threshold=0.50)) == {"c"}


# ------------------------------------------------------- hỏi khi chia tập
def test_touching_names_the_train_images_that_must_go():
    g = chain(6)                       # i00-i01-...-i05
    block = {"i04", "i05"}             # val
    outside = {f"i0{k}" for k in range(4)}
    assert g.touching(block, outside) == {"i03"}


def test_dirty_names_the_val_images_that_still_see_train():
    g = chain(6)
    block = {"i04", "i05"}
    outside = {f"i0{k}" for k in range(4)}
    assert g.dirty(block, outside) == ["i04"]
    # Bỏ i03 khỏi train là val sạch.
    assert g.dirty(block, outside - {"i03"}) == []


def test_cut_profile_finds_the_thin_spot():
    """Chuỗi 10 ảnh, cụm 4 ảnh cuối dính chặt nhau nhưng không dính về trước:
    cắt ngay trước cụm là 0 cạnh bị cắt, cắt giữa cụm thì không."""
    g = og.Graph()
    for k in range(5):                              # i00..i05 dính nhau
        g.add(f"i{k:02d}", f"i{k + 1:02d}", 0.6, gap=1)
    for k in range(6, 9):                           # i06..i09 dính nhau
        g.add(f"i{k:02d}", f"i{k + 1:02d}", 0.6, gap=1)
    seq = [f"i{k:02d}" for k in range(10)]
    profile = dict(g.cut_profile(seq, 4, 9))
    assert profile[6] == 0                          # đúng khe giữa hai cụm
    assert profile[8] > 0                           # cắt giữa cụm sau
    assert min(profile, key=profile.get) == 6


# ---------------------------------------------------------- xác suất p(k)
def test_p_overlap_uses_the_measured_denominator():
    g = og.Graph()
    g.pairs_by_gap = {1: 100, 2: 200}
    g.edges_by_gap = {1: 60, 2: 10}
    assert g.p_overlap(1) == pytest.approx(0.60)
    assert g.p_overlap(2) == pytest.approx(0.05)


def test_p_overlap_is_zero_where_nothing_was_measured():
    """Ước thấp còn hơn bịa: khoảng cách chưa đo thì trả 0, không nội suy."""
    g = og.Graph()
    g.pairs_by_gap = {1: 100}
    g.edges_by_gap = {1: 60}
    assert g.p_overlap(7) == 0.0


def test_expected_leak_falls_with_distance_and_with_the_buffer():
    g = og.Graph()
    g.pairs_by_gap = {k: 1000 for k in range(1, 40)}
    g.edges_by_gap = {k: (600 if k == 1 else 50) for k in range(1, 40)}
    near, far = g.expected_leak(1), g.expected_leak(8)
    assert near > far > 0
    assert g.expected_leak(2, dropped=0) > g.expected_leak(2, dropped=10)
    assert g.expected_leak(2, dropped=60) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------- đọc / ghi
def test_round_trip_keeps_edges_counts_and_the_scope_warning(tmp_path):
    g = chain(4)
    g.pairs_by_gap, g.edges_by_gap = {1: 30, 2: 20}, {1: 3, 2: 1}
    g.sources = ["runs/overlap/x"]
    p = og.save(g, tmp_path / "edges.json")
    back = og.load(p)
    assert back.n_edges == g.n_edges
    assert back.neighbours("i01") == g.neighbours("i01")
    assert back.pairs_by_gap == g.pairs_by_gap
    assert back.scope == "flight"
    assert back.sources == ["runs/overlap/x"]
    # File phải đọc được bằng mắt: mỗi cạnh một dòng.
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert len(doc["edges"]) == g.n_edges
    assert p.read_text(encoding="utf-8").count('\n    ["') == g.n_edges


def test_loading_a_missing_file_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_overlap_edges"):
        og.load(tmp_path / "khong-co.json")


def test_an_empty_graph_says_its_scope_is_none():
    """Chia tập ở máy chưa xuất bảng cạnh vẫn chạy, nhưng nơi dùng phải phân
    biệt được 'rò rỉ 0 vì sạch' với 'rò rỉ 0 vì không biết'."""
    g = og.empty()
    assert g.scope == "none"
    assert g.n_edges == 0
    assert g.dirty({"a"}, {"b"}) == []
