"""Nhật ký bay đọc ra từ tên file.

Test ở đây giữ đúng một tính chất mà cả việc chia val dựa vào: nhận ra hai
thư mục thật ra là một chuyến bay liên tục, và KHÔNG nhận nhầm khi bộ đếm của
máy bay nhảy về đầu.
"""

from __future__ import annotations

import datetime as dt

from canopyseg.datasets import flightlog


# ------------------------------------------------------------------ đọc tên
def test_parse_reads_field_flight_time_and_counter():
    f = flightlog.parse("field_1__10__1__DJI_20260301075324_0001_D.jpg")
    assert (f.field, f.flight, f.counter) == ("field_1", "10/1", 1)
    assert f.taken == dt.datetime(2026, 3, 1, 7, 53, 24)
    assert f.base == "DJI_20260301075324_0001_D.jpg"


def test_parse_handles_a_flight_without_a_sub_number():
    f = flightlog.parse("field_3__10__DJI_20260301154200_0126_D.jpg")
    assert (f.field, f.flight, f.counter) == ("field_3", "10", 126)


def test_parse_returns_none_instead_of_raising_on_a_foreign_name(flight_run):
    assert flightlog.parse("readme.txt") is None
    assert flightlog.parse("field_1__10__1__IMG_0001.jpg") is None
    # Bản xuất lẫn một file lạ thì phần còn lại vẫn phải đọc được.
    fs = flightlog.frames(["readme.txt"] + flight_run("field_1", "10/1", "20260301", 75324, 1, 3))
    assert len(fs) == 3


def test_flights_group_and_sort_by_capture_time(flight_run):
    names = flight_run("field_1", "10/1", "20260301", 75324, 1, 4)
    fs = flightlog.frames(list(reversed(names)))
    seq = flightlog.flights(fs)[("field_1", "10/1")]
    assert [f.counter for f in seq] == [1, 2, 3, 4]


# --------------------------------------------------------------- ranh giới
def test_a_junction_is_found_when_the_drone_counter_keeps_going(flight_run):
    """10/1 dừng ở 55, 10/2 bắt đầu ở 63: máy bay không hạ cánh."""
    fs = flightlog.frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 55)
                          + flight_run("field_1", "10/2", "20260301", 80700, 63, 56))
    (j,) = flightlog.junctions(fs)
    assert (j.field, j.before, j.after) == ("field_1", "10/1", "10/2")
    assert j.counter_gap == 8
    assert j.seconds > 0


def test_a_counter_reset_is_not_a_junction(flight_run):
    """Bay ngày khác, thẻ nhớ đánh số lại từ 1. Tên file không nói được hai
    ngày đó có chung mảnh đất hay không, nên không được đoán là nối liền."""
    fs = flightlog.frames(flight_run("field_1", "10/3", "20260228", 161500, 1, 88)
                          + flight_run("field_1", "10/1", "20260301", 75324, 1, 54))
    assert flightlog.junctions(fs) == []


def test_a_counter_jump_too_large_is_not_a_junction(flight_run):
    fs = flightlog.frames(flight_run("field_2", "10/1", "20260301", 151300, 7, 67)
                          + flight_run("field_2", "10/2", "20260301", 152500, 400, 50))
    assert flightlog.junctions(fs) == []


def test_two_fields_never_share_a_junction(flight_run):
    """Bộ đếm chạy tiếp từ ruộng này sang ruộng khác (một phiên bay qua nhiều
    ruộng) vẫn không phải ranh giới cần đệm: hai ruộng là hai mảnh đất."""
    fs = flightlog.frames(flight_run("field_2", "10", "20260301", 151300, 7, 20)
                          + flight_run("field_3", "10", "20260301", 154200, 28, 20))
    assert flightlog.junctions(fs) == []


# ------------------------------------------------------------------ phiên
def test_sessions_chain_the_flights_that_belong_together(flight_run):
    fs = flightlog.frames(
        flight_run("field_1", "10/3", "20260228", 161500, 1, 10)      # phiên 28/02
        + flight_run("field_1", "10/4", "20260228", 164500, 14, 10)
        + flight_run("field_1", "10/1", "20260301", 75324, 1, 10)     # phiên 01/03
        + flight_run("field_1", "10/2", "20260301", 80700, 15, 10)
        + flight_run("field_6", "10", "20260302", 93600, 538, 10))    # đứng một mình
    got = [["/".join(k) for k in chain] for chain in flightlog.sessions(fs)]
    assert got == [["field_1/10/3", "field_1/10/4"],
                   ["field_1/10/1", "field_1/10/2"],
                   ["field_6/10"]]
