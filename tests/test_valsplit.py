"""Chọn val bên trong các ruộng train.

Tính chất phải giữ, và là lý do cả module tồn tại: sau khi chia, KHÔNG ảnh
val nào còn cạnh chồng lấn sang train. Mọi thứ khác (khối to bao nhiêu, cắt ở
đâu) là đánh đổi; cái đó là đúng/sai.
"""

from __future__ import annotations

from canopyseg.datasets import flightlog, valsplit
from canopyseg.datasets.overlap_graph import Graph


def frames(*runs) -> list[flightlog.Frame]:
    names = [n for r in runs for n in r]
    return flightlog.frames(names)


def sequential_graph(fs, *, overlap=0.6, reach=1) -> Graph:
    """Nối mỗi ảnh với `reach` ảnh liền sau trong cùng đường bay."""
    g = Graph()
    for seq in flightlog.flights(fs).values():
        names = [f.file_name for f in seq]
        for i, a in enumerate(names):
            for b in names[i + 1:i + 1 + reach]:
                g.add(a, b, overlap, gap=names.index(b) - i)
    return g


def leak(a: valsplit.Assignment, fs, g: Graph) -> list[str]:
    pool = {f.file_name for f in fs}
    return g.dirty(a.val & pool, pool - a.val - a.drop)


# ------------------------------------------------------------- khối cuối
def test_block_takes_the_tail_of_every_flight(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20),
                flight_run("field_2", "10", "20260301", 151300, 1, 20))
    g = sequential_graph(fs)
    a = valsplit.by_block(fs, g, frac=0.25, slack=0)
    assert len(a.val) == 10                      # 25% của 20, hai đường bay
    assert {f.field for f in fs if f.file_name in a.val} == {"field_1", "field_2"}
    tail = {f.file_name for f in flightlog.flights(fs)[("field_1", "10/1")][-5:]}
    assert tail <= a.val


def test_no_val_image_still_touches_train(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 40))
    g = sequential_graph(fs, reach=3)
    a = valsplit.by_block(fs, g, frac=0.25, slack=0)
    assert leak(a, fs, g) == []
    assert a.drop, "phải bỏ vài ảnh train sát ranh giới, không thì val không thể sạch"


def test_the_buffer_drops_only_what_actually_touches_val(flight_run):
    """Đệm theo đồ thị bỏ đúng ảnh có cạnh, không bỏ cả vùng. Chuỗi 20 ảnh
    nối 1 bước: khối 5 ảnh cuối chỉ dính đúng một ảnh train."""
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20))
    g = sequential_graph(fs, reach=1)
    a = valsplit.by_block(fs, g, frac=0.25, slack=0)
    assert len(a.drop) == 1


def test_the_cut_slides_to_where_the_graph_is_thin(flight_run):
    """Ba cụm ảnh dính nhau, giữa các cụm không có cạnh nào.

    Khối val rộng 5 ảnh. Vị trí danh nghĩa (25% cuối, tức 15..19) xẻ đôi cụm
    cuối. Cụm 13..17 rộng đúng 5 ảnh và nằm trong tầm trượt — cho ranh giới
    xê dịch là nó nhảy sang đó và không cắt cạnh nào.
    """
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20))
    seq = [f.file_name for f in flightlog.flights(fs)[("field_1", "10/1")]]
    g = Graph()
    for lo, hi in ((0, 12), (13, 17), (18, 19)):      # ba cụm, giữa chúng là khe
        for i in range(lo, hi):
            g.add(seq[i], seq[i + 1], 0.6, gap=1)

    tight = valsplit.by_block(fs, g, frac=0.25, slack=0)      # cắt cứng ở 15..19
    loose = valsplit.by_block(fs, g, frac=0.25, slack=0.5)    # cho trượt
    assert tight.why["edges_cut_total"] == 1
    assert tight.drop == {seq[14]}                  # phải bỏ một ảnh train
    assert loose.why["edges_cut_total"] == 0
    assert loose.val == set(seq[13:18])             # đúng trọn cụm giữa
    assert loose.drop == set()                      # không phải bỏ ảnh nào
    assert leak(loose, fs, g) == []


def test_a_fixed_block_keeps_the_same_images_out_of_train_every_run(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 30))
    g = sequential_graph(fs)
    first = valsplit.by_block(fs, g, frac=0.2, slack=0)
    again = valsplit.by_block(fs, g, frac=0.2, slack=0)
    assert first.val == again.val


def test_a_rotating_block_moves_so_every_image_trains_somewhere(flight_run):
    """Khối cố định để lại một dải ảnh không bao giờ vào train. Khối trượt
    thì mỗi lượt lấy một dải khác, gộp 6 lượt là mọi ảnh đều được học."""
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 60))
    g = sequential_graph(fs)
    pool = {f.file_name for f in fs}

    fixed = set()
    for _ in range(6):
        a = valsplit.by_block(fs, g, frac=0.15, slack=0)
        fixed |= pool - a.val - a.drop
    assert pool - fixed, "khối cố định lẽ ra phải bỏ sót một dải ảnh"

    moved = set()
    for slot in range(6):
        a = valsplit.by_block(fs, g, frac=0.15, slack=0, slot=slot, n_slots=6)
        moved |= pool - a.val - a.drop
    assert moved == pool


def test_why_records_every_cut_for_fold_json(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20),
                flight_run("field_1", "10/2", "20260301", 80700, 30, 20))
    g = sequential_graph(fs)
    a = valsplit.by_block(fs, g, frac=0.25, slack=0)
    assert set(a.why["cuts"]) == {"field_1/10/1", "field_1/10/2"}
    one = a.why["cuts"]["field_1/10/1"]
    assert one["images"] == 20 and one["val_images"] == 5
    assert a.why["graph_scope"] == "flight" and a.why["buffer"] == "graph"


# ------------------------------------------------------------------ chấm
def test_audit_counts_leak_density_and_fields(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20),
                flight_run("field_2", "10", "20260301", 151300, 1, 20))
    g = sequential_graph(fs)
    a = valsplit.by_block(fs, g, frac=0.25, slack=0)
    regions = {f.file_name: 10 for f in fs}
    rep = valsplit.audit(a, fs, g, regions)
    assert rep["val"]["images"] == 10 and rep["val"]["regions"] == 100
    assert rep["val"]["regions_per_image"] == 10.0
    assert rep["val_fields"] == {"field_1": 5, "field_2": 5}
    assert rep["leak"]["val_images_touching_train"] == 0
    assert rep["leak"]["graph_scope"] == "flight"
    assert rep["train"]["images"] + rep["val"]["images"] + rep["dropped"]["images"] == 40


def test_audit_reports_leak_when_there_is_no_buffer(flight_run):
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20))
    g = sequential_graph(fs)
    naked = valsplit.Assignment(val={f.file_name for f in fs[-5:]}, drop=set())
    rep = valsplit.audit(naked, fs, g)
    assert rep["leak"]["val_images_touching_train"] == 1
    assert rep["leak"]["percent_of_val"] == 20.0


def test_audit_says_the_graph_knows_nothing_when_it_is_empty(flight_run):
    """Rò rỉ 0 trên đồ thị rỗng nghĩa là KHÔNG BIẾT, không phải sạch. Báo cáo
    phải phân biệt được hai chuyện đó."""
    from canopyseg.datasets.overlap_graph import empty
    fs = frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 20))
    a = valsplit.by_block(fs, empty(), frac=0.25, slack=0)
    rep = valsplit.audit(a, fs, empty())
    assert rep["leak"]["val_images_touching_train"] == 0
    assert rep["leak"]["graph_scope"] == "none"
    assert a.drop == set()


# -------------------------------------------------- trọn một đường bay
def two_segments(flight_run, buffer_gap=8):
    """Một chuyến bay liên tục bị cắt thành hai thư mục: bộ đếm chạy tiếp."""
    return frames(flight_run("field_1", "10/1", "20260301", 75324, 1, 30),
                  flight_run("field_1", "10/2", "20260301", 80700, 30 + buffer_gap, 30),
                  flight_run("field_2", "10", "20260301", 151300, 1, 30))


def test_flight_split_takes_whole_flights(flight_run):
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_1/10/2"], buffer=0)
    assert a.val == {f.file_name for f in fs if f.flight == "10/2"}
    assert len(a.val) == 30
    assert a.why["used"] == ["field_1/10/2"]


def test_a_flight_of_the_test_field_is_skipped_but_written_down(flight_run):
    """Cấu hình khai cả hai đường bay cuối; lượt nào có ruộng đó làm test thì
    đường bay ấy không nằm trong pool, phải tự bỏ qua chứ không gãy."""
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_1/10/2", "field_9/10/1"], buffer=0)
    assert a.why["requested"] == ["field_1/10/2", "field_9/10/1"]
    assert a.why["used"] == ["field_1/10/2"]


def test_the_seam_between_two_folders_of_one_mission_gets_a_buffer(flight_run):
    """10/1 và 10/2 là một chuyến bị cắt. Lấy 10/2 làm val thì phải bỏ ảnh ở
    CUỐI 10/1 — phía train của chỗ nối."""
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_1/10/2"], buffer=5)
    tail = [f.file_name for f in flightlog.flights(fs)[("field_1", "10/1")]][-5:]
    assert set(tail) <= a.drop
    (seam,) = a.why["junctions"]
    assert seam["buffered_side"] == "before" and seam["buffered"] == 5
    assert seam["counter_gap"] == 8


def test_the_buffer_goes_on_the_other_side_when_val_comes_first(flight_run):
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_1/10/1"], buffer=4)
    head = [f.file_name for f in flightlog.flights(fs)[("field_1", "10/2")]][:4]
    assert set(head) <= a.drop
    assert a.why["junctions"][0]["buffered_side"] == "after"


def test_no_buffer_between_two_fields(flight_run):
    """field_2 không nối với field_1, nên lấy nó làm val không tốn ảnh nào."""
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_2/10"], buffer=20)
    assert a.why["junctions"] == []
    assert a.drop == set()


def test_the_seam_record_estimates_what_the_buffer_leaves_behind(flight_run):
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    g.pairs_by_gap = {k: 1000 for k in range(1, 40)}
    g.edges_by_gap = {k: (600 if k == 1 else 40) for k in range(1, 40)}
    wide = valsplit.by_flight(fs, g, flights=["field_1/10/2"], buffer=30)
    narrow = valsplit.by_flight(fs, g, flights=["field_1/10/2"], buffer=0)
    assert (narrow.why["junctions"][0]["estimated_val_images_left_dirty"]
            > wide.why["junctions"][0]["estimated_val_images_left_dirty"])


def test_val_images_are_never_also_dropped(flight_run):
    """Hai đường bay cùng làm val, cạnh nhau: ảnh của val này không được rơi
    vào khoảng đệm của val kia."""
    fs = two_segments(flight_run)
    g = sequential_graph(fs)
    a = valsplit.by_flight(fs, g, flights=["field_1/10/1", "field_1/10/2"], buffer=10)
    assert a.val & a.drop == set()
