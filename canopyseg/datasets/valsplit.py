"""Chọn tập val bên trong các ruộng train, và khoảng đệm đi kèm.

Bài toán: đồ án yêu cầu mỗi lượt train 5 ruộng và test ruộng còn lại. Val
phải cắt ra từ chính 5 ruộng đó. Nhưng ảnh bay liên tục chồng lấn nhau, nên
cắt bừa là val nhìn thấy đúng những gốc cà phê đang có trong train — điểm val
cao giả, đường cong không bao giờ bão hoà, checkpoint giữ lại là cái học vẹt
nhất.

Mọi cách ở đây trả về cùng một thứ:

    Assignment(val, drop, why)

`drop` là khoảng đệm: ảnh KHÔNG vào tập nào. Nó là cái giá phải trả để val
sạch, và `why` ghi lại đã trả bao nhiêu, ở đâu, vì sao — để `fold.json` và
báo cáo nói cùng một con số.

Ba cách đang dùng, xem `configs/dataset/val_block.yaml`, `val_flight.yaml`
và `val_field.yaml`. Chúng xếp thành một thang khoảng cách val<->train:
cùng đường bay, khác đường bay, khác ruộng — và chỉ cách cuối mới đặt val
vào đúng điều kiện mà test sẽ chấm.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from . import flightlog, overlap_graph
from .overlap_graph import Graph

METHODS = ("block", "flight", "field")


@dataclass
class Assignment:
    """Kết quả chia: ảnh nào là val, ảnh nào bỏ, và vì sao."""

    val: set[str] = dc_field(default_factory=set)
    drop: set[str] = dc_field(default_factory=set)
    why: dict = dc_field(default_factory=dict)


def _buffer(val: set[str], pool: set[str], graph: Graph) -> set[str]:
    """Ảnh train có cạnh chồng lấn sang val — đúng những ảnh phải bỏ.

    Đây là khoảng đệm ở dạng chính xác của nó. Đệm theo số thứ tự ảnh phải
    bỏ cả một vùng rộng để chắc chắn bắt hết, phần lớn là ảnh vô tội; tra
    thẳng bảng cạnh thì bỏ đúng cái cần bỏ.
    """
    return graph.touching(val, pool - val)


# ------------------------------------------------------ khối trong đường bay
def _block_bounds(seq: Sequence[str], graph: Graph, *, frac: float,
                  slot: int | None, n_slots: int, slack: float) -> tuple[int, int, int]:
    """Chọn (đầu, cuối, số cạnh bị cắt) của khối val trong một đường bay.

    Vị trí danh nghĩa do `slot` quyết định (None = khối nằm cuối). Quanh vị
    trí đó cho ranh giới trượt trong `slack` lần bề rộng khối, và lấy chỗ cắt
    ít cạnh chồng lấn nhất.

    Vì sao đáng trượt: 15% là con số ta tự đặt, còn chỗ máy bay quay đầu là
    do địa hình. Cắt đúng khúc quay đầu thì hai luống bị xẻ đôi và phải bỏ
    cả chục ảnh train; xê dịch vài ảnh là cắt vào khe giữa hai cụm, không
    tốn gì. 14% hay 21% đều là val hợp lệ như nhau.
    """
    n = len(seq)
    w = max(1, min(n - 1, round(frac * n))) if n > 1 else 1
    lo0 = n - w if slot is None else round(slot * (n - w) / max(n_slots - 1, 1))
    lo0 = max(0, min(n - w, lo0))
    span = max(0, round(slack * w))
    whole = set(seq)
    best: tuple[tuple[int, int], int, int] | None = None
    for lo in range(max(0, lo0 - span), min(n - w, lo0 + span) + 1):
        block = set(seq[lo:lo + w])
        cut = graph.crossing(block, whole - block)
        key = (cut, abs(lo - lo0))
        if best is None or key < best[0]:
            best = (key, lo, cut)
    assert best is not None
    return best[1], best[1] + w, best[2]


def by_block(frames: Iterable[flightlog.Frame], graph: Graph, *,
             frac: float = 0.15, slack: float = 0.5,
             slot: int | None = None, n_slots: int = 6) -> Assignment:
    """val = một khối ảnh liên tiếp trong mỗi đường bay, đệm theo đồ thị.

    `slot=None` đặt khối ở cuối mỗi đường bay và giữ nguyên vị trí đó ở cả 6
    lượt — hệ quả là những ảnh đó không bao giờ vào tập train của lượt nào.
    Truyền `slot` (0..n_slots-1) thì khối trượt theo lượt, mỗi ảnh làm val ở
    một lượt và làm train ở các lượt khác.
    """
    fl = flightlog.flights(frames)
    pool = {f.file_name for f in frames}
    val: set[str] = set()
    cuts: dict[str, dict] = {}
    for key in sorted(fl):
        seq = [f.file_name for f in fl[key]]
        lo, hi, cut = _block_bounds(seq, graph, frac=frac, slot=slot,
                                    n_slots=n_slots, slack=slack)
        val |= set(seq[lo:hi])
        cuts["/".join(key)] = {"images": len(seq), "from": lo, "to": hi,
                               "val_images": hi - lo, "edges_cut": cut}
    drop = _buffer(val, pool, graph)
    return Assignment(val=val, drop=drop, why={
        "method": "block",
        "frac": frac, "slack": slack, "slot": slot, "n_slots": n_slots,
        "buffer": "graph", "threshold": graph.threshold, "graph_scope": graph.scope,
        "cuts": cuts,
        "edges_cut_total": sum(c["edges_cut"] for c in cuts.values()),
    })


# ------------------------------------------------------- trọn một đường bay
def _key(name: str | Sequence[str]) -> tuple[str, str]:
    """'field_1/10/4' hoặc ['field_1', '10/4'] -> ('field_1', '10/4')."""
    if isinstance(name, str):
        field, _, flight = name.partition("/")
        return (field, flight)
    field, flight = name
    return (str(field), str(flight))


def by_flight(frames: Iterable[flightlog.Frame], graph: Graph, *,
              flights: Iterable[str | Sequence[str]], buffer: int = 20) -> Assignment:
    """val = trọn một hoặc vài đường bay, đệm ở ranh giới nối liền.

    Ưu điểm so với cắt khối: không xẻ đôi chuỗi ảnh nào, nên trong lòng val
    không có ranh giới nào để rò rỉ.

    Nhưng thư mục đường bay không phải lúc nào cũng là một chuyến bay riêng.
    Ở bộ này cả ba ruộng nhiều thư mục đều là một chuyến liên tục bị cắt: bộ
    đếm của máy bay chạy tiếp qua ranh giới (xem `flightlog.junctions`). Chỗ
    nối đó cần đệm y như một điểm cắt giữa đường bay, nên ta bỏ `buffer` ảnh
    ở phía train của mỗi ranh giới.

    Đệm ở đây đếm theo ẢNH chứ không tra đồ thị, vì phép đo chồng lấn chạy
    với scope=flight nên KHÔNG có cạnh nào bắc qua ranh giới hai thư mục —
    tra đồ thị ở đúng chỗ cần nhất thì nó im lặng. `why` ghi lại ước lượng rò
    rỉ còn lại theo p(k) để biết `buffer` chọn đã đủ chưa.

    Đường bay nào thuộc ruộng test thì không có trong `frames`, tự động bỏ
    qua; `why` ghi cả danh sách yêu cầu lẫn danh sách thực dùng.
    """
    fl = flightlog.flights(frames)
    pool = {f.file_name for f in frames}
    want = [_key(x) for x in flights]
    used = [k for k in want if k in fl]

    val: set[str] = set()
    for k in used:
        val |= {f.file_name for f in fl[k]}

    drop: set[str] = set()
    seams = []
    for j in flightlog.junctions(frames):
        before, after = (j.field, j.before), (j.field, j.after)
        side = None
        if after in used and before not in used:
            # [-0:] là CẢ danh sách, không phải rỗng — buffer=0 phải là không bỏ gì.
            side, names = "before", [f.file_name for f in fl[before]][len(fl[before]) - buffer:]
        elif before in used and after not in used:
            side, names = "after", [f.file_name for f in fl[after]][:buffer]
        if side is None:
            continue
        drop |= set(names)
        seams.append({
            "field": j.field, "before": j.before, "after": j.after,
            "counter_gap": j.counter_gap, "minutes": round(j.seconds / 60, 1),
            "buffered_side": side, "buffered": len(names),
            # Ước theo p(k): khoảng cách khung hình ở ranh giới là counter_gap,
            # đã bỏ `buffer` ảnh phía train thì còn lại bao nhiêu ảnh val bẩn.
            "estimated_val_images_left_dirty":
                round(graph.expected_leak(j.counter_gap, buffer), 2),
        })

    drop |= _buffer(val, pool - drop, graph)
    return Assignment(val=val, drop=drop - val, why={
        "method": "flight",
        "requested": ["/".join(k) for k in want],
        "used": ["/".join(k) for k in used],
        "buffer": buffer, "threshold": graph.threshold, "graph_scope": graph.scope,
        "junctions": seams,
    })


# ------------------------------------------------------------ trọn một ruộng
def by_field(frames: Iterable[flightlog.Frame], graph: Graph, *,
             fields: Sequence[str] | None = None,
             slot: int | None = None) -> Assignment:
    """val = trọn một ruộng, xoay vòng nên mỗi ruộng làm val đúng một lượt.

    Đây là cách duy nhất trong ba cách mà val chịu cùng loại dịch chuyển với
    test: test là ruộng chưa thấy, và val ở đây cũng là ruộng chưa thấy. Hai
    cách kia chọn checkpoint bằng ảnh cùng mảnh đất với train, tức bằng một
    thước đo lạc quan hơn thước sẽ chấm mình.

    Giá phải trả: lượt nào cũng chỉ còn 4 ruộng để học thay vì 5, và cỡ lẫn
    mật độ của val nhảy theo ruộng được chọn (ở bộ này: 48..280 ảnh, 6.9..25.1
    vùng/ảnh). Số ẢNH train thì gần như không đổi, vì cách này không tốn
    khoảng đệm nào.

    Cố định một ruộng làm val thì không chạy được: lượt lấy đúng ruộng đó làm
    test sẽ không còn val. Nên `slot` xoay vòng — val là ruộng kế tiếp ruộng
    test. Với sáu lượt của folds.yaml, mỗi ruộng test một lần, val một lần,
    train bốn lần: không ruộng nào được ưu ái.

    `fields` thu hẹp vòng xoay lại một nhóm (vd chỉ các ruộng nội suy); ruộng
    đang làm test không có trong `frames` nên tự rơi khỏi vòng.

    Khoảng đệm vẫn tra đồ thị dù hai ruộng là hai mảnh đất khác nhau. Ở bộ
    này nó luôn trả về rỗng — 0/1073 cạnh bắc qua ranh giới ruộng — nhưng tra
    thì con số đó là ĐO, còn bỏ qua thì nó chỉ là niềm tin.
    """
    present = sorted({f.field for f in frames})
    ring = [x for x in fields if x in present] if fields else present
    if not ring:
        raise ValueError(
            f"không còn ruộng nào làm val được: công thức khai {list(fields or [])}, "
            f"mà lượt này chỉ có {present}. Khai thêm ruộng vào 'fields'.")
    chosen = ring[0 if slot is None else slot % len(ring)]

    val = {f.file_name for f in frames if f.field == chosen}
    pool = {f.file_name for f in frames}
    drop = _buffer(val, pool, graph)
    return Assignment(val=val, drop=drop, why={
        "method": "field",
        "val_field": chosen,
        "ring": ring,
        "requested": list(fields) if fields else None,
        "slot": slot,
        "buffer": "graph", "threshold": graph.threshold, "graph_scope": graph.scope,
        "edges_cut": graph.crossing(val, pool - val),
    })


# ------------------------------------------------------------------ chấm điểm
def audit(a: Assignment, frames: Iterable[flightlog.Frame], graph: Graph,
          regions: dict[str, int] | None = None) -> dict:
    """Đo một cách chia: to nhỏ, rò rỉ, mật độ, thành phần ruộng.

    `compare_val_splits.py` và `make_fold.py` đều gọi hàm này, nên con số
    trong báo cáo và con số trong fold.json không thể lệch nhau.
    """
    frames = list(frames)
    pool = {f.file_name for f in frames}
    val = a.val & pool
    train = pool - val - a.drop
    reg = regions or {}
    field = {f.file_name: f.field for f in frames}
    dirty = graph.dirty(val, train)

    def stats(names: set[str]) -> dict:
        n_reg = sum(reg.get(x, 0) for x in names)
        return {"images": len(names), "regions": n_reg,
                "regions_per_image": round(n_reg / len(names), 2) if names else 0.0}

    return {
        "train": stats(train), "val": stats(val), "dropped": stats(a.drop),
        "val_fields": dict(sorted(Counter(field[x] for x in val).items())),
        "leak": {
            "threshold": graph.threshold,
            "graph_scope": graph.scope,
            "val_images_touching_train": len(dirty),
            "percent_of_val": round(100 * len(dirty) / len(val), 2) if val else 0.0,
            "examples": dirty[:5],
        },
    }


# --------------------------------------------------------------- công thức
def load_recipe(path: str | Path) -> dict:
    """Đọc val_block.yaml / val_flight.yaml / val_field.yaml, kiểm trước khi cắt gì.

    Sai một tham số ở đây là sáu fold ra sai theo cùng một kiểu, mà nhìn thư
    mục thì không thấy gì lạ — nên kiểm ngay lúc đọc.
    """
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    method = doc.get("method")
    if method not in METHODS:
        raise ValueError(f"{p}: 'method' phải là một trong {METHODS}, đang là {method!r}")
    if method == "block":
        frac = float(doc.get("frac", 0.15))
        if not 0 < frac < 1:
            raise ValueError(f"{p}: frac phải trong (0, 1), đang là {frac}")
        if float(doc.get("slack", 0.5)) < 0:
            raise ValueError(f"{p}: slack không được âm")
    elif method == "flight":
        if not doc.get("flights"):
            raise ValueError(f"{p}: method=flight thì phải khai 'flights'")
        if int(doc.get("buffer", 20)) < 0:
            raise ValueError(f"{p}: buffer không được âm")
    else:
        fs = doc.get("fields")
        if fs is not None and not fs:
            raise ValueError(f"{p}: khai 'fields' thì không được để rỗng")
        if doc.get("rotate") is False and not fs:
            raise ValueError(
                f"{p}: rotate=false thì phải khai 'fields'. Không khai mà tắt xoay "
                "vòng thì val luôn là ruộng đầu bảng chữ cái, và ruộng đó gần như "
                "không bao giờ vào tập train.")
    doc["_path"] = p.as_posix()
    return doc


def load_graph(recipe: dict, base: str | Path = ".") -> Graph:
    """Bảng cạnh mà công thức khai. Không khai thì trả đồ thị rỗng — chia vẫn
    chạy, nhưng `graph_scope` sẽ là "none" để không ai đọc nhầm "rò rỉ 0" là
    "sạch"."""
    edges = recipe.get("edges")
    if not edges:
        return overlap_graph.empty()
    return overlap_graph.load(Path(base) / edges)


def apply(recipe: dict, frames: Iterable[flightlog.Frame], graph: Graph, *,
          slot: int | None = None, n_slots: int = 6) -> Assignment:
    """Gọi đúng cách chia mà công thức khai. `slot` là thứ tự lượt (0..5),
    chỉ dùng khi công thức bật `rotate`.

    Mặc định `rotate` khác nhau theo cách, và đó là chủ ý: block mặc định KHÔNG
    xoay (khối nằm cuối đường bay là hành vi gốc), còn field mặc định CÓ xoay
    vì không xoay thì một ruộng bị loại khỏi train ở cả sáu lượt."""
    if recipe["method"] == "block":
        return by_block(frames, graph,
                        frac=float(recipe.get("frac", 0.15)),
                        slack=float(recipe.get("slack", 0.5)),
                        slot=slot if recipe.get("rotate") else None,
                        n_slots=n_slots)
    if recipe["method"] == "flight":
        return by_flight(frames, graph,
                         flights=recipe["flights"],
                         buffer=int(recipe.get("buffer", 20)))
    return by_field(frames, graph,
                    fields=recipe.get("fields"),
                    slot=slot if recipe.get("rotate", True) else None)
