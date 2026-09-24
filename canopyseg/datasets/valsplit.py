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

Hai cách đang dùng, xem `configs/dataset/val_block.yaml` và `val_flight.yaml`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dc_field
from typing import Iterable, Sequence

from . import flightlog
from .overlap_graph import Graph


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
