"""Đồ thị chồng lấn giữa các ảnh: đọc, lưu, và các phép hỏi khi chia tập.

Mỗi ảnh là một đỉnh, hai ảnh chồng lấn từ `threshold` trở lên thì có một
cạnh. Số liệu gốc là `runs/overlap/*/pairs.csv` do `scripts/measure_overlap.py`
sinh ra bằng COLMAP; `runs/` nằm ngoài git nên phải xuất sang
`configs/dataset/overlap_edges.json` thì mã mới dùng được ở máy khác.

Ngoài danh sách cạnh, file còn giữ **mẫu số**: mỗi khoảng cách khung hình đã
so bao nhiêu cặp. Không có nó thì không tính lại được xác suất chồng lấn
p(k), mà p(k) là thứ trả lời câu "đệm mấy ảnh là đủ" — nên nó phải đi theo
dữ liệu chứ không nằm trong đầu ai.

Một giới hạn phải nhớ mỗi lần đọc con số từ đây: lần đo dùng `scope=flight`,
tức CHỈ so các cặp trong cùng một đường bay. Hai đường bay khác nhau của cùng
một ruộng chưa được so. `Graph.scope` giữ lại dữ kiện đó để nơi dùng còn biết
"rò rỉ 0" nghĩa là "0 trong phạm vi đã đo".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_THRESHOLD = 0.30


@dataclass
class Graph:
    """Đồ thị chồng lấn, cộng với phần thống kê cần để ước lượng."""

    threshold: float = DEFAULT_THRESHOLD
    scope: str = "flight"
    adj: dict[str, dict[str, float]] = dc_field(default_factory=dict)
    gaps: dict[tuple[str, str], int] = dc_field(default_factory=dict)
    pairs_by_gap: dict[int, int] = dc_field(default_factory=dict)
    edges_by_gap: dict[int, int] = dc_field(default_factory=dict)
    sources: list[str] = dc_field(default_factory=list)
    missing: list[str] = dc_field(default_factory=list)

    # ---------------------------------------------------------------- cạnh
    def add(self, a: str, b: str, overlap: float, gap: int | None = None) -> None:
        if overlap < self.threshold or a == b:
            return
        for x, y in ((a, b), (b, a)):
            if overlap > self.adj.setdefault(x, {}).get(y, 0.0):
                self.adj[x][y] = overlap
        if gap is not None:
            self.gaps[tuple(sorted((a, b)))] = gap

    def neighbours(self, name: str, threshold: float | None = None) -> dict[str, float]:
        thr = self.threshold if threshold is None else threshold
        return {k: v for k, v in self.adj.get(name, {}).items() if v >= thr}

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    def touching(self, block: Iterable[str], outside: set[str],
                 threshold: float | None = None) -> set[str]:
        """Ảnh trong `outside` có cạnh nối vào `block` — tức phần phải bỏ nếu
        muốn `block` sạch."""
        thr = self.threshold if threshold is None else threshold
        out: set[str] = set()
        for k in block:
            out |= {nb for nb, ov in self.adj.get(k, {}).items()
                    if ov >= thr and nb in outside}
        return out

    def dirty(self, block: Iterable[str], outside: set[str],
              threshold: float | None = None) -> list[str]:
        """Ảnh trong `block` còn dính `outside` — tức val bẩn."""
        thr = self.threshold if threshold is None else threshold
        return sorted(k for k in block
                      if any(ov >= thr and nb in outside
                             for nb, ov in self.adj.get(k, {}).items()))

    def crossing(self, block: set[str], outside: set[str]) -> int:
        """Số cạnh bắc cầu giữa `block` và `outside`."""
        return sum(1 for k in block
                   for nb, ov in self.adj.get(k, {}).items()
                   if ov >= self.threshold and nb in outside)

    def cut_profile(self, seq: Sequence[str], lo: int, hi: int) -> list[tuple[int, int]]:
        """Với mỗi điểm cắt trong [lo, hi], đếm số cạnh bị cắt nếu lấy
        seq[cut:] làm một khối riêng. Đây là cách tìm "chỗ mỏng"."""
        whole = set(seq)
        out = []
        for cut in range(lo, hi + 1):
            block = set(seq[cut:])
            out.append((cut, self.crossing(block, whole - block)))
        return out

    # --------------------------------------------------------- xác suất
    def p_overlap(self, gap: int) -> float:
        """Xác suất hai ảnh cách nhau `gap` khung hình chồng lấn >= ngưỡng.

        Dùng mẫu số đã đo, nên chỉ có nghĩa với khoảng cách đã có đủ cặp so.
        Ngoài phạm vi đó trả 0 — ước thấp còn hơn bịa.
        """
        n = self.pairs_by_gap.get(gap, 0)
        return self.edges_by_gap.get(gap, 0) / n if n else 0.0

    def expected_leak(self, distance: int, dropped: int = 0, span: int = 80) -> float:
        """Ước số ảnh val bị nhiễm ở MỘT ranh giới, sau khi bỏ `dropped` ảnh
        train sát ranh giới đó.

        Ảnh val thứ j và ảnh train thứ i (đếm lùi từ ranh giới) cách nhau
        distance + i + j khung hình. Coi các cặp độc lập nhau thì ảnh val thứ
        j sạch với xác suất tích (1 - p), nên kỳ vọng số ảnh bẩn là tổng phần
        bù. Giả định độc lập là chỗ yếu của ước lượng này — nó cho con số để
        chọn khoảng đệm, không thay được phép đo thật.
        """
        return sum(1.0 - math.prod(1.0 - self.p_overlap(distance + i + j)
                                   for i in range(dropped, span))
                   for j in range(span))

    # ------------------------------------------------------------ đọc/ghi
    def to_json(self) -> dict:
        edges = []
        for a, nbs in self.adj.items():
            for b, ov in nbs.items():
                if a < b:
                    edges.append([a, b, round(ov, 4), self.gaps.get((a, b), -1)])
        edges.sort()
        return {
            "threshold": self.threshold,
            "scope": self.scope,
            "note": ("scope=flight: chỉ so các cặp trong cùng một đường bay. "
                     "Chồng lấn giữa hai đường bay của cùng một ruộng CHƯA đo."),
            "sources": self.sources,
            "images_not_in_export": self.missing,
            "pairs_by_gap": {str(k): v for k, v in sorted(self.pairs_by_gap.items())},
            "edges_by_gap": {str(k): v for k, v in sorted(self.edges_by_gap.items())},
            "edges": edges,
        }

    @classmethod
    def from_json(cls, doc: dict) -> "Graph":
        g = cls(threshold=float(doc.get("threshold", DEFAULT_THRESHOLD)),
                scope=str(doc.get("scope", "flight")),
                sources=list(doc.get("sources") or []),
                missing=list(doc.get("images_not_in_export") or []))
        g.pairs_by_gap = {int(k): int(v) for k, v in (doc.get("pairs_by_gap") or {}).items()}
        g.edges_by_gap = {int(k): int(v) for k, v in (doc.get("edges_by_gap") or {}).items()}
        for a, b, ov, gap in doc.get("edges") or []:
            g.add(a, b, float(ov), int(gap) if gap is not None and gap >= 0 else None)
        return g


def load(path: str | Path) -> Graph:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Không thấy {p}. Sinh ra bằng: python scripts/export_overlap_edges.py")
    return Graph.from_json(json.loads(p.read_text(encoding="utf-8")))


def save(graph: Graph, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = graph.to_json()
    # Một cạnh một dòng: file này để người đọc và grep được, không chỉ để máy.
    body = ",\n    ".join(json.dumps(e, ensure_ascii=False) for e in doc.pop("edges"))
    head = json.dumps(doc, indent=2, ensure_ascii=False)[:-2].rstrip()
    p.write_text(f"{head},\n  \"edges\": [\n    {body}\n  ]\n}}\n", encoding="utf-8")
    return p


def empty(threshold: float = DEFAULT_THRESHOLD) -> Graph:
    """Đồ thị rỗng: cho phép chia tập ở máy chưa xuất bảng cạnh, với điều kiện
    nơi gọi biết rằng mọi con số rò rỉ khi đó là 0 vì KHÔNG BIẾT, không phải
    vì sạch."""
    return Graph(threshold=threshold, scope="none")
