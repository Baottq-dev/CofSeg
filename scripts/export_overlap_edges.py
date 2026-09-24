"""Xuất đồ thị chồng lấn đã đo sang configs/ để mã dùng được ở máy khác.

    python scripts/export_overlap_edges.py
    python scripts/export_overlap_edges.py --threshold 0.5 --out configs/dataset/overlap_50.json
    python scripts/export_overlap_edges.py --dry-run

Đọc mọi `runs/overlap/*/pairs.csv` (do scripts/measure_overlap.py sinh ra),
đổi tên ảnh sang tên đã làm phẳng của bản xuất, rồi ghi
`configs/dataset/overlap_edges.json`.

`runs/` nằm ngoài git. Nếu không xuất thì việc chia val phải chạy lại COLMAP
ở mọi máy, và con số trong báo cáo không ai kiểm lại được.

File ghi ra giữ cả mẫu số — mỗi khoảng cách khung hình đã so bao nhiêu cặp —
nên xác suất chồng lấn p(k) tính lại được từ chính file đó.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.datasets import overlap_graph as og  # noqa: E402

console.setup()

ROOT = Path(__file__).resolve().parent.parent


def flat_name(field: str, flight: str, base: str) -> str:
    """('field_2', '10/1', 'DJI_x.jpg') -> 'field_2__10__1__DJI_x.jpg'."""
    return f"{field}__{flight.replace('/', '__')}__{base}"


def build(runs: list[Path], known: set[str], threshold: float) -> og.Graph:
    g = og.Graph(threshold=threshold)
    scopes, missing = set(), set()
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        field = summary["field"]
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        scopes.add(cfg.get("scope", "flight"))
        here = run.resolve()
        g.sources.append(here.relative_to(ROOT).as_posix()
                         if here.is_relative_to(ROOT) else here.as_posix())
        with (run / "pairs.csv").open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                a = flat_name(field, r["flight"], r["a"])
                b = flat_name(field, r["flight"], r["b"])
                if a not in known or b not in known:
                    missing.update(x for x in (a, b) if x not in known)
                    continue
                gap = int(r["seq_gap"])
                g.pairs_by_gap[gap] = g.pairs_by_gap.get(gap, 0) + 1
                ov = max(float(r["overlap_ab"] or 0), float(r["overlap_ba"] or 0))
                if ov >= threshold:
                    g.edges_by_gap[gap] = g.edges_by_gap.get(gap, 0) + 1
                    g.add(a, b, ov, gap)
    g.scope = scopes.pop() if len(scopes) == 1 else "mixed"
    g.missing = sorted(missing)
    return g


def report(g: og.Graph, known: set[str]) -> None:
    deg = Counter(len(g.adj.get(n, {})) for n in known)
    isolated = deg.get(0, 0)
    total_deg = sum(k * v for k, v in deg.items())
    print(f"{g.n_edges} cạnh >= {g.threshold:.0%} trên {len(known)} ảnh "
          f"(scope={g.scope})")
    print(f"  bậc trung bình {total_deg / len(known):.2f}, "
          f"{isolated} ảnh không có hàng xóm nào, bậc lớn nhất {max(deg)}")
    if g.missing:
        print(f"  {len(g.missing)} ảnh có trong lần đo nhưng không có trong bản xuất, bỏ qua")

    print(f"\n{'cách k khung':>13s} {'cặp đã so':>10s} {'cặp chồng lấn':>14s} "
          f"{'p(k)':>7s}")
    shown = 0
    for gap in sorted(g.pairs_by_gap):
        n, e = g.pairs_by_gap[gap], g.edges_by_gap.get(gap, 0)
        if gap > 12 and shown >= 12:
            break
        print(f"{gap:13d} {n:10d} {e:14d} {100 * e / n:6.1f}%")
        shown += 1
    far_n = sum(v for k, v in g.pairs_by_gap.items() if k > 12)
    far_e = sum(v for k, v in g.edges_by_gap.items() if k > 12)
    if far_n:
        print(f"{'>12':>13s} {far_n:10d} {far_e:14d} {100 * far_e / far_n:6.1f}%")

    tot = sum(g.edges_by_gap.values())
    near = sum(v for k, v in g.edges_by_gap.items() if k <= 2)
    print(f"\nTrong {tot} cặp chồng lấn: {near} cặp cách <= 2 khung ({100 * near / tot:.0f}%), "
          f"{tot - near} cặp cách >= 3 khung ({100 * (tot - near) / tot:.0f}%).")
    print("Khoảng cách trong chuỗi ảnh không phải khoảng cách trên mặt đất: máy bay "
          "bay luống,\nquay đầu là chụp lại ngay cạnh chỗ cũ sau cả chục khung hình.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/export/dataset_v1",
                    help="bản xuất chưa chia, để lấy danh sách tên ảnh hợp lệ")
    ap.add_argument("--runs", default="runs/overlap", help="thư mục chứa các lần đo")
    ap.add_argument("--out", default="configs/dataset/overlap_edges.json")
    ap.add_argument("--threshold", type=float, default=og.DEFAULT_THRESHOLD)
    ap.add_argument("--dry-run", action="store_true", help="in thống kê, không ghi file")
    a = ap.parse_args(argv)

    ann = Path(a.export) / "annotations" / "instances.json"
    if not ann.exists():
        raise SystemExit(f"Không thấy {ann}. Cần bản xuất chưa chia (split_by=none).")
    known = {im["file_name"] for im in json.loads(ann.read_text(encoding="utf-8"))["images"]}

    runs = sorted(p for p in Path(a.runs).glob("*") if (p / "pairs.csv").exists())
    if not runs:
        raise SystemExit(
            f"Không thấy lần đo nào trong {a.runs}.\n"
            "    python scripts/measure_overlap.py --field field_1 --pairs all --method colmap")
    print(f"{len(runs)} lần đo: " + ", ".join(r.name for r in runs) + "\n")

    g = build(runs, known, a.threshold)
    report(g, known)
    if a.dry_run:
        print("\n(--dry-run: chưa ghi file)")
        return 0
    out = og.save(g, a.out)
    print(f"\n-> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
