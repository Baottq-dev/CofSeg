"""In nhật ký bay của một bản xuất: đường bay, ngày giờ, bộ đếm, ranh giới.

    python scripts/inspect_flights.py --export data/export/dataset_v1
    python scripts/inspect_flights.py --export data/export/dataset_v1 --json

Đây là bước xem dữ liệu trước khi chọn cách chia val. Câu hỏi cần trả lời:
những thư mục ta gọi là "đường bay" có thật sự là các chuyến bay riêng không,
hay chỉ là một chuyến liên tục bị cắt thành nhiều thư mục?

Bằng chứng nằm ở bộ đếm của máy bay (bốn số trong tên file). Nếu thư mục sau
bắt đầu ở số ngay sau thư mục trước thì máy bay chưa hạ cánh giữa hai thư mục
— và ranh giới giữa chúng cần được đối xử y như một điểm cắt giữa đường bay.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import console  # noqa: E402
from canopyseg.datasets import flightlog  # noqa: E402

console.setup()


def load(export: Path) -> tuple[list[flightlog.Frame], dict[str, int], int]:
    """Frame của mọi ảnh trong bản xuất + số vùng tán mỗi ảnh."""
    ann = export / "annotations" / "instances.json"
    if not ann.exists():
        raise SystemExit(f"Không thấy {ann}. Cần bản xuất chưa chia (split_by=none).")
    raw = json.loads(ann.read_text(encoding="utf-8"))
    per_image = Counter(int(a["image_id"]) for a in raw["annotations"])
    names = {im["file_name"]: per_image.get(int(im["id"]), 0) for im in raw["images"]}
    fs = flightlog.frames(names)
    return fs, names, len(names) - len(fs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/export/dataset_v1")
    ap.add_argument("--max-gap", type=int, default=flightlog.MAX_COUNTER_GAP,
                    help="bộ đếm chênh tối đa còn coi là một chuyến liên tục")
    ap.add_argument("--json", action="store_true", help="in JSON thay vì bảng")
    a = ap.parse_args(argv)

    export = Path(a.export)
    fs, regions, unparsed = load(export)
    fl = flightlog.flights(fs)
    js = flightlog.junctions(fs, a.max_gap)
    ss = flightlog.sessions(fs, a.max_gap)

    rows = []
    for key in sorted(fl, key=lambda k: fl[k][0].taken):
        seq = fl[key]
        n_reg = sum(regions[f.file_name] for f in seq)
        rows.append({
            "field": key[0], "flight": key[1], "images": len(seq), "regions": n_reg,
            "first_counter": seq[0].counter, "last_counter": seq[-1].counter,
            "start": seq[0].taken.isoformat(sep=" "),
            "end": seq[-1].taken.isoformat(sep=" "),
            "minutes": round((seq[-1].taken - seq[0].taken).total_seconds() / 60, 1),
        })

    if a.json:
        print(json.dumps({
            "export": str(export).replace("\\", "/"),
            "images": len(fs), "unparsed": unparsed,
            "flights": rows,
            "junctions": [j.__dict__ for j in js],
            "sessions": [["/".join(k) for k in chain] for chain in ss],
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"{export}: {len(fs)} ảnh, {sum(r['regions'] for r in rows)} vùng tán, "
          f"{len(fl)} đường bay, {len(ss)} phiên bay")
    if unparsed:
        print(f"  {unparsed} tên file không theo quy ước, bỏ qua")

    print(f"\n{'đường bay':16s} {'ảnh':>4s} {'vùng':>6s} {'vùng/ảnh':>9s} "
          f"{'đếm đầu':>8s} {'đếm cuối':>9s}  {'bắt đầu':16s} {'phút':>5s}")
    for r in rows:
        print(f"{r['field'] + '/' + r['flight']:16s} {r['images']:4d} {r['regions']:6d} "
              f"{r['regions'] / r['images']:9.1f} {r['first_counter']:8d} "
              f"{r['last_counter']:9d}  {r['start'][:16]:16s} {r['minutes']:5.1f}")

    print("\nRanh giới nối liền (bộ đếm chạy tiếp qua hai thư mục):")
    if not js:
        print("  không có — mỗi thư mục là một chuyến bay riêng")
    for j in js:
        print(f"  {j.field}/{j.before} -> {j.field}/{j.after}: "
              f"bộ đếm chênh {j.counter_gap} khung, cách {j.seconds / 60:.0f} phút"
              f"   <- máy bay KHÔNG hạ cánh giữa hai thư mục này")

    print("\nPhiên bay liên tục:")
    for chain in ss:
        head = flightlog.flights(fs)[chain[0]][0]
        print(f"  {head.taken:%d/%m %H:%M}  " + "  ->  ".join("/".join(k) for k in chain))

    if js:
        print("\nHệ quả cho việc chia val: lấy trọn một thư mục ở trên làm val thì "
              "ranh giới\nnối liền của nó cần khoảng đệm, y như cắt giữa một đường bay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
