"""Đọc nhật ký bay ra từ tên file ảnh đã làm phẳng.

Tên ảnh trong bản xuất mang đủ thông tin để dựng lại chuyến bay:

    field_1__10__1__DJI_20260301075324_0001_D.jpg
    └ruộng┘ └bay┘   └ thời điểm ──┘ └đếm┘

Hai con số cuối là thứ đáng giá nhất và dễ bỏ qua nhất:

- **thời điểm** cho biết ảnh chụp lúc nào, nên biết được hai ảnh cách nhau bao
  lâu kể cả khi số thứ tự liền nhau (máy bay dừng giữa chừng).
- **bộ đếm của máy bay** cho biết hai thư mục có phải là một chuyến bay liên
  tục bị cắt đôi hay không. Nếu thư mục sau bắt đầu ở số ngay sau thư mục
  trước thì máy bay chưa hạ cánh: "hai đường bay" thật ra là một.

Điều đó quyết định cách chia val. Lấy trọn một thư mục làm val mà thư mục đó
nối liền thư mục khác đang ở train thì ranh giới giữa chúng rò rỉ y hệt như
cắt giữa một đường bay — chỉ khác là điểm cắt do người xuất dữ liệu chọn hộ.

    from canopyseg.datasets import flightlog
    frames = flightlog.frames(names)
    for j in flightlog.junctions(frames):
        print(j.before, j.after, j.counter_gap)
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable

#: field_1__10__1__DJI_20260301075324_0001_D.jpg
_DJI = re.compile(r"^DJI_(\d{14})_(\d{3,5})_\w+\.\w+$")

#: Bộ đếm nhảy quá xa thì coi như hai chuyến khác nhau chứ không phải một
#: chuyến bị cắt. 30 khung ở nhịp ~10 s/ảnh là khoảng 5 phút bay.
MAX_COUNTER_GAP = 30


@dataclass(frozen=True)
class Frame:
    """Một ảnh, kèm chỗ đứng của nó trong chuyến bay."""

    file_name: str
    field: str
    flight: str
    base: str
    taken: dt.datetime
    counter: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.field, self.flight)


@dataclass(frozen=True)
class Junction:
    """Ranh giới giữa hai thư mục nối liền nhau trong cùng một chuyến bay."""

    field: str
    before: str          # đường bay kết thúc
    after: str           # đường bay bắt đầu
    counter_gap: int     # số khung hình máy bay chụp mà bản xuất không có
    seconds: float       # thời gian giữa ảnh cuối và ảnh đầu


def parse(file_name: str) -> Frame | None:
    """Tên ảnh đã làm phẳng -> Frame. Trả None nếu tên không theo quy ước."""
    parts = file_name.split("__")
    if len(parts) < 3:
        return None
    m = _DJI.match(parts[-1])
    if not m:
        return None
    try:
        taken = dt.datetime.strptime(m[1], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return Frame(
        file_name=file_name,
        field=parts[0],
        flight="/".join(parts[1:-1]),
        base=parts[-1],
        taken=taken,
        counter=int(m[2]),
    )


def frames(names: Iterable[str]) -> list[Frame]:
    """Bỏ qua tên không đọc được thay vì nổ: một bản xuất lẫn file lạ vẫn
    phân tích được phần còn lại, và `inspect_flights.py` sẽ đếm phần bỏ qua."""
    out = [parse(n) for n in names]
    return [f for f in out if f is not None]


def flights(fs: Iterable[Frame]) -> dict[tuple[str, str], list[Frame]]:
    """Gom theo (ruộng, đường bay), mỗi nhóm sắp theo thứ tự chụp."""
    out: dict[tuple[str, str], list[Frame]] = {}
    for f in fs:
        out.setdefault(f.key, []).append(f)
    for k in out:
        out[k].sort(key=lambda f: (f.taken, f.counter))
    return out


def sequence(fs: Iterable[Frame]) -> list[Frame]:
    """Một chuỗi ảnh đã sắp theo thứ tự chụp."""
    return sorted(fs, key=lambda f: (f.taken, f.counter))


def junctions(fs: Iterable[Frame], max_gap: int = MAX_COUNTER_GAP) -> list[Junction]:
    """Các chỗ hai thư mục thật ra là một chuyến bay liên tục.

    Xét từng ruộng, sắp các đường bay theo thời gian, rồi hỏi: bộ đếm của máy
    bay có chạy tiếp qua ranh giới không? Chạy tiếp (chênh 1..max_gap) nghĩa
    là máy bay không hạ cánh giữa hai thư mục.

    Bộ đếm nhảy về 1 (đổi ngày, đổi thẻ nhớ) cho chênh âm, không phải ranh
    giới liên tục — hai lần bay đó có chung mảnh đất hay không thì tên file
    không trả lời được, phải đo chồng lấn.
    """
    by_field: dict[str, dict[tuple[str, str], list[Frame]]] = {}
    for key, seq in flights(fs).items():
        by_field.setdefault(key[0], {})[key] = seq

    out: list[Junction] = []
    for field, fl in by_field.items():
        order = sorted(fl, key=lambda k: fl[k][0].taken)
        for a, b in zip(order, order[1:]):
            gap = fl[b][0].counter - fl[a][-1].counter
            if not 0 < gap <= max_gap:
                continue
            out.append(Junction(
                field=field,
                before=a[1],
                after=b[1],
                counter_gap=gap,
                seconds=(fl[b][0].taken - fl[a][-1].taken).total_seconds(),
            ))
    return sorted(out, key=lambda j: (j.field, j.before))


def sessions(fs: Iterable[Frame], max_gap: int = MAX_COUNTER_GAP) -> list[list[tuple[str, str]]]:
    """Gom các đường bay nối liền nhau thành từng phiên bay liên tục.

    Trả về danh sách phiên, mỗi phiên là danh sách khoá đường bay theo thứ tự
    bay. Đường bay đứng một mình cũng là một phiên một phần tử.
    """
    joined = {(j.field, j.before): (j.field, j.after) for j in junctions(fs, max_gap)}
    fl = flights(fs)
    later = set(joined.values())
    out: list[list[tuple[str, str]]] = []
    for key in sorted(fl, key=lambda k: fl[k][0].taken):
        if key in later:
            continue                      # sẽ được nối vào phiên của đường bay trước
        chain = [key]
        while chain[-1] in joined:
            chain.append(joined[chain[-1]])
        out.append(chain)
    return out
