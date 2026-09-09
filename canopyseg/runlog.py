"""Ghi lại toàn bộ những gì hiện trên terminal vào một file text trong run dir.

Không dùng module logging: ultralytics, torch và tqdm in thẳng ra stdout/stderr
chứ không đi qua logger nào, nên cách duy nhất bắt được TẤT CẢ là chen vào giữa
hai luồng đó.

Ba việc phải xử lý, nếu không file log sẽ vô dụng:

1. THANH TIẾN TRÌNH. tqdm vẽ lại một dòng bằng ký tự \\r hàng chục lần mỗi
   giây. Ghi thô thì một epoch thành vài megabyte trên đúng một dòng. Ở đây
   chỉ giữ trạng thái CUỐI của mỗi dòng — mỗi epoch còn lại một dòng gọn.

2. MÃ MÀU ANSI. Trên terminal chúng là màu, trong file text chúng là rác dạng
   ESC[34m. Lọc bỏ.

3. XẢ NGAY. Lần train hôm trước chết ở epoch 24 mà không để lại dấu vết nào.
   Log chỉ có ích khi nó đã nằm trên đĩa TRƯỚC lúc tiến trình chết, nên xả đĩa
   sau mỗi dòng hoàn chỉnh.

Điểm thứ ba cho thêm một thứ chẩn đoán: nếu tiến trình chết vì lỗi Python thì
log kết thúc bằng traceback; nếu bị giết từ bên ngoài (hết commit, Ctrl+C ở
tầng hệ điều hành) thì log đứt giữa chừng, không có dòng kết. Hai trường hợp
phân biệt được ngay khi mở file.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# CSI: ESC [ ... chữ cái cuối. Đủ cho màu, in đậm, xoá dòng của tqdm.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _Tee(io.TextIOBase):
    """Ghi ra console y nguyên, ra file thì đã dọn sạch."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh
        self._buf = ""

    # Console phải nhận nguyên bản: giữ màu và thanh tiến trình như khi chạy tay.
    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            self.stream.write(s)
        except (ValueError, OSError):
            pass  # terminal đã đóng thì vẫn phải ghi được file
        self._to_file(s)
        return len(s)

    def _to_file(self, s: str) -> None:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._write_line(line)
        # Phần đuôi chưa xuống dòng mà đã bị \r đè: bỏ phần cũ đi, nếu không
        # bộ đệm phình theo từng nhịp cập nhật của thanh tiến trình.
        if "\r" in self._buf:
            self._buf = self._buf.rsplit("\r", 1)[-1]

    def _write_line(self, line: str) -> None:
        # Chỉ giữ đoạn sau dấu \r cuối cùng — đó là thứ người dùng thật sự thấy.
        text = _ANSI.sub("", line.rsplit("\r", 1)[-1]).rstrip()
        self.fh.write(text + "\n")
        self.fh.flush()

    def flush(self) -> None:
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass
        self.fh.flush()

    def close_buffer(self) -> None:
        if self._buf.strip():
            self._write_line(self._buf)
        self._buf = ""

    # tqdm và một số thư viện hỏi hai thuộc tính này trước khi vẽ.
    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self.stream, "encoding", "utf-8")

    def writable(self) -> bool:
        return True


@contextlib.contextmanager
def capture(path: str | Path, header: dict | None = None):
    """Chuyển hướng stdout/stderr vào console VÀ file, trong phạm vi khối with."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8")
    t0 = time.time()

    fh.write(f"# bắt đầu   : {datetime.now().isoformat(timespec='seconds')}\n")
    fh.write(f"# lệnh      : {' '.join(sys.argv)}\n")
    for k, v in (header or {}).items():
        fh.write(f"# {k:<10}: {v}\n")
    fh.write("#" + "-" * 70 + "\n")
    fh.flush()

    out, err = sys.stdout, sys.stderr
    t_out, t_err = _Tee(out, fh), _Tee(err, fh)
    sys.stdout, sys.stderr = t_out, t_err
    status = "hoàn tất"
    try:
        yield path
    except KeyboardInterrupt:
        # Ghi rõ để lần sau phân biệt được "người dừng" với "bị giết".
        status = "NGƯỜI DỪNG (Ctrl+C)"
        traceback.print_exc()
        raise
    except BaseException:
        status = "LỖI"
        traceback.print_exc()
        raise
    finally:
        t_out.close_buffer()
        t_err.close_buffer()
        sys.stdout, sys.stderr = out, err
        fh.write("#" + "-" * 70 + "\n")
        fh.write(f"# kết thúc  : {status} sau {time.time() - t0:.1f}s\n")
        fh.close()
