"""Chép nguyên văn phiên làm việc trên terminal ra một file text trong run dir.

Nguyên tắc: file log phải là BẢN SAO ĐÚNG của những gì hiện trên màn hình.
Không gộp dòng, không lọc mã màu, không thêm header. `type run.log` trong
terminal sẽ phát lại y hệt lần chạy, kể cả thanh tiến trình.

Không dùng module logging làm cơ chế: ultralytics, torch và tqdm in thẳng ra
stdout/stderr chứ không đi qua logger nào, nên cách duy nhất bắt được tất cả là
chen vào giữa hai luồng đó.

Một cạm bẫy phải xử lý riêng: logging.StreamHandler GIỮ THAM CHIẾU tới luồng
ngay lúc handler được tạo, nên gán lại sys.stdout không hề tác động tới nó.
Ultralytics dựng LOGGER lúc import — trước khi tee được cài — nên nếu bỏ qua
bước trỏ lại handler, log sẽ mất toàn bộ dòng đi qua LOGGER: banner phiên bản,
bảng kiến trúc model, dòng optimizer, và dòng số liệu val mỗi epoch. tqdm thì
ngược lại, nó tra sys.stdout ở từng lần gọi nên vẫn vào log bình thường.

Xả đĩa sau mỗi lần ghi: log chỉ có ích khi nó đã nằm trên đĩa TRƯỚC lúc tiến
trình chết. Nhờ vậy phân biệt được hai kiểu kết thúc — lỗi Python để lại
traceback trong file, còn bị giết từ bên ngoài thì file đứt ngang.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import sys
import traceback
from pathlib import Path

# Chỉ dùng cho chế độ gộp (raw=False). Mặc định KHÔNG lọc gì.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _Tee(io.TextIOBase):
    """Ghi ra console và ra file. Mặc định file nhận y hệt console."""

    def __init__(self, stream, fh, raw: bool = True):
        self.stream = stream
        self.fh = fh
        self.raw = raw
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            self.stream.write(s)
        except (ValueError, OSError):
            pass  # terminal đóng rồi thì vẫn phải ghi được file
        if self.raw:
            self.fh.write(s)
            self.fh.flush()
        else:
            self._collapse(s)
        return len(s)

    # ------------------------------------------------------------- chế độ gộp
    def _collapse(self, s: str) -> None:
        """Giữ mỗi dòng một lần, bỏ mã màu — dùng khi cần file nhỏ."""
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._write_collapsed(line)
        if "\r" in self._buf:
            self._buf = self._buf.rsplit("\r", 1)[-1]

    def _write_collapsed(self, line: str) -> None:
        self.fh.write(_ANSI.sub("", line.rsplit("\r", 1)[-1]).rstrip() + "\n")
        self.fh.flush()

    def close_buffer(self) -> None:
        if not self.raw and self._buf.strip():
            self._write_collapsed(self._buf)
        self._buf = ""

    # ------------------------------------------------------------- giao diện
    def flush(self) -> None:
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass
        self.fh.flush()

    def isatty(self) -> bool:
        # Trả về đúng trạng thái của console: tqdm dựa vào đây để quyết định vẽ
        # thanh tiến trình hay in từng dòng. Nói dối ở đây là làm thay đổi thứ
        # ultralytics in ra, tức đúng cái phải tránh.
        return getattr(self.stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self.stream, "encoding", "utf-8")

    def writable(self) -> bool:
        return True


def _stream_handlers():
    """Mọi StreamHandler đang tồn tại, kể cả của thư viện bên thứ ba."""
    mgr = logging.root.manager
    loggers = [logging.root] + [
        lg for lg in mgr.loggerDict.values() if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler):
                yield h


def _redirect_log_handlers(old_streams: dict, new_streams: dict) -> list:
    """Trỏ lại các StreamHandler đang bám vào luồng cũ (xem ghi chú đầu file)."""
    changed = []
    for h in _stream_handlers():
        s = getattr(h, "stream", None)
        for key, old in old_streams.items():
            if s is old:
                changed.append((h, old))
                h.setStream(new_streams[key])
                break
    return changed


@contextlib.contextmanager
def capture(path: str | Path, raw: bool = True):
    """Chuyển hướng stdout/stderr vào console VÀ file trong phạm vi khối with.

    raw=True (mặc định): file là bản sao đúng từng ký tự của terminal.
    raw=False: mỗi dòng chỉ giữ trạng thái cuối và bỏ mã màu — file nhỏ hơn
    nhiều nhưng không còn là bản sao trung thực.

    Không ghi thêm header hay footer: xuất xứ của lần chạy đã nằm trong
    env.json và config.yaml cùng thư mục.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline mặc định (dịch "\n" thành xuống dòng của hệ điều hành) là đúng
    # thứ Python làm với stdout, nên file trùng KHỚP TỪNG BYTE với kết quả của
    # `python benchmark/maskrcnn/train.py > terminal.txt`. Ký tự \r của thanh tiến trình
    # không bị dịch nên vẫn nguyên vẹn.
    fh = path.open("w", encoding="utf-8")

    out, err = sys.stdout, sys.stderr
    t_out, t_err = _Tee(out, fh, raw), _Tee(err, fh, raw)
    sys.stdout, sys.stderr = t_out, t_err
    rebound = _redirect_log_handlers(
        {"out": out, "err": err}, {"out": t_out, "err": t_err}
    )
    try:
        yield path
    except BaseException:
        # Python chỉ in traceback SAU khi khối with đã trả lại luồng gốc, nên
        # nó không tự vào file. Ghi thẳng vào file (không qua tee, để console
        # không thấy hai lần) rồi ném tiếp.
        fh.write(traceback.format_exc())
        fh.flush()
        raise
    finally:
        for handler, original in rebound:
            handler.setStream(original)
        t_out.close_buffer()
        t_err.close_buffer()
        sys.stdout, sys.stderr = out, err
        fh.close()
