"""Cho phép in tiếng Việt ra console Windows.

Console mặc định của Windows dùng cp1252, gặp chữ có dấu là ném
UnicodeEncodeError giữa chừng — mất luôn cả kết quả đã tính xong.

Chỉ SCRIPT được gọi hàm này. Thư viện không tự đổi stdout khi được import:
đó là tác dụng phụ toàn cục, và code gọi thư viện có quyền tự quyết.
"""

from __future__ import annotations

import sys


def setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
