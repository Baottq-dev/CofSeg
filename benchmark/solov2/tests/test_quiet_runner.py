"""Dựng Runner của mmengine mà không để nó đổ 480 dòng ra màn hình.

Cả hai chuyện phải xử lý đều nằm BÊN TRONG `Runner.from_cfg`, nên bịt log sau
khi nó trả về là muộn:

1. `from_cfg` in bảng môi trường, toàn bộ config và bảng thứ tự hook.
2. `MMLogger` tạo `StreamHandler(stream=sys.stdout)` và GIỮ tham chiếu tới
   luồng ngay lúc tạo. Hứng stdout xong mà không trỏ handler về chỗ cũ thì mọi
   dòng về sau rơi vào một StringIO đã bỏ đi — im lặng hoàn toàn, kể cả cảnh
   báo, và người chạy không biết.
"""

from __future__ import annotations

import io
import logging
import sys

from cofseg.training.mmdet import MMDetTrainer


class _FakeRunner:
    """Bắt chước đúng hai hành vi đáng quan tâm của Runner thật."""

    def __init__(self, logger):
        self.logger = logger

    @classmethod
    def from_cfg(cls, cfg):
        # MMLogger: không đăng ký vào logging.Logger.manager, và bám vào
        # sys.stdout TẠI THỜI ĐIỂM NÀY.
        logger = logging.Logger("mmengine")
        h = logging.StreamHandler(stream=sys.stdout)
        h.setLevel(logging.DEBUG)
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        logger.info("bảng môi trường và toàn bộ config, 480 dòng")
        return cls(logger)


def _console(runner):
    return [h for h in runner.logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)][0]


def test_dump_luc_dung_runner_khong_ra_man_hinh(capsys):
    runner = MMDetTrainer._build_runner(_FakeRunner, None, quiet=True)
    assert "480 dòng" not in capsys.readouterr().out


def test_handler_duoc_tra_ve_stdout_that(capsys):
    """Không trỏ lại thì im lặng TOÀN BỘ phần sau, kể cả cảnh báo."""
    runner = MMDetTrainer._build_runner(_FakeRunner, None, quiet=True)
    h = _console(runner)
    assert h.stream is sys.stdout, "handler còn bám vào bộ đệm đã bỏ đi"

    h.setLevel(logging.DEBUG)          # bỏ phần hush để kiểm riêng đường ra
    runner.logger.warning("dòng này phải thấy được")
    assert "dòng này phải thấy được" in capsys.readouterr().out


def test_hush_cham_dung_logger_that():
    """Bịt theo tên sẽ trượt; _build_runner phải đưa thẳng đối tượng."""
    runner = MMDetTrainer._build_runner(_FakeRunner, None, quiet=True)
    assert _console(runner).level == logging.WARNING


def test_verbose_thi_khong_dung_gi_ca(capsys):
    runner = MMDetTrainer._build_runner(_FakeRunner, None, quiet=False)
    assert "480 dòng" in capsys.readouterr().out
    assert _console(runner).level == logging.DEBUG
