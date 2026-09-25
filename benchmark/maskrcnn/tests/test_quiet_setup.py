"""Gọi `default_setup` của detectron2 mà không để nó đổ config ra màn hình.

Đo trên một lượt Mask2Former thật: 452 / 655 dòng (69%) là bảng môi trường và
bản dump config, cả hai in BÊN TRONG `default_setup`. Bịt log sau khi nó trả về
là muộn — đúng lỗi đã mắc ở đây trước đó.

Cái bẫy thứ hai giống hệt mmengine: `setup_logger` tạo
`StreamHandler(stream=sys.stdout)` và giữ tham chiếu ngay lúc tạo, mà nó chạy
bên trong `default_setup`, tức lúc stdout đang bị hứng. Không trỏ lại thì mọi
dòng về sau im lặng hoàn toàn, kể cả "Skip loading parameter ...".
"""

from __future__ import annotations

import logging
import sys

import pytest

from cofseg.training.detectron2 import Detectron2Trainer


@pytest.fixture(autouse=True)
def _sach():
    """detectron2/fvcore là logger toàn cục; trả lại nguyên trạng sau mỗi test."""
    cu = {n: list(logging.getLogger(n).handlers) for n in ("detectron2", "fvcore")}
    yield
    for n, hs in cu.items():
        lg = logging.getLogger(n)
        lg.handlers[:] = hs


def _fake_default_setup(cfg, args):
    """Bắt chước đúng hai hành vi đáng quan tâm của default_setup thật."""
    for n in ("detectron2", "fvcore"):
        lg = logging.getLogger(n)
        lg.handlers[:] = []
        h = logging.StreamHandler(stream=sys.stdout)   # bám vào stdout LÚC NÀY
        h.setLevel(logging.DEBUG)
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    logging.getLogger("detectron2").info("bảng môi trường và full config, 452 dòng")


def _console(name="detectron2"):
    return [h for h in logging.getLogger(name).handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)][0]


def test_dump_config_khong_ra_man_hinh(capsys):
    Detectron2Trainer._default_setup(_fake_default_setup, None, quiet=True)
    assert "452 dòng" not in capsys.readouterr().out


def test_handler_duoc_tra_ve_stdout_that(capsys):
    """Không trỏ lại thì im lặng TOÀN BỘ phần sau, kể cả cảnh báo."""
    Detectron2Trainer._default_setup(_fake_default_setup, None, quiet=True)
    h = _console()
    assert h.stream is sys.stdout, "handler còn bám vào bộ đệm đã bỏ đi"

    h.setLevel(logging.DEBUG)          # bỏ phần hush để kiểm riêng đường ra
    logging.getLogger("detectron2").warning("Skip loading parameter ...")
    assert "Skip loading parameter" in capsys.readouterr().out


def test_ca_fvcore_lan_detectron2_deu_duoc_bit():
    """'Skip loading parameter' là của fvcore; quên nó là mất cảnh báo đáng đọc."""
    Detectron2Trainer._default_setup(_fake_default_setup, None, quiet=True)
    for n in ("detectron2", "fvcore"):
        assert _console(n).level == logging.WARNING
        assert _console(n).stream is sys.stdout


def test_traceback_cua_train_loop_chi_in_mot_lan(capsys):
    """detectron2 log exception rồi raise lại; Python in bản đầy đủ hơn ở ngoài."""
    Detectron2Trainer._default_setup(_fake_default_setup, None, quiet=True)
    lg = logging.getLogger("detectron2.engine.train_loop")
    lg.error("Exception during training:\nTraceback (most recent call last): ...")
    lg.error("Một lỗi khác hẳn, phải thấy được")
    out = capsys.readouterr().out
    assert "Exception during training" not in out
    assert "Một lỗi khác hẳn" in out


def test_verbose_thi_khong_dung_gi_ca(capsys):
    Detectron2Trainer._default_setup(_fake_default_setup, None, quiet=False)
    assert "452 dòng" in capsys.readouterr().out
    assert _console().level == logging.DEBUG
