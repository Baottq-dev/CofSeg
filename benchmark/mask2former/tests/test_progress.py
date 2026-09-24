"""Phần thuần Python của cofseg/progress.py.

Ở đây kiểm được tất cả những gì quan trọng mà không cần detectron2, mmdet hay
GPU: cách dựng dòng epoch, cách gộp khối tổng kết, cách đọc kết quả prepare(),
và quan trọng nhất là hush() không được gỡ nhầm đường ghi ra file.
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

import pytest

from cofseg import progress


# ------------------------------------------------------------------ định dạng
def test_fmt_time_duoi_mot_gio_khong_co_phan_gio():
    assert progress.fmt_time(132.4) == "2:12"
    assert progress.fmt_time(0) == "0:00"
    assert progress.fmt_time(59.6) == "1:00"


def test_fmt_time_tren_mot_gio():
    assert progress.fmt_time(4521) == "1:15:21"


def test_fmt_time_khong_co_so_thi_khong_gay():
    assert progress.fmt_time(None) == "—"


def test_epoch_line_co_du_cac_phan():
    line = progress.epoch_line(1, 3, 30, 90, loss=2.627, lr=5.8e-05, mem=9.3,
                               metrics={"mAP50-95": 3.7593, "mAP50": 13.5186})
    assert "epoch 1/3" in line
    assert "iter 30/90" in line
    assert "loss 2.627" in line
    assert "mAP50-95   3.76" in line
    assert "mAP50  13.52" in line
    assert "tốt nhất" not in line


def test_epoch_line_danh_dau_epoch_tot_nhat():
    assert progress.epoch_line(2, 3, 60, 90, best=True).endswith("* tốt nhất")


def test_epoch_line_thang_hang_khi_so_epoch_nhieu_chu_so():
    """Cột phải thẳng giữa epoch 9 và epoch 10, nếu không thì bảng lệch."""
    a = progress.epoch_line(9, 50, 270, 1500)
    b = progress.epoch_line(10, 50, 300, 1500)
    assert len(a) == len(b)
    assert a.index("iter") == b.index("iter")


def test_epoch_line_bo_qua_chi_so_khong_co():
    """Epoch không chấm val thì không được bịa ra số."""
    line = progress.epoch_line(1, 3, 30, 90, loss=1.0, metrics={"mAP50": 12.0})
    assert "mAP50 " in line
    assert "mAP50-95" not in line


def test_summary_gach_ngang_bao_het_dong_dai_nhat():
    out = progress.summary("t", [("a", "giá trị rất dài " * 3)])
    rows = out.strip().splitlines()
    assert len(rows[0]) >= max(len(r) for r in rows)


def test_summary_bo_dong_rong():
    out = progress.summary("t", [("có", "1"), ("không", None), ("rỗng", "")])
    assert "có" in out and "không" not in out and "rỗng" not in out


# ------------------------------------------------------------------- dữ liệu
def test_data_line_gom_ba_tap_va_ten_field_test():
    info = {"train": {"images": 470, "regions": 7101},
            "val": {"images": 85, "regions": 1078},
            "test": {"images": 280, "regions": 3720, "fields": {"field_1": 280}}}
    line = progress.data_line(info)
    assert "train 470" in line and "val 85" in line and "test 280" in line
    assert "(field_1)" in line


def test_data_line_tra_none_cho_dang_khac():
    """prepare() của YOLO trả về hình dạng khác; gọi vào đây phải trả None để
    nơi gọi rơi về cách in cũ, chứ không phải nổ."""
    assert progress.data_line({"data_yaml": "x", "splits": {"train": "a"}}) is None
    assert progress.data_line({}) is None
    assert progress.data_line(None) is None


# -------------------------------------------------------------------- logger
def test_hush_bit_info_ra_man_hinh_nhung_giu_canh_bao(tmp_path, capsys):
    """Cốt lõi của module: giấu dump config (INFO) mà vẫn thấy WARNING."""
    lg = logging.getLogger("thu_nghiem_hush")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    lg.addHandler(console)

    progress.hush("thu_nghiem_hush")
    lg.info("dump config dài dòng")
    lg.warning("Skip loading parameter")

    out = capsys.readouterr().err
    assert "dump config" not in out
    assert "Skip loading parameter" in out
    lg.handlers.clear()


def test_hush_khong_dung_den_duong_ghi_ra_file(tmp_path):
    """logging.FileHandler là lớp con của StreamHandler; lọc cẩu thả là mất
    luôn bản log đầy đủ trong run dir."""
    lg = logging.getLogger("thu_nghiem_hush_file")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    f = tmp_path / "log.txt"
    to_file = logging.FileHandler(f, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    lg.addHandler(to_file)

    progress.hush("thu_nghiem_hush_file")

    assert to_file.level == logging.DEBUG, "ngưỡng của handler ra file phải nguyên vẹn"
    lg.info("dòng INFO vẫn phải vào file")
    to_file.flush()
    assert "dòng INFO vẫn phải vào file" in f.read_text(encoding="utf-8")
    lg.handlers.clear()
    to_file.close()


def test_unhush_tra_lai_nguong_cu():
    lg = logging.getLogger("thu_nghiem_unhush")
    lg.handlers.clear()
    lg.propagate = False
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    lg.addHandler(console)

    changed = progress.hush("thu_nghiem_unhush")
    assert console.level == logging.WARNING
    progress.unhush(changed)
    assert console.level == logging.DEBUG
    lg.handlers.clear()


def test_hush_logger_khong_ton_tai_khong_gay():
    assert progress.hush("khong_he_co_logger_nay") == []


# ---------------------------------------------------------------------- thanh
def test_bar_tat_khi_stdout_khong_phai_terminal(monkeypatch, capsys):
    """Chạy nohup / redirect thì không được rải ký tự \r vào file."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    bar = progress.Bar(10, "thử")
    assert bar.bar is None
    bar.advance(3)
    assert bar.n == 3
    bar.write("dòng này vẫn phải in")
    bar.close()
    assert "dòng này vẫn phải in" in capsys.readouterr().out


def test_bar_dem_du_so_buoc_ke_ca_khi_tat():
    bar = progress.Bar(5, enabled=False)
    for _ in range(5):
        bar.advance()
    assert bar.n == 5
    bar.close()


@pytest.mark.parametrize("enabled", [True, False])
def test_bar_close_goi_hai_lan_khong_gay(enabled):
    bar = progress.Bar(3, enabled=enabled)
    bar.close()
    bar.close()


# ------------------------------------------------------------------ cảnh báo
# Chạy trong TIẾN TRÌNH CON, không phải trong pytest: pytest bọc mỗi test
# trong warnings.catch_warnings() và tự thu cảnh báo, nên đo trong đó là đo
# máy móc của pytest chứ không phải hành vi lúc train thật.
_KICH_BAN = """
import sys, warnings
sys.path.insert(0, {root!r})
{setup}
for _ in range(30):
    if {ctx}:
        # Thu vien nao do trong vong lap train dung catch_warnings: luc thoat
        # no goi _filters_mutated(), va CPython xoa sach registry danh dau
        # "canh bao nay hien roi". Day la thu da danh bai bo loc "once".
        with warnings.catch_warnings():
            pass
    warnings.warn("autocast is deprecated", FutureWarning)
    warnings.warn("mot canh bao khac han", FutureWarning)
"""


def _chay(setup: str, ctx: bool = True):
    root = str(pathlib.Path(__file__).resolve().parents[1])
    code = _KICH_BAN.format(root=root, setup=setup, ctx=ctx)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-800:]
    return [l for l in r.stderr.splitlines() if "Warning:" in l]


def test_khong_lam_gi_thi_canh_bao_ngap_man_hinh():
    """Mốc để so: 30 iteration x 2 cảnh báo = 60 dòng ra stderr."""
    assert len(_chay("")) == 60


def test_bo_loc_once_cua_python_khong_cuu_duoc():
    """Ghi lại VÌ SAO không dùng cách đơn giản đó: catch_warnings vô hiệu nó.

    Không có catch_warnings thì "once" ăn; có thì thua hẳn. Vòng lặp train
    thật rơi vào vế thứ hai.
    """
    setup = 'warnings.filterwarnings("once", category=FutureWarning)'
    assert len(_chay(setup, ctx=False)) == 2
    assert len(_chay(setup, ctx=True)) == 60


def test_chuyen_canh_bao_vao_file_thi_man_hinh_sach(tmp_path):
    """Cách đang dùng: stderr không còn dòng nào, kể cả khi bị catch_warnings
    quấy mỗi iteration."""
    f = tmp_path / "warnings.log"
    setup = ("from cofseg import progress; "
             f"progress.warnings_to_file({str(f)!r})")
    assert _chay(setup, ctx=True) == []


def test_file_giu_moi_loai_dung_mot_lan(tmp_path):
    """Không giấu cảnh báo nào: mỗi nội dung khác nhau vẫn còn nguyên."""
    f = tmp_path / "warnings.log"
    setup = ("from cofseg import progress; "
             f"progress.warnings_to_file({str(f)!r})")
    _chay(setup, ctx=True)
    lines = [l for l in f.read_text(encoding="utf-8").splitlines() if "Warning:" in l]
    assert len(lines) == 2, lines
    assert any("autocast" in l for l in lines)
    assert any("khac han" in l for l in lines)


def test_dem_duoc_so_loai_canh_bao(tmp_path):
    """Khối tổng kết cần con số này để trỏ người đọc tới file."""
    import warnings as w

    dedupe = progress.warnings_to_file(tmp_path / "warnings.log")
    for _ in range(10):
        w.warn("lặp lại y hệt", FutureWarning)
        w.warn("khác hẳn", FutureWarning)
    assert len(dedupe.seen) == 2

    logging.captureWarnings(False)
    lg = logging.getLogger("py.warnings")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def test_khong_dung_vao_bo_loc_cua_python(tmp_path):
    """Đụng vào warnings.filters là BẬT LÊN những cảnh báo Python đang ẩn.

    Lần trước tôi thêm "once" cho DeprecationWarning và làm lộ ra cảnh báo
    của pycocotools vốn đang bị ẩn — màn hình bẩn thêm chứ không sạch đi.
    """
    import warnings as w

    truoc = list(w.filters)
    progress.warnings_to_file(tmp_path / "warnings.log")
    assert list(w.filters) == truoc

    logging.captureWarnings(False)
    lg = logging.getLogger("py.warnings")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
