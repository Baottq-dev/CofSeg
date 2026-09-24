"""Trainer YOLO của thư mục này: hợp đồng tham số và vách bộ nhớ.

Bản sao lõi trong cofseg/ là của riêng thư mục, nên test cũng ở đây: sửa
trainer mà quên chạy test thì không ai khác biết.
"""

from __future__ import annotations

import pytest


def test_yolo_trainer_declares_ultralytics_params():
    """Danh sách sống của ultralytics: phải có các tham số hay dùng và bốn khoá."""
    from cofseg.training.yolo import YoloTrainer

    names = YoloTrainer.param_names()
    assert names is not None and {"imgsz", "batch", "epochs", "mask_ratio"} <= names
    assert YoloTrainer.locked_params() == {"data", "project", "name", "exist_ok"}
    assert YoloTrainer.locked_params() <= names


def test_memory_wall_scales_with_the_card(monkeypatch):
    """0.88 x VRAM: vách 7.0 GB trên card 8 GB, 21.1 GB trên card 24 GB."""
    from cofseg.training import memory

    monkeypatch.setattr(memory, "total_gb", lambda device=0: 8.0)
    assert memory.wall_gb() == pytest.approx(7.0)
    monkeypatch.setattr(memory, "total_gb", lambda device=0: 24.0)
    assert memory.wall_gb() == pytest.approx(21.1)
    monkeypatch.setattr(memory, "total_gb", lambda device=0: None)
    assert memory.wall_gb() is None


def test_summary_lay_chi_so_mat_na_chu_khong_phai_hop(tmp_path, capsys):
    """(M) là mặt nạ, (B) là hộp. Bài toán là phân vùng, nên lấy (M).

    Hai số này khác nhau thật, nên nhầm cột là báo cáo sai kết quả chứ không
    chỉ là in xấu.
    """
    from cofseg.training.yolo import YoloTrainer

    t = YoloTrainer({"trainer": "yolo", "model": {"name": "yolo11s-seg.pt"},
                     "data": {"yaml": str(tmp_path / "data.yaml")},
                     "train": {"epochs": 50}}, tmp_path)
    t._print_summary({"metrics/mAP50-95(B)": 0.9999, "metrics/mAP50(B)": 0.8888,
                      "metrics/mAP50-95(M)": 0.2761, "metrics/mAP50(M)": 0.5579},
                     132.4, tmp_path / "ultralytics" / "weights" / "best.pt")
    out = capsys.readouterr().out
    assert "mAP50-95 27.61" in out and "mAP50 55.79" in out
    assert "99.99" not in out and "88.88" not in out
    assert "train 2:12" in out
    assert "ultralytics" in out


def test_summary_khong_co_chi_so_van_in_duoc(tmp_path, capsys):
    """Lượt chạy hỏng giữa chừng vẫn phải ra khối tổng kết, không kéo theo
    ngoại lệ thứ hai che mất lỗi thật."""
    from cofseg.training.yolo import YoloTrainer

    t = YoloTrainer({"trainer": "yolo", "model": {"name": "yolo11s-seg.pt"},
                     "data": {"yaml": str(tmp_path / "data.yaml")}}, tmp_path)
    t._print_summary({}, None, tmp_path / "w.pt")
    assert "—" in capsys.readouterr().out
