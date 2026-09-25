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
    # "model" bị khoá có chủ đích: nó nằm trong 114 tham số của ultralytics
    # nên không khoá thì --model lọt vào khối train: rồi KHÔNG đổi trọng số
    # thật, chỉ đổi args.yaml của lần chạy. Đổi model là việc của apply_model.
    assert YoloTrainer.locked_params() == {"data", "project", "name", "exist_ok", "model"}
    assert YoloTrainer.locked_params() <= names


def test_doi_model_ghi_vao_khoa_cap_cao_nhat():
    """--model phải đổi đúng thứ YOLO() nạp, và trả về tên cho thư mục run."""
    from cofseg.training.yolo import YoloTrainer

    cfg = {"trainer": "yolo", "model": "weights/yolo11s-seg.pt"}
    assert YoloTrainer.apply_model(cfg, "yolo11m-seg") == "yolo11m-seg"
    assert cfg["model"] == "weights/yolo11m-seg.pt"
    # Đường dẫn có sẵn thì giữ nguyên, không nở thêm lần nữa.
    assert YoloTrainer.apply_model(cfg, "weights/yolov9c-seg.pt") == "yolov9c-seg"
    assert cfg["model"] == "weights/yolov9c-seg.pt"


def test_yolo_khong_doi_backbone_duoc():
    """Bảng backbone rỗng, và thông báo phải chỉ sang --model chứ không chỉ
    nói 'không được'."""
    from cofseg.training.yolo import YoloTrainer

    cfg = {"trainer": "yolo", "model": "weights/yolo11s-seg.pt"}
    assert YoloTrainer.backbones(cfg) == {}
    with pytest.raises(SystemExit, match="--model"):
        YoloTrainer.apply_backbone(cfg, "r101")
    assert "yolo26x-seg" in YoloTrainer.describe_backbones(cfg)


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
    # ultralytics val ngay trong model.train(), nên 2:12 là TỔNG train + val;
    # nhãn phải nói đúng điều đó thay vì gọi tất cả là "train".
    assert "train + val 2:12" in out
    assert "ultralytics" in out


def test_summary_khong_co_chi_so_van_in_duoc(tmp_path, capsys):
    """Lượt chạy hỏng giữa chừng vẫn phải ra khối tổng kết, không kéo theo
    ngoại lệ thứ hai che mất lỗi thật."""
    from cofseg.training.yolo import YoloTrainer

    t = YoloTrainer({"trainer": "yolo", "model": {"name": "yolo11s-seg.pt"},
                     "data": {"yaml": str(tmp_path / "data.yaml")}}, tmp_path)
    t._print_summary({}, None, tmp_path / "w.pt")
    assert "—" in capsys.readouterr().out
