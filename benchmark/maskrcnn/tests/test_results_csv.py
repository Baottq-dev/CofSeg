"""metrics.json của detectron2 -> results.csv, và cái bẫy epoch cuối.

detectron2 tăng `self.iter` lên `max_iter` TRƯỚC khi gọi `after_train`, để
`after_train` phân biệt được "train xong tử tế" với "chết giữa chừng". Mà
`EvalHook` lại cố tình để dành val của epoch cuối cho `after_train`. Hậu quả:
AP quan trọng nhất của cả lượt chạy được ghi ở `iteration = max_iter`, và
phép chia thẳng cho ra một epoch KHÔNG TỒN TẠI — lượt 3 epoch báo "epoch tốt
nhất 4/3".
"""

from __future__ import annotations

import json

from cofseg.training.detectron2 import D2_DEFAULTS, Detectron2Trainer


def _trainer(tmp_path, epochs=3, batch=16, n_train=470):
    cfg = {"trainer": "detectron2", "model": {"arch": "maskrcnn"},
           "data": {"root": str(tmp_path)},
           "train": {**D2_DEFAULTS, "epochs": epochs, "batch": batch}}
    t = Detectron2Trainer(cfg, tmp_path)
    t.n_train = n_train
    return t


def _metrics(tmp_path, entries):
    out = tmp_path / "d2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        "\n".join(json.dumps(e) for e in entries), encoding="utf-8")


def test_epoch_cuoi_khong_vuot_qua_so_epoch_that(tmp_path):
    """Lượt 3 epoch: AP ghi ở iteration 90 vẫn phải là epoch 3, không phải 4."""
    t = _trainer(tmp_path, epochs=3)
    _metrics(tmp_path, [
        {"iteration": 29, "segm/AP": 3.76, "segm/AP50": 13.52},
        {"iteration": 59, "segm/AP": 12.81, "segm/AP50": 33.70},
        {"iteration": 90, "segm/AP": 27.61, "segm/AP50": 56.00},   # after_train
    ])
    rows = t._results_csv(30)
    assert [r["epoch"] for r in rows] == [1, 2, 3]


def test_epoch_giua_chung_van_tinh_dung(tmp_path):
    """Chặn trên không được làm sai các dòng bình thường."""
    t = _trainer(tmp_path, epochs=3)
    _metrics(tmp_path, [{"iteration": i} for i in (0, 19, 29, 30, 39, 59, 60, 89)])
    rows = t._results_csv(30)
    assert [r["epoch"] for r in rows] == [1, 1, 1, 2, 2, 2, 3, 3]


def test_epoch_tot_nhat_lay_tu_dong_co_AP_cao_nhat(tmp_path):
    """AP tụt ở epoch cuối thì epoch tốt nhất là epoch 2."""
    t = _trainer(tmp_path, epochs=3)
    _metrics(tmp_path, [
        {"iteration": 29, "segm/AP": 3.76},
        {"iteration": 59, "segm/AP": 30.0},
        {"iteration": 90, "segm/AP": 12.0},
    ])
    rows = t._results_csv(30)
    best = max((r for r in rows if r.get("segm/AP")), key=lambda r: float(r["segm/AP"]))
    assert best["epoch"] == 2


def test_khong_co_metrics_json_thi_tra_rong(tmp_path):
    assert _trainer(tmp_path)._results_csv(30) == []
