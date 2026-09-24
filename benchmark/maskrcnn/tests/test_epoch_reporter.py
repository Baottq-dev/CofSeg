"""Hook báo cáo epoch, chạy với detectron2 GIẢ LẬP.

detectron2 không cài trên máy phát triển, nhưng phần logic dễ sai nhất của
hook lại không dính gì tới GPU: dịch iteration sang epoch, và xử lý epoch
cuối. EvalHook của detectron2 CỐ TÌNH bỏ qua iteration cuối trong after_step
rồi chấm val trong after_train — nên nếu hook báo cáo epoch cuối ở after_step
thì dòng quan trọng nhất của cả lượt train sẽ thiếu mất AP.

Ở đây dựng một `detectron2.engine` tối thiểu trong sys.modules rồi cho hook
chạy qua một vòng lặp giống hệt TrainerBase.train, kể cả thứ tự after_step /
after_train. Cái được kiểm là HÀNH VI, không phải hình dạng của mã.
"""

from __future__ import annotations

import sys
import types

import pytest

from cofseg.training.detectron2 import D2_DEFAULTS, Detectron2Trainer


class _History:
    def __init__(self, value):
        self.value = value

    def median(self, _window):
        return self.value


class _Storage:
    """Đủ phần EventStorage mà hook đụng tới: latest() và history()."""

    def __init__(self):
        self.values: dict[str, float] = {}
        self.iter = 0

    def put(self, **kw):
        self.values.update(kw)

    def latest(self):
        return {k: (v, self.iter) for k, v in self.values.items()}

    def history(self, key):
        if key not in self.values:
            raise KeyError(key)
        return _History(self.values[key])


class _FakeTrainer:
    def __init__(self, max_iter):
        self.iter = 0
        self.start_iter = 0
        self.max_iter = max_iter
        self.storage = _Storage()


@pytest.fixture
def fake_d2(monkeypatch):
    """`from detectron2.engine import HookBase` phải chạy được."""
    class HookBase:
        trainer = None

        def before_train(self): ...
        def after_step(self): ...
        def after_train(self): ...

    engine = types.ModuleType("detectron2.engine")
    engine.HookBase = HookBase
    root = types.ModuleType("detectron2")
    root.engine = engine
    monkeypatch.setitem(sys.modules, "detectron2", root)
    monkeypatch.setitem(sys.modules, "detectron2.engine", engine)
    return HookBase


def _trainer(tmp_path, epochs=3, batch=16, n_train=470):
    cfg = {"trainer": "detectron2", "model": {"arch": "maskrcnn"},
           "data": {"root": str(tmp_path)},
           "train": {**D2_DEFAULTS, "epochs": epochs, "batch": batch}}
    t = Detectron2Trainer(cfg, tmp_path)
    t.n_train = n_train
    return t


def _run(reporter, epochs, per_epoch, ap_at):
    """Vòng lặp y như TrainerBase.train, kể cả chỗ EvalHook bỏ qua iter cuối.

    `ap_at` ánh xạ epoch -> AP; epoch cuối được ghi ở after_train để mô phỏng
    đúng hành vi của EvalHook.
    """
    trainer = _FakeTrainer(epochs * per_epoch)
    reporter.trainer = trainer
    reporter.before_train()
    for i in range(trainer.max_iter):
        trainer.iter = i
        trainer.storage.iter = i
        trainer.storage.put(total_loss=2.5 - i * 0.01, lr=1e-4)
        done = i + 1
        ep = done // per_epoch
        # EvalHook: chấm val ở cuối epoch, TRỪ iteration cuối cùng.
        if done % per_epoch == 0 and done != trainer.max_iter and ep in ap_at:
            trainer.storage.put(**{"segm/AP": ap_at[ep], "segm/AP50": ap_at[ep] * 2})
        reporter.after_step()
    if epochs in ap_at:                      # EvalHook.after_train
        trainer.storage.put(**{"segm/AP": ap_at[epochs], "segm/AP50": ap_at[epochs] * 2})
    reporter.after_train()
    return trainer


def test_in_du_mot_dong_moi_epoch(tmp_path, fake_d2, capsys):
    t = _trainer(tmp_path, epochs=3)
    _run(t._epoch_reporter(), 3, 30, {1: 3.76, 2: 12.81, 3: 27.61})
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch")]
    assert len(lines) == 3
    assert lines[0].startswith("epoch 1/3   iter 30/90")
    assert lines[1].startswith("epoch 2/3   iter 60/90")
    assert lines[2].startswith("epoch 3/3   iter 90/90")


def test_epoch_cuoi_van_co_AP(tmp_path, fake_d2, capsys):
    """Cái bẫy chính: AP của epoch cuối chỉ có ở after_train."""
    t = _trainer(tmp_path, epochs=3)
    _run(t._epoch_reporter(), 3, 30, {1: 3.76, 2: 12.81, 3: 27.61})
    last = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch 3/3")][0]
    assert "mAP50-95  27.61" in last
    assert "mAP50  55.22" in last


def test_danh_dau_dung_epoch_tot_nhat(tmp_path, fake_d2, capsys):
    """AP tụt ở epoch cuối thì dấu sao phải ở epoch 2, không phải epoch 3."""
    t = _trainer(tmp_path, epochs=3)
    _run(t._epoch_reporter(), 3, 30, {1: 3.76, 2: 30.0, 3: 12.0})
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch")]
    assert lines[0].endswith("* tốt nhất")
    assert lines[1].endswith("* tốt nhất")
    assert not lines[2].endswith("* tốt nhất")


def test_epoch_khong_cham_val_thi_khong_bia_so(tmp_path, fake_d2, capsys):
    """val_every=2: epoch 1 và 3 không có AP, dòng của chúng không được lặp
    lại số của epoch khác."""
    t = _trainer(tmp_path, epochs=4)
    _run(t._epoch_reporter(), 4, 30, {2: 10.0, 4: 20.0})
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch")]
    assert len(lines) == 4
    assert "mAP" not in lines[0]
    assert "mAP50-95  10.00" in lines[1]
    assert "mAP50-95  20.00" in lines[3]


def test_khong_no_khi_chua_co_so_lieu_nao(tmp_path, fake_d2, capsys):
    """Lượt train chết ở epoch 1 vẫn phải in được, không kéo theo lỗi thứ hai."""
    t = _trainer(tmp_path, epochs=1)
    _run(t._epoch_reporter(), 1, 30, {})
    assert capsys.readouterr().out.count("epoch 1/1") == 1


def test_so_iteration_moi_epoch_lam_tron_len(tmp_path, fake_d2, capsys):
    """470 ảnh / batch 16 = 29,4 -> 30 iteration, và 3 epoch = 90 iteration."""
    t = _trainer(tmp_path, epochs=3, batch=16, n_train=470)
    trainer = _run(t._epoch_reporter(), 3, 30, {3: 1.0})
    assert trainer.max_iter == 90
    assert "iter 90/90" in capsys.readouterr().out


def test_tach_thoi_gian_train_khoi_thoi_gian_val(tmp_path, fake_d2, capsys):
    """Evaluator gọi close_train_bar() lúc bắt đầu chấm, nên dòng epoch nói
    được 'train bao lâu + val bao lâu' thay vì một con số gộp.

    Một epoch 0:53 mà 0:33 trong đó là chấm val thì con số gộp nói sai hẳn
    về tốc độ huấn luyện.
    """
    import time

    t = _trainer(tmp_path, epochs=1)
    r = t._epoch_reporter()
    trainer = _FakeTrainer(30)
    r.trainer = trainer
    r.before_train()
    for i in range(30):
        trainer.iter = i
        trainer.storage.iter = i
        r.after_step()
    # EvalHook.after_train -> evaluator.reset() -> close_train_bar()
    r.close_train_bar()
    time.sleep(0.05)                      # thời gian chấm val
    trainer.storage.put(**{"segm/AP": 27.61})
    r.after_train()

    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch")][0]
    assert "+ val " in line, line


def test_close_train_bar_goi_hai_lan_khong_doi_so(tmp_path, fake_d2):
    """reset() có thể chạy nhiều lần trong một lượt chấm; lần sau không được
    cộng thêm thời gian val vào thời gian train."""
    import time

    t = _trainer(tmp_path, epochs=1)
    r = t._epoch_reporter()
    r.trainer = _FakeTrainer(30)
    r.before_train()
    r.close_train_bar()
    lan_dau = r.trained
    time.sleep(0.05)
    r.close_train_bar()
    assert r.trained == lan_dau
