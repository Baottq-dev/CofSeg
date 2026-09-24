"""Hook báo cáo epoch của trainer mmdet, chạy với mmengine GIẢ LẬP.

mmengine gọi hook theo thứ tự: các iteration, rồi `after_train_epoch`, RỒI
ValLoop mới chạy và gọi `after_val_epoch`. Nghĩa là lúc `after_train_epoch`
chạy thì AP của epoch đó chưa tồn tại. In ngay tại đó là in ra một dòng không
có AP dù một giây sau AP đã có — nên hook phải giữ dòng lại.

Epoch không chấm val (val_every > 1) thì không có `after_val_epoch` nào tới,
và dòng bị giữ phải được xả ra ở đầu epoch sau hoặc ở `after_train`. Đó là
hai đường đi dễ quên nhất, và cả hai được kiểm ở đây.
"""

from __future__ import annotations

import sys
import types

import pytest

from cofseg.training.mmdet import MM_DEFAULTS, MMDetTrainer


@pytest.fixture
def fake_mmengine(monkeypatch):
    """`from mmengine.hooks import Hook` phải chạy được."""
    class Hook:
        priority = "NORMAL"

        def before_train_epoch(self, runner): ...
        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None): ...
        def after_train_epoch(self, runner): ...
        def before_val_epoch(self, runner): ...
        def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None): ...
        def after_val_epoch(self, runner, metrics=None): ...
        def after_train(self, runner): ...

    hooks = types.ModuleType("mmengine.hooks")
    hooks.Hook = Hook
    root = types.ModuleType("mmengine")
    root.hooks = hooks
    monkeypatch.setitem(sys.modules, "mmengine", root)
    monkeypatch.setitem(sys.modules, "mmengine.hooks", hooks)
    return Hook


class _FakeRunner:
    """Chỉ những thứ hook thật sự hỏi tới; cái gì không có thì phải trả None
    chứ không được làm gãy lượt train."""

    def __init__(self, n_val=85):
        self.message_hub = None
        self.optim_wrapper = None
        self.val_dataloader = [None] * n_val


def _trainer(tmp_path, epochs=3, batch=16, n_train=470):
    cfg = {"trainer": "mmdet", "model": {"arch": "solov2"},
           "data": {"root": str(tmp_path)},
           "train": {**MM_DEFAULTS, "epochs": epochs, "batch": batch}}
    t = MMDetTrainer(cfg, tmp_path)
    t.n_train = n_train
    return t


def _run(reporter, epochs, per_epoch, ap_at):
    """Vòng đời của EpochBasedTrainLoop + ValLoop, đúng thứ tự mmengine gọi."""
    runner = _FakeRunner()
    for ep in range(1, epochs + 1):
        reporter.before_train_epoch(runner)
        for i in range(per_epoch):
            reporter.after_train_iter(runner, i, outputs={"loss": 2.5 - i * 0.01})
        reporter.after_train_epoch(runner)
        if ep in ap_at:                     # ValLoop chỉ chạy ở epoch có val
            reporter.before_val_epoch(runner)
            for i in range(len(runner.val_dataloader)):
                reporter.after_val_iter(runner, i)
            reporter.after_val_epoch(runner, {"coco/segm_mAP": ap_at[ep],
                                              "coco/segm_mAP_50": ap_at[ep] * 2})
    reporter.after_train(runner)


def _lines(capsys):
    return [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch")]


def test_in_du_mot_dong_moi_epoch(tmp_path, fake_mmengine, capsys):
    t = _trainer(tmp_path, epochs=3)
    _run(t._epoch_reporter(), 3, 30, {1: 0.0376, 2: 0.1281, 3: 0.2761})
    lines = _lines(capsys)
    assert len(lines) == 3
    assert lines[0].startswith("epoch 1/3   iter 30/90")
    assert lines[2].startswith("epoch 3/3   iter 90/90")


def test_doi_thang_0_1_cua_mmengine_sang_0_100(tmp_path, fake_mmengine, capsys):
    """mmengine trả 0,2761; ba model kia in 27,61. Bảng so sánh chỉ đọc được
    nếu bốn model cùng thang."""
    t = _trainer(tmp_path, epochs=1)
    _run(t._epoch_reporter(), 1, 30, {1: 0.2761})
    line = _lines(capsys)[0]
    assert "mAP50-95  27.61" in line
    assert "mAP50  55.22" in line


def test_dong_bi_giu_lai_cho_toi_khi_co_AP(tmp_path, fake_mmengine, capsys):
    """after_train_epoch KHÔNG được in: lúc đó ValLoop còn chưa chạy."""
    t = _trainer(tmp_path, epochs=1)
    r = t._epoch_reporter()
    runner = _FakeRunner()
    r.before_train_epoch(runner)
    for i in range(30):
        r.after_train_iter(runner, i, outputs={"loss": 1.0})
    r.after_train_epoch(runner)
    assert _lines(capsys) == [], "chưa được in khi AP còn chưa có"
    r.after_val_epoch(runner, {"coco/segm_mAP": 0.2761, "coco/segm_mAP_50": 0.5522})
    assert "mAP50-95  27.61" in _lines(capsys)[0]


def test_epoch_khong_cham_val_van_duoc_in(tmp_path, fake_mmengine, capsys):
    """val_every=2: epoch 1 và 3 không có after_val_epoch nào tới. Dòng bị giữ
    phải được xả ra, không thì mất hẳn."""
    t = _trainer(tmp_path, epochs=4)
    _run(t._epoch_reporter(), 4, 30, {2: 0.10, 4: 0.20})
    lines = _lines(capsys)
    assert len(lines) == 4
    assert "mAP" not in lines[0]
    assert "mAP50-95  10.00" in lines[1]
    assert "mAP" not in lines[2]
    assert "mAP50-95  20.00" in lines[3]


def test_epoch_cuoi_khong_cham_val_van_ra_o_after_train(tmp_path, fake_mmengine, capsys):
    t = _trainer(tmp_path, epochs=2)
    _run(t._epoch_reporter(), 2, 30, {1: 0.10})
    lines = _lines(capsys)
    assert len(lines) == 2
    assert lines[1].startswith("epoch 2/2")


def test_danh_dau_dung_epoch_tot_nhat(tmp_path, fake_mmengine, capsys):
    t = _trainer(tmp_path, epochs=3)
    _run(t._epoch_reporter(), 3, 30, {1: 0.05, 2: 0.30, 3: 0.12})
    lines = _lines(capsys)
    assert lines[0].endswith("* tốt nhất")
    assert lines[1].endswith("* tốt nhất")
    assert not lines[2].endswith("* tốt nhất")


def test_runner_thieu_thong_tin_khong_lam_gay_train(tmp_path, fake_mmengine, capsys):
    """optim_wrapper là None ở đây: thiếu lr thì bỏ cột đó, không được ném
    ngoại lệ ra giữa lượt train.

    Loss thì vẫn phải có: nó đến từ `outputs` của after_train_iter, thứ
    mmengine đưa thẳng vào tay hook, chứ không phải hỏi ngược runner."""
    t = _trainer(tmp_path, epochs=1)
    _run(t._epoch_reporter(), 1, 30, {1: 0.1})
    line = _lines(capsys)[0]
    assert "lr" not in line
    assert "loss " in line
    assert "mAP50-95  10.00" in line


def test_tach_thoi_gian_train_khoi_thoi_gian_val(tmp_path, fake_mmengine, capsys):
    """Dòng epoch phải nói được train bao lâu, val bao lâu — không gộp."""
    t = _trainer(tmp_path, epochs=1)
    _run(t._epoch_reporter(), 1, 30, {1: 0.2761})
    line = _lines(capsys)[0]
    assert "+ val " in line, line


def test_thanh_val_dem_du_so_anh(tmp_path, fake_mmengine):
    """Tổng của thanh val lấy từ val_dataloader, không phải đoán."""
    t = _trainer(tmp_path, epochs=1)
    r = t._epoch_reporter()
    runner = _FakeRunner(n_val=85)
    r.before_train_epoch(runner)
    r.before_val_epoch(runner)
    assert r.val_bar.total == 85
    for i in range(85):
        r.after_val_iter(runner, i)
    assert r.val_bar.n == 85
    r.after_val_epoch(runner, {})


def test_runner_khong_co_val_dataloader_khong_gay(tmp_path, fake_mmengine):
    """Cấu hình không chấm val thì vẫn phải chạy, chỉ là thanh rỗng."""
    t = _trainer(tmp_path, epochs=1)
    r = t._epoch_reporter()

    class Troi:
        message_hub = None
        optim_wrapper = None

    r.before_train_epoch(Troi())
    r.before_val_epoch(Troi())
    assert r.val_bar.total == 0
