"""Hợp đồng Trainer: phần scripts/train.py dựa vào mà không cần biết họ model.

Ba classmethod về tham số là thứ quyết định gõ sai có bị bắt hay không, nên
mặc định của chúng phải rõ ràng: không khai gì = không kiểm (chủ đích), khai
defaults = tên suy từ defaults, không phải hai danh sách có thể lệch nhau.
"""

from canopyseg.training.base import Trainer


class Silent(Trainer):
    def fit(self):
        return {"weights": {}}


class Declared(Trainer):
    @classmethod
    def param_defaults(cls):
        return {"imgsz": 640, "batch": 2, "data": None}

    @classmethod
    def locked_params(cls):
        return frozenset({"data"})

    def fit(self):
        return {"weights": {}}


def test_default_contract_declares_nothing():
    assert Silent.param_defaults() is None
    assert Silent.param_names() is None
    assert Silent.locked_params() == frozenset()
    assert Silent.run_tag({}) == ""


def test_names_follow_defaults():
    assert Declared.param_names() == {"imgsz", "batch", "data"}
    assert "data" in Declared.locked_params()


def test_lifecycle_defaults_are_inert(tmp_path):
    t = Silent({}, tmp_path)
    assert t.prepare() == {}
    assert t.probe() == {"supported": False}


def test_yolo_trainer_declares_ultralytics_params():
    """Danh sách sống của ultralytics: phải có các tham số hay dùng và bốn khoá."""
    from canopyseg.training.yolo import YoloTrainer

    names = YoloTrainer.param_names()
    assert names is not None and {"imgsz", "batch", "epochs", "mask_ratio"} <= names
    assert YoloTrainer.locked_params() == {"data", "project", "name", "exist_ok"}
    assert YoloTrainer.locked_params() <= names


def test_memory_wall_scales_with_the_card(monkeypatch):
    """0.88 x VRAM: tái tạo đúng vách 7.0 GB đo trên card 8 GB, và không còn là
    hằng số sai khi chạy trên card 24 GB."""
    from canopyseg.training import memory

    monkeypatch.setattr(memory, "total_gb", lambda device=0: 8.0)
    assert memory.wall_gb() == 7.0
    monkeypatch.setattr(memory, "total_gb", lambda device=0: 24.0)
    assert memory.wall_gb() == 21.1
    monkeypatch.setattr(memory, "total_gb", lambda device=0: None)
    assert memory.wall_gb() is None
