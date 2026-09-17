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
