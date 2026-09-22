"""Các file config đi kèm repo phải nối đúng vào registry và vào nhau.

Phần lớn đường chạy trên máy lab (detectron2, mmdet) không kiểm được ở nhà,
nên ít nhất tên trainer/model, tên file config mà run_fold.sh gọi, và mọi
tham số trong khối train: phải đúng — gõ sai một chữ ở đây là mất một đêm
máy thuê.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from canopyseg import config, registry
from canopyseg.models.build import model_param_names

ROOT = Path(__file__).resolve().parents[1]


# Config chung ở configs/, config của từng model trong thư mục người phụ trách.
def _configs(kind: str) -> list[Path]:
    found = list((ROOT / "configs" / kind).glob("*.yaml"))
    found += ROOT.glob(f"members/*/configs/{kind}/*.yaml")
    return sorted(p for p in found if not p.name.startswith("_"))


TRAIN = _configs("train")
EVAL = _configs("eval")


@pytest.mark.parametrize("path", TRAIN, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_train_config_names_a_trainer_and_only_its_params(path):
    import canopyseg.training  # noqa: F401

    cfg = config.load(path)
    cls = registry.resolve("trainer", cfg["trainer"])
    names = cls.param_names()
    if names is not None:
        unknown = set(cfg.get("train") or {}) - names
        assert not unknown, f"{path.name}: khối train có khoá trainer không nhận: {sorted(unknown)}"
    assert cls.run_tag(cfg)  # tên thư mục run phải dựng được trước khi có trainer


@pytest.mark.parametrize("path", EVAL, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_eval_config_names_a_model_and_only_its_params(path):
    import canopyseg.models  # noqa: F401

    cfg = config.load(path)
    m = cfg["model"]
    names = model_param_names(m["name"])
    if names is not None:
        unknown = set(m) - names - {"name"}
        assert not unknown, f"{path.name}: khối model có khoá {m['name']} không nhận: {sorted(unknown)}"


def test_run_fold_calls_only_configs_that_exist():
    sh = (ROOT / "scripts" / "remote" / "run_fold.sh").read_text(encoding="utf-8")
    calls = re.findall(r"^want (\w+)\s+&& run_cfg (\w+)\s+(\S+)", sh, flags=re.M)
    assert calls, "run_fold.sh không còn dòng `want X && run_cfg X <config>` nào"
    for want, name, cfg in calls:
        assert want == name
        assert (ROOT / cfg).exists(), f"run_fold.sh gọi config thiếu: {cfg}"
    default = re.search(r'ONLY="([^"]+)"', sh).group(1).split(",")
    assert default == ["yolo11s", "maskrcnn", "solov2", "mask2former"]
    assert {n for _, n, _ in calls} >= set(default) - {"yolo11s"}
    assert "members/yolo11/configs/train/yolo11s.yaml" in sh
