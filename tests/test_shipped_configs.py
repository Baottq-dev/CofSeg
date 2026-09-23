"""Các file config đi kèm repo phải nối đúng vào registry và vào nhau.

Bốn model của bảng benchmark có bản code riêng trong `benchmark/<model>/`,
nên test ở đây chỉ phủ phần của gốc (đường chạy nhanh, nhánh promptable) cộng
những ràng buộc BẮC CẦU giữa các thư mục: `benchmark/run.py` phải trỏ đúng
thư mục và config có thật, và mỗi thư mục phải tự chứa. Test cho model của
từng người nằm trong `benchmark/<model>/tests/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from canopyseg import config, registry
from canopyseg.models.build import model_param_names

ROOT = Path(__file__).resolve().parents[1]
TRAIN = sorted(p for p in (ROOT / "configs" / "train").glob("*.yaml") if not p.name.startswith("_"))
EVAL = sorted(p for p in (ROOT / "configs" / "eval").glob("*.yaml") if not p.name.startswith("_"))
MEMBERS = sorted(d for d in (ROOT / "benchmark").iterdir()
                 if d.is_dir() and (d / "train.py").exists())


@pytest.mark.parametrize("path", TRAIN, ids=lambda p: p.name)
def test_train_config_names_a_trainer_and_only_its_params(path):
    import canopyseg.training  # noqa: F401

    cfg = config.load(path)
    cls = registry.resolve("trainer", cfg["trainer"])
    names = cls.param_names()
    if names is not None:
        unknown = set(cfg.get("train") or {}) - names
        assert not unknown, f"{path.name}: khối train có khoá trainer không nhận: {sorted(unknown)}"
    assert cls.run_tag(cfg)  # tên thư mục run phải dựng được trước khi có trainer


@pytest.mark.parametrize("path", EVAL, ids=lambda p: p.name)
def test_eval_config_names_a_model_and_only_its_params(path):
    import canopyseg.models  # noqa: F401

    cfg = config.load(path)
    m = cfg["model"]
    names = model_param_names(m["name"])
    if names is not None:
        unknown = set(m) - names - {"name"}
        assert not unknown, f"{path.name}: khối model có khoá {m['name']} không nhận: {sorted(unknown)}"


def test_runner_knows_every_member_folder():
    """benchmark/run.py là chỗ duy nhất biết bốn model; bảng trong đó phải khớp
    thư mục có thật, và mỗi mục phải trỏ vào config có thật."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bench_run", ROOT / "benchmark" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert set(mod.ORDER) == set(mod.MODELS) == {d.name for d in MEMBERS}
    for name, spec_ in mod.MODELS.items():
        folder = ROOT / "benchmark" / spec_["dir"]
        assert folder.is_dir(), f"{name}: không có thư mục {spec_['dir']}"
        for key in ("train", "eval"):
            assert (folder / spec_[key]).exists(), f"{name}: thiếu {spec_[key]}"
        assert spec_["score"] in ("predictions", "weights")
        assert spec_["data"] in ("root", "yaml")


@pytest.mark.parametrize("member", MEMBERS, ids=lambda d: d.name)
def test_member_folder_is_self_contained(member):
    """Thư mục thành viên phải chạy được một mình: bản sao lõi, hai script,
    config, test. Và KHÔNG được nhắc tới canopyseg/ — import ngược ra gốc là
    hết độc lập mà không ai nhận ra."""
    for need in ("cofseg", "configs", "train.py", "evaluate.py", "tests"):
        assert (member / need).exists(), f"{member.name}: thiếu {need}"

    leaked = [p.relative_to(member).as_posix()
              for p in member.rglob("*.py") if "canopyseg" in p.read_text(encoding="utf-8")]
    assert not leaked, f"{member.name}: còn nhắc canopyseg ở {leaked}"


@pytest.mark.parametrize("member", MEMBERS, ids=lambda d: d.name)
def test_member_configs_resolve(member):
    """Mọi config của thành viên phải nạp được (chuỗi `base:` không gãy) và
    khai đúng trainer/model mà bản sao của họ có."""
    trainers = ({p.stem for p in (member / "cofseg" / "training").glob("*.py")}
                - {"__init__", "base", "memory"})
    models = ({p.stem for p in (member / "cofseg" / "models").glob("*.py")}
              - {"__init__", "base", "build"})
    seen = 0
    for p in sorted((member / "configs").rglob("*.yaml")):
        if p.name.startswith("_"):
            continue
        cfg = config.load(p)
        seen += 1
        if "trainer" in cfg:
            assert cfg["trainer"] in trainers, f"{p.name}: trainer {cfg['trainer']!r} không có trong cofseg/"
        if isinstance(cfg.get("model"), dict) and "name" in cfg["model"]:
            assert cfg["model"]["name"] in models, f"{p.name}: model {cfg['model']['name']!r} không có"
    assert seen, f"{member.name}: không có config nào"
