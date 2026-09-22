"""Các file config đi kèm repo phải nối đúng vào registry và vào nhau.

Bốn model của bảng benchmark có bản code riêng trong `benchmark/<model>_<người>/`,
nên test ở đây chỉ phủ phần của gốc (đường chạy nhanh, nhánh promptable) cộng
những ràng buộc BẮC CẦU giữa các thư mục: `run_fold.sh` phải gọi đúng thư mục
có thật, và mỗi thư mục phải tự chứa. Test cho model của từng người nằm trong
`benchmark/<...>/tests/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from canopyseg import config, registry
from canopyseg.models.build import model_param_names

ROOT = Path(__file__).resolve().parents[1]
TRAIN = sorted(p for p in (ROOT / "configs" / "train").glob("*.yaml") if not p.name.startswith("_"))
EVAL = sorted(p for p in (ROOT / "configs" / "eval").glob("*.yaml") if not p.name.startswith("_"))
MEMBERS = sorted(d for d in (ROOT / "benchmark").iterdir() if d.is_dir() and (d / "run.sh").exists())


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


def test_run_fold_calls_member_folders_that_exist():
    sh = (ROOT / "scripts" / "remote" / "run_fold.sh").read_text(encoding="utf-8")
    calls = re.findall(r"^want (\S+)\s+&& run (\S+)\s+(\S+)", sh, flags=re.M)
    assert len(calls) == 4, "run_fold.sh phải gọi đúng bốn thư mục thành viên"
    for want, short, member in calls:
        assert want == short
        assert (ROOT / "benchmark" / member / "run.sh").exists(), f"thiếu benchmark/{member}/run.sh"
    default = re.search(r'ONLY="([^"]+)"', sh).group(1).split(",")
    assert set(default) == {s for _, s, _ in calls}
    assert {m for _, _, m in calls} == {d.name for d in MEMBERS}


@pytest.mark.parametrize("member", MEMBERS, ids=lambda d: d.name)
def test_member_folder_is_self_contained(member):
    """Thư mục thành viên phải chạy được một mình: có bản sao lõi, script, và
    mọi config mà run.sh nhắc tới. Và KHÔNG được nhắc tới canopyseg/ — import
    ngược ra gốc là hết độc lập mà không ai nhận ra."""
    for need in ("cofseg", "configs", "scripts/train.py", "scripts/evaluate.py", "run.sh"):
        assert (member / need).exists(), f"{member.name}: thiếu {need}"

    sh = (member / "run.sh").read_text(encoding="utf-8")
    for rel in re.findall(r'"\$HERE/(configs/\S+?\.yaml)"', sh):
        assert (member / rel).exists(), f"{member.name}: run.sh gọi config thiếu: {rel}"

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
