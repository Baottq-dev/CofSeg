"""Ràng buộc bắc cầu giữa các thư mục benchmark/.

Gốc repo không còn model nào: bốn model nằm trong `benchmark/<model>/`, mỗi
thư mục một bản độc lập. Test ở đây chỉ kiểm những thứ nối các thư mục lại —
mỗi thư mục phải tự chứa và config trong đó phải nạp được. Test cho từng
model nằm trong `benchmark/<model>/tests/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from canopyseg import config

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = sorted(d for d in (ROOT / "benchmark").iterdir()
                 if d.is_dir() and (d / "train.py").exists())


@pytest.mark.parametrize("member", MEMBERS, ids=lambda d: d.name)
def test_member_folder_is_self_contained(member):
    """Thư mục thành viên phải chạy được một mình: bản sao lõi, hai script,
    config, test. Và KHÔNG được nhắc tới canopyseg/ — gốc không còn model nào
    nên import ngược ra đó là gãy, mà gãy lúc chạy thật trên máy thuê."""
    for need in ("cofseg", "configs", "train.py", "evaluate.py", "tests"):
        assert (member / need).exists(), f"{member.name}: thiếu {need}"

    leaked = [p.relative_to(member).as_posix()
              for p in member.rglob("*.py")
              if "upstream" not in p.parts and "canopyseg" in p.read_text(encoding="utf-8")]
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


def test_root_no_longer_ships_models():
    """Gốc chỉ còn phần dùng chung. Thêm lại model vào canopyseg/ là đi ngược
    thoả thuận: model mới thì thêm một thư mục trong benchmark/."""
    for gone in ("models", "training", "metrics", "registry.py"):
        assert not (ROOT / "canopyseg" / gone).exists(), f"canopyseg/{gone} quay lại rồi"
    assert not (ROOT / "configs" / "train").exists()
    assert not (ROOT / "configs" / "eval").exists()
    kept = {p.name for p in (ROOT / "canopyseg").glob("*.py")}
    assert {"weights.py", "config.py", "console.py"} <= kept
