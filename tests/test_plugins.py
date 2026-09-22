"""members/<model>/plugin.py: chỗ mỗi người viết code riêng mà không sửa lõi.

Điều phải giữ: plugin được nạp trước khi tra registry (nếu không, `trainer:`
trỏ vào lớp của member sẽ báo "không có"); nạp hai lần không chạy lại module;
plugin hỏng thì báo rõ file nào chứ không im lặng bỏ qua — chạy tiếp với
registry thiếu model sẽ gãy ở chỗ khó hiểu hơn nhiều.
"""

from __future__ import annotations

import sys

import pytest

from canopyseg import plugins, registry


@pytest.fixture
def fake_repo(tmp_path):
    """Một repo giả có members/<hai model>/, chỉ một cái có plugin.py."""
    (tmp_path / "members" / "solov2").mkdir(parents=True)
    (tmp_path / "members" / "yolo11").mkdir(parents=True)     # không có plugin.py
    (tmp_path / "members" / "solov2" / "plugin.py").write_text(
        "from canopyseg.registry import register\n"
        "from canopyseg.training.base import Trainer\n"
        "\n"
        "LOADS = []\n"
        "LOADS.append(1)\n"
        "\n"
        '@register("trainer", "solov2_thu_nghiem")\n'
        "class Thu(Trainer):\n"
        "    def fit(self):\n"
        '        return {"weights": {}}\n',
        encoding="utf-8")
    yield tmp_path
    for name in [n for n in sys.modules if n.startswith("members_")]:
        del sys.modules[name]
    registry._REG.get("trainer", {}).pop("solov2_thu_nghiem", None)


def test_plugin_registers_a_trainer_and_loads_once(fake_repo):
    assert [p.parent.name for p in plugins.plugin_files(fake_repo)] == ["solov2"]

    loaded = plugins.load_members(fake_repo)
    assert loaded == ["members_solov2_plugin"]
    cls = registry.resolve("trainer", "solov2_thu_nghiem")
    assert cls.fit(None) == {"weights": {}}

    # Gọi lại: không chạy lại module (danh sách LOADS vẫn một phần tử).
    assert plugins.load_members(fake_repo) == loaded
    assert len(sys.modules["members_solov2_plugin"].LOADS) == 1


def test_missing_members_dir_is_not_an_error(tmp_path):
    assert plugins.plugin_files(tmp_path) == []
    assert plugins.load_members(tmp_path) == []


def test_broken_plugin_names_the_file(tmp_path):
    (tmp_path / "members" / "hong").mkdir(parents=True)
    (tmp_path / "members" / "hong" / "plugin.py").write_text(
        "import khong_co_goi_nay\n", encoding="utf-8")
    with pytest.raises(ImportError, match=r"hong[/\\]plugin\.py"):
        plugins.load_members(tmp_path)
    # Không để lại module nửa vời trong sys.modules.
    assert "members_hong_plugin" not in sys.modules


def test_shipped_repo_loads_whatever_members_have():
    names = {d.name for d in (plugins.ROOT / "members").iterdir() if d.is_dir()}
    assert {"maskrcnn", "solov2", "yolo11", "mask2former"} <= names
    # Chưa ai có plugin.py thì trả rỗng; ai có thì nạp đúng file của người đó.
    expect = [f"members_{p.parent.name}_plugin" for p in plugins.plugin_files()]
    assert plugins.load_members() == expect
