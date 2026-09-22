"""Nạp code riêng của từng thành viên trong members/<model>/plugin.py.

Vì sao có cơ chế này: lõi trong canopyseg/ dùng chung cho cả bốn model, nên
người phụ trách một model không nên sửa nó để thử một ý riêng — sửa là đổi số
của ba người kia. `plugin.py` là chỗ viết thứ của riêng mình: một trainer khác,
một hook, một phép tăng cường, một model ghép. File được import trước khi
scripts/train.py và scripts/evaluate.py tra registry, nên @register trong đó
có hiệu lực y như code trong canopyseg/.

    members/solov2/plugin.py
    ------------------------
    from canopyseg.registry import register
    from canopyseg.training.mmdet import MMDetTrainer

    @register("trainer", "solov2_tta")
    class Solov2TTA(MMDetTrainer):
        ...

    members/solov2/configs/train/solov2_tta.yaml -> trainer: solov2_tta

Plugin hỏng thì báo rõ tên file rồi dừng: chạy tiếp với một registry thiếu
model sẽ hỏng ở chỗ khó hiểu hơn nhiều.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: Gốc repo, tính từ file này (canopyseg/plugins.py -> ..).
ROOT = Path(__file__).resolve().parents[1]
MEMBERS_DIR = "members"
PLUGIN_NAME = "plugin.py"


def plugin_files(root: str | Path | None = None) -> list[Path]:
    """members/<tên>/plugin.py đang có, sắp theo tên thư mục."""
    base = Path(root or ROOT) / MEMBERS_DIR
    if not base.is_dir():
        return []
    return sorted(p / PLUGIN_NAME for p in base.iterdir()
                  if p.is_dir() and (p / PLUGIN_NAME).is_file())


def load_members(root: str | Path | None = None) -> list[str]:
    """Import mọi plugin và trả về tên module đã nạp.

    Gọi nhiều lần không nạp lại (module đã có trong sys.modules).
    """
    loaded = []
    for path in plugin_files(root):
        name = f"members_{path.parent.name}_plugin"
        if name in sys.modules:
            loaded.append(name)
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:      # pragma: no cover - lỗi hệ thống tệp
            raise ImportError(f"Không nạp được plugin {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[name]
            raise ImportError(f"Lỗi trong {path.relative_to(Path(root or ROOT))}: {e}") from e
        loaded.append(name)
    return loaded
