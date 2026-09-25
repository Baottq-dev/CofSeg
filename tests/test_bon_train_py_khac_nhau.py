"""Bốn `train.py` là bốn bản riêng, và mỗi bản chỉ khai cờ model của nó nhận.

Trước đây bốn file giống hệt nhau tới từng byte trừ đường dẫn trong docstring,
nên `--help` của YOLO quảng cáo `--backbone` (họ này không có backbone để đổi)
còn `--help` của Mask R-CNN quảng cáo `--model` (chỉ YOLO nhận). Một cờ hiện
trong `--help` nhưng luôn báo lỗi là nhiễu, không phải tính năng.

Test này không ép bốn file phải khác nhau cho có; nó khoá đúng một tính chất:
cờ nào có mặt thì trainer phía sau phải thật sự dùng được nó.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"

#: thư mục -> (cờ phải có, cờ không được có)
CO = {
    "maskrcnn": ({"--backbone", "--list-backbones"}, {"--model", "--list-models"}),
    "mask2former": ({"--backbone", "--list-backbones"}, {"--model", "--list-models"}),
    "solov2": ({"--backbone", "--list-backbones"}, {"--model", "--list-models"}),
    "yolo": ({"--model", "--list-models"}, {"--backbone", "--list-backbones"}),
}
MODELS = tuple(CO)


def _src(model: str) -> str:
    return (BENCH / model / "train.py").read_text(encoding="utf-8")


def _co_khai(src: str) -> set[str]:
    """Cờ mà argparse thật sự khai, không tính cờ nhắc trong chú thích."""
    return set(re.findall(r'ap\.add_argument\(\s*"(--[a-z0-9-]+)"', src))


@pytest.mark.parametrize("model", MODELS)
def test_train_py_khai_dung_co_cua_minh(model):
    phai_co, khong_duoc = CO[model]
    khai = _co_khai(_src(model))
    thieu = phai_co - khai
    assert not thieu, f"{model}/train.py thiếu {sorted(thieu)}"
    thua = khong_duoc & khai
    assert not thua, f"{model}/train.py khai {sorted(thua)} mà trainer của nó không nhận"


def test_bon_train_py_khong_con_giong_het_nhau():
    """Bốn bản phải khác nhau ở phần cờ, không chỉ ở docstring."""
    than = {}
    for m in MODELS:
        # Bỏ mọi dòng có đường dẫn thư mục: đó là khác biệt cũ, không tính.
        than[m] = "\n".join(d for d in _src(m).splitlines()
                            if f"benchmark/{m}" not in d)
    assert than["yolo"] != than["maskrcnn"], \
        "train.py của YOLO và Mask R-CNN vẫn giống hệt nhau"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former", "solov2"))
def test_ba_model_backbone_van_dung_chung_mot_luong(model):
    """Ba model đổi backbone thì phần xử lý phải giống nhau — khác nhau ở đây
    nghĩa là ba bảng so sánh đi qua ba đường khác nhau."""
    than = "\n".join(d for d in _src(model).splitlines() if f"benchmark/{model}" not in d)
    goc = "\n".join(d for d in _src("maskrcnn").splitlines() if "benchmark/maskrcnn" not in d)
    assert than == goc, f"{model}/train.py đã trôi khỏi bản của maskrcnn"


@pytest.mark.parametrize("model", MODELS)
def test_khong_cho_viet_tat_co(model):
    """argparse nhận mọi tiền tố không nhập nhằng, nên `--conf 0.05` từng bị
    nuốt thành `--config 0.05`. Siêu tham số phải rơi xuống parse_known_args."""
    assert "allow_abbrev=False" in _src(model), f"{model}/train.py còn cho viết tắt"
