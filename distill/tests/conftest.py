"""Fixture dùng chung cho test của distill/.

Như test của benchmark/yolo: không test nào dựa vào data/ hay weights/ (cả hai nằm ngoài git). Thầy YOLO26s
dựng từ yaml của gói ultralytics với trọng số ngẫu nhiên; ảnh và nhãn dựng tại chỗ.

Chạy từ gốc repo:  python -m pytest distill/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import distill  # noqa: E402,F401 - đặt sys.path tới benchmark/yolo cho `import cofseg`

H = 32  # lưới proto của ảnh 128 px (stride 4)


@pytest.fixture(scope="session")
def teacher():
    """YOLO26s-seg 80 lớp, trọng số ngẫu nhiên nhưng cố định: đúng kiến trúc của checkpoint COCO."""
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    m = SegmentationModel("yolo26s-seg.yaml", nc=80, verbose=False)
    # BN có thống kê khác 0/1 để phép so lớp đầu không đúng nhờ may.
    g = torch.Generator().manual_seed(1)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
    return m.eval()


@pytest.fixture(scope="session")
def teacher_n():
    """YOLO26n-seg 80 lớp, trọng số ngẫu nhiên cố định: nguồn của trò 0,25."""
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(2)
    return SegmentationModel("yolo26n-seg.yaml", nc=80, verbose=False).eval()


@pytest.fixture
def batch():
    """Ba tán hình chữ nhật không chồng nhau: hai ở ảnh 0, một ở ảnh 1 (cùng dạng test loss của benchmark)."""
    torch = pytest.importorskip("torch")
    boxes = torch.tensor([[0.25, 0.25, 0.30, 0.30], [0.70, 0.70, 0.35, 0.40], [0.50, 0.50, 0.50, 0.50]])
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    masks = torch.zeros(2, H, H)
    for k, (b, bi) in enumerate(zip(boxes, batch_idx)):
        j = int((batch_idx[:k] == bi).sum())
        cx, cy, w, h = (b * H).tolist()
        masks[int(bi), int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)] = j + 1
    return {"batch_idx": batch_idx, "cls": torch.zeros(3, 1), "bboxes": boxes,
            "masks": masks, "sem_masks": torch.zeros(2, H, H)}
