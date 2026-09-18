"""Mask R-CNN: bộ đọc mẫu, tăng cường, một bước huấn luyện, lưu/nạp, dự đoán.

Chạy hoàn toàn trên CPU với model không trọng số ở cỡ ảnh nhỏ. Không kiểm
"model có học được không" — chỉ kiểm rằng mọi mảnh khớp nhau: nhãn đi đúng
đường vào loss, checkpoint đi đúng đường ra lớp bọc, mặt nạ về đúng cỡ gốc.
"""

import numpy as np
import pytest

from canopyseg.datasets import CocoDataset
from canopyseg.datasets.instances import (
    InstanceDataset, Ops, apply_ops, collate, rasterize, resize_long_side,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------- hình học
def _tri(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[10:20, 5:30] = 255
    poly = np.array([[5, 10], [30, 10], [30, 20], [5, 20]], np.float64)
    return img, poly


@pytest.mark.parametrize("ops", [
    Ops(), Ops(hflip=True), Ops(vflip=True), Ops(rot90=1), Ops(rot90=2), Ops(rot90=3),
    Ops(hflip=True, vflip=True, rot90=1), Ops(hflip=True, rot90=3),
])
def test_polygon_and_image_transform_agree(ops):
    """Rasterise sau khi biến đổi polygon == biến đổi mặt nạ đã rasterise."""
    img, poly = _tri(40, 30)
    mask_before = rasterize([poly], 30, 40)[0]
    img2, polys2 = apply_ops(img, [poly], ops)
    mask_direct = apply_ops(mask_before[..., None].repeat(3, 2), [], ops)[0][..., 0]
    mask_via_poly = rasterize(polys2, *img2.shape[:2])[0]
    assert img2.shape[:2] == mask_via_poly.shape
    inter = np.count_nonzero(mask_direct & mask_via_poly)
    union = np.count_nonzero(mask_direct | mask_via_poly)
    assert inter / union > 0.9
    # Nội dung ảnh (vùng trắng) phải nằm đúng chỗ polygon sau biến đổi.
    white = img2[..., 0] == 255
    assert np.count_nonzero(white & mask_via_poly) / np.count_nonzero(white) > 0.9


def test_resize_long_side_scales_both_axes():
    img = np.zeros((300, 400, 3), np.uint8)
    out, s = resize_long_side(img, 100)
    assert out.shape[:2] == (75, 100) and s == pytest.approx(0.25)
    same, s1 = resize_long_side(img, 400)
    assert same is img and s1 == 1.0


# ---------------------------------------------------------------- bộ đọc
def test_dataset_yields_torchvision_targets(tiny_dataset):
    coco = CocoDataset(tiny_dataset, "train")
    ds = InstanceDataset(coco, imgsz=100, fliplr=0.0, flipud=0.0, rot90=False)
    assert len(ds) == 3
    img, t = ds.load(0, Ops())
    assert img.shape[1] == 100                      # cạnh dài = imgsz
    assert t["boxes"].shape == (2, 4) and t["masks"].shape[0] == 2
    assert t["masks"].shape[1:] == img.shape[:2]
    assert (t["boxes"][:, 2] > t["boxes"][:, 0]).all()
    # Vùng đã thu 4 lần: hình vuông 60 -> 15 px.
    x0, y0, x1, y1 = t["boxes"][0]
    assert (x1 - x0) == pytest.approx(15, abs=0.5)

    x, target = ds[0]
    assert x.shape == (3, img.shape[0], img.shape[1]) and x.max() <= 1.0
    assert target["masks"].dtype.is_floating_point is False
    images, targets = collate([ds[0], ds[2]])
    assert len(images) == 2 and targets[1]["boxes"].shape == (0, 4)   # ảnh không vùng


def test_limit_and_seed_are_respected(tiny_dataset):
    coco = CocoDataset(tiny_dataset, "train")
    assert len(InstanceDataset(coco, 64, limit=2)) == 2
    a = InstanceDataset(coco, 64, fliplr=0.5, flipud=0.5, rot90=True, seed=7)
    b = InstanceDataset(coco, 64, fliplr=0.5, flipud=0.5, rot90=True, seed=7)
    assert [a.sample_ops() for _ in range(5)] == [b.sample_ops() for _ in range(5)]


# ---------------------------------------------------------------- model
@pytest.mark.slow
def test_one_training_step_then_predict_roundtrip(tiny_dataset, tmp_path):
    import torch

    from canopyseg.models.maskrcnn import (
        MaskRCNNModel, build_maskrcnn, load_checkpoint, save_checkpoint,
    )

    coco = CocoDataset(tiny_dataset, "train")
    ds = InstanceDataset(coco, imgsz=96, fliplr=0.0, flipud=0.0, rot90=False)
    images, targets = collate([ds[0], ds[1]])
    model = build_maskrcnn(num_classes=2, pretrained=False, imgsz=96, max_det=10)
    model.train()
    losses = model(list(images), list(targets))
    total = sum(losses.values())
    assert torch.isfinite(total) and {"loss_mask", "loss_classifier"} <= set(losses)
    total.backward()

    ck = save_checkpoint(tmp_path / "w" / "last.pt", model, imgsz=96, num_classes=2, max_det=10)
    model2, meta = load_checkpoint(ck, device="cpu")
    assert meta["imgsz"] == 96 and meta["arch"] == "maskrcnn_resnet50_fpn_v2"
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)

    wrapper = MaskRCNNModel(weights=str(ck), conf=0.0, device="cpu")
    img = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    preds = wrapper.predict(img)
    assert isinstance(preds, list)
    for p in preds:
        assert p.mask.dtype == bool and p.mask.ndim == 2
        ox, oy = p.origin
        assert 0 <= ox and 0 <= oy
        assert oy + p.mask.shape[0] <= 300 and ox + p.mask.shape[1] <= 400
    assert wrapper.describe["imgsz"] == 96


def test_trainer_declares_params_and_tag():
    from canopyseg.training.maskrcnn import MASKRCNN_DEFAULTS, MaskRCNNTrainer

    assert MaskRCNNTrainer.param_names() == set(MASKRCNN_DEFAULTS)
    assert MaskRCNNTrainer.run_tag({"train": {"imgsz": 640}}) == "i640b2e50"
    assert MaskRCNNTrainer.locked_params() == frozenset()


def test_lr_schedule_warms_up_then_decays(tmp_path):
    from canopyseg.training.maskrcnn import MaskRCNNTrainer

    t = MaskRCNNTrainer({"train": {"lr": 1.0, "warmup_iters": 10, "cos_lr": True}}, tmp_path)
    assert t._lr_at(0, 100) < 0.01
    assert t._lr_at(10, 100) == pytest.approx(1.0)
    assert t._lr_at(55, 100) == pytest.approx(0.5, abs=0.02)
    assert t._lr_at(100, 100) == pytest.approx(0.0, abs=1e-6)
    flat = MaskRCNNTrainer({"train": {"lr": 1.0, "warmup_iters": 0, "cos_lr": False}}, tmp_path)
    assert flat._lr_at(50, 100) == 1.0
