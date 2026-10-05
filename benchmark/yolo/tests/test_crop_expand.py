"""`crop_expand` của YoloSegModel: cắt mặt nạ bằng box nới thay vì box sát."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from cofseg.models import model_param_names  # noqa: E402
from cofseg.models.yolo_seg import _crop_expand_predictor, expand_boxes  # noqa: E402


def test_noi_quanh_tam_va_kep_trong_anh():
    b = torch.tensor([[10.0, 10.0, 30.0, 50.0], [0.0, 90.0, 20.0, 100.0]])
    out = expand_boxes(b, 1.5, (100, 120))
    assert out[0].tolist() == [5.0, 0.0, 35.0, 60.0]       # tâm (20, 30), nửa cạnh (15, 30)
    assert out[1].tolist() == [0.0, 87.5, 25.0, 100.0]     # kẹp ở mép trái và mép dưới


def test_he_so_1_giu_nguyen_box():
    b = torch.tensor([[3.0, 4.0, 13.0, 24.0]])
    assert torch.equal(expand_boxes(b, 1.0, (50, 50)), b)


def _construct(factor, box):
    """Logit dương ở mọi chỗ, nên mặt nạ sau khi cắt đúng bằng vùng box đã nới."""
    P = _crop_expand_predictor(factor)
    p = P(overrides={"task": "segment", "retina_masks": True, "verbose": False})
    p.model = types.SimpleNamespace(names={0: "crown"})
    pred = torch.tensor([[*box, 0.9, 0.0, *([1.0] * 32)]])
    proto = torch.ones(32, 16, 16)
    img = torch.zeros(1, 3, 64, 64)
    orig = __import__("numpy").zeros((64, 64, 3), "uint8")
    return p.construct_result(pred, img, orig, "x.jpg", proto)


def test_mat_na_phu_box_noi_con_box_tra_ve_la_box_goc():
    r1 = _construct(1.0, [16.0, 16.0, 32.0, 32.0])
    r2 = _construct(1.5, [16.0, 16.0, 32.0, 32.0])
    a1 = int(r1.masks.data.sum()); a2 = int(r2.masks.data.sum())
    assert a1 == 16 * 16 and a2 == 24 * 24
    assert r2.boxes.xyxy[0].tolist() == [16.0, 16.0, 32.0, 32.0]


def test_tham_so_vao_duoc_tu_dong_lenh():
    assert "crop_expand" in model_param_names("yolo_seg")


@pytest.mark.parametrize("bad", [0.9, True, "1.2"])
def test_gia_tri_sai_bi_chan(bad):
    from cofseg.models.yolo_seg import YoloSegModel

    with pytest.raises(SystemExit, match="crop_expand"):
        YoloSegModel.__init__(object.__new__(YoloSegModel), "weights/yolo26s-seg.pt", crop_expand=bad)
