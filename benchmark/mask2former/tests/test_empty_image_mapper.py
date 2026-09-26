"""Mapper LSJ của Mask2Former gãy trên ẢNH NỀN — ảnh không có vùng nào.

    instances = utils.annotations_to_instances(annos, image_shape)
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()   # <- gãy
    ...
    if hasattr(instances, 'gt_masks'):                             # <- có canh

`annotations_to_instances` chỉ gắn `gt_masks` khi `len(annos)` > 0. Dòng canh ở
dưới cho thấy tác giả biết trường đó có thể vắng, chỉ là sót một chỗ. Upstream
không lộ vì recipe COCO của họ để `FILTER_EMPTY_ANNOTATIONS: True`, còn ta cố ý
để False để giữ ảnh nền.

Bộ block/f1 có ĐÚNG 2 ảnh nền trên 470 ảnh train — 0.4%, đủ để giết lượt chạy ở
iteration 43/235.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


class _Instances:
    """Đủ giống detectron2.structures.Instances cho phép kiểm ở đây."""

    def __init__(self, image_size):
        self.image_size = image_size
        self._f: dict = {}

    def __setattr__(self, k, v):
        if k in ("image_size", "_f"):
            return object.__setattr__(self, k, v)
        self._f[k] = v

    def __getattr__(self, k):
        try:
            return self._f[k]
        except KeyError:
            raise AttributeError(
                f"Cannot find field '{k}' in the given Instances!") from None

    def __len__(self):
        return len(next(iter(self._f.values()), []))


class _Boxes:
    def __init__(self, t):
        self.tensor = t

    def __len__(self):
        return len(self.tensor)


class _UpstreamMapper:
    """Chép đúng khúc gãy của COCOInstanceNewBaselineDatasetMapper."""

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

    def __call__(self, dataset_dict):
        out = {"image": torch.zeros((3, 64, 96), dtype=torch.uint8),
               "file_name": dataset_dict["file_name"]}
        if not self.is_train:
            return out
        if "annotations" in dataset_dict:
            annos = [a for a in dataset_dict["annotations"]
                     if a.get("iscrowd", 0) == 0]
            inst = _Instances((64, 96))
            inst.gt_classes = torch.zeros(len(annos), dtype=torch.int64)
            if annos:                       # <- điều kiện của detectron2
                inst.gt_masks = torch.ones((len(annos), 64, 96), dtype=torch.uint8)
            inst.gt_boxes = _Boxes(inst.gt_masks)     # <- dòng gãy
            out["instances"] = inst
        return out


@pytest.fixture
def mapper_cls(monkeypatch):
    """Dựng sẵn hai module mà empty_safe_mapper đi tìm."""
    structures = types.ModuleType("detectron2.structures")
    structures.Instances = _Instances
    structures.Boxes = _Boxes
    d2 = types.ModuleType("detectron2")
    d2.structures = structures
    monkeypatch.setitem(sys.modules, "detectron2", d2)
    monkeypatch.setitem(sys.modules, "detectron2.structures", structures)

    leaf = types.ModuleType(
        "mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper")
    leaf.COCOInstanceNewBaselineDatasetMapper = _UpstreamMapper
    for name in ("mask2former", "mask2former.data",
                 "mask2former.data.dataset_mappers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper",
        leaf)

    from cofseg.models.detectron2 import empty_safe_mapper
    return empty_safe_mapper(object())


ANH_NEN = {"file_name": "field_3__10__DJI_20260301160126_0265_D.jpg",
           "annotations": []}
ANH_CO_VUNG = {"file_name": "field_1__10__1__a.jpg",
               "annotations": [{"iscrowd": 0}, {"iscrowd": 0}]}


def test_mapper_goc_that_su_gay_tren_anh_nen():
    """Nếu ngày nào đó upstream tự vá, test này đỏ và ta bỏ được lớp bọc."""
    with pytest.raises(AttributeError, match="gt_masks"):
        _UpstreamMapper(None, True)(dict(ANH_NEN))


def test_anh_nen_khong_con_gay(mapper_cls):
    out = mapper_cls(None, True)(dict(ANH_NEN))
    assert len(out["instances"]) == 0


def test_anh_nen_co_du_ba_truong_model_can(mapper_cls):
    inst = mapper_cls(None, True)(dict(ANH_NEN))["instances"]
    assert inst.gt_masks.shape == (0, 64, 96)
    assert inst.gt_masks.dtype == torch.uint8      # như convert_coco_poly_to_mask
    assert inst.gt_classes.shape == (0,)
    assert inst.gt_boxes.tensor.shape == (0, 4)


def test_anh_co_vung_van_di_duong_cu(mapper_cls):
    inst = mapper_cls(None, True)(dict(ANH_CO_VUNG))["instances"]
    assert inst.gt_masks.shape == (2, 64, 96)


def test_chi_toan_iscrowd_cung_tinh_la_anh_nen(mapper_cls):
    d = {"file_name": "x.jpg", "annotations": [{"iscrowd": 1}, {"iscrowd": 1}]}
    assert len(mapper_cls(None, True)(d)["instances"]) == 0


def test_khong_dung_vao_dataset_dict_goc(mapper_cls):
    d = dict(ANH_NEN)
    mapper_cls(None, True)(d)
    assert d["annotations"] == [], "dataset_dict dùng chung giữa các epoch"


def test_luc_val_thi_khong_gan_instances(mapper_cls):
    assert "instances" not in mapper_cls(None, False)(dict(ANH_NEN))
