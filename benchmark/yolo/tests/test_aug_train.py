"""--aug nối vào dataloader của ultralytics: thứ tự phép, chuyển vị, dán, bộ số riêng.

Bộ dữ liệu dựng tại chỗ: ba ảnh 1280 x 720 cùng "ruộng" field_9, mỗi ảnh một
lưới đĩa (có cặp chạm nhau), imgsz 512 nên khung nạp 512 x 288 với r = 0,4 như
khung thật.
"""

from __future__ import annotations

import random
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.utils.instance import Instances  # noqa: E402

from cofseg.training import aug as augmod  # noqa: E402
from cofseg.training.aug_train import AugDataset, PasteCrowns, Transpose, add_instances, make_trainer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
IW, IH = 1280, 720


def _disk(cx, cy, r, n=90):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("field9")
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for k in range(3):
        img = np.full((IH, IW, 3), (40, 90, 120), np.uint8)
        img += rng.integers(0, 20, img.shape, np.uint8)
        polys = []
        for i in range(8):
            cx = 160 + (i % 4) * 300 + (110 if i in (1, 5) else 0) * (k == 0)
            cy = 180 + (i // 4) * 360
            polys.append(_disk(cx + rng.uniform(-10, 10), cy + rng.uniform(-10, 10), rng.uniform(95, 125)))
        for i, p in enumerate(polys):
            cv2.fillPoly(img, [np.round(p).astype(np.int32)], (20 + 25 * i, 150, 70))
        for split in ("train", "val"):
            name = f"field_9__{k:04d}"
            cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), img)
            with open(root / "labels" / split / f"{name}.txt", "w") as fh:
                for p in polys:
                    q = p / [IW, IH]
                    fh.write("0 " + " ".join(f"{v:.6f}" for v in q.reshape(-1)) + "\n")
    data = {"path": str(root), "train": str(root / "images" / "train"), "val": str(root / "images" / "val"),
            "names": {0: "canopy"}, "nc": 1, "channels": 3}
    return root, data


def _hyp(seed=0):
    import yaml

    tr = yaml.safe_load(open(REPO / "benchmark/yolo/configs/train/yolo26s.yaml", encoding="utf-8"))["train"]
    cfg = get_cfg(overrides={**tr, "imgsz": 512, "seed": seed})
    cfg.task = "segment"
    return cfg


def _dataset(tiny, spec, tmp_path, **params):
    _root, data = tiny
    if spec:
        c = {"aug": params} if params else {}
        augmod.apply(c, spec)
        cls = type("AugDataset", (AugDataset,), {"AUG": c["aug"], "LOG_DIR": str(tmp_path / "aug")})
    else:
        cls = YOLODataset
    hyp = _hyp()
    return cls(img_path=data["train"], imgsz=512, batch_size=2, augment=True, hyp=copy(hyp), rect=False,
               cache=None, single_cls=False, stride=32, pad=0.0, prefix="t: ", task="segment", classes=None,
               data=data, fraction=1.0)


def _sample(ds, i, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[i]


def _states():
    return random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone()


def _same_states(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and torch.equal(a[2], b[2])


# ------------------------------------------------------------------ thứ tự phép

def test_thu_tu_phep(tiny, tmp_path):
    ds = _dataset(tiny, "ag1+pa1", tmp_path)
    names = [type(t).__name__ for t in ds.transforms.transforms]
    assert names[0] == "PasteCrowns"
    assert names[-2:] == ["Transpose", "Format"]
    assert names[1:-2] == [type(t).__name__ for t in _dataset(tiny, None, tmp_path).transforms.transforms[:-1]]


def test_khong_co_aug_trung_dataset_goc(tiny, tmp_path):
    base = _dataset(tiny, None, tmp_path)
    for spec, params in (("ag1", {"ag1": {"p": 0.0}}), ("pa1", {"pa1": {"p": 0.0}}),
                         ("ag1+ag4", {"ag1": {"p": 0.0}, "ag4": {"p": 0.0}})):
        ds = _dataset(tiny, spec, tmp_path, **params)
        for i in range(3):
            a, b = _sample(base, i, 11 + i), _sample(ds, i, 11 + i)
            assert torch.equal(a["img"], b["img"]) and torch.equal(a["masks"], b["masks"])
            assert torch.equal(a["bboxes"], b["bboxes"])


def test_bo_so_rieng_khong_cham_bo_so_chung(tiny, tmp_path):
    ds = _dataset(tiny, "ag1+pa1", tmp_path, ag1={"p": 1.0}, pa1={"p": 1.0})
    lab = ds.get_image_and_label(0)
    before = _states()
    lab = ds.transforms.transforms[0](lab)
    t = Transpose(1.0, 0)
    lab["img"] = np.zeros((64, 64, 3), np.uint8)
    lab["instances"] = Instances(np.zeros((0, 4)), np.zeros((0, 1000, 2), np.float32), None, "xywh", False)
    t(lab)
    assert _same_states(before, _states())


# ------------------------------------------------------------------ chuyển vị

def test_chuyen_vi_anh_va_nhan():
    img = np.random.default_rng(0).integers(0, 255, (40, 40, 3), np.uint8)
    seg = np.array([[[2, 5], [30, 5], [30, 9], [2, 9]]], np.float32)
    inst = Instances(np.array([[16, 7, 28, 4]], np.float32), seg.copy(), None, "xywh", False)
    lab = Transpose(1.0, 0)({"img": img, "instances": inst})
    assert np.array_equal(lab["img"], img.transpose(1, 0, 2))
    assert np.array_equal(lab["instances"].segments, seg[..., ::-1])
    assert np.array_equal(lab["instances"].bboxes, np.array([[7, 16, 4, 28]], np.float32))
    lab2 = Transpose(0.0, 0)({"img": img, "instances": Instances(np.array([[16, 7, 28, 4]], np.float32),
                                                                  seg.copy(), None, "xywh", False)})
    assert lab2["img"] is img


def test_chuyen_vi_mot_nua(tiny, tmp_path):
    t = Transpose(0.5, 3)
    hits = 0
    for _ in range(400):
        img = np.zeros((4, 8, 3), np.uint8)
        lab = t({"img": img, "instances": Instances(np.zeros((0, 4)), np.zeros((0, 10, 2), np.float32),
                                                       None, "xywh", False)})
        hits += lab["img"].shape[0] == 8
    assert 160 < hits < 240


# ------------------------------------------------------------------ dán

def test_dan_them_tan_va_nhat_ky(tiny, tmp_path):
    ds = _dataset(tiny, "pa1", tmp_path, pa1={"p": 1.0, "k_min": 3, "k_max": 3})
    s = ds.aug_summary
    assert s["mode"] == "pa1" and s["images"] == 3 and s["specs"] > 0
    assert set(s["bank"]) == {"field_9"}
    paste = ds.transforms.transforms[0]
    assert isinstance(paste, PasteCrowns)
    added = 0
    for i in range(3):
        lab = ds.get_image_and_label(i)
        n0, img0 = len(lab["instances"]), lab["img"].copy()
        out = paste(lab)
        n1 = len(out["instances"])
        assert len(out["cls"]) == n1 and (out["cls"][n0:] == 0).all()
        assert out["instances"].normalized and out["instances"]._bboxes.format == "xywh"
        assert out["instances"].segments.shape[1:] == (1000, 2)
        if n1 > n0:
            assert not np.array_equal(out["img"], img0)
        added += n1 - n0
    assert added > 0
    d = tmp_path / "aug"
    for f in ("summary.json", "donors.csv", "pool.csv", "pool_log.csv"):
        assert (d / f).is_file(), f
    assert list(d.glob("pasted_*.csv"))
    # Ảnh sau cả chuỗi vẫn ra đúng dạng tensor.
    out = _sample(ds, 0, 5)
    assert out["img"].shape == (3, 512, 512) and len(out["cls"]) == len(out["bboxes"])


def test_ag4_dan_xa(tiny, tmp_path):
    ds = _dataset(tiny, "ag4", tmp_path, ag4={"p": 1.0})
    assert ds.aug_summary["mode"] == "ag4"
    assert ds.aug_summary["specs"] > 0
    lab = ds.transforms.transforms[0](ds.get_image_and_label(1))
    assert len(lab["instances"]) >= 8


def test_them_nhan_giu_dang():
    seg = np.random.default_rng(0).uniform(0.2, 0.4, (2, 1000, 2)).astype(np.float32)
    lab = {"instances": Instances(np.array([[.3, .3, .1, .1], [.3, .3, .1, .1]], np.float32), seg, None,
                                  "xywh", True), "cls": np.zeros((2, 1), np.float32)}
    add_instances(lab, [np.array([[10, 10], [50, 10], [50, 40], [10, 40]], np.float32)], 100, 50)
    inst = lab["instances"]
    assert len(inst) == 3 and inst.normalized and inst._bboxes.format == "xywh"
    assert np.allclose(inst.bboxes[2], [0.3, 0.5, 0.4, 0.6], atol=1e-5)
    assert lab["cls"].shape == (3, 1)


# ------------------------------------------------------------------ trainer

def test_trainer_chi_doi_dataset_train(monkeypatch, tmp_path):
    import ultralytics.data.build as build
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer

    seen = []
    monkeypatch.setattr(DetectionTrainer, "build_dataset",
                        lambda self, img_path, mode="train", batch=None: seen.append((mode, build.YOLODataset)))
    blk = {}
    augmod.apply(blk, "ag1+pa1")
    cls = make_trainer(blk["aug"], tmp_path)
    assert issubclass(cls, SegmentationTrainer)
    obj = object.__new__(cls)
    obj.build_dataset("x", "train")
    obj.build_dataset("x", "val")
    assert issubclass(seen[0][1], AugDataset) and seen[0][1].AUG["tag"] == "ag1-pa1"
    assert seen[0][1].LOG_DIR == str(tmp_path / "aug")
    assert seen[1][1] is YOLODataset
    assert build.YOLODataset is YOLODataset
