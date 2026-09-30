"""Trainer detectron2: phần thuần Python kiểm được ở nhà.

detectron2 không cài trên máy phát triển, nên ở đây chỉ kiểm việc dịch khối
train: sang khoá config (số iteration, lịch lr, độ phân giải, số lớp), hợp
đồng trainer, và prepare() — thứ chạy trước khi đụng tới detectron2. Test cần
detectron2 thật nằm cuối và tự bỏ qua khi thiếu.
"""

from __future__ import annotations

import json

import pytest

from cofseg.models.detectron2 import ARCHS, ZOO, class_opts, size_opts
from cofseg.training.detectron2 import (
    BASE_LR, D2_DEFAULTS, Detectron2Trainer, build_opts, iters_per_epoch,
)


def _opts(arch, n_train=600, **over):
    args = {**D2_DEFAULTS, **over}
    o = build_opts(arch, n_train, args, names={"train": "tr", "val": "va"}, out_dir="out")
    return dict(zip(o[::2], o[1::2]))


def test_epochs_become_iterations_and_lr_scales_with_batch():
    d = _opts("maskrcnn", 600, batch=4, epochs=50)
    assert d["SOLVER.MAX_ITER"] == 7500                     # 150 it/epoch x 50
    assert d["SOLVER.STEPS"] == (5250, 6750)                # 70 % và 90 %
    assert d["SOLVER.CHECKPOINT_PERIOD"] == d["TEST.EVAL_PERIOD"] == 150
    assert d["SOLVER.BASE_LR"] == pytest.approx(0.02 * 4 / 16)
    # warmup_iters < 1 là PHẦN của lịch: 0.03 x 7500 = 225.
    assert d["SOLVER.WARMUP_ITERS"] == 225
    assert d["DATASETS.TRAIN"] == ("tr",) and d["DATASETS.TEST"] == ("va",)
    assert d["MODEL.ROI_HEADS.NUM_CLASSES"] == 1
    assert d["MODEL.ROI_HEADS.SCORE_THRESH_TEST"] == 0.05
    assert d["DATALOADER.FILTER_EMPTY_ANNOTATIONS"] is False
    assert d["OUTPUT_DIR"] == "out" and d["SEED"] == 0

    # Batch không chia hết số ảnh: làm tròn lên, warmup không dài hơn cả lần chạy.
    d = _opts("cascade", 10, batch=4, epochs=1, warmup_iters=200)
    assert d["SOLVER.MAX_ITER"] == 3 and d["SOLVER.WARMUP_ITERS"] == 3
    assert iters_per_epoch(10, 4) == 3

    # >= 1 vẫn là số vòng tuyệt đối, để cách viết cũ không bị đổi nghĩa ngầm.
    assert _opts("maskrcnn", warmup_iters=500)["SOLVER.WARMUP_ITERS"] == 500
    # Phần nhỏ tới mức ra 0 thì vẫn phải là 1 vòng.
    assert _opts("maskrcnn", 10, batch=10, epochs=1,
                 warmup_iters=0.001)["SOLVER.WARMUP_ITERS"] == 1

    # Đầu mask của R-CNN: 14 -> 28x28 như recipe gốc, đổi được khi cần biên mịn hơn.
    assert d["MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION"] == 14
    assert _opts("maskrcnn", mask_resolution=28)["MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION"] == 28
    # Mask2Former không có ROI head nên không được nhận khoá đó.
    assert "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION" not in _opts("mask2former")

    # lr đặt tay thì thắng recipe.
    assert _opts("maskrcnn", lr=0.001)["SOLVER.BASE_LR"] == 0.001


def test_imgsz_means_long_side_like_the_other_trainers():
    d = _opts("maskrcnn", imgsz=1024)
    assert d["INPUT.MAX_SIZE_TRAIN"] == d["INPUT.MAX_SIZE_TEST"] == 1024
    assert d["INPUT.MIN_SIZE_TRAIN"] == (576,) and d["INPUT.MIN_SIZE_TEST"] == 576
    # Ảnh 4:3 thì cạnh ngắn theo tỉ lệ thật, không cố định 9/16.
    o = size_opts(1024, aspect=3 / 4)
    assert dict(zip(o[::2], o[1::2]))["INPUT.MIN_SIZE_TRAIN"] == (768,)


def test_mask2former_keys_differ_from_rcnn():
    d = _opts("mask2former", batch=4, epochs=100, num_queries=100)
    assert d["MODEL.SEM_SEG_HEAD.NUM_CLASSES"] == 1
    assert d["MODEL.MASK_FORMER.NUM_OBJECT_QUERIES"] == 100
    assert d["INPUT.IMAGE_SIZE"] == 1024
    assert d["SOLVER.BASE_LR"] == pytest.approx(1e-4 * 4 / 16)
    assert "MODEL.ROI_HEADS.NUM_CLASSES" not in d
    assert "MODEL.ROI_HEADS.SCORE_THRESH_TEST" not in d
    assert class_opts("pointrend")[-2:] == ["MODEL.POINT_HEAD.NUM_CLASSES", 1]


def test_bad_inputs_fail_early():
    with pytest.raises(ValueError, match="arch"):
        build_opts("yolo", 100, D2_DEFAULTS)
    with pytest.raises(ValueError):
        build_opts("maskrcnn", 0, D2_DEFAULTS)
    with pytest.raises(ValueError, match="arch"):
        Detectron2Trainer({"model": {"arch": "swin"}}, "x")
    assert set(BASE_LR) == set(ARCHS) == set(ZOO)


def test_trainer_contract_and_run_tag():
    names = Detectron2Trainer.param_names()
    assert {"imgsz", "batch", "epochs", "lr", "num_queries", "workers"} <= names
    assert Detectron2Trainer.locked_params() == frozenset()
    assert Detectron2Trainer.run_tag({"train": {"imgsz": 2048, "batch": 1}}) == "i2048b1e50"
    assert Detectron2Trainer.run_tag({}) == "i1024b16e50"


def test_prepare_counts_train_images_without_detectron2(three_splits, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    t = Detectron2Trainer({"model": {"arch": "cascade"},
                           "data": {"root": str(three_splits), "min_area": 50.0},
                           "train": {"batch": 2}}, run)
    info = t.prepare()
    assert info["arch"] == "cascade"
    assert info["train"]["images"] == 3 and info["train"]["dropped"] == 1
    assert info["iters_per_epoch"] == 2 and t.n_train == 3
    assert t.aspect == pytest.approx(300 / 400)
    assert info["sample"]["shape"] == [300, 400, 3]
    saved = json.loads((run / "dataset_check.json").read_text(encoding="utf-8"))
    assert saved["test"]["images"] == 3
    assert t._dataset_names() == {"train": "coffee_dataset_train", "val": "coffee_dataset_val",
                                  "test": "coffee_dataset_test"}

    # data.limit cắt số ảnh train dùng để tính iteration (khói).
    t = Detectron2Trainer({"model": {"arch": "maskrcnn"},
                           "data": {"root": str(three_splits), "limit": 2},
                           "train": {"batch": 1}}, run)
    assert t.prepare()["iters_per_epoch"] == 2


@pytest.mark.gpu
def test_probe_runs_two_real_iterations(three_splits, tmp_path):
    pytest.importorskip("detectron2")
    run = tmp_path / "run"
    run.mkdir()
    t = Detectron2Trainer({"model": {"arch": "maskrcnn"},
                           "data": {"root": str(three_splits)},
                           "train": {"batch": 1, "imgsz": 256}}, run)
    p = t.probe()
    assert p["supported"] and p["peak_gb"] > 0 and p["objects_in_probe"] >= 0
