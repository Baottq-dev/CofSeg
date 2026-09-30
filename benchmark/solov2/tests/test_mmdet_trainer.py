"""Trainer mmdet (SOLOv2): phần kiểm được ở nhà.

mmcv không cài trên máy phát triển, nên ở đây chỉ kiểm việc dịch khối
train: sang config mmengine (dataset, pipeline, lịch lr, hook), việc gộp lên
config zoo (mmengine thuần Python, có ở nhà), hợp đồng trainer và prepare().
Test cần mmcv/GPU nằm cuối và tự bỏ qua khi thiếu.
"""

from __future__ import annotations

import json

import pytest

from cofseg.training.mmdet import (
    ARCHS, CLASSES, MM_DEFAULTS, ZOO, MMDetTrainer, build_overrides, flip_transform,
    iters_per_epoch, load_config,
)

SPLITS = {"train": "train", "val": "val", "test": "test"}


def _over(n_train=600, aspect=9 / 16, limit=None, **kw):
    args = {**MM_DEFAULTS, **kw}
    return build_overrides("solov2", "data/export/block/f4", SPLITS, n_train, args,
                           aspect=aspect, out_dir="out", load_from="w.pth", limit=limit)


def test_schedule_follows_epochs_and_lr_scales_with_batch():
    o = _over(600, batch=4, epochs=50)
    assert o["train_cfg"] == {"type": "EpochBasedTrainLoop", "max_epochs": 50, "val_interval": 1}
    warm, steps = o["param_scheduler"]
    # warmup_iters < 1 là PHẦN của lịch: 0.03 x 7500 = 225.
    assert warm["end"] == 225 and warm["by_epoch"] is False
    assert steps["milestones"] == [35, 45] and steps["end"] == 50    # 70 % và 90 %
    assert o["optim_wrapper"]["optimizer"]["lr"] == pytest.approx(0.01 * 4 / 16)
    assert o["optim_wrapper"]["type"] == "AmpOptimWrapper"
    assert o["train_dataloader"]["batch_size"] == 4
    assert o["load_from"] == "w.pth" and o["work_dir"] == "out"
    assert o["default_hooks"]["checkpoint"]["save_best"] == "coco/segm_mAP"
    assert o["model"]["test_cfg"] == {"score_thr": 0.05, "max_per_img": 100}
    assert iters_per_epoch(600, 4) == 150

    # Warmup không dài hơn cả lần chạy; epoch ít thì mốc giảm lr không rơi về 0.
    o = _over(10, batch=4, epochs=1, warmup_iters=200)
    assert o["param_scheduler"][0]["end"] == 3
    assert o["param_scheduler"][1]["milestones"] == [1]
    assert _over(lr=0.001)["optim_wrapper"]["optimizer"]["lr"] == 0.001
    assert _over(amp=False)["optim_wrapper"]["type"] == "OptimWrapper"


def test_dataset_blocks_point_at_the_fold_and_keep_background_images():
    o = _over(limit=16, workers=8)
    tr = o["train_dataloader"]["dataset"]
    assert tr["type"] == "CocoDataset" and tr["metainfo"]["classes"] == CLASSES
    assert tr["ann_file"] == "annotations/instances_train.json"
    assert tr["data_prefix"] == {"img": "images/train/"}
    assert tr["filter_cfg"]["filter_empty_gt"] is False
    assert tr["indices"] == 16
    assert "indices" not in o["val_dataloader"]["dataset"]
    assert o["val_dataloader"]["dataset"]["test_mode"] is True
    assert o["train_dataloader"]["num_workers"] == 8 and o["train_dataloader"]["persistent_workers"]
    assert _over(workers=0)["train_dataloader"]["persistent_workers"] is False
    assert o["val_evaluator"]["ann_file"].endswith("instances_val.json")
    te = o["test_evaluator"]
    assert te["ann_file"].endswith("instances_test.json") and te["format_only"] is False
    assert te["outfile_prefix"].replace("\\", "/").endswith("out/test/pred")


def test_imgsz_means_long_side_and_flips_are_independent():
    o = _over(imgsz=1024, aspect=9 / 16)
    pipe = o["train_dataloader"]["dataset"]["pipeline"]
    resize = next(t for t in pipe if t["type"] == "Resize")
    assert resize["scale"] == (1024, 576) and resize["keep_ratio"] is True
    assert _over(imgsz=1024, aspect=3 / 4)["test_dataloader"]["dataset"]["pipeline"][1]["scale"] == (1024, 768)
    flip = next(t for t in pipe if t["type"] == "RandomFlip")
    assert flip["direction"] == ["horizontal", "vertical", "diagonal"]
    assert flip["prob"] == [0.25, 0.25, 0.25]                     # p=q=0.5 độc lập
    assert [t["type"] for t in pipe] == ["LoadImageFromFile", "LoadAnnotations", "Resize",
                                         "RandomFlip", "PackDetInputs"]
    assert flip_transform(0.5, 0.0) == {"type": "RandomFlip", "prob": [0.5], "direction": ["horizontal"]}
    assert flip_transform(0, 0) is None
    assert "RandomFlip" not in [t["type"] for t in _over(fliplr=0, flipud=0)["train_dataloader"]["dataset"]["pipeline"]]


def test_bad_inputs_fail_early():
    with pytest.raises(ValueError, match="arch"):
        build_overrides("yolo", "r", SPLITS, 100, MM_DEFAULTS)
    with pytest.raises(ValueError):
        _over(0)
    with pytest.raises(ValueError, match="arch"):
        MMDetTrainer({"model": {"arch": "condinst"}}, "x")
    assert set(ZOO) == set(ARCHS) and all({"config", "checkpoint", "overrides", "lr"} <= set(v) for v in ZOO.values())


def test_trainer_contract_and_run_tag():
    names = MMDetTrainer.param_names()
    assert {"imgsz", "batch", "epochs", "lr", "workers", "val_conf"} <= names
    assert MMDetTrainer.locked_params() == frozenset()
    assert MMDetTrainer.run_tag({"train": {"imgsz": 2048, "batch": 1}}) == "i2048b1e50"
    assert MMDetTrainer.run_tag({}) == "i1024b16e50"


def test_prepare_counts_train_images_and_checks_class_names(three_splits, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    t = MMDetTrainer({"model": {"arch": "solov2"},
                      "data": {"root": str(three_splits), "min_area": 50.0},
                      "train": {"batch": 2}}, run)
    info = t.prepare()
    assert info["arch"] == "solov2" and info["classes"] == ["canopy"]
    assert info["train"]["images"] == 3 and info["train"]["dropped"] == 1
    assert info["iters_per_epoch"] == 2 and t.n_train == 3
    assert t.aspect == pytest.approx(300 / 400)
    assert json.loads((run / "dataset_check.json").read_text(encoding="utf-8"))["test"]["images"] == 3
    assert t.checkpoint().endswith(("solov2_r50_fpn_3x_coco.pth", "fed092d4.pth"))
    assert MMDetTrainer({"model": {"weights": "x.pth"}}, run).checkpoint() == "x.pth"

    # Tên lớp trong bản xuất lệch với config -> dừng trước khi đụng GPU.
    ann = three_splits / "annotations" / "instances_train.json"
    doc = json.loads(ann.read_text(encoding="utf-8"))
    doc["categories"][0]["name"] = "tree"
    ann.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="categories"):
        MMDetTrainer({"data": {"root": str(three_splits)}}, run).prepare()


def test_zoo_config_merges_with_overrides_without_mmcv(three_splits, tmp_path):
    pytest.importorskip("mmengine")
    run = tmp_path / "run"
    run.mkdir()
    t = MMDetTrainer({"data": {"root": str(three_splits), "limit": 2},
                      "train": {"batch": 1, "epochs": 2, "imgsz": 256, "val_conf": 0.1}}, run)
    cfg = t._cfg()
    assert cfg.model.type == "SOLOv2" and cfg.model.mask_head.num_classes == 1
    assert cfg.model.test_cfg.score_thr == 0.1 and cfg.model.test_cfg.mask_thr == 0.5
    assert cfg.train_cfg.max_epochs == 2 and cfg.train_dataloader.dataset.indices == 2
    assert cfg.train_dataloader.dataset.pipeline[2].scale == (256, 192)
    assert cfg.default_scope == "mmdet" and cfg.work_dir.endswith("mmdet")
    assert cfg.val_evaluator.metric == "segm" and cfg.test_evaluator.outfile_prefix.endswith("pred")
    # Recipe gốc của SOLOv2 phải còn nguyên chỗ không ghi đè.
    assert cfg.model.mask_head.num_grids == [40, 36, 24, 16, 12]
    assert cfg.optim_wrapper.clip_grad.max_norm == 35
    # Config dump được (đây là file cạnh best.pth mà MMDetModel đọc lại).
    cfg.dump(str(tmp_path / "dump.py"))
    assert "num_classes=1" in (tmp_path / "dump.py").read_text(encoding="utf-8")
    assert load_config("solov2", {}).model.mask_head.num_classes == 1


@pytest.mark.gpu
def test_probe_runs_two_real_iterations(three_splits, tmp_path):
    pytest.importorskip("mmcv")
    run = tmp_path / "run"
    run.mkdir()
    t = MMDetTrainer({"data": {"root": str(three_splits)},
                      "train": {"batch": 1, "imgsz": 256}}, run)
    p = t.probe()
    assert p["supported"] and p["peak_gb"] > 0
