"""`path:` trong data.yaml là đường dẫn của MÁY ĐÃ CẮT FOLD, không phải máy train.

Cắt fold ở Windows rồi rsync sang máy lab Linux: `path: F:/CoffeeSeg/...` không
phải đường dẫn tuyệt đối trên POSIX, nên ultralytics nối nó vào thư mục dataset
của chính nó và báo thiếu một đường dẫn ghép chẳng ai viết bao giờ:

    /home/student/DE170271_Baottq/LightCoral/F:/CoffeeSeg/data/export/block/f1/images/val

Thông tin đó vốn thừa — data.yaml luôn nằm ở gốc fold — nên trainer ghi đè bằng
chính thư mục chứa nó.
"""

from __future__ import annotations

import json

import yaml

from cofseg.training.yolo import YoloTrainer

FOREIGN = "F:/CoffeeSeg/data/export/block/f1"


def _fold(tmp_path, path_value=FOREIGN):
    fold = tmp_path / "f1"
    for sp in ("train", "val", "test"):
        (fold / "images" / sp).mkdir(parents=True)
        (fold / "labels" / sp).mkdir(parents=True)
    doc = {"path": path_value, "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {0: "canopy"}}
    (fold / "data.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return fold


def _trainer(tmp_path, fold):
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"trainer": "yolo", "model": "yolo11s-seg.pt",
           "data": {"yaml": str(fold / "data.yaml")}, "train": {"epochs": 1}}
    return YoloTrainer(cfg, run)


def _used(tmp_path, fold):
    return yaml.safe_load(
        _trainer(tmp_path, fold)._portable_yaml().read_text(encoding="utf-8"))


def test_path_cua_may_khac_bi_ghi_de_bang_thu_muc_fold(tmp_path):
    fold = _fold(tmp_path)
    assert _used(tmp_path, fold)["path"] == str(fold.resolve()).replace("\\", "/")


def test_path_moi_la_tuyet_doi_that(tmp_path):
    """Chỉ cần tuyệt đối là ultralytics thôi nối vào DATASETS_DIR của nó."""
    from pathlib import Path
    assert Path(_used(tmp_path, _fold(tmp_path))["path"]).is_absolute()


def test_cac_khoa_con_lai_giu_nguyen(tmp_path):
    doc = _used(tmp_path, _fold(tmp_path))
    assert doc["train"] == "images/train" and doc["val"] == "images/val"
    assert doc["test"] == "images/test" and doc["names"] == {0: "canopy"}


def test_khong_dung_vao_data_goc(tmp_path):
    """data/ là dữ liệu dùng chung của nhóm; bản sửa nằm trong run_dir."""
    fold = _fold(tmp_path)
    goc = (fold / "data.yaml").read_text(encoding="utf-8")
    t = _trainer(tmp_path, fold)
    out = t._portable_yaml()
    assert (fold / "data.yaml").read_text(encoding="utf-8") == goc
    assert out == t.run_dir / "data.yaml"


def test_thieu_han_khoa_path_van_duoc_dien(tmp_path):
    fold = _fold(tmp_path)
    doc = yaml.safe_load((fold / "data.yaml").read_text(encoding="utf-8"))
    del doc["path"]
    (fold / "data.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert _used(tmp_path, fold)["path"] == str(fold.resolve()).replace("\\", "/")


def test_goi_hai_lan_chi_ghi_mot_lan(tmp_path):
    t = _trainer(tmp_path, _fold(tmp_path))
    assert t._portable_yaml() is t._portable_yaml()


def test_prepare_bao_ca_hai_duong_dan(tmp_path, monkeypatch):
    """dataset_check.json phải nói rõ đã nạp file NÀO, không chỉ file gốc."""
    fold = _fold(tmp_path)
    t = _trainer(tmp_path, fold)
    import sys
    import types
    mod = types.ModuleType("ultralytics.data.utils")
    mod.check_det_dataset = lambda p, autodownload=False: {
        "nc": 1, "names": {0: "canopy"}, "train": p}
    pkg = types.ModuleType("ultralytics")
    data = types.ModuleType("ultralytics.data")
    monkeypatch.setitem(sys.modules, "ultralytics", pkg)
    monkeypatch.setitem(sys.modules, "ultralytics.data", data)
    monkeypatch.setitem(sys.modules, "ultralytics.data.utils", mod)

    t.prepare()
    out = json.loads((t.run_dir / "dataset_check.json").read_text(encoding="utf-8"))
    assert out["data_yaml"] == str(fold / "data.yaml")
    assert out["data_yaml_used"] == str(t.run_dir / "data.yaml")
