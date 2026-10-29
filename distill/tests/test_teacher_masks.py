"""Phần thầy và dataset của chắt lọc: tìm trọng số và lượt chấm của thầy, ghép dự đoán với tán nhãn, ảnh đích
4 kênh, và đích đi qua đúng phép biến đổi của ảnh lúc train (kd_data.py). Fold, ảnh và dự đoán dựng tại chỗ."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mu = pytest.importorskip("pycocotools.mask")

from distill import teacher_masks as tm  # noqa: E402
from distill.teachers import TEACHERS, find_eval, find_weights, parse_teachers  # noqa: E402

W0, H0 = 200, 120
FOLD = "f9"
# Hai tán chữ nhật không chạm nhau ở ảnh a, một tán ở ảnh b, ảnh c không có tán (ảnh nền). Toạ độ px.
RECTS = {"a": [(20, 20, 80, 70), (110, 30, 180, 100)], "b": [(40, 30, 150, 95)], "c": []}


def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32) / np.array([W0, H0], np.float32)


def _mask(x0, y0, x1, y1):
    """Mặt nạ của đa giác, đúng phép tô của pycocotools (cùng cách bộ dựng tính IoU)."""
    xy = (_poly(x0, y0, x1, y1) * [W0, H0]).astype(np.float64).ravel().tolist()
    return mu.decode(mu.merge(mu.frPyObjects([xy], H0, W0))).astype(bool)


def _rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


@pytest.fixture
def fold(tmp_path):
    root = tmp_path / FOLD
    for d in ("images/train", "labels/train", "annotations"):
        (root / d).mkdir(parents=True)
    (root / "fold.json").write_text(json.dumps({"fold": FOLD, "source_sha1": "abc"}), encoding="utf-8")
    rng = np.random.default_rng(0)
    images = []
    for k, (name, rects) in enumerate(RECTS.items()):
        img = rng.integers(60, 200, (H0, W0, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / "train" / f"{name}.jpg"), img)
        lines = ["0 " + " ".join(f"{v:.6f}" for v in _poly(*r).ravel()) for r in rects]
        (root / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                              encoding="utf-8")
        images.append({"id": k + 1, "file_name": f"{name}.jpg", "width": W0, "height": H0})
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps({"images": images, "annotations": [], "categories": [{"id": 1, "name": "canopy"}]}),
        encoding="utf-8")
    return root


def _eval(evals, teacher, preds, split="train", fold=FOLD, limit=None, stamp="2026-10-29_000000"):
    t = TEACHERS[teacher]
    d = evals / f"{stamp}_{t.run_name}_field-{fold}_{split}_i1024"
    d.mkdir(parents=True)
    m = {"data": {"root": f"data/export/field/{fold}", "split": split, "limit": limit},
         "model": {"weights": f"runs/train_2/x_{t.run_name}_field-{fold}_i1024/weights/{t.weights}"}}
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    out = [{"image_id": i, "category_id": 1, "segmentation": _rle(mk), "score": s} for i, mk, s in preds]
    (d / "predictions.json").write_text(json.dumps(out), encoding="utf-8")
    return d


@pytest.fixture
def evals(tmp_path):
    """yolo26s: tán 0 khớp đúng, tán 1 lệch 10 px, một dự đoán thừa; solov2: tán 0 bị co 5 px mỗi cạnh."""
    e = tmp_path / "eval"
    a0, a1 = RECTS["a"]
    shifted = (a1[0] + 10, a1[1], a1[2] + 10, a1[3])
    _eval(e, "yolo26s", [(1, _mask(*a0), 0.9), (1, _mask(*shifted), 0.8), (1, _mask(85, 85, 105, 115), 0.6),
                         (2, _mask(*RECTS["b"][0]), 0.95)])
    shrunk = (a0[0] + 5, a0[1] + 5, a0[2] - 5, a0[3] - 5)
    _eval(e, "solov2", [(1, _mask(*shrunk), 0.7)])
    return e


# ---------------------------------------------------------------- bảng thầy, tìm file
def test_bang_thay_va_tap_con():
    assert [t.key for t in parse_teachers(None)] == ["yolo26s", "solov2", "m2f"]
    assert [t.key for t in parse_teachers("m2f,yolo26s")] == ["yolo26s", "m2f"]
    with pytest.raises(SystemExit, match="không có"):
        parse_teachers("yolo26s,abc")


def test_tim_trong_so_dung_mot_thu_muc(tmp_path):
    t = TEACHERS["solov2"]
    w = tmp_path / f"2026-09-29_1_{t.run_name}_field-f1_i1024b16e100" / "weights"
    w.mkdir(parents=True)
    (w / "best.pth").write_bytes(b"x")
    assert find_weights(tmp_path, t, "f1") == w / "best.pth"
    with pytest.raises(SystemExit, match="đúng một"):
        find_weights(tmp_path, t, "f2")
    w2 = tmp_path / f"2026-09-30_1_{t.run_name}_field-f1_i1024b16e100" / "weights"
    w2.mkdir(parents=True)
    (w2 / "best.pth").write_bytes(b"x")
    with pytest.raises(SystemExit, match="đúng một"):
        find_weights(tmp_path, t, "f1")


def test_lenh_suy_luan_thay(tmp_path, fold, evals):
    from distill.teacher_infer import commands

    runs = tmp_path / "train_2"
    for t in TEACHERS.values():
        w = runs / f"2026-09-29_1_{t.run_name}_field-{FOLD}_i1024b16e100" / "weights"
        w.mkdir(parents=True)
        (w / t.weights).write_bytes(b"x")
    plan = commands(str(fold), str(runs), str(tmp_path / "out"))
    assert [t.key for t, _, _ in plan] == ["yolo26s", "solov2", "m2f"]
    for t, cmd, done in plan:
        assert done is None and cmd[1] == f"benchmark/{t.bench}/evaluate.py"
        assert cmd[cmd.index("--split") + 1] == "train" and cmd[cmd.index("--runs") + 1] == str(tmp_path / "out")
        assert cmd[cmd.index("--config") + 1] == f"benchmark/{t.bench}/configs/eval/{t.config}"
    # thầy đã có lượt chấm train thì bỏ qua (chạy lại sau khi bị ngắt)
    plan = commands(str(fold), str(runs), str(tmp_path))
    skipped = {t.key for t, cmd, _ in plan if cmd is None}
    assert skipped == {"yolo26s", "solov2"}


# ---------------------------------------------------------------- ghép, ảnh đích
def test_work_shape_nhu_load_image():
    assert tm.work_shape(1440, 2560, 1024) == (576, 1024)
    assert tm.work_shape(120, 200, 192) == (116, 192)       # ceil(115,2)
    assert tm.work_shape(100, 100, 100) == (100, 100)


def test_ghep_mot_mot_theo_iou_giam_dan():
    iou = np.array([[0.9, 0.0], [0.7, 0.6], [0.95, 0.1], [0.0, 0.4]])
    # tán 0: dự đoán 2 (0,95) thắng dự đoán 0; tán 1: dự đoán 1 (0,6); dự đoán 3 dưới ngưỡng
    assert tm.match(iou) == {0: (2, 0.95), 1: (1, 0.6)}
    assert tm.match(np.zeros((0, 3))) == {}


def test_anh_dich_giu_hai_tan_lon_nhat():
    s = {0: np.array([[1.0, 0.6, 0.0]], np.float32), 1: np.array([[0.2, 0.8, 0.0]], np.float32),
         2: np.array([[0.1, 0.3, 0.0]], np.float32)}
    m = tm.top2_map(s, (1, 3))
    assert m[0, 0].tolist() == [0, 255, 1, 51]
    assert m[0, 1].tolist() == [1, 204, 0, 153]                # tán 2 (0,3) bị loại: chỉ giữ hai tán lớn nhất
    assert m[0, 2].tolist() == [tm.NONE, 0, tm.NONE, 0]


def _decode(m, j):
    return (m[..., 1] / 255.0) * (m[..., 0] == j) + (m[..., 3] / 255.0) * (m[..., 2] == j)


def test_dung_dich_trung_binh_va_thay_khop_nhat(fold, evals, tmp_path):
    meta = tm.build(str(fold), str(evals), str(tmp_path / "t"), "yolo26s,solov2", "mean", imgsz=192, log=lambda s: None)
    ra = meta["images"]["a"]
    assert ra["has"] == [1, 1]
    assert ra["iou"]["yolo26s"][0] > 0.98 and 0.7 < ra["iou"]["yolo26s"][1] < 0.9
    assert 0.6 < ra["iou"]["solov2"][0] < 0.7 and ra["iou"]["solov2"][1] == -1
    assert meta["images"]["c"] == {"n": 0, "boxes": [], "iou": {"yolo26s": [], "solov2": []}, "has": []}
    assert meta["summary"]["gt_by_teacher_count"] == [0, 2, 1]   # tán a1, b0: một thầy; tán a0: hai thầy
    m = cv2.imread(str(tmp_path / "t" / FOLD / "maps" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert m.shape == (116, 192, 4)
    r = 192 / 200
    at = lambda x, y: (int(y * r), int(x * r))               # noqa: E731
    assert _decode(m, 0)[at(50, 45)] == 1.0                     # lòng tán 0: cả hai thầy
    assert abs(_decode(m, 0)[at(22, 45)] - 0.5) < 0.02          # vành ngoài phần co của solov2: một nửa
    assert _decode(m, 1)[at(150, 60)] == 1.0
    assert _decode(m, 1)[at(185, 60)] == 1.0                    # phần lệch ra ngoài nhãn của thầy vẫn giữ
    assert (m[..., 0][at(95, 100)] == tm.NONE)                  # dự đoán thừa không ghép được bị bỏ
    best = tm.build(str(fold), str(evals), str(tmp_path / "b"), "yolo26s,solov2", "best", imgsz=192, log=lambda s: None)
    mb = cv2.imread(str(tmp_path / "b" / FOLD / "maps" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert best["mode"] == "best" and _decode(mb, 0)[at(22, 45)] == 1.0   # thầy khớp nhất là yolo26s


@pytest.mark.parametrize("kw,match", [({"split": "test"}, "split"), ({"fold": "f8"}, "thư mục dữ liệu"),
                                      ({"limit": 5}, "allow-partial")])
def test_luot_cham_sai_bi_tu_choi(fold, tmp_path, kw, match):
    e = tmp_path / "eval"
    d = _eval(e, "yolo26s", [], **{k: v for k, v in kw.items() if k != "fold"})
    if "fold" in kw:  # lượt chấm của fold khác đặt nhầm tên thư mục
        mt = json.loads((d / "metrics.json").read_text())
        mt["data"]["root"] = "data/export/field/f8"
        (d / "metrics.json").write_text(json.dumps(mt))
    if kw.get("split") == "test":
        with pytest.raises(SystemExit, match="Không thấy lượt chấm train"):
            tm.build(str(fold), str(e), str(tmp_path / "t"), "yolo26s", imgsz=192, log=lambda s: None)
        assert find_eval(e, TEACHERS["yolo26s"], FOLD, "test") is not None
        return
    with pytest.raises(SystemExit, match=match):
        tm.build(str(fold), str(e), str(tmp_path / "t"), "yolo26s", imgsz=192, log=lambda s: None)


# ---------------------------------------------------------------- dataset
def _gt_teacher(fold, tmp_path, imgsz):
    """Thầy cho đúng mặt nạ nhãn: đích thầy phải trùng mặt nạ nhãn sau mọi phép biến đổi."""
    e = tmp_path / "eval_gt"
    preds = [(k + 1, _mask(*r), 0.9) for k, rects in enumerate(RECTS.values()) for r in rects]
    _eval(e, "yolo26s", preds)
    tm.build(str(fold), str(e), str(tmp_path / "gt"), "yolo26s", imgsz=imgsz, log=lambda s: None)
    return tmp_path / "gt"


def _dataset(fold, targets, imgsz, **hyp):
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg

    from distill.kd_data import make_dataset

    h = get_cfg(overrides={"flipud": 0.5, "fliplr": 0.5, "scale": 0.5, "translate": 0.1, "mosaic": 0.0,
                           "mask_ratio": 4, "overlap_mask": True, "imgsz": imgsz, **hyp})
    ds_cls = make_dataset(targets, fold)
    return ds_cls(img_path=str(fold / "images" / "train"), imgsz=imgsz, augment=True, hyp=h, batch_size=2,
                  data={"names": {0: "canopy"}, "nc": 1, "channels": 3}, task="segment")


def test_dich_di_cung_tan_qua_lat_co_gian_dich(fold, tmp_path):
    torch = pytest.importorskip("torch")
    import random

    targets = _gt_teacher(fold, tmp_path, 192)
    ds = _dataset(fold, targets, 192)
    random.seed(0)
    np.random.seed(0)
    ious = []
    for _ in range(12):
        for i in range(len(ds)):
            s = ds[i]
            m, ids, has, masks = s["kd_map"], s["kd_ids"], s["kd_has"], s["masks"][0]
            assert m.shape == (4, 48, 48) and m.shape[-2:] == masks.shape[-2:]
            assert len(ids) == len(has) == len(s["cls"]) and (s["cls"] == 0).all()
            assert len(set(ids.tolist())) == len(ids)
            for k in range(len(ids)):
                g = masks == k + 1
                soft = (m[1].float() / 255) * (m[0] == ids[k]) + (m[3].float() / 255) * (m[2] == ids[k])
                assert has[k]
                if g.sum() >= 20:
                    b = soft > 0.5
                    ious.append(float((b & g).sum() / (b | g).sum()))
                for j in range(len(ids)):                     # không rơi vào tán khác
                    if j != k:
                        assert soft[masks == j + 1].mean() < 0.2
    assert len(ious) > 20 and np.mean(ious) > 0.85 and min(ious) > 0.6
    b = ds.collate_fn([ds[0], ds[1], ds[2]])
    assert b["kd_map"].shape == (3, 4, 48, 48) and b["kd_map"].dtype == torch.uint8
    assert len(b["kd_ids"]) == len(b["kd_has"]) == len(b["cls"]) == len(b["batch_idx"])


def test_tan_khong_co_thay_va_anh_nen(fold, evals, tmp_path):
    tm.build(str(fold), str(evals), str(tmp_path / "t"), "solov2", imgsz=192, log=lambda s: None)
    ds = _dataset(fold, tmp_path / "t", 192, flipud=0.0, fliplr=0.0, scale=0.0, translate=0.0)
    names = [p.split("\\")[-1].split("/")[-1] for p in ds.im_files]
    sa, sb, sc = (ds[names.index(f"{n}.jpg")] for n in "abc")
    assert sorted(zip(sa["kd_ids"].tolist(), sa["kd_has"].tolist())) == [(0, True), (1, False)]
    assert sb["kd_has"].tolist() == [False] and len(sc["kd_ids"]) == 0
    assert (sc["kd_map"][[1, 3]] == 0).all()


def test_dich_lech_nhan_hoac_imgsz_thi_dung(fold, tmp_path):
    targets = _gt_teacher(fold, tmp_path, 192)
    with pytest.raises(SystemExit, match="imgsz"):
        _dataset(fold, targets, 160)
    p = targets / FOLD / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["images"]["a"]["boxes"][0][0] += 0.05
    p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="không khớp nhãn"):
        _dataset(fold, targets, 192)


def test_align_theo_hop():
    from distill.kd_data import align

    b = np.array([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.2, 0.1, 0.3]])
    assert align(b, b).tolist() == [0, 1, 2]
    assert align(b[[2, 0, 1]], b).tolist() == [2, 0, 1]      # ultralytics bỏ dòng trùng làm đổi thứ tự
    assert align(b[:2] + 0.01, b) is None
    assert align(np.zeros((0, 4)), np.zeros((0, 4))).tolist() == []
