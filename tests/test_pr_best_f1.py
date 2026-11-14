"""P, R, F1 của bộ chấm phải tính đúng như ultralytics, và tính lại được từ per_region.csv.

`summary.precision` / `summary.recall` tính trên mọi dự đoán qua ngưỡng chấm (0.05):
ngưỡng đó để giữ đủ dự đoán cho đường PR của mAP, nên hai số đó không phải điểm
vận hành. `summary.pr_best_f1` báo P, R, F1 tại ngưỡng cho F1 lớn nhất, như
ultralytics làm khi in P, R cạnh mAP. Ba bất biến ở đây:

1. Trùng `ultralytics.utils.metrics.ap_per_class` từng chữ số trên cùng TP và điểm.
2. Tính lại từ per_region.csv (chuỗi, điểm đã làm tròn) ra đúng số của lượt chấm mới.
3. `resummarize` chỉ thêm khối mới; đếm lại không khớp summary đã ghi thì không ghi gì.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo")


def _runner(model: str = "yolo"):
    """Nạp `cofseg.evaluation.runner` của riêng một thư mục model."""
    d = BENCH / model
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for k in [m for m in sys.modules if m == "cofseg" or m.startswith("cofseg.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cofseg"] = m
    spec.loader.exec_module(m)
    from cofseg.evaluation import runner

    return runner


def _rows(tp: list[int], scores: list[float], n_missed: int) -> list[dict]:
    """Hàng kiểu per_region: tán khớp (1, có điểm), tán bỏ sót (0), dự đoán thừa (-1)."""
    rows = [{"matched": 1 if t else -1, "score": s, "field": "field_1"} for t, s in zip(tp, scores)]
    rows += [{"matched": 0, "score": "", "field": "field_1"} for _ in range(n_missed)]
    return rows


def test_tinh_tay_mot_vi_du_nho():
    """Mười tán thật, mười dự đoán đúng điểm 0.90-0.99, năm dự đoán sai điểm 0.05-0.09.

    Ở ngưỡng chấm 0.05, P = 10 / 15 (khoá `precision` cũ). Giữa ngưỡng 0.09 và 0.9
    không có dự đoán nào, nên R = 1 cả khoảng. Ultralytics nội suy P TUYẾN TÍNH
    theo điểm giữa hai dự đoán liền nhau (1.0 ở 0.9, 10/11 ở 0.09) rồi làm mượt
    đường F1 bằng cửa sổ 0.1, nên điểm chọn rơi ngay dưới 0.9 với P gần 1 chứ
    không đúng bằng 1."""
    r = _runner()
    rows = _rows([1] * 10 + [0] * 5, [0.9 + 0.01 * i for i in range(10)] + [0.05 + 0.01 * i for i in range(5)],
                 n_missed=0)
    out = r.pr_best_f1(rows)
    assert set(out) == {"conf", "precision", "recall", "f1"}
    assert out["recall"] == 1.0
    assert 0.99 <= out["precision"] < 1.0
    assert out["f1"] == round(2 * out["precision"] / (out["precision"] + 1.0), 4)
    assert 0.8 <= out["conf"] <= 0.9
    assert r.summarize(rows, [{"count_error": 0, "ms": 1.0}])["precision"] == round(10 / 15, 4)


def test_khong_co_du_doan_hay_khong_co_tan():
    r = _runner()
    assert r.pr_best_f1(_rows([], [], n_missed=3)) == {"conf": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert r.pr_best_f1(_rows([0, 0], [0.9, 0.5], n_missed=0))["recall"] == 0.0


def test_trung_ultralytics_tung_chu_so():
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    r = _runner()
    rng = np.random.default_rng(0)
    for n_pred, n_gt in ((50, 40), (400, 300), (1000, 120)):
        scores = np.round(rng.uniform(0.05, 1.0, n_pred), 4)
        tp = (rng.uniform(size=n_pred) < scores).astype(int)
        n_tp = int(tp.sum())
        n_gt = max(n_gt, n_tp)
        ours = r.pr_best_f1(_rows(tp.tolist(), scores.tolist(), n_missed=n_gt - n_tp))
        ref = metrics.ap_per_class(tp[:, None].astype(float), scores, np.zeros(n_pred), np.zeros(n_gt))
        assert ours["precision"] == round(float(ref[2][0]), 4)
        assert ours["recall"] == round(float(ref[3][0]), 4)
        assert ours["f1"] == round(float(ref[4][0]), 4)


def test_tinh_lai_tu_csv_ra_dung_so_luot_cham(tmp_path):
    """Đường vừa chấm (số) và đường đọc lại per_region.csv (chuỗi) phải ra cùng một khối."""
    r = _runner()
    rows = _rows([1, 0, 1, 1, 0, 1, 0], [0.91, 0.88, 0.7, 0.66, 0.4, 0.2, 0.07], n_missed=2)
    summary = r.summarize(rows, [{"count_error": 0, "ms": 1.0}])
    run = tmp_path / "run"
    run.mkdir()
    with (run / "per_region.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["matched", "score", "field"])
        w.writeheader()
        w.writerows(rows)
    cu = dict(summary)
    del cu["pr_best_f1"]
    (run / "metrics.json").write_text(json.dumps({"summary": cu, "coco": {"mask": {"AP": 0.5}}}), encoding="utf-8")

    out = r.resummarize(run)
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert out == summary["pr_best_f1"] == m["summary"]["pr_best_f1"]
    assert m["coco"] == {"mask": {"AP": 0.5}}
    assert {k: v for k, v in m["summary"].items() if k != "pr_best_f1"} == cu
    assert m["resummarized"]["keys"] == ["summary.pr_best_f1"]


def test_dem_khong_khop_thi_khong_ghi(tmp_path):
    r = _runner()
    rows = _rows([1, 0, 1], [0.9, 0.5, 0.3], n_missed=1)
    run = tmp_path / "run"
    run.mkdir()
    with (run / "per_region.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["matched", "score", "field"])
        w.writeheader()
        w.writerows(rows)
    goc = json.dumps({"summary": {"n_gt": 99, "true_positive": 2, "false_positive": 1}})
    (run / "metrics.json").write_text(goc, encoding="utf-8")
    with pytest.raises(ValueError, match="không khớp"):
        r.resummarize(run)
    assert (run / "metrics.json").read_text(encoding="utf-8") == goc


@pytest.mark.parametrize("model", MODELS)
def test_bon_ban_cung_mot_ham(model):
    """Bốn thư mục model dùng chung một bộ chấm; P, R phải tính cùng một cách ở cả bốn."""
    for ten in ("runner.py", "__init__.py"):
        a = (BENCH / "yolo" / "cofseg" / "evaluation" / ten).read_bytes()
        b = (BENCH / model / "cofseg" / "evaluation" / ten).read_bytes()
        assert a == b, f"{model}/{ten} đã lệch với bản của yolo"


@pytest.mark.parametrize("model", MODELS)
def test_evaluate_co_co_resummarize(model):
    src = (BENCH / model / "evaluate.py").read_text(encoding="utf-8")
    assert '"--resummarize"' in src and "run_resummarize(a.resummarize)" in src
