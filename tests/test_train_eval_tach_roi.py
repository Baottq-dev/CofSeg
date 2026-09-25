"""train.py chỉ train và val; chấm là việc của evaluate.py.

Trước đây trainer chấm luôn split test ở cuối rồi ghi predictions.json. Hai
việc hỏng vì thế:

- cùng một checkpoint ra HAI bộ số. Bộ chấm của khung (COCOEvaluator,
  CocoMetric) chỉ cho mAP; evaluate.py cho thêm Boundary AP và chỉ số biên
  từng vùng. Hai con số mAP cũng không bằng nhau vì hai đường suy luận khác
  nhau — và không ai biết nên tin con nào.
- test bị đụng vào mỗi lượt train. Trong một bảng benchmark thì split test
  chỉ được chạm đúng một lần, ở bước chấm cuối.

Test ở đây đọc MÃ NGUỒN chứ không chạy trainer: ba trong bốn trainer cần
detectron2/mmcv, thứ không có trên máy phát triển.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"

#: thư mục model -> file trainer của nó
TRAINERS = {
    "maskrcnn": "cofseg/training/detectron2.py",
    "mask2former": "cofseg/training/detectron2.py",
    "solov2": "cofseg/training/mmdet.py",
    "yolo": "cofseg/training/yolo.py",
}
MODELS = tuple(TRAINERS)


def _src(model: str) -> str:
    return (BENCH / model / TRAINERS[model]).read_text(encoding="utf-8")


def _readme(model: str) -> str:
    return (BENCH / model / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("model", MODELS)
def test_trainer_khong_cham_split_test(model):
    """Không trainer nào được gọi vòng chấm trên split test."""
    src = _src(model)
    for goi in ("cls.test(", "Runner.from_cfg(cfg_t)", ".test() or {}"):
        assert goi not in src, f"{model}: còn gọi vòng chấm test trong trainer"
    assert 'work_dir = str(self.out_dir / "test")' not in src, \
        f"{model}: còn dựng thư mục chấm test"


@pytest.mark.parametrize("model", MODELS)
def test_trainer_khong_ghi_predictions(model):
    """predictions.json chỉ do evaluate.py sinh ra, ở runs/eval/."""
    src = _src(model)
    assert 'run_dir / "predictions.json"' not in src, \
        f"{model}: trainer còn ghi predictions.json"
    assert 'run_dir / "test_metrics.json"' not in src, \
        f"{model}: trainer còn ghi test_metrics.json"


@pytest.mark.parametrize("model", MODELS)
def test_tom_tat_cuoi_khong_co_dong_test(model):
    """Khối cuối lượt train chỉ nói về val.

    Một dòng "test" ở đây là mời người ta chép số vào báo cáo mà không qua
    evaluate.py, tức báo cáo thiếu Boundary AP — chỉ số mà cả dự án dựa vào.
    """
    src = _src(model)
    assert "test_segm" not in src, f"{model}: khối cuối còn nhận số test"
    assert 'label = f"test' not in src, f"{model}: khối cuối còn in dòng test"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_datasets_test_cua_d2_van_tro_vao_val(model):
    """detectron2 gọi split đánh giá lúc train là DATASETS.TEST.

    Tên dễ gây hiểu nhầm: nó phải trỏ vào VAL. Trỏ vào test thật thì
    BestCheckpointer chọn checkpoint bằng chính bộ sẽ chấm nó.
    """
    src = _src(model)
    assert '"DATASETS.TEST", (names.get("val", "coffee_val"),)' in src, \
        f"{model}: DATASETS.TEST không còn trỏ vào val"


@pytest.mark.parametrize("model", MODELS)
def test_readme_lay_predictions_tu_buoc_cham(model):
    """README phải chép predictions.json từ runs/eval, không phải runs/train."""
    doc = _readme(model)
    assert re.search(r'cp "\$EV/predictions\.json" preds/', doc), \
        f"{model}: README không lấy predictions từ thư mục eval"
    assert '"$RUN/predictions.json"' not in doc, \
        f"{model}: README còn lấy predictions từ thư mục train"


@pytest.mark.parametrize("model", MODELS)
def test_readme_cham_bang_trong_so_tot_nhat(model):
    """Bước chấm phải trỏ vào best của lượt train vừa xong."""
    doc = _readme(model)
    assert "evaluate.py" in doc and "--split test" in doc, f"{model}: thiếu bước chấm"
    assert re.search(r'--weights "\$RUN/\S*best\.p', doc), \
        f"{model}: bước chấm không nhận trọng số tốt nhất từ lượt train"


@pytest.mark.parametrize("model", MODELS)
def test_dong_thoi_gian_khong_goi_val_la_train(model):
    """Val tốn gấp đôi train trên bộ này (batch 1), nên gộp hai số rồi dán
    nhãn "train" là nói dối về chỗ hết giờ máy."""
    src = _src(model)
    assert "self._time_row(seconds)" in src, \
        f"{model}: khối cuối còn gộp val vào nhãn train"


def test_time_row_noi_ro_khi_khong_tach_duoc():
    """Không có reporter thì phải nói đó là TỔNG, không phải train."""
    base = (BENCH / "yolo" / "cofseg" / "training" / "base.py").read_text(encoding="utf-8")
    assert "def _time_row" in base, "thiếu _time_row trong Trainer"
    assert '"train + val ' in base, "nhánh không tách được phải nói rõ là tổng"
