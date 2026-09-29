"""AP ở ngưỡng IoU cao phải đọc đúng ô, và phải khớp với AP đã ghi.

Vì sao cần test riêng cho hai con số tưởng như tầm thường: pycocotools dò
ngưỡng bằng `np.where(iouThr == p.iouThrs)`, mà `np.linspace(.5, .95, 10)[8]`
là 0.8999999999999999. Phép so bằng đó TRƯỢT với 0.90 và hàm trả về -1 mà
không báo lỗi — 0.50, 0.75, 0.95 thì khớp, đúng 0.90 là không. Một cột toàn
-1 trông y hệt một cột "model không đạt ngưỡng nào", nên sai này đi thẳng vào
báo cáo mà không ai nhận ra.

Bất biến thứ hai, mạnh hơn: trung bình mười ngưỡng PHẢI bằng đúng `AP`.
Không xấp xỉ — cùng một mảng `precision`, cùng một phép `mean`, nên lệch một
chữ số nghĩa là đọc nhầm trục.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo")

pytest.importorskip("pycocotools.mask")
cv2 = pytest.importorskip("cv2")

H, W = 300, 400


def _cofseg(model: str):
    """Nạp gói `cofseg` của riêng một thư mục model."""
    d = BENCH / model
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for k in [m for m in sys.modules if m == "cofseg" or m.startswith("cofseg.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cofseg"] = m
    spec.loader.exec_module(m)
    return m


def _ce(model: str):
    _cofseg(model)
    from cofseg.evaluation import coco_eval

    return coco_eval


def _o_vuong(x0, y0, s):
    return [float(x0), float(y0), float(x0 + s), float(y0),
            float(x0 + s), float(y0 + s), float(x0), float(y0 + s)]


@pytest.fixture
def bo_nho(tmp_path):
    """Bộ COCO bé + dự đoán LỆCH DẦN, để mười ngưỡng IoU ra mười giá trị khác
    nhau thay vì cùng bằng 0 hoặc cùng bằng 1.

    Mỗi ảnh một vùng 100x100; dự đoán là chính ô đó dịch sang phải i px, nên
    IoU = (100-i)/(100+i) chạy từ 1.0 xuống ~0.6. Nhờ vậy AP50 khác AP90 và
    test phân biệt được "đọc đúng ô" với "đọc nhầm ô nhưng may mà giống nhau".
    """
    n = 12
    images = [{"id": i + 1, "file_name": f"a{i}.jpg", "width": W, "height": H}
              for i in range(n)]
    anns = [{"id": i + 1, "image_id": i + 1, "category_id": 1,
             "segmentation": [_o_vuong(50, 50, 100)], "area": 10000.0,
             "bbox": [50, 50, 100, 100], "iscrowd": 0} for i in range(n)]
    gt = tmp_path / "instances_test.json"
    gt.write_text(json.dumps({
        "info": {}, "licenses": [], "images": images, "annotations": anns,
        "categories": [{"id": 1, "name": "canopy", "supercategory": "canopy"}],
    }), encoding="utf-8")

    from pycocotools import mask as mask_utils

    det = []
    for i in range(n):
        m = np.zeros((H, W), np.uint8)
        m[50:150, 50 + i:150 + i] = 1
        rle = mask_utils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("ascii")
        det.append({"image_id": i + 1, "category_id": 1, "segmentation": rle,
                    "score": 1.0 - 0.01 * i})
    return str(gt), det, [i + 1 for i in range(n)]


@pytest.mark.parametrize("model", MODELS)
def test_trung_binh_muoi_nguong_bang_dung_AP(model, bo_nho):
    """Bất biến gốc: AP = trung bình AP tại mười ngưỡng IoU.

    Đây là phép kiểm mạnh nhất có thể viết mà không cần dữ liệu thật — nó bắt
    mọi kiểu đọc nhầm trục (lấy nhầm area, nhầm maxDets, nhầm thứ tự).
    """
    ce = _ce(model)
    gt, det, ids = bo_nho
    out = ce.evaluate(gt, det, ids, boundary=False, box=False, show_progress=False)
    dai = out["mask"]["AP_by_iou"]
    assert len(dai) == 10, f"{model}: mong mười ngưỡng, có {len(dai)}"
    assert abs(float(np.mean(list(dai.values()))) - out["mask"]["AP"]) < 1e-12, (
        f"{model}: trung bình mười ngưỡng lệch với AP -> đọc nhầm trục")


@pytest.mark.parametrize("model", MODELS)
def test_AP90_khong_phai_la_am_mot(model, bo_nho):
    """Cái bẫy chính: `_summarize(1, iouThr=0.9)` trả -1 vì so bằng dấu phẩy động.

    Bộ dữ liệu ở trên có cặp IoU = 1.0 nên AP90 BẮT BUỘC phải dương. Thấy -1 ở
    đây nghĩa là ai đó đã thay `ap_at_iou` bằng lối gọi cũ.
    """
    ce = _ce(model)
    gt, det, ids = bo_nho
    out = ce.evaluate(gt, det, ids, boundary=False, box=False, show_progress=False)
    for k in ("AP90", "AP95"):
        assert k in out["mask"], f"{model}: thiếu {k}"
        assert out["mask"][k] > 0, (
            f"{model}: {k} = {out['mask'][k]} — dấu hiệu của phép so bằng trượt")
    assert out["mask"]["AP50"] >= out["mask"]["AP90"] >= out["mask"]["AP95"], (
        f"{model}: AP phải giảm khi ngưỡng IoU tăng")


@pytest.mark.parametrize("model", MODELS)
def test_nguong_ngoai_dai_thi_bao_loi_chu_khong_lay_gan_nhat(model, bo_nho):
    """Dò bằng khoảng cách nhỏ nhất mà không chặn thì gọi 0.99 sẽ lặng lẽ trả
    về ô 0.95 — đúng cái kiểu sai mà test này ra đời để chặn."""
    ce = _ce(model)
    gt, det, ids = bo_nho
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(gt)
        ev = COCOeval(c, c.loadRes(list(det)), iouType="segm")
        ev.params.imgIds = sorted(ids)
        ev.evaluate()
        ev.accumulate()
    with pytest.raises(ValueError, match="0.99"):
        ce.ap_at_iou(ev, 0.99)


@pytest.mark.parametrize("model", MODELS)
def test_bon_ban_cofseg_van_giong_nhau(model):
    """Bốn thư mục model dùng CHUNG một bộ chấm; lệch một bản là bảng so sánh
    đo hai luật khác nhau mà không dòng log nào nói ra."""
    for ten in ("coco_eval.py", "report.py"):
        a = (BENCH / "yolo" / "cofseg" / "evaluation" / ten).read_bytes()
        b = (BENCH / model / "cofseg" / "evaluation" / ten).read_bytes()
        assert a == b, f"{model}/{ten} đã lệch với bản của yolo"


def test_cot_AP90_co_trong_bang_tong_hop():
    """Thêm khoá vào metrics.json mà quên bảng thì con số chấm rồi không ai thấy."""
    sys.path.insert(0, str(ROOT))
    from canopyseg.evaluation import folds as rep

    assert ("AP90", ("coco", "mask", "AP90"), 3, 1) in rep.METRICS
    md = rep.to_markdown([{"dataset": "field", "field": "f6", "model": "m", "images": 48,
                           "mAP": 0.687, "dmAP_pct": 0.0, "AP50": 0.95, "AP75": 0.82,
                           "AP90": 0.184, "run": "x"}])
    assert "AP90" in md and "0.184" in md
    # Lượt chấm cũ chưa có khoá -> phải in "—" chứ không được nổ.
    assert "—" in rep.to_markdown([{"dataset": "field", "field": "f6", "model": "m",
                                    "images": 48, "mAP": 0.687, "run": "x"}])
