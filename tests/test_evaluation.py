"""Kiểm phần ghép instance bằng hình dựng sẵn có đáp án biết trước.

Cùng lý do như tests/test_metrics.py: nếu phần ghép sai, bảng kết quả vẫn in ra
những con số trông rất hợp lý và không có gì báo động. Recall tụt vì model tệ
hay vì hàm ghép bỏ sót cặp — nhìn bảng thì không phân biệt được.
"""

import numpy as np
import pytest

from canopyseg.datasets.region import Region
from canopyseg.evaluation import align_masks, match_instances
from canopyseg.models.base import Prediction

W, H = 400, 300


def square_region(x0, y0, size, ann_id=1, image="field_1__10__a.jpg") -> Region:
    p = np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
        dtype=np.float64,
    )
    return Region(polygon=p, image_file=image, image_id=1, width=W, height=H,
                  area=float(size * size), ann_id=ann_id)


def square_pred(x0, y0, size, score=0.9) -> Prediction:
    m = np.zeros((H, W), bool)
    m[y0 : y0 + size, x0 : x0 + size] = True
    return Prediction(mask=m, origin=(0, 0), score=score)


# ------------------------------------------------------------------ căn khung
def test_align_puts_both_masks_in_one_frame():
    r = square_region(50, 40, 60)
    p = square_pred(50, 40, 60)
    g, pm = align_masks(r, p)
    assert g.shape == pm.shape
    # Cùng một hình vuông vẽ hai đường khác nhau phải trùng khớp gần như hoàn toàn.
    inter = np.count_nonzero(g & pm)
    union = np.count_nonzero(g | pm)
    assert inter / union > 0.95


def test_align_handles_prediction_touching_image_edge():
    r = square_region(0, 0, 40)
    p = square_pred(0, 0, 40)
    g, pm = align_masks(r, p)
    assert g.shape == pm.shape and g.any() and pm.any()


# -------------------------------------------------------------------- ghép cặp
def test_perfect_prediction_matches():
    regions = [square_region(50, 50, 60, ann_id=1)]
    preds = [square_pred(50, 50, 60)]
    m, missed, spur = match_instances(regions, preds)
    assert len(m) == 1 and not missed and not spur
    assert m[0].iou > 0.9


def test_disjoint_prediction_does_not_match():
    regions = [square_region(10, 10, 40)]
    preds = [square_pred(200, 200, 40)]
    m, missed, spur = match_instances(regions, preds)
    assert not m and len(missed) == 1 and len(spur) == 1


def test_each_ground_truth_is_claimed_at_most_once():
    """Hai dự đoán chồng lên cùng một tán: một khớp, một thành thừa."""
    regions = [square_region(50, 50, 60)]
    preds = [square_pred(50, 50, 60, score=0.9), square_pred(52, 52, 60, score=0.8)]
    m, missed, spur = match_instances(regions, preds)
    assert len(m) == 1 and len(spur) == 1
    # Tham lam theo điểm tin cậy: cái chắc chắn hơn phải được ưu tiên.
    assert m[0].pred.score == pytest.approx(0.9)


def test_threshold_decides_marginal_overlap():
    regions = [square_region(50, 50, 100)]
    preds = [square_pred(90, 90, 100)]           # chồng nhau một phần
    assert not match_instances(regions, preds, iou_thr=0.5)[0]
    assert match_instances(regions, preds, iou_thr=0.1)[0]


def test_all_regions_accounted_for():
    """Mỗi vùng thật phải rơi vào đúng một nhóm: khớp hoặc bỏ sót."""
    regions = [square_region(10, 10, 40, ann_id=1), square_region(200, 150, 50, ann_id=2)]
    preds = [square_pred(10, 10, 40)]
    m, missed, spur = match_instances(regions, preds)
    assert len(m) + len(missed) == len(regions)
    assert not spur


def test_empty_prediction_mask_is_ignored():
    regions = [square_region(50, 50, 60)]
    empty = Prediction(mask=np.zeros((H, W), bool), origin=(0, 0), score=0.9)
    m, missed, spur = match_instances(regions, [empty])
    assert not m and len(missed) == 1
