"""SAM 2.1 thật, trên ảnh dựng tại chỗ. Cần weights/sam2.1_hiera_tiny.pt và
package sam2; tự bỏ qua khi thiếu, nên chạy được trên máy khác mà không đỏ.

Ảnh là một đĩa sáng trên nền tối — hình mà mọi SAM đều phải tách được. Cái
được kiểm là đường ống: box vào đúng, mặt nạ ra đúng khung, cắt sát, mặt nạ
gợi ý đi đúng đường, và refine giữ nguyên số dự đoán.
"""

from pathlib import Path

import numpy as np
import pytest

WEIGHTS = Path("weights/sam2.1_hiera_tiny.pt")
pytestmark = [pytest.mark.weights, pytest.mark.slow]


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("sam2")
    if not WEIGHTS.exists():
        pytest.skip(f"thiếu {WEIGHTS}")
    from canopyseg.models.sam_prompt import Sam2Backend

    return Sam2Backend(size="t", checkpoint=str(WEIGHTS), box_batch=2)


@pytest.fixture(scope="module")
def disk_image():
    h, w = 240, 320
    img = np.full((h, w, 3), 30, np.uint8)
    y, x = np.ogrid[:h, :w]
    disk = (x - 160) ** 2 + (y - 120) ** 2 <= 50 ** 2
    img[disk] = (220, 220, 220)
    return img, disk


def _iou(a, b):
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


def test_box_prompt_recovers_the_disk(backend, disk_image):
    img, disk = disk_image
    backend.set_image(img)
    masks, ious = backend.masks_for_boxes(np.array([[105, 65, 215, 175]], np.float32))
    assert masks.shape == (1, 240, 320) and 0 <= ious[0] <= 1
    assert _iou(masks[0], disk) > 0.85


def test_boxes_are_chunked_without_mixing_order(backend, disk_image):
    img, disk = disk_image
    backend.set_image(img)
    boxes = np.array([[105, 65, 215, 175], [0, 0, 40, 40], [105, 65, 215, 175]], np.float32)
    masks, _ = backend.masks_for_boxes(boxes)          # box_batch=2 -> hai lô
    assert masks.shape[0] == 3
    assert _iou(masks[0], disk) > 0.85 and _iou(masks[2], disk) > 0.85
    assert masks[1].sum() < masks[0].sum()


def test_oracle_and_refine_wrappers(backend, disk_image):
    from canopyseg.models.compose import RefineModel
    from canopyseg.models.sam_prompt import OracleSAMModel, SAMRefiner

    img, disk = disk_image
    oracle = OracleSAMModel.__new__(OracleSAMModel)
    oracle.backend = backend
    preds = oracle.predict(img, np.array([[105, 65, 215, 175]], np.float32))
    assert len(preds) == 1
    p = preds[0]
    assert p.origin != (0, 0) or p.mask.shape != (240, 320)   # đã cắt sát bbox
    full = np.zeros((240, 320), bool)
    oy, ox = p.origin[1], p.origin[0]
    full[oy : oy + p.mask.shape[0], ox : ox + p.mask.shape[1]] = p.mask
    assert _iou(full, disk) > 0.85

    refiner = SAMRefiner.__new__(SAMRefiner)
    refiner.backend, refiner.prompt = backend, "box+mask"
    m = RefineModel(base=oracle, refiner=refiner)
    out = m.predict(img, np.array([[105, 65, 215, 175]], np.float32))
    assert len(out) == 1 and out[0].meta["refined"] and out[0].score == p.score
