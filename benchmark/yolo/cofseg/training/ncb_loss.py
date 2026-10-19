"""TH1 (`--loss ncb`): nhãn mặt nạ mềm theo tỉ lệ phủ (NCB), với Proto gốc.

BCE mặt nạ gốc so với nhãn NHỊ PHÂN ở lưới 1/4 (`mask_ratio: 4`): ô vắt qua
biên tán bị gán hẳn 0 hoặc hẳn 1. Riêng việc làm tròn đó đặt trần IoU 0,962
trong khi nhãn theo tỉ lệ phủ cho trần 0,986. Đo trên R2 trước khi train, mặt
nạ hụt 0,059 IoU so với mức prototype vẽ được ngay trên ẢNH TRAIN (0,078 trên
ảnh thử), nên loss mặt nạ còn dư địa trên chính dữ liệu đã học.

NCB đã có trong T1 (gpr_train.py) nhưng chạy chung với GPR, nên không tách
được phần lợi của nhãn mềm. Ở đây chỉ còn NCB:

1. nhãn raster ở đủ độ phân giải đầu vào (`--set train.mask_ratio=1`; losses.py
   chặn lượt chạy nếu thiếu), avg-pool về lưới 1/4 của Proto gốc -> tỉ lệ phủ
   của từng ô;
2. làm mềm bằng Gauss σ (mặc định 2,4 px đầu vào, bằng nhiễu nhãn đã đo);
3. BCE với đích mềm, cùng vùng (trong box GT) và cùng cách chuẩn hoá như gốc.

Khác T1: không GPR, không giới hạn số anchor dương (T1 lấy mẫu 64 mỗi ảnh vì
lưới 512² của GPR; ở lưới 256² của Proto gốc, mỗi nhãn mềm tốn đúng bằng một
nhãn nhị phân của gốc). Hộp, điểm, gán nhãn và cả hai đầu giữ nguyên. Nhãn của
đầu ngữ nghĩa phụ lấy từ cùng nhãn đủ độ phân giải, thu về cỡ dự đoán bằng
nearest như `BCEDiceLoss` vẫn làm.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

from ultralytics.utils.loss import E2ELoss

from .gpr_train import NCBSegLoss, _parts
from .mask_iou_loss import is_end2end


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(NCBSegLoss, sigma=p["sigma"], max_pos=0)
    return E2ELoss(model, make) if is_end2end(model) else make(model)


class EpochLog:
    """Mỗi epoch một dòng vào loss_ncb.csv: với từng đầu, số anchor dương đã
    tính loss, BCE trung bình với đích mềm, và IoU cứng (logit > 0 so với đích
    > 0,5) trong box GT ở lưới proto."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        for _, part in _parts(crit):
            part.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        parts = _parts(crit)
        if not parts:
            return
        row: dict = {"epoch": epoch + 1}
        for name, part in parts:
            n, s_bce, s_iou = part.stats.tolist()
            avg = (lambda s: round(s / n, 4)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_bce": avg(s_bce), f"{name}_iou": avg(s_iou)})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
