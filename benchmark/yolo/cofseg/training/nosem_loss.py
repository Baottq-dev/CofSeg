"""Đối chứng của TC2 (`--loss nosem`): tắt loss ngữ nghĩa phụ của Proto26.

TC2 đổi nhãn của đầu `semseg` từ "hợp mọi tán" sang nhãn có dải tiếp giáp.
Lượt này trả lời câu hỏi đi trước: chính loss ngữ nghĩa gốc có ảnh hưởng gì
tới cách model xử lý chỗ hai tán chạm nhau không.

- Tắt mà lỗi ở chỗ tiếp giáp giữ nguyên: loss gốc gần như không tác động
  vùng đó, nên TC2 phải được hiểu là THÊM tín hiệu tách tán.
- Tắt mà lỗi giảm: nhãn hợp-mọi-tán đang cản việc tách tán, và TC2 vừa bỏ một
  tín hiệu cản vừa thêm một tín hiệu đúng.

Thay đổi duy nhất: loss gốc được gọi với proto KHÔNG kèm đầu ngữ nghĩa, nên
ô `sem_loss` bằng 0. Hộp, phân loại, mặt nạ từng tán, gán nhãn và ProgLoss giữ
nguyên. Đầu `semseg` vẫn chạy (nó nằm trong forward lúc train) nhưng không
nhận gradient; nó bị xoá khi fuse như mọi lượt khác. Chỉ có nghĩa với YOLO26:
YOLOv8/11 không có đầu này.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

from .mask_iou_loss import is_end2end


def check_model(model, params: dict) -> None:
    from ultralytics.nn.modules.block import Proto26

    if not any(isinstance(m, Proto26) for m in model.modules()):
        raise SystemExit("--loss nosem chỉ dùng cho yolo26*-seg: YOLOv8/YOLO11 không có đầu semseg để tắt.")


class NoSemSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với loss ngữ nghĩa phụ bằng 0."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.batches = 0

    def loss(self, preds, batch):
        proto = preds["proto"]
        if not (isinstance(proto, tuple) and len(proto) == 2):
            return super().loss(preds, batch)
        proto_t, sem = proto
        total, items = super().loss({**preds, "proto": proto_t}, batch)
        # Giữ đầu semseg trong đồ thị (như nhánh `else` của gốc) để DDP không
        # báo tham số không dùng; giá trị cộng vào bằng 0.
        add = torch.zeros_like(total)
        add[4] = (sem * 0).sum()
        self.batches += 1
        return total + add, items


def build_criterion(model, p: dict):
    return E2ELoss(model, NoSemSegLoss) if is_end2end(model) else NoSemSegLoss(model)


class EpochLog:
    """Mỗi epoch một dòng vào loss_nosem.csv: số batch đã tắt loss ngữ nghĩa.

    Không có gì để đo thêm (cột sem_loss của results.csv đã bằng 0); file này
    chỉ là dấu vết cho thấy biến thể thật sự chạy ở từng epoch.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _part(crit):
        return getattr(crit, "one2many", crit)

    def start(self, crit, epoch: int) -> None:
        part = self._part(crit)
        if isinstance(part, NoSemSegLoss):
            part.batches = 0

    def end(self, crit, epoch: int) -> None:
        part = self._part(crit)
        if not isinstance(part, NoSemSegLoss):
            return
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=["epoch", "batches_sem_off"])
            if new:
                wr.writeheader()
            wr.writerow({"epoch": epoch + 1, "batches_sem_off": part.batches})
