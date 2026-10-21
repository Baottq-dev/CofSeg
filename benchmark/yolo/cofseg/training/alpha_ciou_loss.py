"""TH5 (`--loss alpha_ciou`): α-CIoU thay CIoU trong loss hộp.

α-IoU (He et al. 2021) lấy luỹ thừa α cho cả ba phần của CIoU:

    α-CIoU = IoU^α − (ρ²/c²)^α − (β·v)^α,     loss = 1 − α-CIoU

với ρ khoảng cách hai tâm, c đường chéo hộp bao, v độ lệch tỉ lệ khung và
β = v / (v − IoU + 1) tách gradient, đúng như CIoU của `bbox_iou`. Với α = 3,
gradient ở IoU 0,9 lớn gấp khoảng 3,2 lần ở IoU 0,5, nên loss dồn vào các hộp
đã gần đúng. Lý do: AP50 của R2 khoảng 93 nhưng AP90 chỉ 16,8, và mặt nạ bị
cắt bằng hộp dự đoán lúc suy luận.

Dư địa đã đo trước khi train là nhỏ: ở tán khớp, phần IoU mặt nạ mất vì hộp
dự đoán lệch so với hộp thật chỉ 0,011 trên ảnh train và 0,015 trên ảnh thử.

Chỉ loss hộp đổi, ở cả hai đầu. Bộ gán TAL gọi CIoU riêng (`utils/tal.py`),
nên anchor nào dương và đích điểm của nó giữ nguyên; term L1 khoảng cách cạnh
cũng giữ nguyên. α = 1 cho đúng CIoU gốc — test giữ điều đó.
"""

from __future__ import annotations

import csv
import math
from functools import partial
from pathlib import Path

import torch
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8SegmentationLoss

from .mask_iou_loss import is_end2end


def alpha_ciou(box1: torch.Tensor, box2: torch.Tensor, alpha: float = 3.0, eps: float = 1e-7) -> torch.Tensor:
    """α-CIoU giữa hai dãy hộp xyxy cùng cỡ (…, 4), trả về (…, 1) như `bbox_iou`.

    Các bước và vị trí eps chép theo `bbox_iou(..., xywh=False, CIoU=True)` của
    ultralytics 8.4.143, chỉ thêm luỹ thừa α, để α = 1 ra đúng từng số.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        beta = v / (v - iou + (1 + eps))
    if alpha == 1:
        return iou - (rho2 / c2 + v * beta)
    return iou.pow(alpha) - ((rho2 / c2).pow(alpha) + (v * beta).pow(alpha))


class AlphaCIoUBboxLoss(BboxLoss):
    """BboxLoss với α-CIoU; term L1 (hoặc DFL) giữ nguyên của gốc."""

    def __init__(self, reg_max: int = 16, alpha: float = 3.0):
        super().__init__(reg_max)
        self.alpha = float(alpha)
        # n hộp dương, ΣIoU thường, số hộp IoU >= 0,8, Σ(1 − α-CIoU)
        self.stats = torch.zeros(4, dtype=torch.float64)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        # Gốc tính CIoU rồi bỏ đi: rẻ, và không phải chép phần L1/DFL.
        _, loss_l1 = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                     target_scores_sum, fg_mask, imgsz, stride)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        a = alpha_ciou(pb, tb, self.alpha)
        loss_iou = ((1.0 - a) * weight).sum() / target_scores_sum
        with torch.no_grad():
            plain = _plain_iou(pb.detach().float(), tb.float())
            self.stats = self.stats.to(pb.device)
            self.stats += torch.stack([
                torch.tensor(float(len(plain)), dtype=torch.float64, device=pb.device),
                plain.double().sum(), (plain >= 0.8).sum().double(), (1.0 - a.detach()).double().sum(),
            ])
        return loss_iou, loss_l1


def _plain_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    iw = (torch.minimum(b1[:, 2], b2[:, 2]) - torch.maximum(b1[:, 0], b2[:, 0])).clamp(min=0)
    ih = (torch.minimum(b1[:, 3], b2[:, 3]) - torch.maximum(b1[:, 1], b2[:, 1])).clamp(min=0)
    inter = iw * ih
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1 + a2 - inter).clamp(min=1e-9)


class AlphaCIoUSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với loss hộp α-CIoU."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *, alpha: float = 3.0):
        super().__init__(model, tal_topk, tal_topk2)
        self.bbox_loss = AlphaCIoUBboxLoss(self.reg_max, alpha).to(self.device)


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(AlphaCIoUSegLoss, alpha=p["alpha"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def _parts(crit) -> list[tuple[str, AlphaCIoUBboxLoss]]:
    if isinstance(crit, AlphaCIoUSegLoss):
        return [("one", crit.bbox_loss)]
    return [(name, getattr(crit, name).bbox_loss) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), AlphaCIoUSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào loss_alpha_ciou.csv: với từng đầu, số hộp dương,
    IoU hộp trung bình, phần hộp có IoU >= 0,8 (chỉ số cơ chế: phải tăng), và
    1 − α-CIoU trung bình."""

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
            n, s_iou, n_hi, s_loss = part.stats.tolist()
            row.update({f"{name}_n": int(n),
                        f"{name}_iou": round(s_iou / n, 4) if n else float("nan"),
                        f"{name}_iou_08": round(n_hi / n, 4) if n else float("nan"),
                        f"{name}_loss": round(s_loss / n, 4) if n else float("nan")})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
