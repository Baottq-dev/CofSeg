"""L2 (`--loss dice`): cộng Dice vào loss mặt nạ của từng tán.

Loss mặt nạ gốc của YOLO-seg chỉ có BCE từng pixel, lấy trung bình trong box
GT (`utils/loss.py:586-588`). BCE chấm từng pixel riêng lẻ; không số hạng nào
đo độ chồng của cả mặt nạ, trong khi AP chấm đúng thứ đó. Với mặt nạ nhị phân
Dice = 2·IoU / (1 + IoU), tăng đơn điệu theo IoU. Hai model có mặt nạ khít
nhất của benchmark đều có Dice: SOLOv2 (Dice x3) và Mask2Former (BCE x5 +
Dice x5). Nhánh ngữ nghĩa phụ của chính YOLO26 cũng dùng BCE + Dice
(`loss.py:502`); chỉ mặt nạ từng tán là chưa có.

Thay đổi duy nhất, cộng vào mỗi anchor dương:

    p = sigmoid(logit),   B = vùng trong box GT,   g = mặt nạ GT
    dice = 1 - (2·Σ_B p·g + 1) / (Σ_B p + Σ_B g + 1)
    mask_loss = BCE_trung_bình_trong_B + weight · dice

Giữ nguyên: anchor nào dương, vùng giám sát (trong box GT — nới vùng là việc
của L3, tách ra để mỗi dòng ablation chỉ đo một thứ), nguyên số hạng BCE, cách
chia cho số anchor dương, seg gain, loss hộp/phân loại/ngữ nghĩa, cả hai đầu
của YOLO26. weight = 0 cho loss trùng từng số với v8SegmentationLoss — test
giữ điều đó.

Chỉ cài đè `single_mask_loss`. Gốc gọi nó qua `self.` trong
calculate_segmentation_loss (`loss.py:643`), nên khác mask_iou, ở đây không
phải chép đoạn gán nhãn nào của thư viện.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask

from .mask_iou_loss import is_end2end


def dice_and_iou(pred_logits: torch.Tensor, gt_masks: torch.Tensor,
                 boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(1 − Dice mềm, IoU cứng) của từng tán, chỉ tính trong box.

    pred_logits (n, h, w) logit trên lưới proto; gt_masks (n, h, w) nhị phân;
    boxes (n, 4) xyxy trên cùng lưới. IoU cứng (ngưỡng logit 0, như lúc suy
    luận) không có gradient, chỉ để ghi nhật ký.

    Cộng bằng float32: dưới AMP logit là fp16, mà một box có thể phủ tới
    65 536 ô của lưới 256x256 — quá ngưỡng 65 504 của fp16.
    """
    # crop_mask nhân TẠI CHỖ. Cắt thẳng lên sigmoid hay gt_mask thì hỏng
    # autograd (sigmoid và BCE giữ chính các tensor đó cho lượt lùi), nên cắt
    # một tấm toàn 1 riêng rồi nhân ra ngoài.
    region = crop_mask(torch.ones_like(pred_logits, dtype=torch.float32), boxes)
    p = pred_logits.float().sigmoid() * region
    g = gt_masks.float() * region
    inter = (p * g).flatten(1).sum(1)
    total = p.flatten(1).sum(1) + g.flatten(1).sum(1)
    dice = 1.0 - (2.0 * inter + 1.0) / (total + 1.0)
    with torch.no_grad():
        pm = (pred_logits > 0) & (region > 0)
        gm = g > 0.5
        iou = (pm & gm).flatten(1).sum(1).float() / (pm | gm).flatten(1).sum(1).clamp(min=1).float()
    return dice, iou


class DiceSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với Dice cộng vào loss mặt nạ."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 weight: float = 1.0):
        super().__init__(model, tal_topk, tal_topk2)
        self.weight = float(weight)
        # n, ΣBCE, Σdice, ΣIoU của anchor dương trong epoch.
        self.stats = torch.zeros(4, dtype=torch.float64, device=self.device)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Ba dòng đầu là nguyên văn gốc (8.4.143), chỉ tách tổng ra khỏi trung
        # bình để cộng Dice từng tán.
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        bce = crop_mask(bce, xyxy).mean(dim=(1, 2)) / area
        loss = bce.sum()
        if self.weight > 0:
            dice, iou = dice_and_iou(pred_mask, gt_mask, xyxy)
            loss = loss + self.weight * dice.sum()
        else:
            with torch.no_grad():
                dice, iou = dice_and_iou(pred_mask, gt_mask, xyxy)
        with torch.no_grad():
            self.stats += torch.stack([
                torch.tensor(float(len(bce)), dtype=torch.float64, device=bce.device),
                bce.double().sum(), dice.double().sum(), iou.double().sum(),
            ])
        return loss


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(DiceSegLoss, weight=p["weight"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def _parts(crit) -> list[tuple[str, DiceSegLoss]]:
    if isinstance(crit, DiceSegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), DiceSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào loss_dice.csv: với từng đầu, số anchor dương, BCE
    và 1 − Dice trung bình, và IoU cứng của mặt nạ trong box GT.

    Cột `_iou` là chỉ số cơ chế đọc được ngay lúc train. Ghi ra file riêng vì
    cùng lý do như loss_mask_iou.csv: thêm khoá vào dict loss là lệch cột val.
    """

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
            n, s_bce, s_dice, s_iou = part.stats.tolist()
            avg = (lambda s: round(s / n, 4)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_bce": avg(s_bce),
                        f"{name}_dice": avg(s_dice), f"{name}_iou": avg(s_iou)})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
