"""TH4 (`--loss repgt`): CIoU + λ·RepGT ở loss hộp của đầu one-to-many.

RepGT là một nửa của Repulsion Loss (Wang et al. 2018): phạt hộp dự đoán lấn
sang GT BÊN CẠNH, đo bằng phần diện tích của GT bên cạnh bị hộp phủ (IoG):

    với mỗi anchor dương P, mục tiêu t:
        G_rep = GT khác t, cùng ảnh, có IoU lớn nhất với hộp dự đoán của P
        IoG   = |B_P ∩ G_rep| / |G_rep|
    RepGT = trung bình trên mọi anchor dương của Smooth_ln(IoG)
    Smooth_ln(x) = −ln(1 − x)                     khi x <= sigma
                 = (x − sigma)/(1 − sigma) − ln(1 − sigma)   khi x > sigma
    box_loss = (CIoU_gốc + λ·RepGT) · box gain

Bài gốc chọn G_rep theo hộp đề xuất; YOLO26 không có hộp đề xuất nên dùng
hộp dự đoán (tách gradient khi chọn). sigma = 0,5 giữ gradient bị chặn (độ dốc
2) khi một hộp gộp phủ trọn tán bên cạnh; Smooth_ln với sigma gần 1 thì vô
hạn ở IoG = 1.

Lý do và giới hạn đã biết trước khi train: kiểm trên R2 cho thấy hộp gộp chỉ
là nguyên nhân trực tiếp của 8,6% số tán bị bỏ lỡ từ 500 px (19,8% ở tán chạm
nhiều); phần lớn tán bỏ lỡ có ứng viên đúng nhưng điểm thấp. Lượt này kiểm
lại kết luận đó bằng train thật.

Chỉ ở đầu one-to-many: đó là đầu lượt chấm benchmark dùng, và đầu one-to-one
chỉ có một anchor dương mỗi tán. Không chép đoạn gán nhãn nào: loss gốc được
tính trước, rồi hộp dự đoán và GT được dựng lại từ cùng `preds`/`batch` bằng
chính các hàm của lớp (`bbox_decode`, `preprocess`). λ = 0 cho loss trùng gốc.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

from .mask_iou_loss import is_end2end


def smooth_ln(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth_ln của Repulsion Loss. Nhánh log nhận x đã kẹp ở sigma để
    gradient của nhánh không chọn không thành inf·0."""
    inner = -torch.log1p(-x.clamp(max=sigma))
    outer = (x - sigma) / (1 - sigma) - math.log(1 - sigma)
    return torch.where(x <= sigma, inner, outer)


def repgt_terms(pred: torch.Tensor, gts: torch.Tensor, valid: torch.Tensor,
                sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RepGT của từng anchor dương.

    pred (n, 4) hộp dự đoán xyxy; gts (n, M, 4) mọi GT của ảnh chứa anchor đó;
    valid (n, M) GT nào được xét (đã bỏ GT đệm và GT mục tiêu). Trả về (loss
    (n,), IoG (n,), has (n,)) — has = anchor có GT bên cạnh chồng lên hộp.
    """
    with torch.no_grad():
        p = pred.detach()[:, None]
        iw = (torch.minimum(p[..., 2], gts[..., 2]) - torch.maximum(p[..., 0], gts[..., 0])).clamp(min=0)
        ih = (torch.minimum(p[..., 3], gts[..., 3]) - torch.maximum(p[..., 1], gts[..., 1])).clamp(min=0)
        inter = iw * ih
        ap = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
        ag = (gts[..., 2] - gts[..., 0]).clamp(min=0) * (gts[..., 3] - gts[..., 1]).clamp(min=0)
        iou = torch.where(valid, inter / (ap + ag - inter).clamp(min=1e-9), torch.full_like(inter, -1.0))
        best, j = iou.max(1)
        has = best > 0
    g = gts[torch.arange(len(j), device=j.device), j]
    iw = (torch.minimum(pred[:, 2], g[:, 2]) - torch.maximum(pred[:, 0], g[:, 0])).clamp(min=0)
    ih = (torch.minimum(pred[:, 3], g[:, 3]) - torch.maximum(pred[:, 1], g[:, 1])).clamp(min=0)
    area_g = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).clamp(min=1e-9)
    iog = (iw * ih / area_g).clamp(max=1.0)
    loss = smooth_ln(iog.float(), sigma) * has
    return loss, iog.detach(), has


class RepGTSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với RepGT cộng vào loss hộp."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 weight: float = 0.5, sigma: float = 0.5):
        super().__init__(model, tal_topk, tal_topk2)
        self.weight = float(weight)
        self.sigma = float(sigma)
        # n anchor dương, số có GT bên cạnh chồng lên, ΣIoG của chúng, số IoG > 0,5, ΣRepGT
        self.stats = torch.zeros(5, dtype=torch.float64, device=self.device)

    def get_assigned_targets_and_loss(self, preds, batch):
        out, loss, items = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx = out[0], out[1]
        if not fg_mask.any():
            return out, loss, items
        # Dựng lại hộp dự đoán (còn gradient) và GT bằng đúng các bước của gốc.
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        pred_px = self.bbox_decode(anchor_points, pred_distri) * stride_tensor
        bs = pred_distri.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_distri.dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_bboxes = targets[..., 1:5]
        bi, ai = fg_mask.nonzero(as_tuple=True)
        valid = gt_bboxes.sum(2)[bi] > 0
        valid[torch.arange(len(bi), device=bi.device), target_gt_idx[bi, ai]] = False
        per, iog, has = repgt_terms(pred_px[bi, ai].float(), gt_bboxes[bi].float(), valid, self.sigma)
        rep = per.sum() / len(per)
        if self.weight > 0:
            loss[0] = loss[0] + self.weight * rep * self.hyp.box
        with torch.no_grad():
            self.stats += torch.stack([
                torch.tensor(float(len(per)), dtype=torch.float64, device=per.device),
                has.sum().double(), iog[has].double().sum(), (iog > 0.5).sum().double(), per.double().sum(),
            ])
        return out, loss, items


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    if not is_end2end(model):
        return RepGTSegLoss(model, weight=p["weight"], sigma=p["sigma"])
    crit = E2ELoss(model, v8SegmentationLoss)
    crit.one2many = RepGTSegLoss(model, tal_topk=crit.one2many.assigner.topk, weight=p["weight"], sigma=p["sigma"])
    return crit


def _rep(crit) -> RepGTSegLoss | None:
    part = getattr(crit, "one2many", crit)
    return part if isinstance(part, RepGTSegLoss) else None


class EpochLog:
    """Mỗi epoch một dòng vào loss_repgt.csv cho đầu dùng RepGT:

    - `n_pos`: số anchor dương;
    - `share_overlap`: phần anchor dương có hộp chồng lên một GT bên cạnh;
    - `iog`: IoG trung bình của các anchor đó;
    - `share_iog_05`: phần anchor dương phủ quá nửa một GT bên cạnh, tức hộp
      kiểu gộp. Đây là chỉ số cơ chế: RepGT có tác dụng thì nó phải giảm;
    - `repgt`: RepGT trung bình mỗi anchor dương.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        part = _rep(crit)
        if part is not None:
            part.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        part = _rep(crit)
        if part is None:
            return
        n, n_has, s_iog, n_half, s_rep = part.stats.tolist()
        row = {"epoch": epoch + 1, "n_pos": int(n),
               "share_overlap": round(n_has / n, 4) if n else float("nan"),
               "iog": round(s_iog / n_has, 4) if n_has else float("nan"),
               "share_iog_05": round(n_half / n, 4) if n else float("nan"),
               "repgt": round(s_rep / n, 5) if n else float("nan")}
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
