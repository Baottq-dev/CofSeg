"""LM3 (`--loss lm3`): α-CIoU chỉ cho những tán mô hình đã chắc chắn; tán yếu giữ CIoU gốc.

TH5 (α-CIoU, α = 3, cho mọi hộp) tăng AP90 1,74 nhưng mất 0,91 AP50 và bỏ lỡ thêm
24 tán mỗi ruộng. Lý do: với α = 3, gradient của hộp IoU 0,3 chỉ còn 0,27 lần CIoU,
nên hộp của tán còn yếu kém đi; đích điểm TAL lại tính từ IoU hộp, nên điểm của
chúng tụt theo. 242 tán TH5 làm mất có điểm R2 trung vị 0,288 và CIoU hộp 0,886:
hộp vốn tốt, chỉ điểm thấp.

LM3 bật α-CIoU theo từng tán, bằng một cổng mềm tính từ chính dự đoán của mô hình:

    với mỗi tán g:  s*_g = điểm lớn nhất trong P(g),  u*_g = CIoU hộp lớn nhất trong P(g)
                    (P(g) = neo dương TAL của g; cả hai tách gradient)
    γ_g  = σ((s*_g − τ_s)/κ) · σ((u*_g − τ_u)/κ)
    loss hộp = Σ_a t̂_a · [ (1 − γ_g(a))·(1 − CIoU_a) + γ_g(a)·(1 − αCIoU_a) ] / Σ t̂

Tán chưa chắc (γ ≈ 0) nhận đúng gradient như R2; tán đã chắc (γ ≈ 1) được mài như
TH5. Ngưỡng τ_s = 0,6 chọn bằng phép kiểm P3: ở ngưỡng này 81% tán TH5 làm mất nằm
dưới cổng, trong khi 89% tán trên ảnh train vẫn bật cổng. Ngưỡng 0,3 chỉ giữ được 56%.

Chỉ loss hộp đổi, ở cả hai đầu, mỗi đầu dùng điểm và hộp của chính nó. Bộ gán TAL,
đích điểm, term L1 khoảng cách cạnh, loss mặt nạ giữ nguyên. α = 1, hoặc cổng không
bao giờ bật, cho đúng loss gốc — test giữ điều đó.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8SegmentationLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from .alpha_ciou_loss import _plain_iou, alpha_ciou
from .mask_iou_loss import is_end2end


class RecordingAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner y hệt gốc, giữ lại điểm đầu vào và kết quả gán của lần gọi gần nhất."""

    last = None

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        out = super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        self.last = (pd_scores.detach(), out[3], out[4])  # điểm (sigmoid), fg_mask, target_gt_idx
        return out


@torch.no_grad()
def crown_gate(s: torch.Tensor, u: torch.Tensor, group: torch.Tensor, *, tau_s: float, tau_u: float,
               kappa: float) -> torch.Tensor:
    """γ của từng neo dương: cổng của tán chứa nó, tính từ điểm và CIoU lớn nhất trong tán.

    s, u (P,) điểm và CIoU của neo dương; group (P,) mã tán (duy nhất trong batch).
    """
    n = int(group.max()) + 1 if len(group) else 0
    s_max = torch.full((n,), -1.0, device=s.device).scatter_reduce(0, group, s.float(), reduce="amax")
    u_max = torch.full((n,), -1.0, device=s.device).scatter_reduce(0, group, u.float(), reduce="amax")
    gate = torch.sigmoid((s_max - tau_s) / kappa) * torch.sigmoid((u_max - tau_u) / kappa)
    return gate[group]


class GatedAlphaCIoUBboxLoss(BboxLoss):
    """BboxLoss với α-CIoU có cổng theo tán; term L1 (hoặc DFL) giữ nguyên của gốc."""

    def __init__(self, reg_max: int, assigner: RecordingAssigner, *, alpha: float = 3.0, tau_s: float = 0.6,
                 tau_u: float = 0.7, kappa: float = 0.05):
        super().__init__(reg_max)
        self.assigner = assigner
        self.alpha, self.tau_s, self.tau_u, self.kappa = float(alpha), float(tau_s), float(tau_u), float(kappa)
        # n hộp dương, Σγ, số hộp γ ≥ 0,5, ΣIoU thường, số hộp IoU ≥ 0,8, số tán, số tán γ ≥ 0,5
        self.stats = torch.zeros(7, dtype=torch.float64)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        # Gốc tính CIoU rồi bỏ đi: rẻ, và không phải chép phần L1/DFL.
        _, loss_l1 = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                     target_scores_sum, fg_mask, imgsz, stride)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        scores, fg, tgi = self.assigner.last
        assert torch.equal(fg, fg_mask), "lần gán gần nhất không khớp với fg_mask đang tính loss hộp"
        ciou = alpha_ciou(pb, tb, 1.0)
        s = scores[fg_mask].amax(-1)
        group = torch.nonzero(fg_mask)[:, 0] * (int(tgi.max()) + 1) + tgi[fg_mask]
        gamma = crown_gate(s, ciou.detach().squeeze(-1), group, tau_s=self.tau_s, tau_u=self.tau_u,
                           kappa=self.kappa)[:, None]
        if self.alpha == 1:
            per = 1.0 - ciou
        else:
            per = (1.0 - gamma) * (1.0 - ciou) + gamma * (1.0 - alpha_ciou(pb, tb, self.alpha))
        loss_iou = (per * weight).sum() / target_scores_sum
        with torch.no_grad():
            plain = _plain_iou(pb.detach().float(), tb.float())
            gm = gamma.squeeze(-1)
            _, first = torch.unique(group, return_inverse=True)
            n_crowns = int(first.max()) + 1 if len(first) else 0
            crown_on = torch.zeros(n_crowns, device=pb.device).scatter_reduce(0, first, (gm >= 0.5).float(), "amax")
            self.stats = self.stats.to(pb.device)
            self.stats += torch.tensor([float(len(plain)), float(gm.sum()), float((gm >= 0.5).sum()),
                                        float(plain.sum()), float((plain >= 0.8).sum()), float(n_crowns),
                                        float(crown_on.sum())], dtype=torch.float64, device=pb.device)
        return loss_iou, loss_l1


class LM3SegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với loss hộp α-CIoU có cổng theo tán."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *, alpha: float = 3.0,
                 tau_s: float = 0.6, tau_u: float = 0.7, kappa: float = 0.05):
        super().__init__(model, tal_topk, tal_topk2)
        a = self.assigner
        self.assigner = RecordingAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta,
                                          stride=a.stride, eps=a.eps, topk2=a.topk2)
        self.bbox_loss = GatedAlphaCIoUBboxLoss(self.reg_max, self.assigner, alpha=alpha, tau_s=tau_s,
                                                tau_u=tau_u, kappa=kappa).to(self.device)


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(LM3SegLoss, alpha=p["alpha"], tau_s=p["tau_s"], tau_u=p["tau_u"], kappa=p["kappa"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def _parts(crit) -> list[tuple[str, GatedAlphaCIoUBboxLoss]]:
    if isinstance(crit, LM3SegLoss):
        return [("one", crit.bbox_loss)]
    return [(name, getattr(crit, name).bbox_loss) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), LM3SegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào loss_lm3.csv, với từng đầu:

    - `n`: số hộp dương; `gate`: γ trung bình; `gate_on`: phần hộp có γ ≥ 0,5;
    - `crown_on`: phần tán có cổng bật (chỉ số cơ chế: phải cao, khoảng 0,9 như phép kiểm P3);
    - `iou`, `iou_08`: IoU hộp trung bình và phần hộp có IoU ≥ 0,8 (phải tăng như TH5).
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
        nan = float("nan")
        for name, part in parts:
            n, s_g, n_on, s_iou, n_hi, n_cr, n_cr_on = part.stats.tolist()
            row.update({f"{name}_n": int(n),
                        f"{name}_gate": round(s_g / n, 4) if n else nan,
                        f"{name}_gate_on": round(n_on / n, 4) if n else nan,
                        f"{name}_crown_on": round(n_cr_on / n_cr, 4) if n_cr else nan,
                        f"{name}_iou": round(s_iou / n, 4) if n else nan,
                        f"{name}_iou_08": round(n_hi / n, 4) if n else nan})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
