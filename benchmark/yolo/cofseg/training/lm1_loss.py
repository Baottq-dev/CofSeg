"""LM1 (`--loss lm1`): đẩy ứng viên vẽ đúng của mỗi tán vượt mức nền thật của chính ảnh đó.

Đo trên R2 trước khi train: 65% số tán bị bỏ lỡ đã có ứng viên vẽ đúng (IoU mặt nạ
≥ 0,5) nhưng điểm chỉ trung vị 0,015, nên bị loại ở ngưỡng 0,05 (nhóm A1). Trên ảnh
train qua tăng cường, A1 vẫn chiếm 1,18% số tán, 96% có ứng viên đúng nằm ngay
trong top-10 của bộ gán TAL. Varifocal (TH2) đẩy được các tán này lên trên 0,05
nhưng bằng cách nới lực ép lên nền, nên nền nổi theo và AP50 đứng yên.

LM1 là một term xếp hạng MỘT CHIỀU cộng thêm vào loss gốc, chỉ ở đầu one-to-many:

    với mỗi ảnh b, mỗi tán g có nhãn:
      P(g)   neo dương TAL (top-10) của g
      q_a    IoU mặt nạ của neo a (cắt theo hộp dự đoán của chính a, như lúc suy
             luận), không lan gradient
      i*     neo trong P(g) có q lớn nhất (bằng nhau thì logit lớn hơn)
      τ_b    mức nền: logit lớn thứ k trong các neo âm "sạch" của ảnh có mặt nạ
             IoU < bg_iou với MỌI tán. Neo âm sạch = không thuộc P nào và IoU hộp
             ≤ 0,7 với mọi neo dương (bỏ bóng NMS). Tách gradient.
      cổng   q_i* ≥ q_min  VÀ  ( max_{a∈P(g)} z_a < τ_b + Δ
                                 HOẶC  [neo điểm cao nhất của g có q < q_min VÀ z_i* < τ_b + Δ] )
      term   T · softplus((τ_b + Δ − z_i*) / T)

    L_LM1 = Σ term / số tán có q_i* ≥ q_min trong batch
    loss  = loss gốc (đã trộn one2many/one2one) + λ(e) · L_LM1 · batch_size

- Gradient chỉ đi vào logit của i* ở những tán bị cổng bật: không đẩy nền xuống,
  không chạm hộp, hệ số mặt nạ hay Proto. BCE gốc giữ nguyên làm neo hiệu chỉnh.
  Khoảng 70% dự đoán nền điểm cao là tán thật thiếu nhãn, nên loss nào ép nền
  mạnh hơn BCE sẽ ép đúng vào chúng.
- Mức nền tính TRONG ẢNH: R2 tắt mosaic, mỗi ảnh train là một ảnh gốc của một
  ruộng, nên lệch thang điểm theo ruộng không lẫn vào phép so.
- Mức nền chỉ lấy neo có mặt nạ không chạm tán: phép kiểm P1 cho thấy nếu chỉ lọc
  theo IoU hộp thì khoảng một nửa số neo âm đứng đầu là mảnh hoặc dự đoán trùng
  của chính tán có nhãn, và loss sẽ đẩy tán lên trên mảnh của nó thay vì lên trên nền.
- Vế thứ hai của cổng phủ kiểu "lệch điểm–mặt nạ" (24% tán A1 trên ảnh train): một
  neo khác của tán có điểm khá nhưng mặt nạ sai, còn neo vẽ đúng thì điểm gần 0.
- λ(e) = 0 trong start_epoch epoch đầu (q chưa có nghĩa khi đầu mặt nạ còn non),
  rồi tăng tuyến tính lên weight trong ramp_epochs epoch. Term cộng NGOÀI hệ số
  trộn one2many/one2one của E2ELoss, vì hệ số one2many giảm từ 0,8 xuống 0,1 về
  cuối lượt train mà `best.pt` của R2 rơi vào khoảng 0,32–0,45.

YOLOv8/11 chỉ có một đầu thì term cộng thẳng vào loss của đầu đó.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import crop_mask

from .mask_iou_loss import is_end2end, mask_iou

NMS_IOU = 0.7  # ngưỡng NMS lúc chấm: neo âm trùng hộp quá mức này là bóng của neo dương


@torch.no_grad()
def background_level(z: torch.Tensor, boxes: torch.Tensor, pos: torch.Tensor, coef: torch.Tensor,
                     proto: torch.Tensor, gt: torch.Tensor, scale: torch.Tensor, *, k: int, top: int,
                     bg_iou: float) -> float | None:
    """τ của một ảnh: logit lớn thứ k trong các neo âm sạch có mặt nạ IoU < bg_iou với mọi tán.

    z (N,) logit, boxes (N, 4) xyxy px, pos (N,) bool neo dương, coef (N, nm), proto (nm, h, w),
    gt (n, h, w) mặt nạ từng tán trên lưới proto, scale (4,) px -> lưới proto. Xét `top` neo âm
    sạch điểm cao nhất. Trả về None nếu không đủ k neo nền.
    """
    clean = ~pos
    if pos.any():
        clean &= ~(box_iou(boxes, boxes[pos]) > NMS_IOU).any(1)
    cand = torch.nonzero(clean)[:, 0]
    if len(cand) > top:
        cand = cand[torch.topk(z[cand], top).indices]
    if len(cand) < k:
        return None
    pm = crop_mask((torch.einsum("pc,chw->phw", coef[cand], proto) > 0).float(), boxes[cand] * scale)
    pm = pm.flatten(1)
    g = gt.flatten(1).float()
    inter = pm @ g.T
    union = pm.sum(1, keepdim=True) + g.sum(1)[None] - inter
    iou = (inter / union.clamp(min=1.0)).amax(1) if len(g) else torch.zeros(len(cand), device=z.device)
    zb = z[cand][iou < bg_iou]
    if len(zb) < k:
        return None
    return float(torch.topk(zb, k).values[-1])


def _first_per_group(order: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    gs = g[order]
    first = torch.ones_like(gs, dtype=torch.bool)
    first[1:] = gs[1:] != gs[:-1]
    return order[first]


def crown_terms(z: torch.Tensor, q: torch.Tensor, g: torch.Tensor, tau: float, *, margin: float, temp: float,
                q_min: float) -> tuple[torch.Tensor, dict]:
    """Term LM1 của các tán trong một ảnh.

    z (P,) logit của neo dương (giữ gradient), q (P,) IoU mặt nạ, g (P,) chỉ số tán của từng neo.
    Trả về tổng term và các bộ đếm cho nhật ký.
    """
    zd = z.detach()
    o = torch.argsort(zd, descending=True, stable=True)
    o = o[torch.argsort(q[o], descending=True, stable=True)]
    star = _first_per_group(o[torch.argsort(g[o], stable=True)], g)         # i*: q lớn nhất, rồi logit
    o2 = torch.argsort(zd, descending=True, stable=True)
    top = _first_per_group(o2[torch.argsort(g[o2], stable=True)], g)        # neo điểm cao nhất của tán
    q_star, z_star = q[star], z[star]
    edge = tau + margin
    in_r = q_star >= q_min
    lose = zd[top] < edge
    mis = (q[top] < q_min) & (z_star.detach() < edge) & ~lose
    gate = in_r & (lose | mis)
    term = (temp * F.softplus((edge - z_star) / temp) * gate).sum()
    gap = ((edge - z_star.detach()) * gate).sum()
    return term, {"n": len(star), "r": int(in_r.sum()), "gate": int(gate.sum()), "mis": int((gate & mis).sum()),
                  "lose0": int((in_r & (zd[top] < tau)).sum()), "gap": float(gap)}


class LM1SegLoss(v8SegmentationLoss):
    """v8SegmentationLoss của đầu one-to-many, tính thêm term LM1 vào `self.extra`."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *, weight: float = 0.5,
                 margin: float = 1.0, temp: float = 0.25, k: int = 3, start_epoch: int = 5, ramp_epochs: int = 5,
                 bg_iou: float = 0.1, bg_top: int = 300, q_min: float = 0.5, standalone: bool = False):
        super().__init__(model, tal_topk, tal_topk2)
        self.weight, self.margin, self.temp = float(weight), float(margin), float(temp)
        self.k, self.bg_top, self.bg_iou, self.q_min = int(k), int(bg_top), float(bg_iou), float(q_min)
        self.start_epoch, self.ramp_epochs = int(start_epoch), int(ramp_epochs)
        self.standalone = standalone
        self.epoch = 0
        self.extra: torch.Tensor | None = None
        self._assigned = None
        # số batch, số tán có P(g), tán có q* ≥ q_min, tán bị cổng bật, trong đó do vế lệch điểm–mặt nạ,
        # tán thua thật (< τ), Στ, số ảnh có τ, Σ khoảng (τ+Δ−z*) của tán bị cổng bật, ΣL_LM1, Σ BCE điểm gốc
        self.stats = torch.zeros(11, dtype=torch.float64)

    @property
    def lam(self) -> float:
        if self.epoch < self.start_epoch:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.weight
        return self.weight * min(1.0, (self.epoch - self.start_epoch) / self.ramp_epochs)

    def get_assigned_targets_and_loss(self, preds, batch):
        out = super().get_assigned_targets_and_loss(preds, batch)
        self._assigned = out[0]  # fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor
        return out

    def loss(self, preds, batch):
        total, items = super().loss(preds, batch)
        lam = self.lam
        lm1, counts = self._lm1(preds, batch)
        self.extra = lm1 * lam * preds["scores"].shape[0] if lam > 0 else None
        cls = float(items["cls_loss"]) / max(float(self.hyp.cls), 1e-12)
        self.stats += torch.tensor([1.0, *counts, float(lm1.detach()), cls], dtype=torch.float64)
        if self.standalone and self.extra is not None:
            total = total.clone()
            total[2] = total[2] + self.extra
        return total, items

    def _lm1(self, preds, batch) -> tuple[torch.Tensor, list[float]]:
        fg_mask, tgi, _, anchor_points, stride_tensor = self._assigned
        scores = preds["scores"].permute(0, 2, 1)[..., 0].float()
        boxes = (self.bbox_decode(anchor_points, preds["boxes"].permute(0, 2, 1).contiguous()).detach()
                 * stride_tensor).float()
        coef = preds["mask_coefficient"].permute(0, 2, 1).detach().float()
        proto = preds["proto"]
        if isinstance(proto, tuple):
            proto = proto[0]
        proto = proto.detach().float()
        masks = batch["masks"].to(self.device).float()
        if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
            proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
        h, w = proto.shape[-2:]
        H, W = (torch.tensor(preds["feats"][0].shape[2:], device=self.device) * self.stride[0]).tolist()
        scale = torch.tensor([w / W, h / H, w / W, h / H], device=self.device)
        bidx = batch["batch_idx"].view(-1).to(self.device)

        total = scores.new_zeros(())
        n = r = gate = mis = lose0 = n_tau = 0
        s_tau = s_gap = 0.0
        for i in range(scores.shape[0]):
            n_i = int((bidx == i).sum())
            pos = torch.nonzero(fg_mask[i])[:, 0]
            if n_i == 0 or not len(pos):
                continue
            if self.overlap:
                gt = (masks[i][None] == torch.arange(1, n_i + 1, device=self.device)[:, None, None]).float()
            else:
                gt = masks[bidx == i]
            g = tgi[i, pos]
            with torch.no_grad():
                q = mask_iou(torch.einsum("pc,chw->phw", coef[i, pos], proto[i]), boxes[i, pos] * scale, gt[g])
            tau = background_level(scores[i].detach(), boxes[i], fg_mask[i], coef[i], proto[i], gt, scale,
                                   k=self.k, top=self.bg_top, bg_iou=self.bg_iou)
            if tau is None:
                continue
            term, c = crown_terms(scores[i, pos], q, g, tau, margin=self.margin, temp=self.temp, q_min=self.q_min)
            total = total + term
            n += c["n"]; r += c["r"]; gate += c["gate"]; mis += c["mis"]; lose0 += c["lose0"]
            s_tau += tau; n_tau += 1; s_gap += c["gap"]
        lm1 = total / max(r, 1)
        return lm1, [n, r, gate, mis, lose0, s_tau, n_tau, s_gap]


class LM1E2ELoss(E2ELoss):
    """E2ELoss cộng term LM1 của đầu one-to-many sau khi trộn hai đầu."""

    def __call__(self, preds, batch):
        total, items = super().__call__(preds, batch)
        extra = self.one2many.extra
        if extra is not None:
            total = total.clone()
            total[2] = total[2] + extra
        return total, items


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    kw = dict(weight=p["weight"], margin=p["margin"], temp=p["temp"], k=p["k"], start_epoch=p["start_epoch"],
              ramp_epochs=p["ramp_epochs"], bg_iou=p["bg_iou"], bg_top=p["bg_top"], q_min=p["q_min"])
    if not is_end2end(model):
        return LM1SegLoss(model, standalone=True, **kw)
    crit = LM1E2ELoss(model, v8SegmentationLoss)
    crit.one2many = LM1SegLoss(model, tal_topk=10, **kw)
    return crit


def _part(crit) -> LM1SegLoss | None:
    part = getattr(crit, "one2many", crit)
    return part if isinstance(part, LM1SegLoss) else None


class EpochLog:
    """Mỗi epoch một dòng vào loss_lm1.csv:

    - `lambda`: λ của epoch;
    - `share_r`: phần tán có ứng viên vẽ đúng (q* ≥ q_min) trong P(g);
    - `share_gate`: phần tán bị cổng bật; `share_mis`: phần bật nhờ vế lệch điểm–mặt nạ;
    - `share_lose0`: phần tán có điểm cao nhất thấp hơn hẳn mức nền (chỉ số cơ chế: phải giảm);
    - `tau`: mức nền trung bình (logit); `gap`: khoảng τ + Δ − z* trung bình ở tán bị cổng bật;
    - `lm1`, `cls`: L_LM1 và BCE điểm gốc (chưa nhân gain) trung bình theo batch; `ratio` = lm1 / cls.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        part = _part(crit)
        if part is not None:
            part.epoch = epoch
            part.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        part = _part(crit)
        if part is None:
            return
        nb, n, r, gate, mis, lose0, s_tau, n_tau, s_gap, s_lm1, s_cls = part.stats.tolist()
        nan = float("nan")
        row = {"epoch": epoch + 1, "lambda": round(part.lam, 4), "n_crowns": int(n),
               "share_r": round(r / n, 4) if n else nan,
               "share_gate": round(gate / n, 4) if n else nan,
               "share_mis": round(mis / n, 4) if n else nan,
               "share_lose0": round(lose0 / n, 4) if n else nan,
               "tau": round(s_tau / n_tau, 3) if n_tau else nan,
               "gap": round(s_gap / gate, 3) if gate else nan,
               "lm1": round(s_lm1 / nb, 5) if nb else nan,
               "cls": round(s_cls / nb, 5) if nb else nan,
               "ratio": round(s_lm1 / s_cls, 4) if s_cls else nan}
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
