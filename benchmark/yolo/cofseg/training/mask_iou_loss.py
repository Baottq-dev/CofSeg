"""L1 (`--loss mask_iou`): điểm phân loại học IoU MẶT NẠ thay cho CIoU của hộp.

Ở ultralytics, mục tiêu phân loại của một anchor dương đã là một "chất lượng"
chứ không phải số 1 (TAL, cùng họ VarifocalNet/GFL), nhưng chất lượng đó là
CIoU của HỘP (`utils/tal.py:167-173, 242`). Với đầu one-to-one của YOLO26 — đầu
dùng lúc suy luận — mỗi tán có đúng một anchor dương và mục tiêu của nó chính
là CIoU hộp của anchor đó. Trong khi AP mặt nạ xếp hạng theo điểm và chấm theo
IoU MẶT NẠ. Đo trên đợt 2: điểm của YOLO26s chỉ tương quan rho = 0,64 với IoU
mặt nạ thật, và xếp lại theo IoU thật thì được +3,58 AP ở cả sáu ruộng
(docs/reports/model/phan_tich_chuyen_sau_va_de_xuat_kien_truc_2026-10-04.md).

Thay đổi duy nhất, chỉ ở mục tiêu phân loại của anchor dương:

    q = IoU( (logit mặt nạ > 0) ∩ hộp DỰ ĐOÁN ,  mặt nạ GT )     không có gradient
    t = (1 - w) * t_TAL + w * q,     w = mix * min(1, epoch / warmup_epochs)

q cắt bằng hộp dự đoán vì lúc suy luận mặt nạ cũng bị cắt đúng như vậy
(`ops.process_mask_native` -> `crop_mask`), nên tán bị hộp cắt cụt phải nhận
điểm thấp. Giữ nguyên: ai được làm dương (TAL theo hộp), loss hộp và trọng số
của nó, mẫu số chuẩn hoá, loss mặt nạ, loss ngữ nghĩa. Với mix = 0 hàm này cho
loss trùng từng số với v8SegmentationLoss — test giữ điều đó.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors


def is_end2end(model) -> bool:
    """YOLO26 có hai đầu (one-to-many + one-to-one); YOLOv8/11 chỉ có một."""
    return getattr(model.model[-1], "one2one_cv2", None) is not None


@torch.no_grad()
def mask_iou(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """IoU giữa mặt nạ dự đoán (đã cắt bằng hộp dự đoán) và mặt nạ GT.

    pred_logits (n, h, w) logit trên lưới proto; pred_boxes (n, 4) xyxy cùng
    lưới đó; gt_masks (n, h, w) nhị phân. Ngưỡng logit 0 = xác suất 0,5, đúng
    ngưỡng lúc suy luận.
    """
    pm = crop_mask((pred_logits > 0).float(), pred_boxes) > 0
    gm = gt_masks > 0.5
    inter = (pm & gm).flatten(1).sum(1).float()
    union = (pm | gm).flatten(1).sum(1).float()
    return inter / union.clamp(min=1.0)


class MaskIoUSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với mục tiêu phân loại theo IoU mặt nạ."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 mix: float = 1.0, warmup_epochs: int = 5):
        super().__init__(model, tal_topk, tal_topk2)
        self.mix = float(mix)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0
        # n, Σs, Σq, Σsq, Σs², Σq² của anchor dương trong epoch — đủ cho trung
        # bình và tương quan Pearson mà không phải chép từng giá trị về CPU.
        self.stats = torch.zeros(6, dtype=torch.float64, device=self.device)

    @property
    def weight(self) -> float:
        if self.warmup_epochs <= 0:
            return self.mix
        return self.mix * min(1.0, self.epoch / self.warmup_epochs)

    def loss(self, preds, batch):
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        else:
            pred_semantic = None

        # ---- gán nhãn: chép từ v8DetectionLoss.get_assigned_targets_and_loss (8.4.143),
        # chỉ khác là giữ lại pred_bboxes và target_labels, và CHƯA tính loss phân loại.
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, đơn vị ô lưới
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # ---- mặt nạ: loss như gốc, kèm q cho từng anchor dương
        _, _, mask_h, mask_w = proto.shape
        q = torch.zeros(fg_mask.shape, device=self.device)
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
            loss[1], q = self._mask_loss_and_quality(
                fg_mask, masks, target_gt_idx, target_bboxes, batch["batch_idx"].view(-1, 1),
                proto, pred_masks, imgsz, (pred_bboxes.detach() * stride_tensor),
            )
            if pred_semantic is not None:  # nguyên văn v8SegmentationLoss.loss
                sem_idx = batch["sem_masks"].to(self.device).long().unsqueeze(1)
                if self.overlap:
                    present = masks != 0
                else:
                    bidx = batch["batch_idx"].view(-1)
                    present = torch.zeros(batch_size, *masks.shape[-2:], dtype=torch.bool, device=self.device)
                    for i in range(batch_size):
                        inst = masks[bidx == i]
                        if len(inst):
                            present[i] = inst.sum(dim=0) != 0
                sem_masks = torch.zeros(sem_idx.shape[0], self.nc, *sem_idx.shape[2:], device=self.device)
                sem_masks.scatter_(1, sem_idx, present.unsqueeze(1).float())
                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        # ---- phân loại: mục tiêu mới chỉ ở anchor dương, mẫu số giữ như gốc
        cls_target = target_scores
        w = self.weight
        if fg_mask.any():
            fg = fg_mask
            t_tal = target_scores[fg].sum(-1)  # đúng một lớp khác 0 ở mỗi anchor dương
            qf = q[fg]
            if w > 0:
                onehot = F.one_hot(target_labels[fg].long(), self.nc).to(target_scores.dtype)
                cls_target = target_scores.clone()
                cls_target[fg] = onehot * ((1 - w) * t_tal + w * qf.to(target_scores.dtype))[:, None]
            with torch.no_grad():
                s = pred_scores.detach()[fg].sigmoid().gather(1, target_labels[fg].long()[:, None]).squeeze(1).double()
                qd = qf.double()
                self.stats += torch.stack([torch.tensor(float(len(qd)), dtype=torch.float64, device=qd.device),
                                           s.sum(), qd.sum(), (s * qd).sum(), (s * s).sum(), (qd * qd).sum()])
        bce = self.bce(pred_scores, cls_target.to(dtype))
        if self.class_weights is not None:
            bce *= self.class_weights
        loss[2] = bce.sum() / target_scores_sum * self.hyp.cls

        # ---- hộp: nguyên văn, với target_scores GỐC
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
            )
        else:
            loss[0] += pred_distri[..., :0].sum()
        loss[0] *= self.hyp.box
        loss[3] *= self.hyp.dfl

        loss[1] *= self.hyp.box  # seg gain, như gốc
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))

    def _mask_loss_and_quality(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                               proto, pred_masks, imgsz, pred_bboxes_px):
        """calculate_segmentation_loss + single_mask_loss của gốc, thêm q."""
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        scale = torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        mxyxy = target_bboxes_normalized * scale
        pxyxy = pred_bboxes_px / imgsz[[1, 0, 1, 0]] * scale  # hộp DỰ ĐOÁN trên lưới proto
        q = torch.zeros(fg_mask.shape, device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, pxyxy)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, pxyxy_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = (masks[i] == (mask_idx + 1).view(-1, 1, 1)).float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                pred_mask = torch.einsum("in,nhw->ihw", pred_masks_i[fg_mask_i], proto_i)
                bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
                loss += (crop_mask(bce, mxyxy_i[fg_mask_i]).mean(dim=(1, 2)) / marea_i[fg_mask_i]).sum()
                q[i, fg_mask_i] = mask_iou(pred_mask.detach(), pxyxy_i[fg_mask_i], gt_mask)
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum(), q


def check_model(model, p: dict) -> None:
    """Chặn trước khi train: heads=o2o cần model có đầu one-to-one."""
    if p["heads"] == "o2o" and not is_end2end(model):
        raise SystemExit(
            "loss.mask_iou.heads=o2o chỉ có nghĩa với model đầu-cuối (YOLO26). "
            "YOLOv8/11 chỉ có một đầu, dùng heads=both."
        )


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(MaskIoUSegLoss, mix=p["mix"], warmup_epochs=p["warmup_epochs"])
    if not is_end2end(model):
        return make(model)
    crit = E2ELoss(model, make)
    if p["heads"] == "o2o":
        # Đầu one-to-many về lại loss gốc, cùng topk mà E2ELoss vừa dựng.
        crit.one2many = v8SegmentationLoss(model, tal_topk=crit.one2many.assigner.topk)
    return crit


def _parts(crit) -> list[tuple[str, MaskIoUSegLoss]]:
    if isinstance(crit, MaskIoUSegLoss):
        return [("one", crit)]
    out = []
    for name in ("one2one", "one2many"):
        part = getattr(crit, name, None)
        if isinstance(part, MaskIoUSegLoss):
            out.append((name, part))
    return out


class EpochLog:
    """Mỗi epoch một dòng vào loss_mask_iou.csv: w, và với từng đầu dùng L1 là
    số anchor dương, q trung bình, điểm trung bình, tương quan Pearson điểm-q.

    Đây là chỉ số cơ chế đọc được ngay lúc train: L1 có tác dụng thì tương
    quan của đầu one-to-one phải tăng dần. Ghi ra file riêng chứ không nhét vào
    dict loss: trainer suy ra cột results.csv từ dict đó, còn validator chấm
    trên model EMA với hàm loss gốc, nên thêm khoá vào đó là lệch cột val.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        for _, part in _parts(crit):
            part.epoch = epoch
            part.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        parts = _parts(crit)
        if not parts:
            return
        row = {"epoch": epoch + 1, "w": round(parts[0][1].weight, 4)}
        for name, part in parts:
            n, ss, sq, ssq, ss2, sq2 = part.stats.tolist()
            if n > 1:
                cov = ssq / n - (ss / n) * (sq / n)
                var_s, var_q = ss2 / n - (ss / n) ** 2, sq2 / n - (sq / n) ** 2
                r = cov / (var_s * var_q) ** 0.5 if var_s > 0 and var_q > 0 else float("nan")
            else:
                r = float("nan")
            row.update({f"{name}_n": int(n), f"{name}_q": round(sq / n, 4) if n else float("nan"),
                        f"{name}_score": round(ss / n, 4) if n else float("nan"),
                        f"{name}_r": round(r, 4)})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
