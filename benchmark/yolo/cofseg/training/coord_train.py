"""Phần train của KT4 (toạ độ tương đối trong phép ghép mặt nạ, xem coord_head.py): loss, trainer, nhật ký.

Loss mặt nạ giữ đúng LOẠI và đúng vùng của gốc — BCE từng pixel trong hộp GT,
chia diện tích hộp, cộng qua tán rồi chia số anchor dương — chỉ đổi logit:
thêm phần toạ độ. Khi 5 hệ số toạ độ bằng 0, loss trùng v8SegmentationLoss
từng số (test giữ điều đó).

Toạ độ tính theo hộp nào lúc train là lựa chọn của config (`mask_head.box`):
- `pred`: hộp do chính anchor đó dự đoán, đã tách gradient. Lúc suy luận cũng
  chỉ có hộp dự đoán, nên toạ độ lúc train và lúc chạy cùng một loại; loss mặt
  nạ không kéo được đầu hộp.
- `gt`: hộp thật của tán được gán cho anchor. Toạ độ lúc train luôn chính xác,
  nhưng lúc chạy thật lại là hộp dự đoán.

Hộp dự đoán được giải lại từ đầu ra hộp (`bbox_decode`, như gốc làm bên trong
get_assigned_targets_and_loss nhưng không trả ra), nên không chép đoạn nào của
phần gán nhãn.
"""

from __future__ import annotations

import csv
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import crop_mask, xyxy2xywh
from ultralytics.utils.tal import make_anchors

from .coord_head import CoordSegValidator, coord_axes, coord_term, find_head, is_coord, to_coord

BOX_CHOICES = ("pred", "gt")


class CoordSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với mặt nạ có thêm 5 kênh toạ độ tương đối với hộp."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        head = find_head(model)
        self.kt4 = head.kt4
        self.nm = head.nm
        if self.kt4["box"] not in BOX_CHOICES:
            raise ValueError(f"mask_head.box phải là một trong {BOX_CHOICES}, nhận {self.kt4['box']!r}")
        # n tán, Σ BCE (trung bình trong hộp GT), Σ phần toạ độ / (phần prototype + phần toạ độ)
        # (TB |logit| trong hộp GT), Σ IoU hộp dự đoán – hộp thật.
        self.stats = torch.zeros(4, dtype=torch.float64, device=self.device)

    def loss(self, preds, batch):
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        pd = preds["boxes"].permute(0, 2, 1).contiguous()
        self._pred_boxes = (self.bbox_decode(anchor_points, pd) * stride_tensor).detach()  # (b, A, 4) pixel đầu vào
        return super().loss(preds, batch)

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz):
        _, _, mask_h, mask_w = proto.shape
        # Gốc phóng proto lên cỡ mặt nạ GT khi mask_ratio khác 4, nên suy bước
        # lưới từ chính proto đang có.
        fstride = float(imgsz[0]) / mask_h
        # Ba dòng dưới y như gốc (loss.py:622-628).
        tbn = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(tbn)[..., 2:].prod(2)
        mxyxy = tbn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        cbox = self._pred_boxes if self.kt4["box"] == "pred" else target_bboxes

        loss = 0
        for i in range(fg_mask.shape[0]):
            fg = fg_mask[i]
            if fg.any():
                mask_idx = target_gt_idx[i][fg]
                if self.overlap:
                    gt = (masks[i] == (mask_idx + 1).view(-1, 1, 1)).float()
                else:
                    gt = masks[batch_idx.view(-1) == i][mask_idx]
                pred = pred_masks[i][fg]
                base = torch.einsum("in,nhw->ihw", pred[:, :self.nm], proto[i])
                u, v = coord_axes(cbox[i][fg], mask_h, mask_w, fstride)
                extra = coord_term(pred[:, self.nm:], u, v)
                logits = base + extra
                bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
                per = crop_mask(bce, mxyxy[i][fg]).mean(dim=(1, 2))
                loss += (per / marea[i][fg]).sum()
                with torch.no_grad():
                    region = crop_mask(torch.ones_like(gt), mxyxy[i][fg])
                    a = (extra.abs() * region).flatten(1).sum(1)
                    b = (base.float().abs() * region).flatten(1).sum(1)
                    biou = box_iou(self._pred_boxes[i][fg].float(), target_bboxes[i][fg].float()).diagonal()
                    self.stats += torch.stack([
                        torch.tensor(float(fg.sum()), dtype=torch.float64, device=per.device),
                        per.double().sum(), (a / (a + b).clamp(min=1e-9)).double().sum(), biou.double().sum(),
                    ])
            else:
                # như gốc: giữ đồ thị cho DDP khi ảnh không có anchor dương
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()


def build_criterion(model):
    """Hàm loss của CoordSegmentationModel.init_criterion(); bọc E2ELoss như gốc."""
    if getattr(model.model[-1], "one2one_cv2", None) is not None:
        return E2ELoss(model, CoordSegLoss)
    return CoordSegLoss(model)


class CoordSegTrainer(SegmentationTrainer):
    """SegmentationTrainer dựng model KT4 và chấm val bằng đầu KT4.

    Dựng YOLO26-seg chuẩn từ yaml, nạp trọng số, RỒI mới chuyển đầu sang
    KT4: 32 kênh hệ số cũ nhận đúng trọng số COCO như R2, 5 kênh mới bằng 0.
    Khi --resume, checkpoint đã là KT4 (37 kênh) nên chuyển trước rồi nạp.
    Cấu hình đi vào qua lớp con do make_trainer() tạo.
    """

    KT4: dict = {}

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, None, verbose)
        if weights is not None and is_coord(weights):
            to_coord(model, self.KT4)
            model.load(weights)
        else:
            if weights:
                model.load(weights)
            to_coord(model, self.KT4)
        return model

    def get_validator(self):
        super().get_validator()  # đặt loss_names như gốc
        return CoordSegValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                 _callbacks=self.callbacks)


def make_trainer(cfg: dict) -> type[CoordSegTrainer]:
    return type("CoordSegTrainer", (CoordSegTrainer,), {"KT4": dict(cfg)})


def _parts(crit) -> list[tuple[str, CoordSegLoss]]:
    if isinstance(crit, CoordSegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2many", "one2one")
            if isinstance(getattr(crit, name, None), CoordSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào mask_head_kt4.csv, với từng nhánh:
    số tán đã tính loss, BCE trung bình trong hộp GT, tỉ phần của phần toạ độ
    trong logit (TB |toạ độ| / (TB |prototype| + TB |toạ độ|) trong hộp GT:
    0 nghĩa là model chưa dùng toạ độ), IoU trung bình giữa hộp dự đoán và hộp
    thật của các anchor dương (độ lệch mà toạ độ `pred` phải chịu)."""

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
            n, s_bce, s_share, s_biou = part.stats.tolist()
            avg = (lambda s: round(s / n, 4)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_bce": avg(s_bce), f"{name}_coord_share": avg(s_share),
                        f"{name}_box_iou": avg(s_biou)})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)


def install_log(yolo_model, run_dir: str | Path) -> None:
    """Gắn nhật ký epoch vào lượt train sắp chạy (criterion tạo lười ở batch đầu)."""
    from ultralytics.utils.torch_utils import unwrap_model

    log = EpochLog(Path(run_dir) / "mask_head_kt4.csv")

    def crit(trainer):
        return getattr(unwrap_model(trainer.model), "criterion", None)

    yolo_model.add_callback("on_train_epoch_start", lambda t: log.start(crit(t), t.epoch))
    yolo_model.add_callback("on_train_epoch_end", lambda t: log.end(crit(t), t.epoch))
