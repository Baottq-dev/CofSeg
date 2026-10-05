"""Phần train của A1 (đầu mặt nạ động, xem dyn_head.py): loss, trainer, nhật ký.

Loss mặt nạ giữ đúng LOẠI của gốc — BCE từng pixel, trung bình theo vùng rồi
theo số tán, nhân seg gain — chỉ đổi vùng: cửa sổ box GT x window thay cho
box GT, vì lúc suy luận mặt nạ không còn bị cắt sát box nữa. Pixel của tán
bên cạnh nằm trong cửa sổ là pixel âm, tức có phạt lấn tán. Không thêm Dice:
L2 (--loss dice) cho thấy trên bộ này Dice làm mặt nạ phình ra, và giữ nguyên
loại loss thì khác biệt đo được là của kiến trúc.

Không chép đoạn nào của thư viện: v8SegmentationLoss.loss() vứt toạ độ anchor
trước khi gọi calculate_segmentation_loss, nên loss() ở đây tính lại chúng
(make_anchors, rẻ) rồi giao phần còn lại cho gốc.
"""

from __future__ import annotations

import csv
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.tal import make_anchors

from .dyn_head import DynSegValidator, dynamic_logits, find_head, to_dynamic, window_boxes


class DynSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với mặt nạ dựng bằng đầu A1, giám sát trong cửa sổ."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.dyn = find_head(model).dyn
        # n, ΣBCE, ΣIoU cứng (trong cửa sổ) của các tán đã tính loss trong epoch.
        self.stats = torch.zeros(3, dtype=torch.float64, device=self.device)

    def loss(self, preds, batch):
        self._anchors = make_anchors(preds["feats"], self.stride, 0.5)
        return super().loss(preds, batch)

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz):
        anchor_points, stride_tensor = self._anchors
        points = anchor_points * stride_tensor          # (A, 2) pixel ảnh đầu vào
        strides = stride_tensor[:, 0]
        _, _, mh, mw = proto.shape
        # Gốc phóng proto lên cỡ mặt nạ GT khi mask_ratio khác 4, nên suy
        # bước lưới từ chính proto đang có thay vì đọc dyn["fstride"].
        fstride = float(imgsz[0]) / mh
        cap = int(self.dyn["max_pos"])
        loss = proto.new_zeros(())
        used = 0
        for i in range(fg_mask.shape[0]):
            if not fg_mask[i].any():
                # như gốc: giữ đồ thị cho DDP khi ảnh không có anchor dương
                loss = loss + (proto * 0).sum() + (pred_masks * 0).sum()
                continue
            idx = fg_mask[i].nonzero(as_tuple=True)[0]
            if cap and len(idx) > cap:
                # Mỗi mặt nạ là một bản đồ cả lưới, nên đầu one-to-many (tới 10
                # anchor dương mỗi tán) phải lấy mẫu để không tràn VRAM.
                idx = idx[torch.randperm(len(idx), device=idx.device)[:cap]]
            mask_idx = target_gt_idx[i, idx]
            if self.overlap:
                gt = (masks[i] == (mask_idx + 1).view(-1, 1, 1)).float()
            else:
                gt = masks[batch_idx.view(-1) == i][mask_idx]
            logits = dynamic_logits(pred_masks[i, idx], proto[i], points[idx], strides[idx],
                                    dims=self.dyn["dims"], coords=self.dyn["coords"], fstride=fstride)
            win = window_boxes(target_bboxes[i, idx], float(self.dyn["window"])) / fstride
            region = crop_mask(torch.ones_like(gt), win)
            bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
            area = region.flatten(1).sum(1).clamp(min=1)
            per = (bce * region).flatten(1).sum(1) / area
            loss = loss + per.sum()
            used += len(idx)
            with torch.no_grad():
                pm = (logits > 0) & (region > 0)
                gm = gt > 0.5
                iou = (pm & gm).flatten(1).sum(1).float() / (pm | gm).flatten(1).sum(1).clamp(min=1).float()
                self.stats += torch.stack([
                    torch.tensor(float(len(idx)), dtype=torch.float64, device=per.device),
                    per.double().sum(), iou.double().sum(),
                ])
        return loss / max(used, 1)


def build_criterion(model):
    """Hàm loss của DynSegmentationModel.init_criterion()."""
    if getattr(model.model[-1], "one2one_cv2", None) is not None:
        return E2ELoss(model, DynSegLoss)
    return DynSegLoss(model)


class DynSegTrainer(SegmentationTrainer):
    """SegmentationTrainer dựng model A1 và chấm val bằng đầu A1.

    Dựng YOLO26-seg chuẩn từ yaml như gốc, chuyển đầu sang A1, RỒI mới nạp
    trọng số: nhờ vậy mọi lớp trùng hình dạng — cả Proto26 — nhận trọng số
    COCO, chỉ bộ sinh θ mới khởi tạo. Cấu hình A1 đi vào qua lớp con do
    make_trainer() tạo, vì Model.train() tự khởi tạo trainer, không nhận thêm
    tham số nào.
    """

    DYN: dict = {}

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, None, verbose)
        to_dynamic(model, self.DYN)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return DynSegValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                               _callbacks=self.callbacks)


def make_trainer(dyn: dict) -> type[DynSegTrainer]:
    return type("DynSegTrainer", (DynSegTrainer,), {"DYN": dict(dyn)})


def _parts(crit) -> list[tuple[str, DynSegLoss]]:
    if isinstance(crit, DynSegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), DynSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào mask_head_dyn.csv: với từng đầu, số tán đã tính
    loss, BCE trung bình trong cửa sổ, IoU cứng của mặt nạ trong cửa sổ."""

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


def install_log(yolo_model, run_dir: str | Path) -> None:
    """Gắn nhật ký epoch vào lượt train sắp chạy (criterion tạo lười ở batch đầu)."""
    from ultralytics.utils.torch_utils import unwrap_model

    log = EpochLog(Path(run_dir) / "mask_head_dyn.csv")

    def crit(trainer):
        return getattr(unwrap_model(trainer.model), "criterion", None)

    yolo_model.add_callback("on_train_epoch_start", lambda t: log.start(crit(t), t.epoch))
    yolo_model.add_callback("on_train_epoch_end", lambda t: log.end(crit(t), t.epoch))
