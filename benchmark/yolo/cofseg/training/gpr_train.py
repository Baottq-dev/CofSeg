"""Phần train của T1 (GPR + NCB, xem gpr_head.py): loss, trainer, nhật ký.

L3 — NCB, loss biên hiệu chỉnh theo nhiễu nhãn. Gốc giám sát mặt nạ bằng nhãn
NHỊ PHÂN ở lưới 1/4 (`mask_ratio: 4`): tự đặt trần IoU 0,962 lên chính model
(docs/reports/model/phan_tich_sau_L1_L2_A1_va_de_xuat_2026-10-05.md), và bắt
model học theo một lần vẽ ngẫu nhiên của người gán nhãn. NCB:

1. nhãn raster ở đủ độ phân giải đầu vào (`mask_ratio: 1`), avg-pool về lưới
   của prototype -> TỈ LỆ PHỦ của từng ô, không làm tròn (trần 0,995 ở s2);
2. làm mềm bằng Gaussian với σ = nhiễu nhãn đã đo: cùng một tán vẽ ở hai ảnh
   chồng nhau chỉ khớp IoU 0,87-0,90, tức khoảng 6 px ảnh gốc mỗi lần vẽ, 2,4 px
   đầu vào. Với nhiễu biên Gauss, xác suất một pixel được vẽ vào tán là Φ(d/σ);
   học theo kỳ vọng đó thay vì theo một lần vẽ. Làm mềm đối xứng quanh biên nên
   không thưởng cho việc phình như Dice (L2).

Còn lại giữ đúng loại của gốc: BCE từng pixel trong box GT, trung bình theo
diện tích box rồi theo số tán, nhân seg gain. Với σ = 0 và nhãn đã ở lưới của
proto, loss mặt nạ trùng từng số với gốc — test giữ điều đó.

Phải viết lại `loss()` (theo loss.py:504-565 của 8.4.143) vì gốc phóng proto
lên cỡ nhãn khi hai cỡ khác nhau (loss.py:522-524): ở `mask_ratio: 1` đó là
32 kênh x 1024² mỗi ảnh, còn NCB cần chiều ngược lại — đưa NHÃN về lưới proto.
"""

from __future__ import annotations

import csv
import math
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask

from .gpr_head import GPRSegValidator, find_proto, to_gpr


def gaussian_kernel(sigma: float, device=None) -> torch.Tensor | None:
    """Nhân Gauss 1 chiều (tổng 1), bán kính ceil(3σ); None khi σ <= 0."""
    if sigma <= 0:
        return None
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def soft_target(gt: torch.Tensor, ratio: int, kernel: torch.Tensor | None) -> torch.Tensor:
    """Nhãn mềm (n, H/ratio, W/ratio) từ mặt nạ nhị phân (n, H, W).

    avg-pool cho tỉ lệ phủ đúng của từng ô; Gauss tách hai chiều, đệm bằng giá
    trị mép để mép ảnh không thành một biên giả (tán bị cắt ở mép vẫn kéo dài ra
    ngoài ảnh).
    """
    x = gt[:, None].float()
    if ratio > 1:
        x = F.avg_pool2d(x, ratio)
    if kernel is not None:
        k = kernel.to(x.device)
        p = k.numel() // 2
        x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, p, p), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0]


class NCBSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với loss mặt nạ NCB ở lưới của prototype."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.gpr = find_proto(model).gpr
        # n, ΣBCE, ΣIoU cứng (trong box) của các tán đã tính loss trong epoch.
        self.stats = torch.zeros(3, dtype=torch.float64, device=self.device)

    def loss(self, preds, batch):
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
        pred_semantic = None
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]
        batch_size = proto.shape[0]
        if fg_mask.sum():
            masks = batch["masks"].to(self.device)
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device,
                                 dtype=torch.float32) * self.stride[0]
            loss[1] = self.ncb_loss(fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch["batch_idx"].view(-1), proto, pred_masks, imgsz)
            if pred_semantic is not None:
                loss[4] = self.semantic_loss(pred_semantic, masks, batch, batch_size)
        else:
            # như gốc: giữ đồ thị cho DDP khi cả batch không có anchor dương
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()
        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, dict(zip(self.loss_names, loss.detach()))

    def semantic_loss(self, pred_semantic, masks, batch, batch_size):
        """Nguyên văn nhánh ngữ nghĩa của gốc (loss.py:540-556)."""
        sem_idx = batch["sem_masks"].to(self.device).long().unsqueeze(1)
        if self.overlap:
            present = masks != 0
        else:
            batch_idx = batch["batch_idx"].view(-1)
            present = torch.zeros(batch_size, *masks.shape[-2:], dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                inst = masks[batch_idx == i]
                if len(inst):
                    present[i] = inst.sum(dim=0) != 0
        sem = torch.zeros(sem_idx.shape[0], self.nc, *sem_idx.shape[2:], device=self.device)
        sem.scatter_(1, sem_idx, present.unsqueeze(1).float())
        return self.bcedice_loss(pred_semantic, sem) * self.hyp.box

    def ncb_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
        _, _, mh, mw = proto.shape
        h1, w1 = masks.shape[-2:]
        ratio = h1 // mh
        if h1 != ratio * mh or w1 != ratio * mw:
            raise RuntimeError(
                f"Nhãn mặt nạ {h1}x{w1} không phải bội nguyên của lưới proto {mh}x{mw}. "
                "Config GPR cần train.mask_ratio: 1."
            )
        # box theo pixel đầu vào -> ô lưới proto; σ theo pixel đầu vào -> ô
        scale = torch.stack([mw / imgsz[1], mh / imgsz[0]] * 2).to(target_bboxes.device)
        kernel = gaussian_kernel(float(self.gpr["sigma"]) * mh / float(imgsz[0]), proto.device)
        cap = int(self.gpr["max_pos"])
        loss = proto.new_zeros((), dtype=torch.float32)
        used = 0
        for i in range(fg_mask.shape[0]):
            if not fg_mask[i].any():
                loss = loss + (proto * 0).sum() + (pred_masks * 0).sum()
                continue
            idx = fg_mask[i].nonzero(as_tuple=True)[0]
            if cap and len(idx) > cap:
                # Mỗi nhãn mềm là một bản đồ cả lưới proto (512² ở s2, gấp 4 lần
                # gốc), nên đầu one-to-many (tới 10 anchor dương mỗi tán) lấy mẫu.
                idx = idx[torch.randperm(len(idx), device=idx.device)[:cap]]
            gidx = target_gt_idx[i, idx]
            uniq, inv = gidx.unique(return_inverse=True)
            with torch.no_grad():
                if self.overlap:
                    gt = masks[i] == (uniq + 1).view(-1, 1, 1).to(masks.dtype)
                else:
                    gt = masks[batch_idx == i][uniq] > 0.5
                tgt = soft_target(gt, ratio, kernel)[inv]
            logits = torch.einsum("nc,chw->nhw", pred_masks[i, idx], proto[i])
            box = target_bboxes[i, idx] * scale
            bce = F.binary_cross_entropy_with_logits(logits, tgt.to(logits.dtype), reduction="none")
            area = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])).clamp(min=1e-9)
            per = crop_mask(bce, box).flatten(1).sum(1) / area
            loss = loss + per.sum()
            used += len(idx)
            with torch.no_grad():
                region = crop_mask(torch.ones_like(tgt), box) > 0
                pm, gm = (logits > 0) & region, (tgt > 0.5) & region
                iou = (pm & gm).flatten(1).sum(1).float() / (pm | gm).flatten(1).sum(1).clamp(min=1).float()
                self.stats += torch.stack([
                    torch.tensor(float(len(idx)), dtype=torch.float64, device=per.device),
                    per.detach().double().sum(), iou.double().sum(),
                ])
        return loss / max(used, 1)


def build_criterion(model):
    """Hàm loss của GPRSegmentationModel.init_criterion()."""
    if getattr(model.model[-1], "one2one_cv2", None) is not None:
        return E2ELoss(model, NCBSegLoss)
    return NCBSegLoss(model)


class GPRSegTrainer(SegmentationTrainer):
    """SegmentationTrainer dựng model GPR và chấm val trên lưới của proto.

    Dựng YOLO26-seg chuẩn từ yaml như gốc, chuyển Proto sang GPR, RỒI mới nạp
    trọng số: mọi lớp cũ nhận trọng số COCO, chỉ lớp mới của GPR giữ khởi tạo
    (và ConvTranspose2d, nếu GPR-1 bật, đã bị bỏ). Cấu hình đi vào qua lớp con
    do make_trainer() tạo, vì Model.train() tự khởi tạo trainer.
    """

    GPR: dict = {}

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, None, verbose)
        to_gpr(model, self.GPR)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return GPRSegValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                               _callbacks=self.callbacks)


def make_trainer(cfg: dict) -> type[GPRSegTrainer]:
    return type("GPRSegTrainer", (GPRSegTrainer,), {"GPR": dict(cfg)})


def _parts(crit) -> list[tuple[str, NCBSegLoss]]:
    if isinstance(crit, NCBSegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), NCBSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào mask_head_gpr.csv.

    Với từng đầu: số tán đã tính loss, BCE trung bình, IoU cứng trong box ở lưới
    proto. Thêm |Δ| trung bình của GPR-1 và GPR-2 ở batch cuối epoch (ô lưới
    thô): 0 nghĩa là GPR vẫn đang là phép phóng song tuyến, chưa học gì.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        for _, part in _parts(crit):
            part.stats.zero_()

    def end(self, crit, epoch: int, proto=None) -> None:
        parts = _parts(crit)
        if not parts:
            return
        row: dict = {"epoch": epoch + 1}
        for name, part in parts:
            n, s_bce, s_iou = part.stats.tolist()
            avg = (lambda s: round(s / n, 4)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_bce": avg(s_bce), f"{name}_iou": avg(s_iou)})
        for k in ("gpr1", "gpr2"):
            mod = getattr(proto, k, None) if proto is not None else None
            row[f"{k}_offset"] = round(float(mod.last_offset), 4) if mod is not None else ""
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)


def install_log(yolo_model, run_dir: str | Path) -> None:
    """Gắn nhật ký epoch vào lượt train sắp chạy (criterion tạo lười ở batch đầu)."""
    from ultralytics.utils.torch_utils import unwrap_model

    log = EpochLog(Path(run_dir) / "mask_head_gpr.csv")

    def crit(trainer):
        return getattr(unwrap_model(trainer.model), "criterion", None)

    def proto(trainer):
        try:
            return find_proto(unwrap_model(trainer.model))
        except TypeError:
            return None

    yolo_model.add_callback("on_train_epoch_start", lambda t: log.start(crit(t), t.epoch))
    yolo_model.add_callback("on_train_epoch_end", lambda t: log.end(crit(t), t.epoch, proto(t)))
