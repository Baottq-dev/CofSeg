"""TH3 (`--loss boundary`): BCE mặt nạ + λ·Boundary loss, khoảng cách cắt ở ±clip ô.

76% điểm ảnh sai của R2 nằm ở biên tán. Đo trước khi train, mặt nạ R2 hụt
0,059 IoU so với mức prototype vẽ được ngay trên ảnh train, và hụt nhiều nhất
ở tán chạm nhau (0,073 trên ảnh train, 0,092 trên ảnh thử): chỗ ranh giới
giữa hai tán vẽ kém ngay cả trên dữ liệu đã học.

Boundary loss (Kervadec et al. 2019) nhân xác suất dự đoán của mỗi ô với
khoảng cách có dấu từ ô đó tới biên nhãn: ô sai càng xa biên càng bị phạt
nặng, ô sát biên gần như không. Ở đây khoảng cách tính trên lưới mặt nạ và
CẮT ở ±clip ô (mặc định 5 ô, khoảng 50 px ảnh gốc) rồi chia clip:

    φ = −(m + 0,5) trong tán, m = số lần co mà ô còn sống
    φ = +(k − 0,5) ngoài tán, k = số lần giãn để chạm tới ô
    φ̃ = clamp(φ, −clip, clip) / clip ∈ [−1, 1]
    BL = trung bình trong box GT của φ̃ · (σ(logit) − g)

Lệch nửa ô làm hai phía biên đối xứng (±0,5). Trừ g làm BL >= 0 và bằng 0 khi
mặt nạ đúng nhãn, mà gradient không đổi (g không phụ thuộc tham số). Co/giãn
bằng max-pool 3x3 trên GPU, tức khoảng cách Chebyshev; đệm của max-pool là
−inf nên tán bị khung ảnh cắt không bị co từ mép ảnh, như cách NCB đối xử mép.

Vì sao cắt: chênh lề nhãn giữa các ruộng (1,6–8 px ảnh gốc) nằm gọn trong ô
sát biên nên chỉ nhận |φ̃| <= 0,1; còn mảnh vụn, thuỳ sai và phần tràn sâu sang
tán bên cạnh (>= 3 ô) nhận 0,5–1.

    mask_loss = BCE_gốc + λ · BL,   λ = weight · min(1, max(0, (epoch − start_epoch) / ramp_epochs))

λ = 0 trong start_epoch epoch đầu: Boundary loss bật quá sớm hoặc dùng một
mình thì sụp. Chỉ cài đè `single_mask_loss` như L2 (gốc gọi qua `self.`), nên
không chép đoạn gán nhãn nào; λ = 0 cho loss trùng từng số với gốc.
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


@torch.no_grad()
def signed_distance(gt: torch.Tensor, clip: int) -> torch.Tensor:
    """φ̃ (n, h, w) trong [−1, 1] từ mặt nạ nhị phân (n, h, w); âm trong tán."""
    g = (gt > 0.5).float()[:, None]
    grow, keep = g, g
    out_steps = torch.zeros_like(g)   # số lần giãn mà ô (ngoài tán) chưa được chạm
    in_steps = torch.zeros_like(g)    # số lần co mà ô (trong tán) còn sống
    for _ in range(clip):
        grow = F.max_pool2d(grow, 3, 1, 1)
        keep = -F.max_pool2d(-keep, 3, 1, 1)
        out_steps += 1 - grow
        in_steps += keep
    phi = torch.where(g > 0, -(in_steps + 0.5).clamp(max=clip), (out_steps + 0.5).clamp(max=clip))
    return phi[:, 0] / clip


_PROBE: dict = {}


@torch.no_grad()
def signed_distance_dedup(gt: torch.Tensor, clip: int) -> torch.Tensor:
    """Như signed_distance, nhưng chỉ tính một lần cho mỗi mặt nạ khác nhau.

    Ở đầu one-to-many mỗi tán có tới 10 anchor dương mang cùng một mặt nạ GT;
    tính lặp lại tốn 74 ms mỗi ảnh ở 480 anchor x 256² (RTX 4060). Mặt nạ được
    nhận diện qua tích với một vector ngẫu nhiên cố định; hai mặt nạ khác nhau
    trùng khoá là không thực tế, còn hai mặt nạ giống nhau mà lệch khoá chỉ làm
    tính thừa một lần, không làm sai.
    """
    flat = (gt > 0.5).flatten(1).float()
    key = (flat.shape[1], flat.device)
    if key not in _PROBE:
        gen = torch.Generator().manual_seed(0)
        _PROBE[key] = torch.rand(flat.shape[1], generator=gen).to(flat.device)
    uniq, inv = torch.unique(flat @ _PROBE[key], return_inverse=True)
    first = torch.empty(len(uniq), dtype=torch.long, device=gt.device).scatter_(
        0, inv, torch.arange(len(inv), device=gt.device))
    return signed_distance(gt[first], clip)[inv]


def boundary_term(pred_mask: torch.Tensor, gt_mask: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor,
                  clip: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BL của từng tán, chuẩn hoá như BCE gốc (trung bình trong box GT), kèm φ̃ và vùng box.

    pred_mask (n, h, w) logit; gt_mask (n, h, w) nhị phân; xyxy (n, 4) box GT
    trên lưới mặt nạ; area (n,) diện tích box chuẩn hoá về [0, 1].
    """
    phi = signed_distance_dedup(gt_mask, clip)
    # crop_mask nhân tại chỗ ở một nhánh của nó: cắt một tấm toàn 1 riêng rồi
    # nhân ra ngoài, như dice_loss.py.
    region = crop_mask(torch.ones_like(phi), xyxy)
    bl = (phi * (pred_mask.float().sigmoid() - gt_mask.float()) * region).mean(dim=(1, 2)) / area
    return bl, phi, region


class BoundarySegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với Boundary loss cộng vào loss mặt nạ."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 weight: float = 0.1, clip: int = 5, start_epoch: int = 10, ramp_epochs: int = 10):
        super().__init__(model, tal_topk, tal_topk2)
        self.weight = float(weight)
        self.clip = int(clip)
        self.start_epoch = int(start_epoch)
        self.ramp_epochs = int(ramp_epochs)
        self.epoch = 0
        # n, ΣBCE, ΣBL, số ô dự đoán là tán trong box, số ô đó nằm xa >= 2,5 ô ngoài tán
        self.stats = torch.zeros(5, dtype=torch.float64, device=self.device)

    @property
    def lam(self) -> float:
        if self.epoch < self.start_epoch:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.weight
        return self.weight * min(1.0, (self.epoch - self.start_epoch) / self.ramp_epochs)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Ba dòng đầu là nguyên văn gốc (8.4.143), chỉ tách tổng ra để cộng BL.
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        bce = crop_mask(bce, xyxy).mean(dim=(1, 2)) / area
        loss = bce.sum()
        lam = self.lam
        if lam > 0:
            bl, phi, region = boundary_term(pred_mask, gt_mask, xyxy, area, self.clip)
            loss = loss + lam * bl.sum()
        else:
            # vẫn tính để nhật ký có mốc trước khi λ bật, nhưng ngoài đồ thị
            with torch.no_grad():
                bl, phi, region = boundary_term(pred_mask, gt_mask, xyxy, area, self.clip)
        with torch.no_grad():
            pm = (pred_mask > 0) & (region > 0)
            self.stats += torch.stack([
                torch.tensor(float(len(bce)), dtype=torch.float64, device=bce.device),
                bce.double().sum(), bl.double().sum(),
                pm.sum().double(), (pm & (phi >= 0.5)).sum().double(),
            ])
        return loss


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(BoundarySegLoss, weight=p["weight"], clip=p["clip"],
                   start_epoch=p["start_epoch"], ramp_epochs=p["ramp_epochs"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def _parts(crit) -> list[tuple[str, BoundarySegLoss]]:
    if isinstance(crit, BoundarySegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), BoundarySegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào loss_boundary.csv: λ, và với từng đầu số anchor
    dương, BCE và BL trung bình, cùng `far`: phần ô dự đoán là tán trong box mà
    nằm cách tán >= 2,5 ô (mảnh, thuỳ sai, tràn sang tán bên cạnh). `far` phải
    giảm sau khi λ bật, đó là chỉ số cơ chế đọc được lúc train. BL được ghi cả
    khi λ = 0 để có mốc so sánh."""

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
        row: dict = {"epoch": epoch + 1, "lambda": round(parts[0][1].lam, 4)}
        for name, part in parts:
            n, s_bce, s_bl, n_pm, n_far = part.stats.tolist()
            avg = (lambda s: round(s / n, 5)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_bce": avg(s_bce), f"{name}_bl": avg(s_bl),
                        f"{name}_far": round(n_far / n_pm, 5) if n_pm else float("nan")})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
