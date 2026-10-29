"""Giám sát mặt nạ ngoài hộp (K-A, configs/train/yolo26s-ka.yaml).

Lỗi nhắm: mặt nạ bị hộp cắt thẳng (cạnh thẳng ở 32,8% tán khớp kém của mô hình gốc; SOLOv2 và Mask2Former
gần 0). Loss mặt nạ gốc của YOLO-seg chỉ tính trong hộp nhãn (`v8SegmentationLoss.single_mask_loss`):

    loss_i = Σ_{p ∈ hộp_i} BCE(logit_i(p), nhãn_i(p)) / (|lưới| · diện_tích_hộp_i)

nên logit ngoài hộp không bị ràng buộc gì. Lúc suy luận phép cắt theo hộp che phần đó đi; muốn nới phép cắt
thì logit ngoài hộp phải có nghĩa trước. Đo ngày 14/10: nới hộp chỉ lúc suy luận cho −1,37 AP (×1,1) và
−4,67 AP (×1,2), vì logit ngoài hộp dương và dính liền tán.

Thay đổi duy nhất: vùng tổng lấy trong hộp nhãn giãn ×train_expand quanh tâm (kẹp trong lưới), mẫu số vẫn
là diện tích hộp nhãn GỐC. Phần trong hộp giữ y như gốc; vành mới cộng thêm, với tán bên cạnh và đất trong
vành là nhãn âm (overlap_mask: nhãn tán i là `masks == i + 1`). Hộp, điểm, prototype, hệ số, bộ gán nhãn
và cả hai đầu của YOLO26 giữ nguyên. train_expand = 1 trùng từng số với loss gốc — test giữ điều đó.

Lúc chấm: bộ chấm của benchmark đã có `crop_expand` (cắt mặt nạ bằng hộp dự đoán giãn) và vốn lấy hộp từ
mặt nạ, nên phần suy luận không cần code mới; config chấm configs/eval/yolo26s-ka.yaml đặt crop_expand 1.1.

Hàm loss gắn vào model đang train ở on_train_start (như cofseg/training/losses.py); checkpoint không mang
nó, nên best.pt vẫn là SegmentationModel chuẩn. Loss của lượt val lúc train vẫn là loss gốc (model EMA tự
dựng criterion của nó); val chỉ dùng để chọn best.pt theo mask AP, nên không ảnh hưởng.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask

from cofseg.models.yolo_seg import expand_boxes
from cofseg.training.mask_iou_loss import is_end2end


class OutsideBoxSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với loss mặt nạ tính trong hộp nhãn giãn ×expand, chia diện tích hộp gốc."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *, expand: float = 1.25):
        super().__init__(model, tal_topk, tal_topk2)
        self.expand = float(expand)
        # n tán dương, Σ phần trong hộp, Σ phần vành, Σ tỉ lệ logit dương trong vành, của cả epoch.
        self.stats = torch.zeros(4, dtype=torch.float64, device=self.device)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        h, w = bce.shape[-2:]
        # expand = 1 dùng thẳng hộp gốc: tính lại c ± nửa cạnh có thể lệch một ô ở mép do làm tròn.
        region = xyxy if self.expand == 1.0 else expand_boxes(xyxy, self.expand, (h, w))
        # Nhật ký đọc bce TRƯỚC phép cắt tại chỗ bên dưới. Mặt nạ hàng/cột tách rời thay vì n·h·w ô.
        with torch.no_grad():
            ri, ci = _axes(xyxy, h, w)
            rb, cb = _axes(region, h, w)
            b = bce.detach().float()
            inside = torch.einsum("nhw,nh,nw->n", b, ri, ci) / (h * w) / area
            total = torch.einsum("nhw,nh,nw->n", b, rb, cb) / (h * w) / area
            pm = (pred_mask.detach() > 0).float()
            ring_px = rb.sum(1) * cb.sum(1) - ri.sum(1) * ci.sum(1)
            leak = (torch.einsum("nhw,nh,nw->n", pm, rb, cb) - torch.einsum("nhw,nh,nw->n", pm, ri, ci)) \
                / ring_px.clamp(min=1)
            self.stats += torch.stack([
                torch.tensor(float(len(inside)), dtype=torch.float64, device=inside.device),
                inside.double().sum(), (total - inside).double().sum(), leak.double().sum(),
            ])
        # crop_mask nhân tại chỗ lên kết quả của BCE, như gốc: lượt lùi của BCE không cần đầu ra của nó.
        return (crop_mask(bce, region).mean(dim=(1, 2)) / area).sum()


def _axes(boxes: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mặt nạ hàng (n, h) và cột (n, w) của hộp xyxy trên lưới, đúng quy ước của crop_mask: x1 <= x < x2."""
    x1, y1, x2, y2 = boxes.float().unbind(1)
    c = torch.arange(w, device=boxes.device, dtype=torch.float32)[None]
    r = torch.arange(h, device=boxes.device, dtype=torch.float32)[None]
    cols = ((c >= x1[:, None]) & (c < x2[:, None])).float()
    rows = ((r >= y1[:, None]) & (r < y2[:, None])).float()
    return rows, cols


def build_criterion(model, block: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    make = partial(OutsideBoxSegLoss, expand=block["train_expand"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def _parts(crit) -> list[tuple[str, OutsideBoxSegLoss]]:
    if isinstance(crit, OutsideBoxSegLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(crit, name, None), OutsideBoxSegLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào loss_outside_box.csv, với từng đầu:

    - `_n`: số anchor dương đã tính loss mặt nạ;
    - `_inside`: loss mặt nạ trung bình phần trong hộp nhãn (đúng loss gốc);
    - `_ring`: phần cộng thêm từ vành giữa hộp gốc và hộp giãn;
    - `_leak`: tỉ lệ ô trong vành có logit > 0, tức mặt nạ còn tràn ra vành. Giảm dần là dấu hiệu giám sát
      mới đang có tác dụng.
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
            n, s_in, s_ring, s_leak = part.stats.tolist()
            avg = (lambda s: round(s / n, 5)) if n else (lambda s: float("nan"))
            row.update({f"{name}_n": int(n), f"{name}_inside": avg(s_in), f"{name}_ring": avg(s_ring),
                        f"{name}_leak": avg(s_leak)})
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
