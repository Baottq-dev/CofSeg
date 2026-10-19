"""TH2 (`--loss vfl`): Varifocal thay BCE của điểm, chỉ ở đầu one-to-many.

Đo trên R2 trước khi train: 65% số tán bị bỏ lỡ (81% nếu chỉ tính tán từ 500 px
ở 1024x576) đã có ứng viên vẽ đúng trong ~12 000 ứng viên của đầu one-to-many,
với IoU mặt nạ trung vị 0,74, nhưng điểm cao nhất của chúng chỉ trung vị 0,015,
nên bị loại ngay ở ngưỡng 0,05. Ở tán khít, mặt nạ gộp hai tán lại được điểm
cao. Lỗi nằm ở cách chấm điểm, không ở chỗ thiếu hộp đúng.

Varifocal (VFNet, Zhang et al. 2021) giữ nguyên ĐÍCH mềm của TAL và chỉ đổi
TRỌNG SỐ của từng phần tử trong BCE, đúng công thức của lớp `VarifocalLoss`
trong `ultralytics/utils/loss.py`:

    nhãn  l = 1 nếu đích t > 0, ngược lại 0
    w     = alpha · σ(logit)^gamma · (1 − l)  +  t · l
    loss  = BCE(logit, t) · w

- Dương tính nặng theo chính đích t của nó: neo có hộp tốt (t cao) đẩy điểm
  mạnh hơn neo có hộp kém, nên điểm phải tách hai loại này rõ hơn.
- Âm tính nhân alpha·p^gamma: âm tính dễ (p nhỏ) gần như không còn kéo điểm
  xuống; âm tính tự tin, gồm cả tán thật thiếu nhãn, cũng chỉ còn trọng số
  0,37–0,61 ở p 0,7–0,9.

Không dùng thẳng `VarifocalLoss.forward` vì nó tự rút gọn (`.mean(1).sum()`),
lệch với cách chuẩn hoá của YOLO. Ở đây chỉ thay `self.bce` (gốc gọi nó đúng
một lần, `loss.py:438`) bằng hàm trả loss TỪNG PHẦN TỬ như
`BCEWithLogitsLoss(reduction="none")`; phần còn lại giữ nguyên: tổng chia
`target_scores_sum`, nhân cls gain, gán nhãn, loss hộp, mặt nạ, ngữ nghĩa.

Chỉ ở đầu one-to-many: đầu one-to-one học dẹp hộp trùng nhờ chính các âm tính
sát tán, hạ trọng số âm tính ở đó có thể làm nó dẹp kém đi; còn lượt chấm
benchmark dùng đầu one-to-many + NMS. YOLOv8/11 chỉ có một đầu thì đổi ở đầu đó.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

from .mask_iou_loss import is_end2end


class VarifocalBCE(torch.nn.Module):
    """Thế chỗ `v8DetectionLoss.bce`: loss Varifocal từng phần tử.

    Trọng số không tách gradient, giống `VarifocalLoss` của thư viện và bản
    gốc của VFNet. Tính bằng float32 như BCE gốc dưới AMP.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        # n_pos, Σs, Σt, Σs·t, Σs², Σt² của dương tính; Σloss dương, Σloss âm;
        # số âm tính có p >= 0,5. Đủ cho trung bình, tương quan Pearson điểm-đích
        # và tỉ phần loss của âm tính mà không chép từng giá trị về CPU.
        self.stats = torch.zeros(9, dtype=torch.float64)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        label = (target > 0).float()
        p = pred.sigmoid()
        weight = self.alpha * p.pow(self.gamma) * (1 - label) + target * label
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * weight
        with torch.no_grad():
            pos = label > 0
            s, t = p[pos].double(), target[pos].double()
            lp = loss[pos].double().sum()
            self.stats = self.stats.to(pred.device)
            self.stats += torch.stack([
                torch.tensor(float(s.numel()), dtype=torch.float64, device=pred.device),
                s.sum(), t.sum(), (s * t).sum(), (s * s).sum(), (t * t).sum(),
                lp, loss.double().sum() - lp, (p[~pos] >= 0.5).sum().double(),
            ])
        return loss


def build_criterion(model, p: dict):
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    vfl = VarifocalBCE(alpha=p["alpha"], gamma=p["gamma"])
    if not is_end2end(model):
        crit = v8SegmentationLoss(model)
        crit.bce = vfl
        return crit
    crit = E2ELoss(model, v8SegmentationLoss)
    crit.one2many.bce = vfl
    return crit


def _vfl(crit) -> VarifocalBCE | None:
    part = getattr(crit, "one2many", crit)
    bce = getattr(part, "bce", None)
    return bce if isinstance(bce, VarifocalBCE) else None


class EpochLog:
    """Mỗi epoch một dòng vào loss_vfl.csv, cho đầu đã đổi sang Varifocal.

    - `n_pos`, `score_pos`, `target_pos`: số neo dương, điểm và đích trung bình;
    - `r_pos`: tương quan Pearson điểm-đích ở neo dương. Đây là chỉ số cơ chế
      đọc được lúc train: VFL có tác dụng thì nó phải tăng;
    - `neg_share`: phần của âm tính trong tổng loss điểm (BCE gốc gần 1);
    - `neg_over_05`: số âm tính có điểm >= 0,5 trong epoch.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit, epoch: int) -> None:
        vfl = _vfl(crit)
        if vfl is not None:
            vfl.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        vfl = _vfl(crit)
        if vfl is None:
            return
        n, ss, st, sst, ss2, st2, lpos, lneg, nhi = vfl.stats.tolist()
        if n > 1:
            cov = sst / n - (ss / n) * (st / n)
            vs, vt = ss2 / n - (ss / n) ** 2, st2 / n - (st / n) ** 2
            r = cov / (vs * vt) ** 0.5 if vs > 0 and vt > 0 else float("nan")
        else:
            r = float("nan")
        row = {"epoch": epoch + 1, "n_pos": int(n),
               "score_pos": round(ss / n, 4) if n else float("nan"),
               "target_pos": round(st / n, 4) if n else float("nan"),
               "r_pos": round(r, 4),
               "neg_share": round(lneg / (lpos + lneg), 4) if lpos + lneg > 0 else float("nan"),
               "neg_over_05": int(nhi)}
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
