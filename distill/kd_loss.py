"""Loss của phần chắt lọc: mặt nạ (đích trộn với ba thầy) và đặc trưng (PKD từ YOLO26s cùng fold).

**Chắt lọc mặt nạ** (KDSegLoss). Loss mặt nạ gốc của YOLO-seg với mỗi neo dương gán cho tán j:

    loss_j = Σ_{p ∈ hộp_j} BCE(logit(p), y_j(p)) / (|lưới| · diện_tích_hộp_j)

Ở đây chỉ đổi đích: y_j → (1 − α)·y_j + α·t_j, với t_j là đích của thầy cho tán j (kd_data.py) nếu tán đó có
thầy ghép được, còn không thì giữ y_j. BCE tuyến tính theo đích, nên đây đúng bằng (1 − α)·loss_nhãn +
α·loss_thầy: tổng trọng số của loss mặt nạ không đổi. α = 0 trùng từng số với loss gốc (test giữ điều đó).
Vùng tính mặc định là hộp nhãn như gốc; `mask_expand` > 1 nới vùng như outside_box.py. Hộp, điểm, bộ gán
nhãn, prototype và cả hai đầu của YOLO26 giữ nguyên.

**Chắt lọc đặc trưng** (KDCriterion). Thầy là YOLO26s seed 0 cùng fold, đóng băng, chạy trên đúng batch đã
biến đổi. Ba cặp đặc trưng: đầu vào P4, P5 của đầu (neck) và đầu ra feat_fuse của Proto (đặc trưng trộn P3–P5
ở stride 8). Đặc trưng của trò qua một lớp 1×1 (kd_adapter, train cùng trò, gỡ khỏi best.pt sau khi train)
để khớp số kênh của thầy. Loss theo PKD (Cao và cộng sự, NeurIPS 2022): chuẩn hoá từng kênh về trung bình 0,
độ lệch chuẩn 1 trên (batch, h, w) rồi lấy MSE / 2, tức 1 − tương quan Pearson, nên không lệ thuộc độ lớn
đặc trưng của hai model. Trọng số theo vùng hộp nhãn, không theo điểm của thầy (điểm của thầy cao cả ở tán
thiếu nhãn): MSE lấy trung bình riêng trong hộp nhãn và ngoài hộp, ngoài hộp nhân `feat_bg` (FGD dùng tỉ lệ
nền : vật 1 : 2). Loss cộng vào tổng với hệ số `feat_weight` × cỡ batch, như các loss của ultralytics, và
ghi thành khoá `kd_loss` trong nhật ký train.

Hàm loss gắn vào model đang train ở on_train_start; checkpoint không mang nó. Model EMA (dùng chấm val lúc
train) giữ loss gốc.
"""

from __future__ import annotations

import copy
import csv
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask, xyxy2xywh

from cofseg.models.yolo_seg import expand_boxes
from cofseg.training.mask_iou_loss import is_end2end


# ---------------------------------------------------------------- mặt nạ
class KDSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss với đích mặt nạ trộn nhãn và thầy."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *, alpha: float = 0.5,
                 expand: float = 1.0):
        super().__init__(model, tal_topk, tal_topk2)
        self.alpha, self.expand = float(alpha), float(expand)
        self._batch = None
        # neo dương, neo dương có đích thầy, của cả epoch
        self.stats = torch.zeros(2, dtype=torch.float64, device=self.device)

    def loss(self, preds, batch):
        self._batch = batch
        try:
            return super().loss(preds, batch)
        finally:
            self._batch = None

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        if self.expand == 1.0:
            return v8SegmentationLoss.single_mask_loss(gt_mask, pred, proto, xyxy, area)
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        region = expand_boxes(xyxy, self.expand, bce.shape[-2:])
        return (crop_mask(bce, region).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                    pred_masks, imgsz):
        b = self._batch
        kd = self.alpha > 0 and b is not None and "kd_map" in b
        # Chép v8SegmentationLoss.calculate_segmentation_loss (ultralytics 8.4.143); chỉ thêm dòng trộn đích.
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        if kd:
            kd_map = b["kd_map"].to(self.device)
            kd_ids = b["kd_ids"].to(self.device).long()
            kd_has = b["kd_has"].to(self.device)
            if tuple(kd_map.shape[-2:]) != tuple(masks.shape[-2:]):
                raise RuntimeError(f"kd_map {tuple(kd_map.shape)} lệch lưới mặt nạ {tuple(masks.shape)}")
        bi = batch_idx.view(-1)
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[bi == i][mask_idx]
                if kd:
                    sel = bi == i
                    j = kd_ids[sel][mask_idx].view(-1, 1, 1)
                    has = kd_has[sel][mask_idx]
                    m = kd_map[i]
                    soft = (m[1].float() / 255) * (m[0].long() == j) + (m[3].float() / 255) * (m[2].long() == j)
                    gt_mask = gt_mask + (self.alpha * has.float()).view(-1, 1, 1) * (soft - gt_mask)
                    with torch.no_grad():
                        self.stats += torch.stack([has.new_tensor(len(has), dtype=torch.float64),
                                                   has.double().sum()])
                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()


def build_seg_criterion(model, alpha: float, expand: float):
    make = partial(KDSegLoss, alpha=alpha, expand=expand)
    return E2ELoss(model, make) if is_end2end(model) else make(model)


def seg_parts(crit) -> list[tuple[str, KDSegLoss]]:
    base = getattr(crit, "base", crit)
    if isinstance(base, KDSegLoss):
        return [("one", base)]
    return [(name, getattr(base, name)) for name in ("one2one", "one2many")
            if isinstance(getattr(base, name, None), KDSegLoss)]


def carry_schedule(old, new) -> None:
    """Giữ lịch giảm trọng số one2many của E2ELoss khi thay hàm loss (resume đặt `updates` trên hàm cũ)."""
    if old is None or not hasattr(old, "updates") or not hasattr(new, "update"):
        return
    if int(old.updates) > 0:
        new.updates = int(old.updates) - 1
        new.update()


# ---------------------------------------------------------------- đặc trưng
class Grab:
    """Hook giữ đầu ra (hoặc đầu vào) của một lớp. Bản deepcopy là hook rỗng: ultralytics dựng lại EMA bằng
    deepcopy model (sau NaN), mà tensor giữa đồ thị thì không deepcopy được."""

    def __init__(self, inputs: bool = False):
        self.inputs, self.out = inputs, None

    def __call__(self, module, args, out=None):
        self.out = list(args[0]) if self.inputs else out

    def __deepcopy__(self, memo):
        return Grab(self.inputs)

    def take(self):
        out, self.out = self.out, None
        return out


def tap(model) -> tuple[Grab, Grab, list]:
    """Hook ở đầu vào của đầu (P3–P5) và ở đầu ra feat_fuse của Proto."""
    head = model.model[-1]
    neck, fuse = Grab(inputs=True), Grab()
    handles = [head.register_forward_pre_hook(neck), head.proto.feat_fuse.register_forward_hook(fuse)]
    return neck, fuse, handles


def feature_channels(model, imgsz: int = 64) -> list[int]:
    """Số kênh của P4, P5 (đầu vào của đầu) và feat_fuse, đo bằng một lượt chạy thử ở chế độ eval."""
    neck, fuse, handles = tap(model)
    was = model.training
    try:
        model.eval()
        p = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
        x, f = neck.take(), fuse.take()
        return [int(x[-2].shape[1]), int(x[-1].shape[1]), int(f.shape[1])]
    finally:
        model.train(was)
        for h in handles:
            h.remove()


class KDAdapter(nn.Module):
    """Lớp 1×1 đưa P4, P5, feat_fuse của trò về số kênh của thầy. Chỉ dùng lúc train."""

    def __init__(self, student_ch: list[int], teacher_ch: list[int]):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(s, t, 1) for s, t in zip(student_ch, teacher_ch))

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [c(f) for c, f in zip(self.convs, feats)]


def pkd_norm(f: torch.Tensor) -> torch.Tensor:
    """Chuẩn hoá từng kênh về trung bình 0, độ lệch chuẩn 1 trên (batch, h, w), như PKD."""
    f = f.float()
    mu = f.mean(dim=(0, 2, 3), keepdim=True)
    sd = f.std(dim=(0, 2, 3), keepdim=True)
    return (f - mu) / (sd + 1e-6)


def box_mask(bboxes: torch.Tensor, batch_idx: torch.Tensor, bs: int, h: int, w: int) -> torch.Tensor:
    """(bs, h, w) bool: ô nằm trong (hoặc chạm) một hộp nhãn của ảnh. bboxes xywh chuẩn hoá như batch."""
    out = torch.zeros(bs, h, w, dtype=torch.bool, device=bboxes.device)
    if not len(bboxes):
        return out
    cx, cy, bw, bh = bboxes.float().unbind(1)
    x1, x2 = torch.floor((cx - bw / 2) * w), torch.ceil((cx + bw / 2) * w)
    y1, y2 = torch.floor((cy - bh / 2) * h), torch.ceil((cy + bh / 2) * h)
    c = torch.arange(w, device=bboxes.device, dtype=torch.float32)[None]
    r = torch.arange(h, device=bboxes.device, dtype=torch.float32)[None]
    cols = (c >= x1[:, None]) & (c < x2[:, None])
    rows = (r >= y1[:, None]) & (r < y2[:, None])
    inside = rows[:, :, None] & cols[:, None, :]
    bi = batch_idx.view(-1).long()
    for i in range(bs):
        sel = bi == i
        if sel.any():
            out[i] = inside[sel].any(0)
    return out


def pkd_loss(fs: torch.Tensor, ft: torch.Tensor, fg: torch.Tensor, bg_weight: float) -> torch.Tensor:
    """½·[TB trong hộp + bg_weight·TB ngoài hộp] của Σ_c (ŝ − t̂)² / C."""
    d = (pkd_norm(fs) - pkd_norm(ft)).pow(2).mean(1)
    fgf = fg.float()
    inside = (d * fgf).sum() / fgf.sum().clamp(min=1)
    outside = (d * (1 - fgf)).sum() / (1 - fgf).sum().clamp(min=1)
    return 0.5 * (inside + bg_weight * outside)


class KDCriterion:
    """Hàm loss của model đang train: loss mặt nạ (đã trộn đích) + chắt lọc đặc trưng."""

    LEVELS = ("p4", "p5", "fuse")

    def __init__(self, base, model=None, teacher=None, *, feat_weight: float = 0.0, feat_bg: float = 0.5):
        self.base = base
        self.feat_weight, self.feat_bg = float(feat_weight), float(feat_bg)
        self.teacher, self.adapter = teacher, getattr(model, "kd_adapter", None)
        self.handles = []
        self.feat_stats = torch.zeros(1 + len(self.LEVELS), dtype=torch.float64)
        if self.feat_weight > 0:
            if teacher is None or self.adapter is None:
                raise RuntimeError("Chắt lọc đặc trưng cần thầy và model.kd_adapter")
            self.s_neck, self.s_fuse, h1 = tap(model)
            self.t_neck, self.t_fuse, h2 = tap(teacher)
            self.handles = h1 + h2

    def __call__(self, preds, batch):
        loss, items = self.base(preds, batch)
        if self.feat_weight <= 0:
            return loss, items
        s_neck, s_fuse = self.s_neck.take(), self.s_fuse.take()
        with torch.no_grad():
            self.teacher(batch["img"])
        t_neck, t_fuse = self.t_neck.take(), self.t_fuse.take()
        fs = self.adapter([s_neck[-2], s_neck[-1], s_fuse])
        ft = [t_neck[-2], t_neck[-1], t_fuse]
        bs = batch["img"].shape[0]
        parts = []
        for a, t in zip(fs, ft):
            fg = box_mask(batch["bboxes"], batch["batch_idx"], bs, *a.shape[-2:])
            parts.append(pkd_loss(a, t.detach(), fg, self.feat_bg))
        kd = torch.stack(parts).mean() * self.feat_weight
        with torch.no_grad():
            self.feat_stats += torch.tensor([1.0, *[float(p) for p in parts]], dtype=torch.float64)
        items = dict(items)
        items["kd_loss"] = kd.detach()
        return torch.cat([loss.view(-1), (kd * bs).view(1).to(loss.dtype)]), items

    def update(self):
        if hasattr(self.base, "update"):
            self.base.update()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __getattr__(self, name):
        # updates, o2m, one2one... của E2ELoss bên trong; tránh đệ quy khi đối tượng chưa dựng xong.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __deepcopy__(self, memo):
        # EMA dựng lại bằng deepcopy model (sau NaN) không mang theo thầy và hook: chỉ chép loss mặt nạ.
        return copy.deepcopy(self.base, memo)


def build_criterion(model, block: dict, teacher=None) -> KDCriterion:
    """Hàm loss thay cho `model.init_criterion()` của ultralytics."""
    alpha = block["mask_alpha"] if block.get("targets") else 0.0
    base = build_seg_criterion(model, alpha, block["mask_expand"])
    fw = block["feat_weight"] if block.get("teacher_runs") else 0.0
    return KDCriterion(base, model, teacher, feat_weight=fw, feat_bg=block["feat_bg"])


def load_teacher(path: str | Path):
    """YOLO26s thầy từ best.pt: float32, eval, không gradient."""
    from ultralytics.nn.tasks import load_checkpoint

    model, _ = load_checkpoint(str(path))
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class EpochLog:
    """Mỗi epoch một dòng vào loss_kd.csv:

    - `<đầu>_pos`, `<đầu>_pos_kd`: số neo dương của loss mặt nạ và số neo trong đó có đích thầy;
    - `feat_p4`, `feat_p5`, `feat_fuse`: loss PKD trung bình từng cặp đặc trưng (chưa nhân feat_weight).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, crit) -> None:
        for _, part in seg_parts(crit):
            part.stats.zero_()
        if isinstance(crit, KDCriterion):
            crit.feat_stats.zero_()

    def end(self, crit, epoch: int) -> None:
        if not isinstance(crit, KDCriterion):
            return
        row: dict = {"epoch": epoch + 1}
        for name, part in seg_parts(crit):
            n, nk = part.stats.tolist()
            row.update({f"{name}_pos": int(n), f"{name}_pos_kd": int(nk)})
        n, *lv = crit.feat_stats.tolist()
        for k, v in zip(KDCriterion.LEVELS, lv):
            row[f"feat_{k}"] = round(v / n, 5) if n else float("nan")
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
