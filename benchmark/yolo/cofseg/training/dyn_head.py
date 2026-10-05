"""A1: đầu mặt nạ động, không cắt sát bằng box (configs/train/yolo26s-dyn.yaml).

YOLO-seg dựng mặt nạ mỗi tán bằng tổ hợp TUYẾN TÍNH 32 prototype dùng chung cả
ảnh, rồi CẮT bằng box dự đoán (`ops.process_mask_native` -> `crop_mask`). Đo
trên R2 (docs/reports/model/phan_tich_chuyen_sau_va_de_xuat_kien_truc_
2026-10-04.md, mục 3.3-3.4): 21% mặt nạ có cạnh bị box cắt thẳng, 31% pixel
thừa lấn sang tán bên cạnh, 49% pixel sai nằm xa biên hơn 12 px. Chính YOLACT,
gốc của cách dựng này, nhận lỗi "leakage": mặt nạ được cắt SAU khi dựng nên
mạng không học dập nhiễu ngoài box.

A1 giữ nguyên 32 prototype (cả trọng số COCO) làm đặc trưng F. Mỗi anchor
sinh tham số θ của một mạng nhỏ 1x1 kiểu CondInst, chạy trên [F, toạ độ tương
đối so với chính anchor đó]:

    (32 + 2) -> 8 -ReLU-> 8 -ReLU-> 1          θ: 361 số mỗi anchor

và mặt nạ được giữ trong CỬA SỔ = box x 1.5 thay vì cắt sát box. Đầu gốc là
một trường hợp riêng: không lớp ẩn, không toạ độ, cửa sổ x1 — nên thang
ablation A1a/A1b/A1 chỉ là đổi `dims`, `coords`.

File này là phần model và suy luận: phép dựng mặt nạ, lớp đầu, phép chuyển
một YOLO26-seg chuẩn sang A1, và validator/predictor dùng nó. Phần train
(loss, trainer) ở dyn_train.py. Checkpoint A1 chứa các lớp ở đây, nên muốn
nạp nó phải import được `cofseg` — khác L1/L2, nó không còn là YOLO-seg chuẩn.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationValidator
from ultralytics.nn.modules.head import Detect, Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import ops

#: Toạ độ tương đối chia cho stride x 8 của tầng sinh ra anchor — đúng phép
#: chuẩn hoá của RTMDet-Ins (mmdet rtmdet_ins_head: `/ (strides * 8)`).
COORD_SCALE = 8.0

#: Số tán dựng mặt nạ mỗi lượt lúc suy luận: mỗi tán giữ 8 kênh ẩn trên cả
#: lưới 256x256, nên 300 tán một lượt là ~630 MB chỉ cho một lớp.
CHUNK = 64


# ---------------------------------------------------------------- phép dựng
def n_params(c: int, dims, coords: bool) -> int:
    """Số tham số động mỗi tán: trọng số + bias của từng lớp 1x1."""
    chans = [c + (2 if coords else 0), *dims, 1]
    return sum(a * b + b for a, b in zip(chans[:-1], chans[1:]))


def dynamic_logits(theta: torch.Tensor, feats: torch.Tensor, points: torch.Tensor,
                   strides: torch.Tensor, *, dims, coords: bool, fstride: float) -> torch.Tensor:
    """Logit mặt nạ (n, H, W) của n tán trên CẢ lưới F.

    theta (n, P) tham số động, bố cục như CondInst: mọi trọng số trước, rồi
    mọi bias. feats (C, H, W) đặc trưng mặt nạ. points (n, 2) là (x, y) theo
    pixel ảnh đầu vào của anchor sinh ra θ; strides (n,) stride tầng của nó;
    fstride là số pixel ảnh mỗi ô của F (4 với Proto26).

    Lớp đầu tách ra `einsum(W, F)` dùng chung F cho mọi tán cộng phần toạ độ
    riêng từng tán, nên không phải nhân bản F n lần.
    """
    n = theta.shape[0]
    c, h, w = feats.shape
    chans = [c + (2 if coords else 0), *dims, 1]
    n_w = [a * b for a, b in zip(chans[:-1], chans[1:])]
    parts = torch.split(theta, n_w + chans[1:], dim=1)
    ws = [p.reshape(n, o, i) for p, i, o in zip(parts[:len(n_w)], chans[:-1], chans[1:])]
    bs = parts[len(n_w):]

    dt = theta.dtype
    out = torch.einsum("noc,chw->nohw", ws[0][:, :, :c], feats.to(dt))
    if coords:
        scale = (strides.to(dt) * COORD_SCALE)[:, None]
        xs = (torch.arange(w, device=theta.device, dtype=dt) + 0.5) * fstride
        ys = (torch.arange(h, device=theta.device, dtype=dt) + 0.5) * fstride
        rx = (xs[None] - points[:, :1].to(dt)) / scale            # (n, W)
        ry = (ys[None] - points[:, 1:2].to(dt)) / scale           # (n, H)
        out = (out + ws[0][:, :, c, None, None] * rx[:, None, None, :]
               + ws[0][:, :, c + 1, None, None] * ry[:, None, :, None])
    out = out + bs[0][:, :, None, None]
    for wk, bk in zip(ws[1:], bs[1:]):
        out = torch.einsum("noi,nihw->nohw", wk, F.relu(out)) + bk[:, :, None, None]
    return out[:, 0]


def window_boxes(boxes: torch.Tensor, factor: float) -> torch.Tensor:
    """Box xyxy nới quanh tâm, mỗi chiều x factor."""
    if factor == 1.0:
        return boxes
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) * (factor / 2)
    return torch.cat([ctr - half, ctr + half], 1)


# ---------------------------------------------------------------- đầu và model
class Segment26Dyn(Segment26):
    """Segment26 có bộ sinh tham số động thay cho 32 hệ số.

    Chỉ tạo bằng to_dynamic() từ một Segment26 đã dựng, để giữ nguyên trọng
    số và mọi thuộc tính ultralytics gắn vào đầu (stride, i, f, ...).
    `dyn` mang cấu hình: dims, coords, window, max_pos, n_params, fstride.
    """

    dyn: dict

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if mask_head is not None:
            bs, p = x[0].shape[0], self.dyn["n_params"]
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](x[i]).view(bs, p, -1) for i in range(self.nl)], 2)
        return preds

    def _inference(self, x):
        """Như gốc, kèm toạ độ (pixel) và stride của anchor sau θ.

        Bước chọn top-k của đầu-cuối (Detect.postprocess) gom mọi cột sau
        điểm phân loại theo chỉ số đã chọn, nên mỗi tán giữ lại được đúng vị
        trí đã sinh ra θ của nó — thứ phép dựng cần cho toạ độ tương đối.
        """
        y = Detect._inference(self, x)  # đặt self.anchors (2, A), self.strides (1, A)
        bs = y.shape[0]
        pts = (self.anchors * self.strides)[None].expand(bs, -1, -1)
        st = self.strides[None].expand(bs, -1, -1)
        return torch.cat([y, x["mask_coefficient"].to(y.dtype), pts.to(y.dtype), st.to(y.dtype)], 1)


class DynSegmentationModel(SegmentationModel):
    """SegmentationModel có đầu A1. Khác gốc đúng một chỗ: hàm loss.

    Phải là lớp riêng chứ không gắn criterion bằng callback như L1/L2: bản
    EMA mà validator chấm trong lúc train cũng tự tạo hàm loss qua
    init_criterion(), và loss gốc không hiểu θ 361 số.
    """

    def init_criterion(self):
        from .dyn_train import build_criterion

        return build_criterion(self)


def init_controller(conv: nn.Conv2d, c: int, dims, coords: bool) -> None:
    """θ = W·x + b, với W nhỏ (std 0.01 như CondInst) và b là một mạng TĨNH
    khởi tạo như nn.Linear. Lúc đầu mọi tán dùng chung một mạng hợp lệ — khác
    nhau nhờ toạ độ — rồi phần riêng từng tán học dần qua W."""
    nn.init.normal_(conv.weight, std=0.01)
    chans = [c + (2 if coords else 0), *dims, 1]
    ws, bs = [], []
    for i, o in zip(chans[:-1], chans[1:]):
        lin = nn.Linear(i, o)
        ws.append(lin.weight.detach().flatten())
        bs.append(lin.bias.detach())
    with torch.no_grad():
        conv.bias.copy_(torch.cat(ws + bs).to(conv.bias))


def to_dynamic(model: SegmentationModel, cfg: dict) -> SegmentationModel:
    """Chuyển TẠI CHỖ một YOLO26-seg chuẩn sang A1.

    Chỉ lớp 1x1 cuối của cv4 (và one2one_cv4) đổi: 32 -> n_params đầu ra.
    Mọi thứ khác — backbone, neck, đầu box/cls, cả Proto26 — giữ nguyên, nên
    nạp trọng số COCO sau bước này thì chỉ đúng các lớp đó không khớp.
    """
    head = model.model[-1]
    if isinstance(head, Segment26Dyn):
        return model
    if type(head) is not Segment26:
        raise SystemExit(
            f"A1 viết cho đầu Segment26 của YOLO26-seg; model này có {type(head).__name__}. "
            "Dùng một model yolo26*-seg."
        )
    c, dims, coords = head.nm, [int(d) for d in cfg["dims"]], bool(cfg["coords"])
    p = n_params(c, dims, coords)
    for seq in (head.cv4, getattr(head, "one2one_cv4", None)):
        if seq is None:
            continue
        for branch in seq:
            old = branch[-1]
            new = nn.Conv2d(old.in_channels, p, 1).to(device=old.weight.device, dtype=old.weight.dtype)
            init_controller(new, c, dims, coords)
            branch[-1] = new
    head.__class__ = Segment26Dyn
    # Proto26 gộp P3-P5 ở stride của P3 rồi phóng x2: F ở stride[0] / 2 = 4.
    head.dyn = {**cfg, "dims": dims, "coords": coords, "n_params": p,
                "fstride": float(head.stride[0]) / 2}
    model.__class__ = DynSegmentationModel
    return model


def find_head(model) -> Segment26Dyn:
    """Đầu A1 trong một model: nn.Module trần, bản EMA, hay AutoBackend.

    AutoBackend (8.4.143) không đăng ký model làm submodule mà cất trong
    `backend.model` (một BaseBackend, không phải nn.Module), nên phải đi qua
    đó; `.modules()` của AutoBackend không thấy nó.
    """
    backend = getattr(model, "backend", None)
    for cand in (model, getattr(backend, "model", None)):
        if isinstance(cand, nn.Module):
            for m in cand.modules():
                if isinstance(m, Segment26Dyn):
                    return m
    raise TypeError("model không có đầu A1 (Segment26Dyn)")


def is_dynamic(model) -> bool:
    try:
        find_head(model)
        return True
    except TypeError:
        return False


# ---------------------------------------------------------------- suy luận
def _split(extra: torch.Tensor, dyn: dict):
    p = dyn["n_params"]
    return extra[:, :p], extra[:, p:p + 2], extra[:, p + 2]


def _logits(protos, extra, dyn):
    theta, pts, st = _split(extra.float(), dyn)
    feats = protos.float()
    return torch.cat([
        dynamic_logits(theta[i:i + CHUNK], feats, pts[i:i + CHUNK], st[i:i + CHUNK],
                       dims=dyn["dims"], coords=dyn["coords"], fstride=dyn["fstride"])
        for i in range(0, len(theta), CHUNK)
    ])


def process_mask_dyn(protos, extra, bboxes, shape, upsample: bool = False, *, dyn: dict):
    """Bản A1 của `ops.process_mask`: cùng chữ ký, cùng đầu ra; `extra` là
    [θ, x, y, stride] thay cho hệ số, và cắt bằng cửa sổ thay cho box."""
    c, mh, mw = protos.shape
    if extra.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.uint8, device=extra.device)
    masks = _logits(protos, extra, dyn)
    boxes = window_boxes(bboxes.float(), dyn["window"])
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    else:
        ratios = torch.tensor([[mw / shape[1], mh / shape[0]] * 2], device=boxes.device)
        boxes = boxes * ratios
    return ops.crop_mask(masks.gt_(0.0).byte(), boxes)


def process_mask_native_dyn(protos, extra, bboxes, shape, *, dyn: dict):
    """Bản A1 của `ops.process_mask_native` (box theo ảnh gốc, mặt nạ ở cỡ ảnh gốc)."""
    c, mh, mw = protos.shape
    h, w = shape
    if extra.shape[0] == 0:
        return torch.zeros((0, h, w), dtype=torch.uint8, device=extra.device)
    logits = _logits(protos, extra, dyn)
    step = max(1, 32_000_000 // (h * w))  # cùng ngân sách pixel như gốc
    masks = [ops.scale_masks(logits[i:i + step][None], shape)[0].gt_(0.0).byte()
             for i in range(0, logits.shape[0], step)]
    return ops.crop_mask(torch.cat(masks), window_boxes(bboxes.float(), dyn["window"]))


class DynSegValidator(SegmentationValidator):
    """SegmentationValidator dựng mặt nạ bằng đầu A1."""

    def init_metrics(self, model):
        super().init_metrics(model)
        dyn = find_head(model).dyn
        self._stock_process = self.process
        native = self.process is ops.process_mask_native
        self.process = partial(process_mask_native_dyn if native else process_mask_dyn, dyn=dyn)

    def _prepare_batch(self, si, batch):
        # Gốc chọn cỡ mặt nạ GT bằng `self.process is ops.process_mask_native`
        # (val.py:128); trả hàm gốc về trong lúc đó để phép so còn đúng.
        dyn_process, self.process = self.process, self._stock_process
        try:
            return super()._prepare_batch(si, batch)
        finally:
            self.process = dyn_process


class DynSegPredictor(SegmentationPredictor):
    """SegmentationPredictor dựng mặt nạ bằng đầu A1."""

    def construct_result(self, pred, img, orig_img, img_path, proto):
        # Chép SegmentationPredictor.construct_result (8.4.143), chỉ đổi hai
        # phép dựng mặt nạ sang bản A1.
        dyn = find_head(self.model).dyn
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = process_mask_native_dyn(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2], dyn=dyn)
        else:
            masks = process_mask_dyn(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True, dyn=dyn)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not all(keep):
                pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
