"""KT4: 5 kênh toạ độ tương đối với hộp của từng tán trong phép ghép mặt nạ
(configs/train/yolo26s-kt4.yaml).

YOLO-seg dựng mặt nạ mỗi tán bằng tổ hợp tuyến tính 32 prototype dùng chung cả
ảnh, hệ số là MỘT vector cho cả tán, rồi cắt theo hộp. Prototype dùng chung
không mang thông tin "pixel này nằm ở đâu so với tán đang tô", nên ở hai tán
chạm nhau, cả hệ số tối ưu cũng chỉ đạt IoU 0,63 (đo bằng oracle trên R2).
Thêm 5 kênh toạ độ dưới đây nâng trần đó lên 0,74.

KT4 cộng vào logit của mỗi tán, trên lưới proto:

    c_u·u + c_v·v + c_uu·u² + c_vv·v² + c_uv·u·v
    u = (x − cx) / (w/2),  v = (y − cy) / (h/2),  kẹp trong [−2, 2]

với (cx, cy, w, h) là hộp của chính tán đó và (x, y) là tâm ô lưới proto, cùng
theo pixel ảnh đầu vào. Trong hộp, u và v chạy từ −1 ở mép này tới +1 ở mép
kia. Đầu hệ số ra 32 + 5 số; 5 số mới khởi tạo bằng 0, nên lúc bắt đầu train
model trùng R2 từng số và chỉ dùng toạ độ khi điều đó làm loss giảm.
Prototype, đầu hộp, đầu điểm và cách cắt theo hộp giữ nguyên.

Lúc suy luận, hộp là hộp dự đoán. Lúc train, `box: pred` lấy hộp do chính
anchor đó dự đoán (đã tách gradient), `box: gt` lấy hộp thật: xem
coord_train.py.

File này là phần model và suy luận: phép dựng, lớp đầu, phép chuyển một
YOLO26-seg chuẩn sang KT4, validator và predictor. Checkpoint KT4 chứa các lớp
ở đây, nên muốn nạp nó phải import được `cofseg`, như A1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationValidator
from ultralytics.nn.modules.head import Detect, Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import ops

#: Số kênh toạ độ: u, v, u², v², uv.
N_COORD = 5

#: u, v kẹp trong [−CLAMP, CLAMP]. Trong hộp chúng nằm trong [−1, 1]; ngoài hộp
#: mặt nạ bị cắt bỏ, kẹp chỉ để giá trị không lớn vô hạn với hộp rất nhỏ (an
#: toàn cho fp16).
CLAMP = 2.0

#: Số tán dựng mặt nạ mỗi lượt lúc suy luận.
CHUNK = 64


# ---------------------------------------------------------------- phép dựng
def coord_axes(boxes: torch.Tensor, mh: int, mw: int, fstride: float) -> tuple[torch.Tensor, torch.Tensor]:
    """u theo cột (n, mw) và v theo hàng (n, mh), float32.

    boxes (n, 4) xyxy theo pixel ảnh đầu vào; fstride là số pixel ảnh mỗi ô
    lưới proto. Nửa cạnh hộp không nhỏ hơn một ô, để hộp suy biến lúc đầu
    train không làm toạ độ nổ.
    """
    b = boxes.float()
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    hw = ((b[:, 2] - b[:, 0]) / 2).clamp(min=fstride)
    hh = ((b[:, 3] - b[:, 1]) / 2).clamp(min=fstride)
    xs = (torch.arange(mw, device=b.device, dtype=torch.float32) + 0.5) * fstride
    ys = (torch.arange(mh, device=b.device, dtype=torch.float32) + 0.5) * fstride
    u = ((xs[None] - cx[:, None]) / hw[:, None]).clamp(-CLAMP, CLAMP)
    v = ((ys[None] - cy[:, None]) / hh[:, None]).clamp(-CLAMP, CLAMP)
    return u, v


def coord_term(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Phần toạ độ của logit, (n, mh, mw) float32. c (n, 5) theo thứ tự u, v, u², v², uv.

    Viết tách theo hàng và cột: mọi phép nhân có gradient chỉ giữ lại tensor
    (n, mh) hoặc (n, mw), không giữ bản đồ (n, 5, mh, mw) nào cho lượt lùi.
    """
    c = c.float()
    row = c[:, 0:1] * u + c[:, 2:3] * u * u                 # (n, mw)
    col = c[:, 1:2] * v + c[:, 3:4] * v * v                 # (n, mh)
    return row[:, None, :] + col[:, :, None] + (c[:, 4:5] * v)[:, :, None] * u[:, None, :]


def coord_maps(boxes: torch.Tensor, mh: int, mw: int, fstride: float) -> torch.Tensor:
    """Năm bản đồ toạ độ (n, 5, mh, mw). Chỉ dùng để kiểm và để đo; phép dựng dùng coord_term."""
    u, v = coord_axes(boxes, mh, mw, fstride)
    U, V = u[:, None, :].expand(-1, mh, -1), v[:, :, None].expand(-1, -1, mw)
    return torch.stack([U, V, U * U, V * V, U * V], 1)


def mask_logits(protos: torch.Tensor, coefs: torch.Tensor, boxes: torch.Tensor, fstride: float) -> torch.Tensor:
    """Logit mặt nạ (n, mh, mw) float32 của n tán.

    protos (C, mh, mw); coefs (n, C + 5); boxes (n, 4) xyxy theo pixel ảnh đầu vào.
    """
    c, mh, mw = protos.shape
    out = []
    for i in range(0, coefs.shape[0], CHUNK):
        k, b = coefs[i:i + CHUNK], boxes[i:i + CHUNK]
        base = (k[:, :c].float() @ protos.float().view(c, -1)).view(-1, mh, mw)
        u, v = coord_axes(b, mh, mw, fstride)
        out.append(base + coord_term(k[:, c:], u, v))
    return torch.cat(out)


# ---------------------------------------------------------------- đầu và model
class Segment26Coord(Segment26):
    """Segment26 với đầu hệ số ra nm + 5 số. Chỉ tạo bằng to_coord().

    `kt4` mang cấu hình: box, n_coord, fstride.
    """

    kt4: dict

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if mask_head is not None:
            bs, k = x[0].shape[0], self.nm + N_COORD
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, k, -1) for i in range(self.nl)], 2)
        return preds


class CoordSegmentationModel(SegmentationModel):
    """SegmentationModel có đầu KT4. Khác gốc đúng một chỗ: hàm loss.

    Phải là lớp riêng: bản EMA mà validator chấm trong lúc train tự tạo hàm
    loss qua init_criterion(), và loss gốc không hiểu 37 hệ số.
    """

    def init_criterion(self):
        from .coord_train import build_criterion

        return build_criterion(self)


def _expand(conv: nn.Conv2d, extra: int) -> nn.Conv2d:
    """Conv 1x1 thêm `extra` kênh ra, trọng số và bias bằng 0; kênh cũ giữ nguyên.

    Tạo trong fork_rng: trạng thái ngẫu nhiên chung giữ y như lượt R2 cùng
    seed, vì kênh mới không cần số ngẫu nhiên nào.
    """
    with torch.random.fork_rng(devices=[]):
        new = nn.Conv2d(conv.in_channels, conv.out_channels + extra, 1, bias=True)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.zero_()
        new.weight[:conv.out_channels] = conv.weight
        new.bias[:conv.out_channels] = conv.bias
    return new.to(device=conv.weight.device, dtype=conv.weight.dtype)


def to_coord(model: SegmentationModel, cfg: dict) -> SegmentationModel:
    """Chuyển TẠI CHỖ một YOLO26-seg chuẩn sang KT4.

    Chỉ lớp 1x1 cuối của cv4 (và one2one_cv4) đổi: nm -> nm + 5 đầu ra, kênh
    cũ giữ trọng số đang có. Gọi SAU khi nạp trọng số COCO, để 32 kênh cũ
    nhận đúng trọng số pretrained như R2.
    """
    head = model.model[-1]
    if isinstance(head, Segment26Coord):
        return model
    if type(head) is not Segment26:
        raise SystemExit(
            f"KT4 viết cho đầu Segment26 của YOLO26-seg; model này có {type(head).__name__}. "
            "Dùng một model yolo26*-seg."
        )
    for seq in (head.cv4, getattr(head, "one2one_cv4", None)):
        if seq is None:
            continue
        for branch in seq:
            branch[-1] = _expand(branch[-1], N_COORD)
    head.__class__ = Segment26Coord
    # Proto26 gộp P3-P5 ở stride của P3 rồi phóng x2: lưới proto ở stride[0] / 2 = 4.
    head.kt4 = {**cfg, "n_coord": N_COORD, "fstride": float(head.stride[0]) / 2}
    model.__class__ = CoordSegmentationModel
    return model


def find_head(model) -> Segment26Coord:
    """Đầu KT4 trong một model: nn.Module trần, bản EMA, hay AutoBackend (như dyn_head.find_head)."""
    backend = getattr(model, "backend", None)
    for cand in (model, getattr(backend, "model", None)):
        if isinstance(cand, nn.Module):
            for m in cand.modules():
                if isinstance(m, Segment26Coord):
                    return m
    raise TypeError("model không có đầu KT4 (Segment26Coord)")


def is_coord(model) -> bool:
    try:
        find_head(model)
        return True
    except TypeError:
        return False


# ---------------------------------------------------------------- suy luận
def process_mask_coord(protos, coefs, bboxes, shape, upsample: bool = False):
    """Bản KT4 của `ops.process_mask`: cùng chữ ký, cùng đầu ra. bboxes theo
    pixel ảnh đầu vào cỡ `shape`; `coefs` có nm + 5 cột."""
    c, mh, mw = protos.shape
    if coefs.shape[0] == 0:
        return torch.zeros((0, *(shape if upsample else (mh, mw))), dtype=torch.uint8, device=coefs.device)
    masks = mask_logits(protos, coefs, bboxes, float(shape[0]) / mh)
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    else:
        bboxes = bboxes * torch.tensor([[mw / shape[1], mh / shape[0]] * 2], device=bboxes.device)
    return ops.crop_mask(masks.gt_(0.0).byte(), bboxes)


def process_mask_native_coord(protos, coefs, bboxes_in, in_shape, bboxes, shape):
    """Bản KT4 của `ops.process_mask_native`.

    Toạ độ tính theo hộp ở ảnh đầu vào (`bboxes_in`, cỡ `in_shape`), vì lưới
    proto thuộc ảnh đầu vào; mặt nạ phóng về ảnh gốc cỡ `shape` rồi cắt theo
    hộp ở ảnh gốc (`bboxes`), như gốc.
    """
    c, mh, mw = protos.shape
    h, w = shape
    if coefs.shape[0] == 0:
        return torch.zeros((0, h, w), dtype=torch.uint8, device=coefs.device)
    logits = mask_logits(protos, coefs, bboxes_in, float(in_shape[0]) / mh)
    step = max(1, 32_000_000 // (h * w))  # cùng ngân sách pixel như gốc
    masks = [ops.scale_masks(logits[i:i + step][None], shape)[0].gt_(0.0).byte()
             for i in range(0, logits.shape[0], step)]
    return ops.crop_mask(torch.cat(masks), bboxes)


def _process_native_val(protos, coefs, bboxes, shape):
    """Validator gọi bản native với hộp và `shape` cùng ở ảnh đầu vào."""
    return process_mask_native_coord(protos, coefs, bboxes, shape, bboxes, shape)


class CoordSegValidator(SegmentationValidator):
    """SegmentationValidator dựng mặt nạ bằng đầu KT4."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self._stock_process = self.process
        native = self.process is ops.process_mask_native
        self.process = _process_native_val if native else process_mask_coord

    def _prepare_batch(self, si, batch):
        # Gốc chọn cỡ mặt nạ GT bằng `self.process is ops.process_mask_native`
        # (val.py:128); trả hàm gốc về trong lúc đó để phép so còn đúng.
        coord_process, self.process = self.process, self._stock_process
        try:
            return super()._prepare_batch(si, batch)
        finally:
            self.process = coord_process


class CoordSegPredictor(SegmentationPredictor):
    """SegmentationPredictor dựng mặt nạ bằng đầu KT4."""

    def construct_result(self, pred, img, orig_img, img_path, proto):
        # Chép SegmentationPredictor.construct_result (8.4.143), chỉ đổi hai
        # phép dựng mặt nạ sang bản KT4.
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            boxes_in = pred[:, :4].clone()
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = process_mask_native_coord(proto, pred[:, 6:], boxes_in, img.shape[2:], pred[:, :4],
                                              orig_img.shape[:2])
        else:
            masks = process_mask_coord(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not all(keep):
                pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

