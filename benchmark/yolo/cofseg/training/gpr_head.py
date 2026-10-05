"""M1 — GPR: lấy mẫu lại prototype có dẫn hướng (configs/train/yolo26s-gpr.yaml).

Proto26 của YOLO26-seg dựng 32 prototype ở stride 4 HOÀN TOÀN từ đặc trưng
stride 8: P3 + P4↑ + P5↑ -> feat_fuse -> cv1 -> ConvTranspose2d x2 -> cv2 -> cv3
(`block.py:2005-2017`, `block.py:97-104`). Không đặc trưng stride 4 nào của
backbone đi vào, nên thông tin thật của mặt nạ ở lưới 20 px ảnh gốc, còn lưới
10 px chỉ là nội suy. Đo trên R2 (docs/reports/model/de_xuat_module_yolo26_
2026-10-05.md, mục 1.1): 76% pixel sai của tán khớp nằm ở biên tán–đất, dải
sai dày 10,2 px ảnh gốc — đúng một ô prototype.

GPR thay phép phóng mù đó bằng phép LẤY MẪU LẠI có độ lệch, độ lệch tính từ
đặc trưng độ phân giải cao của backbone (bản đồ dẫn hướng):

    out(p) = grid_sample(F, vị_trí_song_tuyến(p) + Δ(p))
    Δ(p)   = scope · tanh(W · [g(guide)(p), l(F)↑(p)])     (ô lưới thô)

- GPR-1 (stride 8 -> 4) thay ConvTranspose2d, dẫn hướng bằng layer 2 (C3k2, s4).
- GPR-2 (stride 4 -> 2) đưa 32 prototype lên lưới 5 px ảnh gốc, dẫn hướng bằng
  layer 0 (Conv, s2). Một trường độ lệch cho cả 32 kênh: mặt nạ tán i là
  c_i · P(p + Δ(p)), tức chính mặt nạ thô của tán đó được lấy mẫu lại, nên một
  trường sửa biên của MỌI tán cùng lúc.

W của lớp cuối khởi tạo 0, nên lúc đầu GPR đúng bằng phóng song tuyến. Đầu ra
chỉ là lấy mẫu lại giá trị đã có, không bịa giá trị mới. Bản đồ dẫn hướng mặc
định đi qua `.detach()`: loss mặt nạ không chảy ngược vào layer 0 và 2 qua
đường mới, backbone nhận đúng những tín hiệu như R2 — bài học từ L1 và A1, nơi
tín hiệu mới chảy vào phần dùng chung làm lệch thang điểm trên ruộng lạ.

File này là phần model và chấm val: toán tử, Proto, đầu, phép chuyển một
YOLO26-seg chuẩn sang GPR, validator. Phần train (loss NCB, trainer) ở
gpr_train.py. Checkpoint GPR chứa các lớp ở đây, nên muốn nạp nó phải import
được `cofseg`, như checkpoint A1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import ops

#: Lớp backbone làm bản đồ dẫn hướng: (layer, stride). Đọc từ yolo26-seg.yaml:
#: 0 = Conv P1/2, 2 = C3k2 sau Conv P2/4.
GUIDE_S4, GUIDE_S2 = 2, 0


# ---------------------------------------------------------------- toán tử
class GuidedResample(nn.Module):
    """Phóng x2 bằng lấy mẫu lại có độ lệch; độ lệch tính từ bản đồ dẫn hướng.

    x (B, C, h, w) ở lưới thô, guide (B, Cg, 2h, 2w) ở lưới mịn. `groups` nhóm
    kênh, mỗi nhóm một trường độ lệch. `scope` là độ lệch tối đa, tính bằng ô
    lưới thô. Δ = 0 cho đúng `F.interpolate(x, scale_factor=2, mode="bilinear")`.
    """

    def __init__(self, c: int, c_guide: int, c_mid: int, groups: int = 1, scope: float = 1.0,
                 detach_guide: bool = True):
        super().__init__()
        if c % groups:
            raise ValueError(f"số kênh {c} không chia hết cho groups={groups}")
        self.groups, self.scope, self.detach_guide = int(groups), float(scope), bool(detach_guide)
        self.g = Conv(c_guide, c_mid, k=1)          # nén bản đồ dẫn hướng, ở lưới mịn
        self.l = Conv(c, c_mid, k=1)                # nén đặc trưng thô, ở lưới thô rồi mới phóng
        self.dw = nn.Conv2d(2 * c_mid, 2 * c_mid, 3, padding=1, groups=2 * c_mid)
        self.off = nn.Conv2d(2 * c_mid, 2 * self.groups, 1)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)
        # |Δ| TB của lượt train gần nhất (ô thô), cho nhật ký. Giữ dạng tensor:
        # đổi sang float ở đây là đồng bộ GPU hai lần mỗi bước train.
        self.last_offset: torch.Tensor | float = 0.0

    def offsets(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Δ (B, 2·groups, 2h, 2w) theo ô lưới thô, thứ tự (dx, dy) từng nhóm."""
        if self.detach_guide:
            guide = guide.detach()
        size = guide.shape[-2:]
        l = F.interpolate(self.l(x), size=size, mode="bilinear", align_corners=False)
        return self.scope * torch.tanh(self.off(self.dw(torch.cat([self.g(guide), l], 1))))

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hh, ww = guide.shape[-2:]
        d = self.offsets(x, guide)
        if self.training:
            self.last_offset = d.detach().abs().mean()
        g = self.groups
        # Toạ độ chuẩn hoá của tâm ô mịn (align_corners=False): cùng vị trí vật lý
        # mà F.interpolate song tuyến lấy mẫu, nên Δ = 0 là phóng song tuyến.
        # Lưới tính bằng fp32: ở fp16 một ô của bản đồ 512 chỉ còn ~4 bước lượng tử.
        f32 = torch.float32
        ys = (torch.arange(hh, device=x.device, dtype=f32) + 0.5) * (2.0 / hh) - 1
        xs = (torch.arange(ww, device=x.device, dtype=f32) + 0.5) * (2.0 / ww) - 1
        base = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)          # (hh, ww, 2) (x, y)
        d = d.float().view(b, g, 2, hh, ww).permute(0, 1, 3, 4, 2).reshape(b * g, hh, ww, 2)
        d = d * torch.tensor([2.0 / w, 2.0 / h], device=d.device, dtype=f32)  # ô thô -> chuẩn hoá
        out = F.grid_sample(x.float().reshape(b * g, c // g, h, w), base[None] + d,
                            mode="bilinear", padding_mode="border", align_corners=False)
        return out.view(b, c, hh, ww).to(x.dtype)


# ---------------------------------------------------------------- Proto, đầu, model
class Proto26GPR(Proto26):
    """Proto26 có GPR. Chỉ tạo bằng to_gpr() từ một Proto26 đã dựng, để mọi lớp
    cũ (feat_refine, feat_fuse, cv1, cv2, cv3, semseg) giữ tên và trọng số.

    forward nhận [P3, P4, P5, dẫn_hướng_s4, dẫn_hướng_s2].
    """

    gpr: dict

    def forward(self, x, return_semantic: bool = True):
        feats, g4, g2 = x[:3], x[3], x[4]
        feat = feats[0]
        for i, f in enumerate(self.feat_refine):
            feat = feat + F.interpolate(f(feats[i + 1]), scale_factor=2 ** (i + 1), mode="nearest")
        y = self.cv1(self.feat_fuse(feat))
        y = self.gpr1(y, g4) if 1 in self.gpr["stages"] else self.upsample(y)
        p = self.cv3(self.cv2(y))
        if 2 in self.gpr["stages"]:
            p = self.gpr2(p, g2)
        if self.training and return_semantic:
            return p, self.semseg(feat)
        return p


class Segment26GPR(Segment26):
    """Segment26 nhận thêm hai bản đồ dẫn hướng sau P3-P5. Phần phát hiện
    (Detect.forward) chỉ thấy P3-P5, đúng như đầu gốc."""

    def forward(self, x):
        feats = x[: self.nl]
        outputs = Detect.forward(self, feats)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(list(x))
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class GPRSegmentationModel(SegmentationModel):
    """SegmentationModel có GPR. Khác gốc ở hàm loss (NCB, xem gpr_train.py):
    bản EMA mà validator chấm lúc train cũng tự tạo loss qua init_criterion()."""

    def init_criterion(self):
        from .gpr_train import build_criterion

        return build_criterion(self)


def _out_channels(m: nn.Module) -> int:
    """Số kênh ra của một lớp backbone: Conv (layer 0) hay C3k2 (layer 2, ra ở cv2)."""
    if isinstance(m, Conv):
        return m.conv.out_channels
    if isinstance(getattr(m, "cv2", None), Conv):
        return m.cv2.conv.out_channels
    raise SystemExit(f"Không đọc được số kênh ra của lớp dẫn hướng {type(m).__name__}")


def to_gpr(model: SegmentationModel, cfg: dict) -> SegmentationModel:
    """Chuyển TẠI CHỖ một YOLO26-seg chuẩn sang GPR.

    Chỉ Proto đổi: thêm gpr1/gpr2, bỏ ConvTranspose2d nếu GPR-1 bật. Backbone,
    neck, đầu box/cls/hệ số giữ nguyên, nên nạp trọng số COCO sau bước này thì
    mọi lớp cũ nhận trọng số, chỉ lớp mới của GPR giữ khởi tạo.
    """
    head = model.model[-1]
    if isinstance(head, Segment26GPR):
        return model
    if type(head) is not Segment26:
        raise SystemExit(
            f"GPR viết cho đầu Segment26 của YOLO26-seg; model này có {type(head).__name__}. "
            "Dùng một model yolo26*-seg."
        )
    stages = sorted(int(s) for s in cfg["stages"])
    detach = bool(cfg["detach_guide"])
    proto = head.proto
    dev, dt = proto.cv3.conv.weight.device, proto.cv3.conv.weight.dtype
    c_mid = proto.cv1.conv.out_channels
    if 1 in stages:
        cg = _out_channels(model.model[GUIDE_S4])
        proto.gpr1 = GuidedResample(c_mid, cg, 32, groups=int(cfg["groups"]), scope=float(cfg["scope"]),
                                    detach_guide=detach).to(dev, dt)
        del proto.upsample
    if 2 in stages:
        cg = _out_channels(model.model[GUIDE_S2])
        proto.gpr2 = GuidedResample(head.nm, cg, 16, groups=1, scope=float(cfg["scope"]),
                                    detach_guide=detach).to(dev, dt)
    proto.__class__ = Proto26GPR
    # Proto26 gộp P3-P5 ở stride của P3 (8), cv1 ở s8, phóng x2 -> s4; GPR-2 thêm x2.
    pstride = float(head.stride[0]) / (4 if 2 in stages else 2)
    proto.gpr = {**cfg, "stages": stages, "pstride": pstride}
    head.__class__ = Segment26GPR
    head.f = [*head.f, GUIDE_S4, GUIDE_S2]
    model.save = sorted(set(model.save) | {GUIDE_S4, GUIDE_S2})
    model.__class__ = GPRSegmentationModel
    return model


def find_proto(model) -> Proto26GPR:
    """Proto GPR trong một model: nn.Module trần, bản EMA, hay AutoBackend
    (AutoBackend cất model trong `backend.model`, xem dyn_head.find_head)."""
    backend = getattr(model, "backend", None)
    for cand in (model, getattr(backend, "model", None)):
        if isinstance(cand, nn.Module):
            for m in cand.modules():
                if isinstance(m, Proto26GPR):
                    return m
    raise TypeError("model không có Proto GPR (Proto26GPR)")


def is_gpr(model) -> bool:
    try:
        find_proto(model)
        return True
    except TypeError:
        return False


# ---------------------------------------------------------------- chấm val lúc train
class GPRSegValidator(SegmentationValidator):
    """SegmentationValidator cho prototype ở stride khác 4.

    Gốc giả định stride 4 ở hai chỗ: cỡ ảnh suy từ proto (`4 * proto.shape`,
    val.py:103) và lưới của mặt nạ GT (`s // 4`, val.py:128). Với GPR-2 prototype
    ở stride 2: giữ nguyên hai phép đó thì box bị thu nhỏ một nửa khi cắt mặt nạ
    và mặt nạ dự đoán lệch cỡ với GT. Ở đây cả hai dùng stride thật của proto, nên
    val lúc train chấm trên lưới của chính model (5 px ảnh gốc), không phải 1/4.
    """

    def init_metrics(self, model):
        super().init_metrics(model)
        self._pstride = int(find_proto(model).gpr["pstride"])

    def postprocess(self, preds):
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        out = super(SegmentationValidator, self).postprocess(preds[0])
        imgsz = [self._pstride * x for x in proto.shape[2:]]
        for i, pred in enumerate(out):
            coefficient = pred.pop("extra")
            pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
        return out

    def _prepare_batch(self, si, batch):
        prepared = super(SegmentationValidator, self)._prepare_batch(si, batch)
        nl = prepared["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            native = self.process is ops.process_mask_native
            size = [s if native else s // self._pstride for s in prepared["imgsz"]]
            if masks.shape[1:] != size:
                masks = F.interpolate(masks[None], size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared["masks"] = masks
        return prepared
