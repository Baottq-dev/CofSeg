"""Mask R-CNN R50-FPN (torchvision v2) trong hợp đồng SegmentationModel.

Đây là mốc nền hai giai đoạn kinh điển của họ task-specific: đầu mặt nạ 28x28
cố định cho mỗi vùng, nên với tán cạnh ~325 px thì mỗi ô mặt nạ nuốt ~12 px
ảnh gốc — trần đường biên thấp hơn YOLO-seg @1024 (5-10 px). Con số đó là lý
do nó đứng trong bảng: mọi model tinh chỉnh biên (PointRend, Mask Transfiner)
đều được đo bằng khoảng cách tới chính mốc này.

Chọn torchvision thay vì detectron2 vì nó chạy trên Windows với env hiện có;
kiến trúc và trọng số COCO là cùng một họ, chỉ khác khung huấn luyện.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..datasets.instances import resize_long_side
from ..registry import register
from .base import Prediction, SegmentationModel

ARCH = "maskrcnn_resnet50_fpn_v2"


def build_maskrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    imgsz: int = 1024,
    max_det: int = 100,
    box_score_thresh: float = 0.05,
):
    """Dựng model với đầu box/mask thay cho `num_classes` (nền + canopy = 2).

    min_size = max_size = imgsz: ảnh đưa vào đã được thu về cạnh dài imgsz, nên
    transform nội bộ của torchvision (scale = min(min/ngắn, max/dài)) ra đúng 1
    và không thu lần hai. Nhờ vậy "imgsz" ở đây cùng nghĩa với imgsz của YOLO.
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        # None khi không có trọng số: torchvision tự mở cả 5 tầng và không cảnh báo.
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        min_size=imgsz,
        max_size=imgsz,
        box_detections_per_img=max_det,
        box_score_thresh=box_score_thresh,
    )
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def save_checkpoint(path: str | Path, model, **meta) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": ARCH, "model": model.state_dict(), **meta}, path)
    return path


def load_checkpoint(path: str | Path, device=None, **overrides):
    """Trọng số -> (model đã nạp, meta). `overrides` đè lên imgsz/max_det đã lưu."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ck.get("arch") != ARCH:
        raise ValueError(f"{path} không phải checkpoint {ARCH} (arch={ck.get('arch')!r})")
    kw = dict(
        num_classes=int(ck.get("num_classes", 2)),
        pretrained=False,
        imgsz=int(ck.get("imgsz", 1024)),
        max_det=int(ck.get("max_det", 100)),
    )
    kw.update({k: v for k, v in overrides.items() if v is not None})
    model = build_maskrcnn(**kw)
    model.load_state_dict(ck["model"])
    if device is not None:
        model.to(device)
    meta = {k: v for k, v in ck.items() if k != "model"}
    meta.update(kw)
    return model, meta


@register("model", "maskrcnn")
class MaskRCNNModel(SegmentationModel):
    """Model đầu-cuối: tự tìm tán, không cần gợi ý.

    Nhận `weights` (checkpoint do trainer ghi) hoặc `module` (model torch đã
    dựng sẵn — trainer dùng đường này để chấm val giữa các epoch mà không phải
    ghi/đọc đĩa).
    """

    needs_prompt = False

    def __init__(
        self,
        weights: str | None = None,
        module=None,
        imgsz: int | None = None,
        conf: float = 0.25,
        max_det: int | None = None,
        mask_thr: float = 0.5,
        device: str | None = None,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights
        if module is None:
            if not weights:
                raise ValueError("MaskRCNNModel cần `weights` hoặc `module`")
            module, meta = load_checkpoint(weights, self.device, imgsz=imgsz, max_det=max_det)
            imgsz = int(meta["imgsz"])
            max_det = int(meta["max_det"])
        elif imgsz is None:
            # torchvision giữ min_size dạng tuple; lấy phần tử đầu.
            imgsz = int(module.transform.min_size[0])
        self.module = module.to(self.device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_det = max_det
        self.mask_thr = float(mask_thr)

    def warmup(self) -> None:
        self.predict(np.zeros((self.imgsz * 9 // 16, self.imgsz, 3), np.uint8))

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        import torch

        H, W = image.shape[:2]
        small, scale = resize_long_side(image, self.imgsz)
        x = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
        self.module.eval()
        with torch.inference_mode():
            out = self.module([x.to(self.device)])[0]

        scores = out["scores"].detach().cpu().numpy()
        bxs = out["boxes"].detach().cpu().numpy()
        # (N, 1, h, w) xác suất trên khung ảnh đã thu; torchvision đã dán 28x28
        # vào đúng box, nên ngoài box là 0 và cắt theo box không mất gì.
        probs = out["masks"].detach()
        preds: list[Prediction] = []
        for i in range(len(scores)):
            if scores[i] < self.conf:
                continue
            x0, y0, x1, y1 = bxs[i]
            # Nới 2 px để không cắt cụt phần mềm của mặt nạ ở mép box.
            sx0, sy0 = max(0, math.floor(x0) - 2), max(0, math.floor(y0) - 2)
            sx1 = min(small.shape[1], math.ceil(x1) + 2)
            sy1 = min(small.shape[0], math.ceil(y1) + 2)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            crop = probs[i, 0, sy0:sy1, sx0:sx1].float().cpu().numpy()
            # Về khung gốc: phóng BẢN ĐỒ XÁC SUẤT rồi mới ngưỡng. Phóng mặt nạ
            # nhị phân sẽ tạo bậc thang đúng ở chỗ dự án đo kỹ nhất.
            ox, oy = int(round(sx0 / scale)), int(round(sy0 / scale))
            ex, ey = min(W, int(round(sx1 / scale))), min(H, int(round(sy1 / scale)))
            if ex <= ox or ey <= oy:
                continue
            big = cv2.resize(crop, (ex - ox, ey - oy), interpolation=cv2.INTER_LINEAR)
            mask = big >= self.mask_thr
            if not mask.any():
                continue
            preds.append(Prediction(mask=mask, origin=(ox, oy), score=float(scores[i])))
        return preds

    @property
    def describe(self) -> dict:
        return {
            **super().describe,
            "arch": ARCH,
            "weights": self.weights,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "max_det": self.max_det,
            "mask_thr": self.mask_thr,
            "device": self.device,
        }
