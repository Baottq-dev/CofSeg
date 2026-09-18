"""SAM 2.1 trong hợp đồng SegmentationModel: oracle, tự động, và bộ tinh chỉnh.

Đây là họ "promptable" của đề tài. SAM không tự biết "tán cà phê" là gì; nó
trả về mặt nạ cho gợi ý (box, điểm, mặt nạ) mà ai đó đưa vào. Vì thế trong
benchmark nó xuất hiện ở ba vai, mỗi vai một lớp ở đây:

  sam2_oracle   box THẬT từ nhãn -> cận trên: SAM vẽ tốt đến đâu nếu biết
                chính xác tán ở đâu. Không phải hệ thống thật, chỉ là trần.
  sam2_auto     lưới điểm tự động (AMG) -> zero-shot thật, không nhãn, không
                phát hiện; trả cả đất, bóng, hàng cây — phải chịu số thật.
  SAMRefiner    cắm sau một model khác (YOLO, Mask R-CNN): lấy box (và tuỳ
                chọn mặt nạ) của dự đoán gốc làm gợi ý, vẽ lại đường biên.
                Dùng qua RefineModel trong compose.py.

Tên file không phải sam2.py để khỏi che package `sam2` khi import tương đối.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..registry import register
from .base import Prediction, SegmentationModel

#: Cỡ model -> (config Hydra nội bộ của package sam2, checkpoint mặc định).
SIZES: dict[str, tuple[str, str]] = {
    "t": ("configs/sam2.1/sam2.1_hiera_t.yaml", "weights/sam2.1_hiera_tiny.pt"),
    "s": ("configs/sam2.1/sam2.1_hiera_s.yaml", "weights/sam2.1_hiera_small.pt"),
    "b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "weights/sam2.1_hiera_base_plus.pt"),
    "l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "weights/sam2.1_hiera_large.pt"),
}

#: Cạnh lưới mặt nạ độ phân giải thấp mà SAM nhận làm gợi ý (logit).
LOW_RES = 256


class Sam2Backend:
    """Một SAM 2.1 đã nạp + predictor, dùng chung cho mọi lớp trong file này.

    `masks_for_boxes` chạy theo lô `box_batch` box một lần: predictor trả mặt
    nạ ở cỡ ảnh gốc dạng float, 48 mặt nạ 2560x1440 là ~700 MB trên GPU, nên
    không đưa cả ảnh 48 tán vào một lần.
    """

    def __init__(
        self,
        size: str = "l",
        checkpoint: str | None = None,
        device: str | None = None,
        box_batch: int = 16,
    ):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if size not in SIZES:
            raise ValueError(f"size phải là một trong {sorted(SIZES)}, nhận {size!r}")
        cfg, default_ck = SIZES[size]
        self.size = size
        self.checkpoint = str(checkpoint or default_ck)
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Không thấy checkpoint SAM 2.1: {self.checkpoint}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(cfg, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.box_batch = int(box_batch)

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Mã hoá ảnh (phần đắt nhất). Gọi đúng một lần cho mỗi ảnh rồi hỏi
        bao nhiêu box cũng được. Không nhớ theo id(mảng): Python tái dùng id
        sau khi mảng cũ bị thu hồi, và mặt nạ sẽ vẽ trên ảnh trước đó."""
        self.predictor.set_image(np.ascontiguousarray(image_bgr[:, :, ::-1]))  # SAM nhận RGB

    def masks_for_boxes(
        self, boxes: np.ndarray, mask_prompts: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box xyxy (N, 4) [+ mặt nạ gợi ý (N, H, W) bool] -> (mặt nạ (N, H, W) bool, IoU dự đoán (N,)).

        Ảnh phải được set_image trước. multimask_output=False vì box đã khử
        mơ hồ; ba mặt nạ ứng viên chỉ có nghĩa cho gợi ý điểm.
        """
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        n = len(boxes)
        h, w = self.predictor._orig_hw[-1]
        masks = np.zeros((n, h, w), bool)
        scores = np.zeros(n, np.float32)
        for s in range(0, n, self.box_batch):
            chunk = boxes[s : s + self.box_batch]
            kw = {}
            if mask_prompts is not None:
                kw["mask_input"] = self._to_logits(mask_prompts[s : s + self.box_batch])
            m, iou, _ = self.predictor.predict(box=chunk, multimask_output=False, **kw)
            # predictor squeeze(0): 1 box -> (1, H, W) hoặc (H, W); B box -> (B, 1, H, W).
            m = np.asarray(m).reshape(-1, h, w)
            iou = np.asarray(iou).reshape(-1)
            masks[s : s + len(chunk)] = m[: len(chunk)] > 0
            scores[s : s + len(chunk)] = iou[: len(chunk)]
        return masks, scores

    @staticmethod
    def _to_logits(masks: np.ndarray) -> np.ndarray:
        """(N, H, W) bool -> (N, 1, 256, 256) logit: +8 trong mặt nạ, -8 ngoài.

        SAM 2 thu ảnh về 1024x1024 bỏ qua tỉ lệ khung, nên lưới 256x256 cũng
        là cả ảnh bị ép vuông — resize thẳng, không đệm.
        """
        out = np.empty((len(masks), 1, LOW_RES, LOW_RES), np.float32)
        for i, m in enumerate(masks):
            small = cv2.resize(m.astype(np.uint8), (LOW_RES, LOW_RES), interpolation=cv2.INTER_AREA)
            out[i, 0] = np.where(small > 0, 8.0, -8.0)
        return out

    @property
    def describe(self) -> dict:
        return {"sam2": self.size, "checkpoint": self.checkpoint, "box_batch": self.box_batch}


def _from_full(mask: np.ndarray, score: float, **meta) -> Prediction | None:
    """Mặt nạ toàn khung -> Prediction cắt sát bbox; None nếu rỗng."""
    if not mask.any():
        return None
    return Prediction(mask=mask, origin=(0, 0), score=float(score), meta=meta).cropped()


@register("model", "sam2_oracle")
class OracleSAMModel(SegmentationModel):
    """SAM 2.1 với box THẬT làm gợi ý. Cận trên của họ promptable, không phải
    hệ thống chạy được ngoài đời — bảng kết quả phải ghi rõ điều đó."""

    needs_prompt = True

    def __init__(self, size: str = "l", checkpoint: str | None = None,
                 device: str | None = None, box_batch: int = 16):
        self.backend = Sam2Backend(size, checkpoint, device, box_batch)

    def warmup(self) -> None:
        img = np.zeros((256, 256, 3), np.uint8)
        self.backend.set_image(img)
        self.backend.masks_for_boxes(np.array([[10, 10, 100, 100]], np.float32))

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        if boxes is None or len(boxes) == 0:
            return []
        self.backend.set_image(image)
        masks, ious = self.backend.masks_for_boxes(boxes)
        out = []
        for m, s, b in zip(masks, ious, boxes):
            p = _from_full(m, s, prompt_box=[float(v) for v in b])
            if p is not None:
                out.append(p)
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "prompt": "gt_box", **self.backend.describe}


@register("model", "sam2_auto")
class SAMAutoModel(SegmentationModel):
    """SAM 2.1 tự động (AMG): lưới điểm, không nhãn, không phát hiện.

    Trả về mọi thứ giống "vật thể" — đất trống, bóng, hàng cây liền nhau. Lọc
    diện tích là cách duy nhất không cần nhãn để bỏ bớt; ngoài ra là số thật.
    """

    needs_prompt = False

    def __init__(
        self,
        size: str = "l",
        checkpoint: str | None = None,
        device: str | None = None,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 0,
        min_area: float = 0.0,
        max_area: float | None = None,
    ):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.backend = Sam2Backend(size, checkpoint, device)
        self.params = dict(
            points_per_side=int(points_per_side), points_per_batch=int(points_per_batch),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
            box_nms_thresh=float(box_nms_thresh), crop_n_layers=int(crop_n_layers),
            min_mask_region_area=int(min_mask_region_area),
        )
        self.generator = SAM2AutomaticMaskGenerator(
            self.backend.model, output_mode="binary_mask", **self.params
        )
        self.min_area, self.max_area = float(min_area), max_area

    def warmup(self) -> None:
        self.generator.generate(np.zeros((256, 256, 3), np.uint8))

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        anns = self.generator.generate(np.ascontiguousarray(image[:, :, ::-1]))
        out = []
        for a in anns:
            area = float(a.get("area", 0))
            if area < self.min_area or (self.max_area is not None and area > self.max_area):
                continue
            p = _from_full(np.asarray(a["segmentation"], bool), a.get("predicted_iou", 1.0),
                           stability=float(a.get("stability_score", 0.0)))
            if p is not None:
                out.append(p)
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "prompt": "auto_grid", **self.backend.describe,
                **self.params, "min_area": self.min_area, "max_area": self.max_area}


@register("refiner", "sam2")
class SAMRefiner:
    """Vẽ lại đường biên của dự đoán có sẵn bằng SAM 2.1.

    prompt = "box": chỉ box của mặt nạ gốc. Rẻ, và là cách "SAM-refine" hay
    được dùng nhất. prompt = "box+mask": thêm mặt nạ gốc làm gợi ý dày — giữ
    SAM gần với dự đoán gốc hơn, hợp khi model gốc đã tốt và chỉ cần sửa mép.
    """

    def __init__(self, size: str = "l", checkpoint: str | None = None,
                 device: str | None = None, prompt: str = "box", box_batch: int = 16):
        if prompt not in ("box", "box+mask"):
            raise ValueError(f"prompt phải là 'box' hoặc 'box+mask', nhận {prompt!r}")
        self.backend = Sam2Backend(size, checkpoint, device, box_batch)
        self.prompt = prompt

    def refine(self, image: np.ndarray, preds: list[Prediction]) -> list[Prediction]:
        """Trả về danh sách cùng độ dài, cùng score; mặt nạ thay bằng của SAM.
        Dự đoán mà SAM trả rỗng thì giữ nguyên (meta["refined"] = False)."""
        keep = [(i, p.bbox_xyxy) for i, p in enumerate(preds)]
        keep = [(i, b) for i, b in keep if b is not None]
        if not keep:
            return list(preds)
        self.backend.set_image(image)
        boxes = np.array([b for _, b in keep], np.float32)
        mask_prompts = None
        if self.prompt == "box+mask":
            from ..evaluation.coco_eval import full_frame

            h, w = image.shape[:2]
            mask_prompts = np.stack([full_frame(preds[i], (h, w)) for i, _ in keep])
        masks, ious = self.backend.masks_for_boxes(boxes, mask_prompts)

        out = list(preds)
        for (i, _), m, iou in zip(keep, masks, ious):
            src = preds[i]
            meta = {**src.meta, "refined": False, "sam_iou": float(iou)}
            meta.pop("bbox", None)
            if m.any():
                meta["refined"] = True
                out[i] = Prediction(mask=m, origin=(0, 0), score=src.score,
                                    polygon=None, meta=meta).cropped()
            else:
                out[i] = Prediction(mask=src.mask, origin=src.origin, score=src.score,
                                    polygon=src.polygon, meta=meta)
        return out

    @property
    def describe(self) -> dict:
        return {"refiner": "sam2", "prompt": self.prompt, **self.backend.describe}
