"""Bọc YOLO-seg đã huấn luyện vào hợp đồng SegmentationModel.

Nhờ lớp bọc này, model YOLO đi qua ĐÚNG đường đo với mọi model khác, nên các
con số so sánh được. Không có nó thì mỗi họ model lại có một đường tính chỉ
số hơi khác nhau và bảng so sánh mất ý nghĩa.
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from .base import Prediction, SegmentationModel


@register("model", "yolo_seg")
class YoloSegModel(SegmentationModel):
    """Model đầu-cuối: tự tìm tán, không cần gợi ý."""

    needs_prompt = False

    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        device: str | int | None = None,
        retina_masks: bool = True,
        crop_expand: float = 1.0,
    ):
        from ultralytics import YOLO

        # Kiểm trước khi nạp trọng số: gõ sai thì báo ngay, không chờ nạp model.
        if not (isinstance(crop_expand, (int, float)) and not isinstance(crop_expand, bool)
                and crop_expand >= 1.0):
            raise SystemExit(f"crop_expand phải là số >= 1.0, nhận {crop_expand!r}")
        self.weights = weights
        self.model = YOLO(weights)
        # Checkpoint A1 (đầu mặt nạ động, cofseg/training/dyn_head.py) cần
        # predictor riêng: predictor gốc dựng mặt nạ bằng hệ số x prototype rồi
        # cắt sát box, còn đầu A1 xuất tham số của một mạng nhỏ cho từng tán.
        self.mask_head = "gốc"
        self._predict_kw: dict = {}
        head = type(self.model.model.model[-1]).__name__
        if head == "Segment26Dyn":
            from ..training.dyn_head import DynSegPredictor

            self.mask_head = "dyn"
            self._predict_kw["predictor"] = DynSegPredictor
        # crop_expand > 1: cắt mặt nạ bằng box dự đoán NỚI theo hệ số này thay
        # vì box sát. Box sát cắt thẳng 21% số tán của R2 (báo cáo 2026-10-04,
        # mục 3.3). Box trả về vẫn là box gốc; chỉ phép cắt mặt nạ đổi. Mặc
        # định 1.0 = đúng đường của ultralytics, nên bảng benchmark không đổi.
        self.crop_expand = float(crop_expand)
        if self.crop_expand != 1.0:
            if self.mask_head == "dyn":
                raise SystemExit("crop_expand không dùng với đầu A1: đầu đó đã cắt bằng cửa sổ riêng.")
            if not retina_masks:
                raise SystemExit("crop_expand chỉ có ở đường retina_masks=True (đường chấm của benchmark).")
            self._predict_kw["predictor"] = _crop_expand_predictor(self.crop_expand)
        # retina_masks: xuất mặt nạ ở độ phân giải ảnh thay vì lưới proto.
        # Với dự án lấy đường biên làm trọng tâm thì không có lý do tắt.
        self.kw = dict(
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            retina_masks=retina_masks,
            verbose=False,
        )
        if device is not None:
            self.kw["device"] = device

    def warmup(self) -> None:
        self.model.predict(
            np.zeros((self.kw["imgsz"], self.kw["imgsz"], 3), np.uint8), **self.kw, **self._predict_kw
        )

    def predict(
        self, image: np.ndarray, boxes: np.ndarray | None = None
    ) -> list[Prediction]:
        """Mỗi dự đoán về khung cục bộ, KHÔNG chạm vào `res.masks.xy`.

        Hai chặng từng nằm ở đây đều làm trên cả khung 1440x2560 cho TỪNG dự
        đoán, và mỗi chặng đắt hơn chính lượt truyền xuôi của mạng:

        - `res.masks.xy` là cached_property gọi `ops.masks2segments(self.data)`
          (ops.py:678). Hàm đó chuyển cả chồng NxHxW từ GPU về CPU LẦN THỨ HAI
          rồi chạy cv2.findContours trên khung đầy đủ cho từng mặt nạ: 2.89 ms
          mỗi dự đoán. Polygon nó trả về không chỗ nào trong đường chấm đọc
          tới — `matching.py` dùng `region.polygon` của NHÃN THẬT, không phải
          của dự đoán.
        - `astype(bool)` trên khung đầy đủ rồi mới trả về mặt nạ toàn khung.
          Mảng đó vứt đi hơn 99%, và vì `origin` là (0, 0) nên `full_frame`
          ở bước mã hoá RLE phải chép lại cả khung thay vì dán một cửa sổ.

        Đo trên đúng trọng số f6 và một ảnh thật của bộ này, 64 dự đoán,
        1440x2560 (ms mỗi dự đoán):

            masks.xy -> masks2segments        1.59   bỏ
            astype(bool) khung đầy đủ         1.38   bỏ
            cắt cửa sổ + astype nhỏ           2.01   thêm
            predictions_to_coco, toàn khung   4.27 -> 1.89 khi đã cắt
            ------------------------------------------------------
            cũ 7.23   mới 3.90   (1.9 lần)

        Con số đó còn là cận DƯỚI: phép đo chạy trên CPU nên hai lần chuyển
        cả chồng NxHxW từ GPU về host mà đường cũ phải trả đều bằng 0 ở đây.
        Trên máy thật, hồi quy 850 ảnh đã chấm cho `ms = -4.01 + 4.109 x
        số_dự_đoán` (R2 = 0.914) — hệ số chặn bằng 0 nghĩa là lượt truyền
        xuôi của mạng còn không hiện ra bên cạnh mấy chặng này.

        Phép quét để cắt (2.01 ms) không phải chi phí mới: `Prediction.
        bbox_xyxy` nhớ kết quả vào meta, và đường cũ vẫn phải quét đúng như
        thế ở `match_instances` — chỉ là quét trên khung đầy đủ, muộn hơn.

        RLE của hai đường trùng khớp trên 122 dự đoán thật của hai ảnh f6.
        """
        res = self.model.predict(image, **self.kw, **self._predict_kw)[0]
        if res.masks is None:
            return []
        h, w = image.shape[:2]
        scores = (
            res.boxes.conf.cpu().numpy()
            if res.boxes is not None
            else np.ones(len(res.masks))
        )
        out: list[Prediction] = []
        for i, m in enumerate(res.masks.data.cpu().numpy()):
            # retina_masks cho uint8 sẵn; nhánh này chỉ chạy khi tắt nó, và
            # khi đó mặt nạ còn ở lưới proto nên bản sao là nhỏ.
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            if m.shape != (h, w):
                import cv2

                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            pred = Prediction(
                mask=m,
                origin=(0, 0),
                score=float(scores[i]) if i < len(scores) else 1.0,
            ).cropped()
            # Sau cropped() mảng là cửa sổ sát tán, nên đổi kiểu ở đây rẻ.
            pred.mask = pred.mask.astype(bool)
            out.append(pred)
        return out

    @property
    def describe(self) -> dict:
        return {**super().describe, "weights": self.weights, "mask_head": self.mask_head,
                "crop_expand": self.crop_expand, **self.kw}


def expand_boxes(boxes, factor: float, shape: tuple[int, int]):
    """Nới box xyxy quanh tâm theo `factor`, kẹp trong ảnh (h, w)."""
    import torch

    c = (boxes[:, :2] + boxes[:, 2:4]) / 2
    half = (boxes[:, 2:4] - boxes[:, :2]) * (factor / 2)
    h, w = shape
    lo = torch.stack([(c[:, 0] - half[:, 0]).clamp(0, w), (c[:, 1] - half[:, 1]).clamp(0, h)], 1)
    hi = torch.stack([(c[:, 0] + half[:, 0]).clamp(0, w), (c[:, 1] + half[:, 1]).clamp(0, h)], 1)
    return torch.cat([lo, hi], 1)


def _crop_expand_predictor(factor: float):
    """SegmentationPredictor cắt mặt nạ bằng box nới `factor` lần."""
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment import SegmentationPredictor
    from ultralytics.utils import ops

    class CropExpandPredictor(SegmentationPredictor):
        def construct_result(self, pred, img, orig_img, img_path, proto):
            # Chép SegmentationPredictor.construct_result (8.4.143), nhánh
            # retina_masks; chỉ đổi box đưa vào phép cắt mặt nạ.
            if pred.shape[0] == 0:
                return super().construct_result(pred, img, orig_img, img_path, proto)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            crop = expand_boxes(pred[:, :4], factor, orig_img.shape[:2])
            masks = ops.process_mask_native(proto, pred[:, 6:], crop, orig_img.shape[:2])
            keep = masks.amax((-2, -1)) > 0
            if not all(keep):
                pred, masks = pred[keep], masks[keep]
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    CropExpandPredictor.__name__ = f"CropExpandPredictor_{factor:g}"
    return CropExpandPredictor
