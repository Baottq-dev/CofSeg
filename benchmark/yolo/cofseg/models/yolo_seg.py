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
    ):
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(weights)
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
            np.zeros((self.kw["imgsz"], self.kw["imgsz"], 3), np.uint8), **self.kw
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
        res = self.model.predict(image, **self.kw)[0]
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
        return {**super().describe, "weights": self.weights, **self.kw}
