"""Model mmdetection (SOLOv2 R50-FPN) qua cùng hợp đồng SegmentationModel.

`weights` là best.pth do trainer mmdet ghi; config đầy đủ của lần chạy nằm
cạnh nó (mmdet_config.py) nên không cần nhắc lại imgsz hay số lớp. Ảnh vào
là BGR như cv2 đọc — cùng quy ước với YOLO và detectron2 trong repo;
data_preprocessor của mmdet tự đổi sang RGB.

mmdet/mmcv chỉ được import BÊN TRONG hàm: repo ở nhà (Windows) không cài
mmcv, nhưng registry và test phần thuần Python vẫn phải chạy.

CHƯA CHẠY THẬT: mmcv không dựng được trên máy phát triển. Phải khói trên máy
Linux (benchmark/solov2/run.sh f4 --smoke) trước khi tin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import register
from .base import Prediction, SegmentationModel


def require_mmdet():
    try:
        import mmcv  # noqa: F401
        import mmdet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Cần mmcv + mmdet (Linux; xem requirements.txt và scripts/setup_env.py). "
            "Ở máy không có, chấm file predictions.json bằng model coco_predictions."
        ) from e


@register("model", "mmdet")
class MMDetModel(SegmentationModel):
    """Suy luận bằng mmdet.apis.inference_detector; mặt nạ đã ở độ phân giải gốc."""

    needs_prompt = False

    def __init__(self, weights: str, config: str | None = None, conf: float = 0.05,
                 max_det: int | None = 100, device: str | None = None):
        require_mmdet()
        import torch
        from mmdet.apis import init_detector

        self.weights = str(weights)
        self.conf, self.max_det = float(conf), max_det
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = Path(config) if config else Path(weights).with_name("mmdet_config.py")
        if not cfg.exists():
            raise FileNotFoundError(
                f"Không thấy {cfg}: đưa model.config, hoặc để mmdet_config.py cạnh best.pth "
                "như trainer mmdet ghi ra")
        self.config = str(cfg)
        # Ngưỡng và số vật thể tối đa của lần chấm, đè lên giá trị lúc train.
        test_cfg = dict(score_thr=self.conf)
        if max_det:
            test_cfg["max_per_img"] = int(max_det)
        self.model = init_detector(self.config, self.weights, device=self.device,
                                   cfg_options=dict(model=dict(test_cfg=test_cfg)))
        self.imgsz = self._imgsz()

    def _imgsz(self) -> int | None:
        try:
            for t in self.model.cfg.test_dataloader.dataset.pipeline:
                if t.get("type") == "Resize":
                    return int(max(t["scale"]))
        except Exception:
            pass
        return None

    def warmup(self) -> None:
        s = self.imgsz or 256
        self.predict(np.zeros((s, s, 3), np.uint8))

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        from mmdet.apis import inference_detector

        inst = inference_detector(self.model, image).pred_instances
        if inst is None or len(inst) == 0 or not hasattr(inst, "masks"):
            return []
        inst = inst.cpu()
        scores = inst.scores.numpy() if hasattr(inst, "scores") else np.ones(len(inst))
        masks = np.asarray(inst.masks).astype(bool)
        order = np.argsort(-scores)
        preds: list[Prediction] = []
        for i in order:
            if scores[i] < self.conf:
                continue
            if self.max_det and len(preds) >= int(self.max_det):
                break
            if not masks[i].any():
                continue
            preds.append(Prediction(mask=masks[i], origin=(0, 0), score=float(scores[i])).cropped())
        return preds

    @property
    def describe(self) -> dict:
        return {**super().describe, "arch": getattr(self.model.cfg.model, "type", "?"),
                "weights": self.weights, "config": self.config, "imgsz": self.imgsz,
                "conf": self.conf, "max_det": self.max_det, "device": self.device}
