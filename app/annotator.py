# Backend cho giao diện sửa seed:
#  - Bọc SAM ở chế độ interactive (click điểm / vẽ box) để đề xuất mask từng cây.
#  - Quản lý danh sách mask đã chấp nhận và xuất ra COCO (để finetune ở bước 03).
# Dùng chung checkpoint với inference Phase 1 (nếu đã finetune thì gợi ý tốt hơn).
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


class SamAnnotator:
    def __init__(self, sam_cfg, checkpoint=None, variant=None, model_cfg=None,
                 device=None):
        # variant: 'sam2' (SAM 2.1) hoặc 'sam3' (SAM 3 / 3.1). checkpoint &
        # model_cfg cho phép chọn model động từ giao diện.
        self.variant = (variant or sam_cfg.get("variant", "sam2")).lower()
        model_cfg = model_cfg or sam_cfg["model_cfg"]
        ckpt = checkpoint or sam_cfg["checkpoint"]
        device = device or sam_cfg.get("device", "cuda")
        if self.variant.startswith("sam3"):
            self._build_sam3(model_cfg, ckpt, device)
        else:
            self._build_sam2(model_cfg, ckpt, device, sam_cfg)
        self.image = None
        self.masks = []  # danh sách mask bool đã chấp nhận

    def _build_sam2(self, model_cfg, checkpoint, device, sam_cfg=None):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # Nếu là checkpoint đã finetune (state_dict) thì build từ base rồi nạp đè.
        base = checkpoint
        is_ft = "finetuned" in os.path.basename(checkpoint).lower()
        if is_ft and sam_cfg:
            base = sam_cfg.get("checkpoint", checkpoint)
        sam = build_sam2(model_cfg, base, device=device)
        if base != checkpoint and os.path.exists(checkpoint):
            import torch
            state = torch.load(checkpoint, map_location="cpu")
            sam.load_state_dict(state, strict=False)
        self.predictor = SAM2ImagePredictor(sam)

    def _build_sam3(self, model_cfg, checkpoint, device):
        # SAM 3 / 3.1 (facebookresearch/sam3). Giữ interface set_image/predict
        # tương thích SAM2 ở chế độ PVS (point/box). Chỉnh nếu API bản cài khác.
        try:
            from sam3.build_sam import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor
            sam = build_sam3(model_cfg, checkpoint, device=device)
            self.predictor = SAM3ImagePredictor(sam)
        except Exception:
            # Một số bản đóng gói builder trả thẳng predictor.
            from sam3.build_sam import build_sam3_image_predictor
            self.predictor = build_sam3_image_predictor(checkpoint, device=device)

    def set_image(self, img_rgb):
        # Nạp 1 tile và encode 1 lần; các click sau dùng lại embedding nên nhanh.
        self.image = img_rgb
        self.predictor.set_image(img_rgb)
        self.masks = []

    def predict_point(self, x, y, positive=True):
        # Click 1 điểm -> SAM trả nhiều mask, chọn mask có score cao nhất.
        pts = np.array([[x, y]])
        lbl = np.array([1 if positive else 0])
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=lbl, multimask_output=True
        )
        return masks[int(np.argmax(scores))].astype(bool)

    def predict_box(self, box):
        # Vẽ box (x1,y1,x2,y2) -> 1 mask.
        masks, _, _ = self.predictor.predict(
            box=np.array(box), multimask_output=False
        )
        return masks[0].astype(bool)

    def add_mask(self, mask):
        self.masks.append(mask)

    def remove_last(self):
        if self.masks:
            self.masks.pop()

    def clear(self):
        self.masks = []

    def overlay(self, pending=None):
        # Vẽ tất cả mask đã chấp nhận (màu ngẫu nhiên) + mask đang chờ (xanh lá).
        vis = self.image.copy()
        rng = np.random.default_rng(0)
        for m in self.masks:
            color = rng.integers(60, 255, size=3)
            vis[m] = (0.5 * vis[m] + 0.5 * color).astype(np.uint8)
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)
        if pending is not None:
            vis[pending] = (
                0.4 * vis[pending] + 0.6 * np.array([0, 255, 0])
            ).astype(np.uint8)
        return vis

    def export_coco(self, tile_name, height, width, out_json):
        # Xuất COCO (compressed RLE) - đúng định dạng bước 03 finetune đọc.
        images = [dict(id=1, file_name=tile_name, height=height, width=width)]
        anns = []
        for i, m in enumerate(self.masks, 1):
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            anns.append(
                dict(
                    id=i,
                    image_id=1,
                    category_id=1,
                    segmentation=rle,
                    area=int(m.sum()),
                    bbox=[
                        int(xs.min()),
                        int(ys.min()),
                        int(np.ptp(xs)),
                        int(np.ptp(ys)),
                    ],
                    iscrowd=0,
                )
            )
        coco = dict(
            images=images,
            annotations=anns,
            categories=[dict(id=1, name="tree")],
        )
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(coco, f)
        return len(anns)