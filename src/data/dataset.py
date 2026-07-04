# Dataset nạp ảnh + prompt (box) + mask GT để finetune SAM.
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils


class SamPromptDataset(Dataset):
    # Trả về: image (H,W,3 uint8 RGB), boxes (N,4 xyxy), masks (N,H,W uint8).
    def __init__(self, coco_json, images_dir):
        self.images_dir = Path(images_dir)
        coco = json.load(open(coco_json))
        self.imgs = {im["id"]: im for im in coco["images"]}
        self.by_img = {}
        for ann in coco["annotations"]:
            self.by_img.setdefault(ann["image_id"], []).append(ann)
        self.ids = [i for i in self.imgs if self.by_img.get(i)]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _to_mask(ann, h, w):
        seg = ann["segmentation"]
        if isinstance(seg, list):                 # polygon
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg["counts"], list):     # RLE chưa nén
            rle = mask_utils.frPyObjects(seg, h, w)
        else:                                     # RLE đã nén
            rle = seg
        return mask_utils.decode(rle).astype(np.uint8)

    def __getitem__(self, idx):
        info = self.imgs[self.ids[idx]]
        img = cv2.cvtColor(
            cv2.imread(str(self.images_dir / info["file_name"])),
            cv2.COLOR_BGR2RGB)
        h, w = info["height"], info["width"]
        masks, boxes = [], []
        for ann in self.by_img[self.ids[idx]]:
            m = self._to_mask(ann, h, w)
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            masks.append(m)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        return dict(
            image=img,
            boxes=torch.as_tensor(np.array(boxes), dtype=torch.float32),
            masks=torch.as_tensor(np.array(masks), dtype=torch.uint8))