# P1 — Sinh mask SEED bằng SAM zero-shot; xuất COCO để sửa tay.
import numpy as np
import cv2
import torch
from pycocotools import mask as mu


def build_mask_generator(sam_cfg):
    # Automatic mask generator cho SAM2 (đổi sang SAM3 ở phase sau).
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    device = sam_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng -> chạy CPU (rất chậm). "
              "Cài torch bản CUDA nếu có GPU.", flush=True)
        device = "cpu"
    pps = int(sam_cfg.get("seed_points_per_side", 32))
    print(f"[seed] device={device} | points_per_side={pps} | "
          f"checkpoint={sam_cfg['checkpoint']}", flush=True)
    sam = build_sam2(sam_cfg["model_cfg"], sam_cfg["checkpoint"], device=device)
    return SAM2AutomaticMaskGenerator(sam, points_per_side=pps,
                                      pred_iou_thresh=0.8,
                                      stability_score_thresh=0.9)


def masks_to_coco(all_masks, tiles):
    images, annotations, ann_id = [], [], 1
    for img_id, (rec, masks) in enumerate(zip(tiles, all_masks), 1):
        images.append(dict(id=img_id, file_name=rec["tile"],
                           height=rec["h"], width=rec["w"]))
        for m in masks:
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            annotations.append(dict(
                id=ann_id, image_id=img_id, category_id=1,
                segmentation=rle, area=int(m.sum()),
                bbox=[int(xs.min()), int(ys.min()),
                      int(np.ptp(xs)), int(np.ptp(ys))], iscrowd=0))
            ann_id += 1
    return dict(images=images, annotations=annotations,
                categories=[dict(id=1, name="tree")])


def generate_seed_auto(tiles, mask_generator, min_area=500):
    n = len(tiles)
    print(f"[seed] Bắt đầu sinh mask cho {n} tile...", flush=True)
    all_masks = []
    for i, rec in enumerate(tiles, 1):
        img = cv2.cvtColor(cv2.imread(rec["path"]), cv2.COLOR_BGR2RGB)
        res = mask_generator.generate(img)
        kept = [r["segmentation"] for r in res if r["area"] > min_area]
        all_masks.append(kept)
        print(f"[seed] {i}/{n} {rec['tile']} -> {len(kept)} mask", flush=True)
    return masks_to_coco(all_masks, tiles)