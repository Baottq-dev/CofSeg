
import cv2
import torch
from src.seed.sam_seed import masks_to_coco


def build_predictor(sam_cfg, checkpoint):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam = build_sam2(sam_cfg["model_cfg"], sam_cfg["checkpoint"],
                     device=sam_cfg.get("device", "cuda"))
    sam.load_state_dict(torch.load(checkpoint, map_location="cpu"),
                        strict=False)
    sam.eval()
    return SAM2AutomaticMaskGenerator(sam)


def run_inference(predictor, tiles, infer_cfg):
    all_masks = []
    for rec in tiles:
        img = cv2.cvtColor(cv2.imread(rec["path"]), cv2.COLOR_BGR2RGB)
        res = predictor.generate(img)
        all_masks.append([r["segmentation"] for r in res
                          if r["area"] > infer_cfg["min_area"]])
    return masks_to_coco(all_masks, tiles)