# 04 — Inference bằng SAM đã finetune + đo IoU/Boundary-F1/PQ trên tập test.
# Chạy: python scripts/04_infer_eval.py
import yaml
from src.data.tiling import load_manifest
from src.inference.infer import build_predictor, run_inference
from src.eval.seg_metrics import evaluate_coco

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
tiles = load_manifest(f"{cfg['data']['tiles_dir']}/manifest.json")
predictor = build_predictor(cfg["sam"], "weights/sam_finetuned_p1.pt")
pred_coco = run_inference(predictor, tiles, cfg["infer"])
metrics = evaluate_coco(pred_coco, f"{cfg['data']['masks_dir']}/test.json")
print("Kết quả Phase 1:", metrics)