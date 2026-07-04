# 02 — Sinh mask SEED bằng SAM zero-shot, xuất COCO để SỬA TAY.
# Chạy toàn bộ:   python scripts/02_gen_seed.py
# Chạy 1 field:   python scripts/02_gen_seed.py \
#                    --tiles_dir data/tiles/field_1 \
#                    --out       data/masks/field_1_seed.json
import argparse
import json
import os
import yaml
from src.data.tiling import load_manifest
from src.seed.sam_seed import build_mask_generator, generate_seed_auto

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--tiles_dir", default=None, help="ghi đè data.tiles_dir")
ap.add_argument("--out", default=None, help="đường dẫn file seed.json xuất ra")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
tiles_dir = args.tiles_dir or cfg["data"]["tiles_dir"]
tiles = load_manifest(f"{tiles_dir}/manifest.json")
mg = build_mask_generator(cfg["sam"])
coco = generate_seed_auto(tiles, mg, min_area=cfg["sam"]["min_seed_area"])
out = args.out or f"{cfg['data']['masks_dir']}/seed.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(coco, open(out, "w", encoding="utf-8"))
print(f"Seed COCO ({len(coco.get('annotations', []))} mask) -> {out}")
print("Mở trong GUI (python -m app.gradio_app) hoặc CVAT để sửa tay.")