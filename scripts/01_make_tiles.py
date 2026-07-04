# 01 — Cắt ảnh drone gốc thành tile để annotate/train.
# Chạy toàn bộ:   python scripts/01_make_tiles.py
# Chạy 1 field:   python scripts/01_make_tiles.py \
#                    --images_dir data/iachim_dataset_export/data_compressed/field_1 \
#                    --tiles_dir  data/tiles/field_1
import argparse
import yaml
from src.data.tiling import tile_dir, save_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--images_dir", default=None, help="ghi đè data.images_dir (vd: 1 field)")
ap.add_argument("--tiles_dir", default=None, help="ghi đè data.tiles_dir")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
d = cfg["data"]
images_dir = args.images_dir or d["images_dir"]
tiles_dir = args.tiles_dir or d["tiles_dir"]
tiles = tile_dir(images_dir, tiles_dir, d["tile_size"], d["tile_overlap"])
save_manifest(tiles, f"{tiles_dir}/manifest.json")
print(f"Đã tạo {len(tiles)} tile từ {images_dir} -> {tiles_dir}")