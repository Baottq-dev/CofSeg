# 03 — Finetune SAM trên seed đã sửa tay (freeze image encoder).
# Chạy: python scripts/03_finetune_sam.py
import yaml
from src.data.dataset import SamPromptDataset
from src.models.sam_finetune import build_sam, train_sam, save_checkpoint

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
ds = SamPromptDataset(f"{cfg['data']['masks_dir']}/train.json",
                      cfg["data"]["tiles_dir"])
sam = build_sam(cfg["sam"])
sam = train_sam(sam, ds, cfg["sam"])
save_checkpoint(sam, "weights/sam_finetuned_p1.pt")
print("Đã lưu weights/sam_finetuned_p1.pt")