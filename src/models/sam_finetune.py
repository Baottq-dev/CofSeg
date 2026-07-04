# P1 — Finetune SAM: freeze image encoder, train prompt encoder + mask decoder.
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def build_sam(sam_cfg):
    from sam2.build_sam import build_sam2
    return build_sam2(sam_cfg["model_cfg"], sam_cfg["checkpoint"],
                      device=sam_cfg.get("device", "cuda"))


def freeze_encoder(sam):
    for p in sam.image_encoder.parameters():
        p.requires_grad = False


def dice_loss(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    num = 2 * (pred * target).sum((-1, -2))
    den = pred.sum((-1, -2)) + target.sum((-1, -2))
    return (1 - (num + eps) / (den + eps)).mean()


def train_sam(sam, dataset, cfg):
    if cfg.get("freeze_image_encoder", True):
        freeze_encoder(sam)
    params = [p for p in sam.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0])
    dev = cfg.get("device", "cuda")
    sam.train()
    for epoch in range(cfg["epochs"]):
        tot = 0.0
        for sample in loader:
            opt.zero_grad()
            img_t = torch.as_tensor(sample["image"]).permute(2, 0, 1)
            emb = sam.image_encoder(img_t[None].float().to(dev) / 255.0)
            loss = 0.0
            for box, gt in zip(sample["boxes"], sample["masks"]):
                sparse, dense = sam.prompt_encoder(
                    points=None, boxes=box[None].to(dev), masks=None)
                low, _ = sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False)
                pred = F.interpolate(low, gt.shape[-2:], mode="bilinear")
                g = gt[None, None].float().to(dev)
                loss = loss + dice_loss(pred, g) + \
                    F.binary_cross_entropy_with_logits(pred, g)
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"epoch {epoch}: loss={tot:.3f}")
    return sam


def save_checkpoint(sam, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sam.state_dict(), path)