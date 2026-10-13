"""Huấn luyện model segment hoa (U-Net nhẹ) trên nhãn hoa đã chốt.

    python scripts/flower_train.py --name r2 --epochs 40
    python scripts/flower_train.py --name lofo_field_3 --holdout-field field_3
    python scripts/flower_train.py --name r3_auto --include-auto

Nhãn đọc từ thư mục nhãn hoa (configs/flower_rounds.yaml, mặc định
data/masks/flower_v1), ảnh gốc theo configs/config.yaml. Mỗi tán đã chốt được
cắt theo khung (nới lề) ở độ phân giải gốc; mỗi bước lấy ngẫu nhiên các ô
`--tile` px trong các crop đó, lật/xoay ngẫu nhiên. Pixel 255 không tính loss.

Bộ chấm (vòng `role: test`, hoặc vòng vẽ tay kind=manual) KHÔNG BAO GIỜ vào
train, ảnh cách ảnh test ≤ buffer khung cũng không. Val tách ra từ chính các ảnh
train (--val-frac, theo cụm ảnh liền khung, xen kẽ theo ruộng): mỗi epoch chấm
IoU / F1 lớp hoa trên val và giữ checkpoint val IoU cao nhất. Bộ test chỉ được
chấm MỘT lần, lúc cuối, bằng checkpoint đó (chỉ ruộng --holdout-field nếu có).
--val-frac 0: không val, giữ checkpoint cuối.

Ghi runs/flower/<name>/: model.pt (tốt nhất), last.pt, config.json, log.csv,
metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import flower_data as fd  # noqa: E402
from app import flower_rounds as fr  # noqa: E402
from app import flower_store as fs  # noqa: E402

PAD = 32          # lề quanh khung tán khi cắt crop


def load_crops(items, images_dir, labels_dir, store, limit=None):
    """[(rgb crop, mask crop, ruộng)] cho mỗi tán; pixel ngoài các tán được
    chọn của ảnh là 255."""
    out = []
    for flat, its in fd.group_by_image(items).items():
        polys, (h, w) = fd.crown_polys(labels_dir, flat)
        rgb = fd.imread_rgb(Path(images_dir) / fd.rel_of(flat))
        if rgb is None or not polys:
            continue
        h, w = rgb.shape[:2]
        mask = fd.image_mask(store, flat, polys, [it["crown"] for it in its], (h, w))
        for it in its:
            x0, y0, x1, y1 = fs.crown_box(polys[it["crown"]], w, h)
            X0, Y0, X1, Y1 = max(0, x0 - PAD), max(0, y0 - PAD), min(w, x1 + PAD), min(h, y1 + PAD)
            out.append((rgb[Y0:Y1, X0:X1].copy(), mask[Y0:Y1, X0:X1].copy(), it["field"]))
            if limit and len(out) >= limit:
                return out
    return out


def sample_tile(rgb, mask, tile, rng):
    """Một ô tile x tile ngẫu nhiên trong crop (đệm phản chiếu ảnh, 255 cho
    mask nếu crop nhỏ hơn ô), lật/xoay ngẫu nhiên."""
    h, w = mask.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=fs.IGNORE)
        h, w = mask.shape
    y = rng.randint(0, h - tile)
    x = rng.randint(0, w - tile)
    r, m = rgb[y:y + tile, x:x + tile], mask[y:y + tile, x:x + tile]
    if rng.random() < 0.5:
        r, m = r[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5:
        r, m = r[::-1], m[::-1]
    k = rng.randint(0, 3)
    return np.rot90(r, k).copy(), np.rot90(m, k).copy()


def evaluate(model, test_items, images_dir, labels_dir, store, tile, device):
    """IoU / F1 lớp hoa trên bộ test, theo ruộng và tất cả."""
    from app import flower_model as fm
    pairs = {}
    for flat, its in fd.group_by_image(test_items).items():
        polys, _ = fd.crown_polys(labels_dir, flat)
        rgb = fd.imread_rgb(Path(images_dir) / fd.rel_of(flat))
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        ref = store.load_mask(flat, (h, w))
        for it in its:
            b = fs.crown_box(polys[it["crown"]], w, h)
            inside = fs.crown_mask(polys[it["crown"]], b)
            prob = fm.predict_tiles(model, rgb[b[1]:b[3], b[0]:b[2]], tile=tile, device=device)
            p = (prob >= 0.5).astype(np.uint8)[inside]
            g = ref[b[1]:b[3], b[0]:b[2]][inside]
            pairs.setdefault(it["field"], []).append((g, p))
    by = {f: dict(fr.agreement(v), crowns=len(v)) for f, v in sorted(pairs.items())}
    flat = [x for v in pairs.values() for x in v]
    return {"by_field": by, "all": dict(fr.agreement(flat), crowns=len(flat))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="tên lượt, ghi vào runs/flower/<name>/")
    ap.add_argument("--runs", default="runs/flower")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200, help="số batch mỗi epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", default="16,32,64,128", help="số kênh từng tầng U-Net")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="tỉ lệ ảnh train tách ra làm val (theo cụm ảnh liền khung); 0 = không val")
    ap.add_argument("--holdout-field", default=None, help="bỏ ruộng này khỏi train, chỉ chấm trên nó")
    ap.add_argument("--include-auto", action="store_true", help="học cả tán model tự nhận (chưa ai xem)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="chỉ nạp ngần ấy tán (thử nhanh)")
    a = ap.parse_args(argv)

    import torch
    from app import flower_model as fm
    images_dir, labels_dir, flower_dir, rounds = fd.default_dirs()
    store = fs.FlowerStore(flower_dir)
    train_items = fd.collect(store, rounds, labels_dir, "train", a.include_auto, a.holdout_field)
    test_items = fd.collect(store, rounds, labels_dir, "test", holdout=a.holdout_field)
    if not train_items:
        raise SystemExit("Chưa có tán nào đã chốt để học (cần trạng thái done/fixed/accepted).")
    buffer = max([int((r.get("pick") or {}).get("buffer_frames", 0)) for r in rounds] + [0])
    train_items, val_items = fd.split_val(train_items, a.val_frac, buffer, a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    crops = load_crops(train_items, images_dir, labels_dir, store, a.limit)
    out = Path(a.runs) / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a), device=device, n_train_crowns=len(train_items), n_crops=len(crops),
               n_val_crowns=len(val_items), n_test_crowns=len(test_items), flower_dir=str(flower_dir),
               n_train_images=len({it["image"] for it in train_items}),
               val_images=sorted({it["image"] for it in val_items}),
               train_fields=sorted({it["field"] for it in train_items}),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    n_img = lambda its: len({it["image"] for it in its})  # noqa: E731
    print("train %d tán / %d ảnh · val %d tán / %d ảnh · test %d tán / %d ảnh · %s" % (
        len(train_items), n_img(train_items), len(val_items), n_img(val_items),
        len(test_items), n_img(test_items), device))

    model = fm.TinyUNet(tuple(int(c) for c in a.ch.split(","))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * a.steps))
    rng = random.Random(a.seed)
    best, log = None, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(a.steps):
            xs, ys = [], []
            for _ in range(a.batch):
                rgb, m, _ = crops[rng.randrange(len(crops))]
                r, mm = sample_tile(rgb, m, a.tile, rng)
                xs.append(fm.to_tensor(r))
                ys.append(torch.from_numpy(mm.astype(np.int64)))
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)[:, None]
            loss = fm.masked_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        ev = evaluate(model, val_items, images_dir, labels_dir, store, a.tile, device) if val_items else None
        iou = (ev or {}).get("all", {}).get("iou")
        row = dict(epoch=ep, loss=float(np.mean(losses)), val_iou=iou,
                   val_f1=(ev or {}).get("all", {}).get("f1"), seconds=round(time.time() - t0, 1))
        log.append(row)
        print("epoch %3d  loss %.4f  val IoU %s  (%.0f s)" % (ep, row["loss"], "—" if iou is None else "%.3f" % iou, row["seconds"]))
        fm.save_model(out / "last.pt", model, dict(epoch=ep))
        score = ep if not val_items else (iou if iou is not None else float("-inf"))
        if best is None or score > best:
            best = score
            fm.save_model(out / "model.pt", model, dict(epoch=ep, val=ev))
        with open(out / "log.csv", "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0]))
            wr.writeheader()
            wr.writerows(log)
    # Test chấm đúng một lần, bằng checkpoint đã chọn theo val.
    model, ck = fm.load_model(out / "model.pt", device)
    test = evaluate(model, test_items, images_dir, labels_dir, store, a.tile, device) if test_items else None
    (out / "metrics.json").write_text(json.dumps(dict(epoch=ck.get("epoch"), val=ck.get("val"), test=test),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    t_all = (test or {}).get("all") or {}
    print("checkpoint epoch %s · test IoU %s · F1 %s" % (
        ck.get("epoch"), "—" if t_all.get("iou") is None else "%.3f" % t_all["iou"],
        "—" if t_all.get("f1") is None else "%.3f" % t_all["f1"]))
    return out


if __name__ == "__main__":
    main()
