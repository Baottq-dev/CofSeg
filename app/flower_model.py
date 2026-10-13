# app/flower_model.py — Model segment hoa nhỏ và các hàm dùng chung cho
# scripts/flower_train.py, scripts/flower_predict.py.
#
# Bài toán là ngữ nghĩa trong tán: mỗi pixel hoa / không hoa, 255 = bỏ qua (mờ,
# không chắc, ngoài tán đã gán). Cụm hoa chỉ 7-10 px ở GSD 0,55 cm/px nên model
# chạy trên crop tán ở ĐỘ PHÂN GIẢI GỐC, không co ảnh. Hai kiến trúc (--arch):
#   unet-r34  U-Net của segmentation_models_pytorch, encoder ResNet34 pretrain
#             ImageNet (24,4 triệu tham số). Nhánh nối tắt mang chi tiết độ phân
#             giải cao về đầu ra, nên giữ được cụm hoa nhỏ; mặc định.
#   tinyunet  U-Net bốn tầng thuần PyTorch học từ đầu (~0,5 triệu tham số), làm mốc.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


ARCHS = ("unet-r34", "tinyunet")


class TinyUNet(nn.Module):
    """U-Net nhẹ: ra một logit mỗi pixel (hoa). Cạnh ảnh vào phải chia hết cho
    2^(số tầng - 1); predict_tiles lo việc đệm."""
    arch = "tinyunet"

    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.ch = tuple(ch)
        self.down = nn.ModuleList()
        cin = 3
        for c in ch:
            self.down.append(Block(cin, c))
            cin = c
        self.up = nn.ModuleList()
        self.merge = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.merge.append(Block(2 * c, c))
            cin = c
        self.head = nn.Conv2d(cin, 1, 1)

    @property
    def stride(self):
        return 2 ** (len(self.ch) - 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, merge in zip(self.up, self.merge):
            x = up(x)
            x = merge(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def build_model(arch="unet-r34", ch=(16, 32, 64, 128), pretrained=True):
    """Model theo tên. unet-r34 tải trọng số ImageNet ở lần đầu (pretrained=True);
    nạp lại từ checkpoint thì không cần (pretrained=False)."""
    if arch == "tinyunet":
        return TinyUNet(tuple(ch))
    if arch == "unet-r34":
        import segmentation_models_pytorch as smp
        m = smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)
        m.arch, m.stride = arch, 32          # 5 lần giảm mẫu: cạnh ảnh vào chia hết cho 32
        return m
    raise ValueError("arch phải là một trong %s" % ", ".join(ARCHS))


def autocast(device, on=True):
    """bf16 trên GPU (RTX 40xx có), không làm gì trên CPU."""
    dev = str(device)
    return torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                          enabled=bool(on) and dev.startswith("cuda"))


def masked_loss(logits, target):
    """BCE + Dice trên đúng các pixel có nhãn (target != 255). Không pixel nào
    có nhãn thì trả 0 có grad, để vòng lặp không vỡ."""
    t = target.float()
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    lg, y = logits[valid], t[valid]
    bce = F.binary_cross_entropy_with_logits(lg, y)
    p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


def to_tensor(rgb):
    """RGB uint8 (H, W, 3) -> tensor chuẩn hoá (3, H, W)."""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


@torch.no_grad()
def predict_tiles(model, rgb, tile=512, overlap=64, device="cpu", tta=False, amp=True):
    """Xác suất hoa (H, W) float32 cho cả crop, cắt ô có chồng mép rồi lấy
    trung bình. tta=True: thêm lật ngang và lật dọc, trả (trung bình, độ lệch
    tuyệt đối trung bình giữa các phép lật). amp: bf16 khi chạy GPU."""
    model.eval()
    h, w = rgb.shape[:2]
    s = model.stride
    step = max(s, tile - overlap)
    acc = np.zeros((h, w), np.float32)
    acc2 = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    x_all = to_tensor(rgb)
    ys = list(range(0, max(1, h - tile) + 1, step)) or [0]
    xs = list(range(0, max(1, w - tile) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    for y0 in ys:
        for x0 in xs:
            patch = x_all[:, y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[1:]
            Ph, Pw = -(-ph // s) * s, -(-pw // s) * s
            # Phản chiếu cần phần đệm nhỏ hơn cạnh ô; tán rất nhỏ (mảnh tán sát mép
            # ảnh) đệm lên 32 px thì lặp pixel biên.
            mode = "reflect" if Ph - ph < ph and Pw - pw < pw else "replicate"
            patch = F.pad(patch[None], (0, Pw - pw, 0, Ph - ph), mode=mode)[0]
            batch = [patch]
            if tta:
                batch += [patch.flip(-1), patch.flip(-2)]
            with autocast(device, amp):
                logits = model(torch.stack(batch).to(device))
            out = torch.sigmoid(logits.float()).cpu()[:, 0]
            if tta:
                out = torch.stack([out[0], out[1].flip(-1), out[2].flip(-2)])
            out = out[:, :ph, :pw].numpy()
            m = out.mean(0)
            acc[y0:y0 + ph, x0:x0 + pw] += m
            acc2[y0:y0 + ph, x0:x0 + pw] += np.abs(out - m).mean(0)
            cnt[y0:y0 + ph, x0:x0 + pw] += 1
    prob = acc / np.maximum(cnt, 1)
    return (prob, acc2 / np.maximum(cnt, 1)) if tta else prob


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def save_model(path, model, extra=None):
    torch.save(dict(state=model.state_dict(), arch=model.arch, ch=list(getattr(model, "ch", ())),
                    **(extra or {})), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck.get("arch", "tinyunet"), tuple(ck.get("ch") or (16, 32, 64, 128)), pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck
