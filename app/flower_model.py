# app/flower_model.py — Model segment hoa nhỏ và các hàm dùng chung cho
# scripts/flower_train.py, scripts/flower_predict.py.
#
# Bài toán là ngữ nghĩa trong tán: mỗi pixel hoa / không hoa, 255 = bỏ qua (mờ,
# không chắc, ngoài tán đã gán). Cụm hoa chỉ 7-10 px ở GSD 0,55 cm/px nên model
# chạy trên crop tán ở ĐỘ PHÂN GIẢI GỐC, không co ảnh. U-Net bốn tầng, thuần
# PyTorch (không thêm thư viện), ~0,5 triệu tham số với kênh mặc định.
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


class TinyUNet(nn.Module):
    """U-Net nhẹ: ra một logit mỗi pixel (hoa). Cạnh ảnh vào phải chia hết cho
    2^(số tầng - 1); predict_tiles lo việc đệm."""

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
def predict_tiles(model, rgb, tile=512, overlap=64, device="cpu", tta=False):
    """Xác suất hoa (H, W) float32 cho cả crop, cắt ô có chồng mép rồi lấy
    trung bình. tta=True: thêm lật ngang và lật dọc, trả (trung bình, độ lệch
    tuyệt đối trung bình giữa các phép lật)."""
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
            patch = F.pad(patch, (0, Pw - pw, 0, Ph - ph), mode="reflect" if min(ph, pw) > 1 else "constant")
            batch = [patch]
            if tta:
                batch += [patch.flip(-1), patch.flip(-2)]
            out = torch.sigmoid(model(torch.stack(batch).to(device))).cpu()[:, 0]
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
    torch.save(dict(state=model.state_dict(), ch=list(model.ch), **(extra or {})), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TinyUNet(tuple(ck.get("ch", (16, 32, 64, 128))))
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck
