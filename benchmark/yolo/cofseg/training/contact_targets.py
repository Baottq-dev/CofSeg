"""Nhãn phụ của TC2, dựng từ bản đồ chỉ số tán của batch.

Đầu `semseg` của Proto26 học nhãn HỢP mọi tán (`augment.py:2385-2416`,
`loss.py:539-555` của 8.4.143): hai tán chạm nhau là một khối liền, nên nó
không cho tín hiệu nào về chỗ hai tán gặp nhau — đúng chỗ R2 gộp tán, chia
tán thành mảnh hình hộp và cho mặt nạ thô ở vùng tán khép. TC2 thêm hai kênh
nhãn cho đầu đó:

- dải tiếp giáp: pixel có ít nhất HAI tán khác nhau trong vùng lân cận bán
  kính `radius` ô. Biên tán-đất không thuộc dải; khe hẹp giữa hai tán gần
  chạm thì có.
- khoảng cách tới biên: số bước giãn từ biên gần nhất của chính tán đó (biên
  giáp đất hoặc giáp tán khác), chia cho giá trị lớn nhất trong tán. Bằng 1 ở
  lõi, nhỏ ở mép; hai tán liền nhau thành hai đỉnh thay vì một khối phẳng.

Đầu vào là `batch["masks"]` khi `overlap_mask: true`: bản đồ (B, H, W), 0 là
nền, k là tán thứ k của ảnh, chỗ chồng thuộc về tán nhỏ hơn. Không cần sửa
dataset: mọi thứ dựng trên GPU, trong lúc tính loss, không có gradient.

Giới hạn đã biết: tán bị mép ảnh cắt nằm cạnh phần đệm xám của letterbox,
mà phần đệm cũng là nền (0), nên mép ảnh thành một biên. Mép ảnh trong ảnh
gốc thì không: đệm bằng giá trị mép khi tìm biên.

File này chỉ dùng torch để test chạy nhanh trên CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_map(index_map: torch.Tensor) -> torch.Tensor:
    if index_map.dim() != 3:
        raise ValueError(
            f"Cần bản đồ chỉ số tán (B, H, W) của overlap_mask, nhận được hình {tuple(index_map.shape)}"
        )
    return index_map.float().unsqueeze(1)


@torch.no_grad()
def contact_band(index_map: torch.Tensor, radius: int) -> torch.Tensor:
    """Dải tiếp giáp (B, H, W) kiểu bool.

    Trong cửa sổ (2r+1)² quanh mỗi pixel: chỉ số lớn nhất (tính cả nền 0) lớn
    hơn chỉ số tán nhỏ nhất (bỏ nền) đúng khi cửa sổ có ít nhất hai tán khác
    nhau. Không tán nào thì "nhỏ nhất" là +inf, nên điều kiện sai.
    """
    if radius < 1:
        raise ValueError(f"radius phải >= 1, nhận được {radius}")
    x = _as_map(index_map)
    k = 2 * radius + 1
    hi = F.max_pool2d(x, k, stride=1, padding=radius)
    big = torch.full_like(x, float("inf"))
    lo = -F.max_pool2d(-torch.where(x > 0, x, big), k, stride=1, padding=radius)
    return (hi > lo)[:, 0]


@torch.no_grad()
def boundary_distance(index_map: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """Khoảng cách tới biên, chuẩn hoá trong từng tán, (B, H, W) trong [0, 1].

    Biên của tán là pixel của nó có láng giềng 4 hướng mang chỉ số khác (nền
    hoặc tán khác) và có giá trị 1. Pixel giãn tới ở bước s (cửa sổ 3x3, chỉ
    trong tiền cảnh) có giá trị s + 1; quá `max_iter` bước thì giữ max_iter + 1.
    Chia cho giá trị lớn nhất của chính tán đó; ngoài tán là 0.
    """
    x = _as_map(index_map)
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    c = p[..., 1:-1, 1:-1]
    diff = ((p[..., :-2, 1:-1] != c) | (p[..., 2:, 1:-1] != c)
            | (p[..., 1:-1, :-2] != c) | (p[..., 1:-1, 2:] != c))
    fg = x > 0
    reached = diff & fg
    d = reached.float()
    for step in range(1, max_iter + 1):
        grown = F.max_pool2d(reached.float(), 3, stride=1, padding=1) > 0
        new = grown & fg & ~reached
        if not bool(new.any()):
            break
        d = d + new.float() * (step + 1)
        reached = reached | new
    d = torch.where(fg & ~reached, torch.full_like(d, float(max_iter + 1)), d)

    # Lớn nhất theo từng (ảnh, tán): khoá phẳng b * (K + 1) + chỉ số tán.
    ids = x.long()
    k1 = int(ids.max().item()) + 1
    b = torch.arange(ids.shape[0], device=ids.device).view(-1, 1, 1, 1)
    key = (b * k1 + ids).flatten()
    dmax = torch.zeros(ids.shape[0] * k1, device=d.device, dtype=d.dtype)
    dmax = dmax.scatter_reduce(0, key, d.flatten(), reduce="amax", include_self=True)
    out = d.flatten() / dmax[key].clamp(min=1.0)
    return (out.view_as(d) * fg.float())[:, 0]


@torch.no_grad()
def to_semseg_grid(band: torch.Tensor, dist: torch.Tensor, fg: torch.Tensor,
                   size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thu ba bản đồ (B, H, W) về lưới của `semseg`, trả về ba tensor (B, 1, h, w).

    - dải: lấy max trong ô, vì lấy mẫu gần nhất làm rơi mất dải mỏng 1-2 ô;
    - khoảng cách: lấy trung bình trong ô;
    - tiền cảnh: lấy mẫu gần nhất, đúng như nhãn ngữ nghĩa gốc (BCEDiceLoss
      thu nhãn bằng `F.interpolate(..., mode="nearest")`), để vùng tính L1 của
      kênh khoảng cách trùng với vùng tiền cảnh mà kênh gốc học.
    """
    size = tuple(int(s) for s in size)
    b = band.float().unsqueeze(1)
    d = dist.float().unsqueeze(1)
    f = fg.float().unsqueeze(1)
    if tuple(b.shape[-2:]) == size:
        return b, d, f
    return (F.adaptive_max_pool2d(b, size), F.adaptive_avg_pool2d(d, size),
            F.interpolate(f, size, mode="nearest"))
