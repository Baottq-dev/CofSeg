"""Phần CPU của `--aug pa1` và `--aug ag4`: bảng tán, chỗ dán, dán sau, nhãn tán dán.

Chỉ numpy và cv2, không torch hay ultralytics. Mọi toạ độ là px của KHUNG NẠP:
ảnh gốc co cạnh dài về imgsz (1024 x 576 với imgsz 1024 và ảnh 2560 x 1440).
Phép dán chạy ngay sau khi ảnh được đọc, trước RandomPerspective, nên đây là
đúng khung mà ultralytics vừa nạp.

Định nghĩa giữ đúng như phép đo gây hại có đối chứng đã chạy trên R2 trước khi
train (dán sau làm R2 mất 7,5–14,4% tán đích chạm, so với 0–0,5% khi dán xa):

- Tán vẽ bằng fillPoly toạ độ x16, shift 4, LINE_8.
- Biên của một tán = điểm của tán mà phép co 3x3 xoá đi, bỏ 2 px sát khung.
- Tiếp xúc c của tán đích = tỉ lệ điểm biên của nó nằm trong TOUCH px (ô
  vuông) của phần nhìn thấy của tán dán.
- Dán sau: tán dán nằm dưới mọi tán có nhãn. Alpha = Gauss(sigma 1) của mặt nạ
  tán nguồn, bằng 0 trên tán có sẵn, nên mép giáp tán có sẵn giữ cứng.
- Chỗ chạm: thử tám hướng theo thứ tự ngẫu nhiên, trượt tâm tán nguồn từ xa
  vào tâm tán đích từng STEP px; vị trí hợp lệ khi tán dán nằm trọn trong khung,
  phần bị che <= hid và c >= c_lo; lấy vị trí có c gần c* nhất.

Ba chỗ thêm so với phép đo, vì giờ tán dán thành NHÃN để học:
- phần nhìn thấy phải liền một khối: chỉ giữ thành phần lớn nhất, alpha ngoài
  nó (quá 3 px) về 0, và từ chối nếu thành phần đó dưới (1 - hid) diện tích
  tán nguồn hoặc có lỗ thủng (một tán có sẵn nằm lọt trong tán dán);
- nhãn là đường viền ngoài của phần nhìn thấy (findContours, không xấp xỉ);
- ag4 không rút ngẫu nhiên 300 lần mà tính mọi vị trí hợp lệ bằng tương quan
  (cv2.filter2D) rồi rút đều trong số đó, nên ảnh còn chỗ thì luôn tìm được.

Tìm chỗ chạm tốn khoảng 50 ms mỗi lần thử, quá chậm cho dataloader. Nhưng nó
chỉ cần nhãn, không cần điểm ảnh, nên được làm MỘT lần lúc dựng dataset:
image_pool() liệt kê mọi chỗ dán hợp lệ của từng ảnh (kho), lúc train chỉ còn
choose() rút vài chỗ có khung cách nhau và apply_specs() dán, vài ms mỗi tán.
"""


from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

#: Các hằng đo bằng px ẢNH GỐC; đổi sang khung nạp bằng tỉ lệ r = cạnh khung / cạnh gốc.
MIN_DIAM = 150.0      # đường kính tương đương tối thiểu của tán đích và tán nguồn
TOUCH = 5.0           # "chạm" = trong 5 px gốc (2 px ở khung 1024)
STEP = 5.0            # bước trượt khi tìm chỗ chạm (2 px ở khung 1024)
#: Các hằng không đổi theo khung.
SIZE = (0.7, 1.4)     # đường kính tán nguồn / tán đích
DONOR_CONTACT = 0.2   # tán nguồn phải đứng riêng: tiếp xúc < 0,2, không bị khung cắt
SIGMA = 1.0           # viền mềm Gauss, px khung
TRIES = 3             # số tán nguồn thử cho mỗi tán đích
PAD = 4               # lề khi cắt tán nguồn, px khung
NEAREST = 20          # không tán nguồn nào trong SIZE thì lấy trong ngần này tán gần cỡ nhất
HOLE_PX = 4           # lỗ thủng lớn hơn ngần này px trong phần nhìn thấy thì từ chối
KEEP_DIL = 3          # alpha giữ trong ngần này px quanh thành phần được giữ
AG4_FRAG = 0.02       # ag4 không bị che; chỉ bỏ được mảnh rời của chính đa giác nguồn, tối đa 2%

K3 = np.ones((3, 3), np.uint8)


def px(orig: float, r: float) -> int:
    """Đổi một khoảng px ảnh gốc sang px khung nạp, tối thiểu 1."""
    return max(1, int(round(orig * r)))


# ------------------------------------------------------------------ bảng tán

@dataclass
class Crown:
    """Một tán có nhãn trên khung nạp. `m` là mặt nạ cục bộ, góc trên trái (x0, y0)."""

    x0: int
    y0: int
    m: np.ndarray
    area: int
    diam: float
    border: bool
    contact: float

    def full(self, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w), np.uint8)
        mh, mw = self.m.shape
        out[self.y0:self.y0 + mh, self.x0:self.x0 + mw] = self.m
        return out


def raster_local(poly: np.ndarray, h: int, w: int) -> tuple[np.ndarray, int, int]:
    """Tô một đa giác (K, 2) px khung vào cửa sổ quanh nó, cắt theo khung.

    Cùng điểm ảnh với tô cả khung rồi cắt: dịch toạ độ đúng bội của 16 (shift 4)
    là dịch nguyên px. Cửa sổ chừa 1 px mỗi phía để phép co tìm đúng biên.
    """
    p = np.round(np.asarray(poly, np.float64).reshape(-1, 2) * 16).astype(np.int32)
    x0 = min(max((int(p[:, 0].min()) >> 4) - 1, 0), w)
    y0 = min(max((int(p[:, 1].min()) >> 4) - 1, 0), h)
    x1 = min(max((int(p[:, 0].max()) >> 4) + 2, 0), w)
    y1 = min(max((int(p[:, 1].max()) >> 4) + 2, 0), h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.uint8), x0, y0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [p - np.array([x0 * 16, y0 * 16], np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m, x0, y0


def boundary_local(m: np.ndarray, x0: int, y0: int, h: int, w: int) -> np.ndarray:
    """Biên của mặt nạ cục bộ, bỏ điểm cách khung dưới 2 px (toạ độ khung)."""
    if not m.size:
        return np.zeros_like(m, bool)
    er = cv2.erode(m, K3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    b = (m > 0) & (er == 0)
    mh, mw = m.shape
    ys = np.arange(y0, y0 + mh)[:, None]
    xs = np.arange(x0, x0 + mw)[None, :]
    return b & (ys >= 2) & (ys < h - 2) & (xs >= 2) & (xs < w - 2)


def crown_table(segs: np.ndarray, h: int, w: int, r: float) -> tuple[list[Crown], np.ndarray]:
    """Mọi tán của một ảnh (segs: N đa giác px khung) và hợp của chúng (uint8, h x w).

    `contact` = tỉ lệ điểm biên của tán nằm trong TOUCH px của một tán KHÁC;
    `border` = tán chạm khung (trong 2 px). Cả hai tính trên khung nạp.
    """
    cnt = np.zeros((h, w), np.uint16)
    raw = []
    for seg in segs:
        m, x0, y0 = raster_local(seg, h, w)
        if m.size:
            cnt[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] += m
        raw.append((m, x0, y0))
    t = px(TOUCH, r)
    k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    out = []
    for m, x0, y0 in raw:
        area = int(m.sum())
        if not area:
            out.append(Crown(x0, y0, m, 0, 0.0, True, 0.0))
            continue
        ys, xs = np.nonzero(m)
        border = bool(x0 + xs.min() <= 1 or y0 + ys.min() <= 1
                      or x0 + xs.max() >= w - 2 or y0 + ys.max() >= h - 2)
        b = boundary_local(m, x0, y0, h, w)
        contact = 0.0
        if b.any():
            mh, mw = m.shape
            wy0, wx0 = max(y0 - t, 0), max(x0 - t, 0)
            wy1, wx1 = min(y0 + mh + t, h), min(x0 + mw + t, w)
            others = cnt[wy0:wy1, wx0:wx1].astype(np.int32)
            others[y0 - wy0:y0 - wy0 + mh, x0 - wx0:x0 - wx0 + mw] -= m
            near = cv2.dilate((others > 0).astype(np.uint8), k)
            near = near[y0 - wy0:y0 - wy0 + mh, x0 - wx0:x0 - wx0 + mw]
            contact = float((near[b] > 0).mean())
        out.append(Crown(x0, y0, m, area, math.sqrt(4 * area / math.pi), border, contact))
    return out, (cnt > 0).astype(np.uint8)


def is_target(c: Crown, r: float) -> bool:
    """Tán đích (pa1) hay tán tham chiếu cỡ (ag4): không bị khung cắt, đủ lớn."""
    return c.area > 0 and not c.border and c.diam >= MIN_DIAM * r


def is_donor(c: Crown, r: float) -> bool:
    """Tán nguồn: như tán đích, cộng thêm đứng riêng."""
    return is_target(c, r) and c.contact < DONOR_CONTACT


# ------------------------------------------------------------------ tán nguồn

def donor_box(c: Crown, h: int, w: int) -> tuple:
    """(y0, y1, x0, x1) của mảnh tán nguồn: khung bao của tán cộng PAD px, cắt theo khung."""
    ys, xs = np.nonzero(c.m)
    y0, y1 = max(c.y0 + int(ys.min()) - PAD, 0), min(c.y0 + int(ys.max()) + 1 + PAD, h)
    x0, x1 = max(c.x0 + int(xs.min()) - PAD, 0), min(c.x0 + int(xs.max()) + 1 + PAD, w)
    return y0, y1, x0, x1


def donor_mask(c: Crown, h: int, w: int) -> np.ndarray:
    """Mặt nạ của tán nguồn trong mảnh donor_box()."""
    y0, y1, x0, x1 = donor_box(c, h, w)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    mh, mw = c.m.shape
    # c.m có thể chìa ra ngoài mảnh một px lề rỗng; chỉ chép phần giao.
    sy0, sx0 = max(y0 - c.y0, 0), max(x0 - c.x0, 0)
    sy1, sx1 = min(y1 - c.y0, mh), min(x1 - c.x0, mw)
    m[c.y0 + sy0 - y0:c.y0 + sy1 - y0, c.x0 + sx0 - x0:c.x0 + sx1 - x0] = c.m[sy0:sy1, sx0:sx1]
    return m


def crop_donor(img: np.ndarray, c: Crown) -> tuple:
    """Mảnh ảnh và mặt nạ của tán nguồn, cắt theo khung bao của tán cộng PAD px."""
    h, w = img.shape[:2]
    y0, y1, x0, x1 = donor_box(c, h, w)
    return img[y0:y1, x0:x1].copy(), donor_mask(c, h, w)


def d4(a: np.ndarray, rot: int, flip: int) -> np.ndarray:
    a = np.rot90(a, rot, axes=(0, 1))
    return np.ascontiguousarray(a[:, ::-1] if flip else a)


def pick_donor(diams: np.ndarray, diam: float, rng: np.random.Generator, skip=()) -> int | None:
    """Chỉ số tán nguồn có đường kính trong SIZE lần `diam`; không có thì trong NEAREST tán gần cỡ nhất.
    Bỏ các chỉ số trong `skip` (tán đã thử); hết thì None."""
    ok = np.nonzero((diams >= SIZE[0] * diam) & (diams <= SIZE[1] * diam))[0]
    if not len(ok):
        ok = np.argsort(np.abs(diams - diam), kind="stable")[:NEAREST]
    ok = [int(i) for i in ok if int(i) not in skip]
    return int(ok[int(rng.integers(len(ok)))]) if ok else None


# ------------------------------------------------------------------ chỗ dán

def _touched(vis: np.ndarray, py: np.ndarray, px_: np.ndarray, t: int) -> int:
    """Số điểm (py, px_) (toạ độ cửa sổ, có thể âm tới -t) nằm trong t px (ô vuông) của `vis`.

    Bằng đúng `dilate(pad(vis, t), (2t+1)^2)[py + t, px_ + t].sum()`, nhưng chỉ
    giãn khung con quanh các điểm cần đọc; mỗi điểm cách mép khung con >= t px.
    """
    h, w = vis.shape
    qr0, qr1 = int(py.min()) - t, int(py.max()) + t + 1
    qc0, qc1 = int(px_.min()) - t, int(px_.max()) + t + 1
    r0, r1, c0, c1 = max(qr0, 0), min(qr1, h), max(qc0, 0), min(qc1, w)
    if r1 <= r0 or c1 <= c0:
        return 0
    sub = np.zeros((qr1 - qr0, qc1 - qc0), np.uint8)
    sub[r0 - qr0:r1 - qr0, c0 - qc0:c1 - qc0] = vis[r0:r1, c0:c1]
    sub = cv2.dilate(sub, np.ones((2 * t + 1, 2 * t + 1), np.uint8))
    return int(sub[py - qr0, px_ - qc0].sum())


def _rays(tc, cs, start: float, step: int, order, W: int, H: int, w: int, h: int):
    """Mọi vị trí góc trên trái của tán nguồn, theo thứ tự thử: từng hướng, từ xa vào."""
    out = []
    for dg in order:
        u = np.array([np.cos(np.radians(dg)), np.sin(np.radians(dg))])
        d = start
        while d >= 0:
            x0, y0 = np.round(tc + d * u - cs).astype(int)
            d -= step
            if 0 <= x0 and 0 <= y0 and x0 + w <= W and y0 + h <= H:
                out.append((int(x0), int(y0), int(dg)))
    return out


def _corr(a: np.ndarray, k: np.ndarray, x_lo: int, y_lo: int, x_hi: int, y_hi: int) -> np.ndarray:
    """sum(k * a[y:y+h, x:x+w]) cho mọi (x, y) trong [x_lo, x_hi] x [y_lo, y_hi], làm tròn về số nguyên."""
    h, w = k.shape
    roi = np.ascontiguousarray(a[y_lo:y_hi + h, x_lo:x_hi + w], np.float32)
    c = cv2.filter2D(roi, cv2.CV_32F, k.astype(np.float32), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    return np.rint(c[:y_hi - y_lo + 1, :x_hi - x_lo + 1]).astype(np.int64)


def place_contact(U: np.ndarray, tgt: Crown, sm: np.ndarray, rng: np.random.Generator, cstar: float,
                  *, r: float, c_min: float, hid_max: float):
    """Chỗ dán sau chạm tán đích. Trả về (c, hid, x0, y0, hướng) hoặc None.

    U: hợp mọi tán đang có trên ảnh (kể cả tán vừa dán). sm: mặt nạ tán nguồn
    sau D4. Không dừng khi gặp vị trí bị che quá nhiều: đi sâu hơn tán dán có
    thể lộ ra ở phía bên kia tán đích.

    Kết quả trùng từng vị trí với cách thử thẳng từng bước (test giữ điều đó).
    Phần bị che ở mọi vị trí tính một lần bằng tương quan của U với mặt nạ tán
    nguồn; tiếp xúc chỉ tính ở vị trí mà tán nguồn có điểm trong t px của biên
    tán đích (cũng tính bằng tương quan), vì ở chỗ khác c = 0 < c_min.
    """
    H, W = U.shape
    t, step = px(TOUCH, r), px(STEP, r)
    b = boundary_local(tgt.m, tgt.x0, tgt.y0, H, W)
    ty, tx = np.nonzero(b)
    nb = len(ty)
    if not nb:
        return None
    ty, tx = ty + tgt.y0, tx + tgt.x0
    ys, xs = np.nonzero(tgt.m)
    tc = np.array([xs.mean() + tgt.x0, ys.mean() + tgt.y0])
    tr = np.sqrt((xs + tgt.x0 - tc[0]) ** 2 + (ys + tgt.y0 - tc[1]) ** 2).max()
    h, w = sm.shape
    smb = sm > 0
    ys, xs = np.nonzero(smb)
    cs = np.array([xs.mean(), ys.mean()])
    area = len(ys)
    rs = np.sqrt((xs - cs[0]) ** 2 + (ys - cs[1]) ** 2).max()
    pos = _rays(tc, cs, tr + rs + 2 * t, step, rng.permutation(8) * 45, W, H, w, h)
    if not pos:
        return None
    xa = np.array([p[0] for p in pos])
    ya = np.array([p[1] for p in pos])
    x_lo, y_lo, x_hi, y_hi = int(xa.min()), int(ya.min()), int(xa.max()), int(ya.max())
    k = smb.astype(np.float32)
    overlap = _corr(U, k, x_lo, y_lo, x_hi, y_hi)[ya - y_lo, xa - x_lo]
    hid_all = 1 - (area - overlap) / area
    near = np.zeros((H, W), np.uint8)
    near[ty, tx] = 1
    near = cv2.dilate(near, np.ones((2 * t + 1, 2 * t + 1), np.uint8))
    reach = _corr(near, k, x_lo, y_lo, x_hi, y_hi)[ya - y_lo, xa - x_lo] > 0
    best = None
    for (x0, y0, dg), hid, can in zip(pos, hid_all, reach):
        if hid > hid_max or (c_min > 0 and not can):
            continue
        sel = (ty >= y0 - t) & (ty < y0 + h + t) & (tx >= x0 - t) & (tx < x0 + w + t)
        if not sel.any():
            continue
        vis = smb & (U[y0:y0 + h, x0:x0 + w] == 0)
        c = _touched(vis, ty[sel] - y0, tx[sel] - x0, t) / nb
        if c >= c_min and (best is None or abs(c - cstar) < abs(best[0] - cstar)):
            best = (float(c), float(hid), x0, y0, dg)
    return best


def far_state(U: np.ndarray, gap: int) -> tuple[np.ndarray, float]:
    """Vùng cấm của ag4 (cách U dưới `gap` px, float32 0/1) và bán kính đường tròn lớn nhất
    lọt trong phần còn lại. Tính lại sau mỗi lần dán, vì U đổi."""
    if not U.any():
        return np.zeros(U.shape, np.float32), float("inf")
    near = cv2.distanceTransform((U == 0).astype(np.uint8), cv2.DIST_L2, 5) < gap
    free = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 5)
    return near.astype(np.float32), float(free.max())


def inradius(sm: np.ndarray) -> float:
    """Bán kính đường tròn lớn nhất lọt trong mặt nạ (px)."""
    return float(cv2.distanceTransform(np.pad((sm > 0).astype(np.uint8), 1), cv2.DIST_L2, 5).max())


def place_far(state: tuple[np.ndarray, float], sm: np.ndarray, rng: np.random.Generator, rin: float):
    """Chỗ dán ag4: không điểm nào của tán nguồn rơi vào vùng cấm, tán trọn trong khung. (x0, y0) hoặc None.

    Vị trí hợp lệ là chỗ tương quan của mặt nạ tán nguồn với vùng cấm bằng 0;
    rút đều trong mọi vị trí hợp lệ. Bỏ qua phép tương quan khi vùng trống
    không chứa nổi đường tròn nội tiếp của tán nguồn (ảnh khép).
    """
    near, free_r = state
    H, W = near.shape
    h, w = sm.shape
    if h > H or w > W or free_r * 1.02 + 1 < rin:
        return None
    if near.any():
        ok = _corr(near, (sm > 0), 0, 0, W - w, H - h) == 0
    else:
        ok = np.ones((H - h + 1, W - w + 1), bool)
    ys, xs = np.nonzero(ok)
    if not len(ys):
        return None
    i = int(rng.integers(len(ys)))
    return int(xs[i]), int(ys[i])


# ------------------------------------------------------------------ dán và nhãn

def visible_part(U: np.ndarray, sm: np.ndarray, x0: int, y0: int):
    """Thành phần lớn nhất của phần nhìn thấy, và số px lỗ thủng lớn nhất trong nó."""
    h, w = sm.shape
    vis = ((sm > 0) & (U[y0:y0 + h, x0:x0 + w] == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(vis, connectivity=8)
    if n < 2:
        return np.zeros_like(vis), 0
    keep = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # Lỗ = vùng nền không nối được ra mép cửa sổ (đã đệm 1 px).
    bg = np.pad(1 - keep, 1, constant_values=1)
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    outer = blab[0, 0]
    holes = [bst[i, cv2.CC_STAT_AREA] for i in range(1, nb) if i != outer]
    return keep, int(max(holes, default=0))


def paste_behind(img: np.ndarray, U: np.ndarray, sp: np.ndarray, sm: np.ndarray, keep: np.ndarray,
                 x0: int, y0: int) -> None:
    """Dán (sp, sm) nằm dưới mọi tán của U, tại chỗ trên `img`. Viền mềm Gauss SIGMA, mép
    giáp tán có sẵn giữ cứng; alpha chỉ giữ trong KEEP_DIL px quanh `keep`."""
    h, w = sm.shape
    Uc = U[y0:y0 + h, x0:x0 + w]
    a = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), SIGMA) * (Uc == 0)
    a *= cv2.dilate(keep, np.ones((2 * KEEP_DIL + 1,) * 2, np.uint8)) > 0
    reg = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
    img[y0:y0 + h, x0:x0 + w] = np.clip(reg * (1 - a[..., None]) + sp.astype(np.float32) * a[..., None] + .5,
                                        0, 255).astype(np.uint8)


def outline(keep: np.ndarray, x0: int, y0: int) -> np.ndarray | None:
    """Đường viền ngoài của `keep`, (K, 2) float32 px khung; None nếu suy biến."""
    cs, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(c) < 3:
        return None
    return c + np.array([x0, y0], np.float32)


# ------------------------------------------------------------------ kho chỗ dán

@dataclass
class Donor:
    """Một tán nguồn của ảnh khác cùng ruộng, đã cắt sẵn (mảnh BGR, mặt nạ, đường kính px khung)."""

    patch: np.ndarray
    mask: np.ndarray
    diam: float
    src: str
    idx: int


@dataclass
class Spec:
    """Một chỗ dán hợp lệ trên một ảnh, tính từ nhãn của ảnh đó.

    donor = ("same", j): tán j của chính ảnh; ("bank", j): tán nguồn j của kho
    cùng ruộng. (x0, y0): góc trên trái của mảnh tán nguồn sau D4. box: khung
    bao (x0, y0, x1, y1) của phần nhìn thấy, px khung.
    """

    tgt: int
    donor: tuple
    rot: int
    flip: int
    x0: int
    y0: int
    c: float
    hid: float
    cstar: float
    vis_px: int
    box: tuple


#: Số vòng qua danh sách tán đích khi dựng kho: mỗi tán đích có tối đa ngần này chỗ dán.
ROUNDS = 2

#: Cột của nhật ký dựng kho, mỗi dòng một lần thử một tán nguồn cho một tán đích.
LOG_COLS = ("im", "mode", "round", "n_tgt", "tgt", "t_diam", "try", "donor", "d_diam", "rot", "flip",
            "cstar", "c", "hid", "vis_px", "src_px", "x0", "y0", "ok", "why")


def image_pool(segs: np.ndarray, h: int, w: int, r: float, mode: str, p: dict, rng: np.random.Generator,
               bank_masks: list, bank_diams, bank_src: list, name: str = ""):
    """Mọi chỗ dán hợp lệ của một ảnh, tính từ nhãn, không cần điểm ảnh.

    Mỗi vòng (ROUNDS) đi qua các tán đích theo thứ tự ngẫu nhiên; mỗi tán đích
    thử tối đa TRIES tán nguồn: tán cùng ảnh vừa cỡ trước, không có thì kho cùng
    ruộng (bỏ tán của chính ảnh này). Mọi chỗ dán tính trên hợp các tán GỐC của
    ảnh; lúc train chỉ dán cùng lúc những chỗ có khung cách nhau (choose()), nên
    chúng không che nhau và phép tính ở đây vẫn đúng.

    Trả về (danh sách Spec, các dòng nhật ký theo LOG_COLS).
    """
    crowns, U = crown_table(segs, h, w, r)
    targets = [i for i, c in enumerate(crowns) if is_target(c, r)]
    base = {"im": name, "mode": mode, "n_tgt": len(targets)}
    if not targets:
        return [], [{**base, "round": 0, "ok": 0, "why": "no_target"}]
    same = [i for i, c in enumerate(crowns) if is_donor(c, r)]
    keep_bank = [j for j, src in enumerate(bank_src) if src != name]
    bdiam = np.asarray(bank_diams, np.float64)[keep_bank] if keep_bank else np.zeros(0)
    gap = px(p["gap"], r) if mode == "ag4" else 0
    state = far_state(U, gap) if mode == "ag4" else None
    specs, rows = [], []
    for rnd in range(ROUNDS):
        for ti in rng.permutation(targets):
            ti = int(ti)
            tgt = crowns[ti]
            cstar = float(rng.uniform(p["c_lo"], p["c_hi"])) if mode == "pa1" else 0.0
            cand = [j for j in same if j != ti and SIZE[0] * tgt.diam <= crowns[j].diam <= SIZE[1] * tgt.diam]
            tried_s, tried_b = set(), set()
            for attempt in range(TRIES):
                row = {**base, "round": rnd, "tgt": ti, "t_diam": round(tgt.diam, 1), "try": attempt,
                       "cstar": round(cstar, 4)}
                left = [j for j in cand if j not in tried_s]
                q = None
                if not left and len(bdiam):
                    q = pick_donor(bdiam, tgt.diam, rng, tried_b)
                if left:
                    j = int(left[int(rng.integers(len(left)))])
                    tried_s.add(j)
                    donor, sm, dd = ("same", j), donor_mask(crowns[j], h, w), crowns[j].diam
                elif q is not None:
                    tried_b.add(q)
                    j = keep_bank[q]
                    donor, sm, dd = ("bank", j), bank_masks[j], float(bank_diams[j])
                else:
                    rows.append({**row, "ok": 0, "why": "no_donor"})
                    break
                rot, flip = int(rng.integers(4)), int(rng.integers(2))
                sm = d4(sm, rot, flip)
                area = int(sm.sum())
                row.update(donor=f"{donor[0]}:{donor[1]}", d_diam=round(dd, 1), rot=rot, flip=flip, src_px=area)
                if mode == "pa1":
                    pos = place_contact(U, tgt, sm, rng, cstar, r=r, c_min=p["c_lo"], hid_max=p["hid"])
                    if pos is None:
                        rows.append({**row, "ok": 0, "why": "no_contact"})
                        continue
                    c, _, x0, y0, _dg = pos
                    lim = p["hid"]
                else:
                    pos = place_far(state, sm, rng, inradius(sm))
                    if pos is None:
                        rows.append({**row, "ok": 0, "why": "no_room"})
                        continue
                    (x0, y0), c, lim = pos, 0.0, AG4_FRAG
                keep, hole = visible_part(U, sm, x0, y0)
                vis_px = int(keep.sum())
                hid = 1 - vis_px / area
                row.update(c=round(c, 4), hid=round(hid, 4), vis_px=vis_px, x0=x0, y0=y0)
                why = "fragment" if hid > lim + 1e-9 else "hole" if hole > HOLE_PX else ""
                if why:
                    rows.append({**row, "ok": 0, "why": why})
                    continue
                ys, xs = np.nonzero(keep)
                box = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
                specs.append(Spec(ti, donor, rot, flip, int(x0), int(y0), round(c, 4), round(hid, 4),
                                  round(cstar, 4), vis_px, box))
                rows.append({**row, "ok": 1, "why": ""})
                break
    return specs, rows


def choose(specs: list, k: int, rng: np.random.Generator, margin: int) -> list:
    """Tối đa k chỗ dán, thứ tự ngẫu nhiên, mỗi tán đích một lần, khung cách nhau ít nhất margin px."""
    out = []
    for i in rng.permutation(len(specs)):
        if len(out) >= k:
            break
        s = specs[int(i)]
        clash = any(s.tgt == o.tgt or not (s.box[2] + margin <= o.box[0] or o.box[2] + margin <= s.box[0]
                                           or s.box[3] + margin <= o.box[1] or o.box[3] + margin <= s.box[1])
                    for o in out)
        if not clash:
            out.append(s)
    return out


def apply_specs(img: np.ndarray, segs: np.ndarray, r: float, chosen: list, bank: list):
    """Dán các chỗ đã chọn vào một bản chép của `img`. Trả về (ảnh, đa giác (K, 2) px khung, chỗ đã dán).

    Phần nhìn thấy được tính lại trên chính ảnh này, nên nhãn là phần nhìn thấy
    thật kể cả khi đa giác đã lấy mẫu lại lệch vài điểm ảnh so với lúc dựng kho.
    Tán có sẵn giữ nguyên điểm ảnh và nhãn.
    """
    h, w = img.shape[:2]
    crowns, U = crown_table(segs, h, w, r)
    out = img.copy()
    polys, done = [], []
    for s in chosen:
        kind, j = s.donor
        if kind == "same":
            if j >= len(crowns) or not crowns[j].area:
                continue
            sp, sm = crop_donor(img, crowns[j])
        else:
            sp, sm = bank[j].patch, bank[j].mask
        sp, sm = d4(sp, s.rot, s.flip), d4(sm, s.rot, s.flip)
        mh, mw = sm.shape
        if s.x0 < 0 or s.y0 < 0 or s.x0 + mw > w or s.y0 + mh > h:
            continue
        keep, _ = visible_part(U, sm, s.x0, s.y0)
        poly = outline(keep, s.x0, s.y0)
        if poly is None:
            continue
        paste_behind(out, U, sp, sm, keep, s.x0, s.y0)
        U[s.y0:s.y0 + mh, s.x0:s.x0 + mw] |= keep
        polys.append(poly)
        done.append(s)
    return out, polys, done


def image_donors(img: np.ndarray, segs: np.ndarray, r: float, src: str, pick=None) -> list:
    """Tán nguồn của một ảnh, đã cắt sẵn. `pick`: chỉ lấy các chỉ số này (None = mọi tán nguồn)."""
    h, w = img.shape[:2]
    crowns, _ = crown_table(segs, h, w, r)
    out = []
    for i, c in enumerate(crowns):
        if (pick is None or i in pick) and is_donor(c, r):
            sp, sm = crop_donor(img, c)
            out.append(Donor(sp, sm, c.diam, src, i))
    return out


# ------------------------------------------------------------------ dựng kho cho cả tập train

#: Số tán nguồn tối đa mỗi ruộng trong kho tán nguồn (dùng khi ảnh không có tán cùng ảnh vừa cỡ).
BANK_PER_FIELD = 400
#: Muối của các bộ số ngẫu nhiên, để kho tán nguồn, kho chỗ dán và phép dán lúc train rút độc lập.
SALT_BANK, SALT_POOL = 4101, 4102


def field_of(name: str) -> str:
    """Ruộng của một ảnh, đọc từ tên file: `field_3__10__DJI_...jpg` -> `field_3`."""
    return name.split("__", 1)[0]


def frame_size(h0: int, w0: int, imgsz: int) -> tuple[int, int, float]:
    """(h, w, r) của khung nạp, đúng như BaseDataset.load_image (cạnh dài về imgsz)."""
    r = imgsz / max(h0, w0)
    if r == 1:
        return h0, w0, 1.0
    return min(math.ceil(h0 * r), imgsz), min(math.ceil(w0 * r), imgsz), r


def read_frame(path: str, h: int, w: int) -> np.ndarray:
    """Đọc ảnh như ultralytics (imdecode, BGR) rồi co về khung nạp bằng INTER_LINEAR."""
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    if im.shape[:2] != (h, w):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    return im


def build_bank(items: list, seed: int, threads: int = 8):
    """Kho tán nguồn theo ruộng. items: (tên ảnh, đường dẫn, segs px khung, h, w, r) của mọi ảnh train.

    Mỗi ruộng giữ tối đa BANK_PER_FIELD tán nguồn, rút đều bằng bộ số có seed.
    Trả về {ruộng: [Donor]} và các dòng mô tả từng tán nguồn.
    """
    from concurrent.futures import ThreadPoolExecutor

    cand: dict = {}
    for k, (name, _path, segs, h, w, r) in enumerate(items):
        crowns, _ = crown_table(segs, h, w, r)
        for j, c in enumerate(crowns):
            if is_donor(c, r):
                cand.setdefault(field_of(name), []).append((k, j, c))
    rng = np.random.default_rng([SALT_BANK, seed])
    chosen: dict = {}
    for field in sorted(cand):
        rows = cand[field]
        for q in sorted(rng.permutation(len(rows))[:BANK_PER_FIELD].tolist()):
            chosen.setdefault(rows[q][0], []).append(rows[q][1:] + (field,))

    def crop(k):
        name, path, _segs, h, w, _r = items[k]
        img = read_frame(path, h, w)
        return [(field, Donor(*crop_donor(img, c), c.diam, name, j), c) for j, c, field in chosen[k]]

    bank: dict = {}
    rows = []
    with ThreadPoolExecutor(max(1, threads)) as ex:
        for got in ex.map(crop, sorted(chosen)):
            for field, d, c in got:
                bank.setdefault(field, []).append(d)
                rows.append({"field": field, "j": len(bank[field]) - 1, "im": d.src, "crown": d.idx,
                             "diam": round(d.diam, 1), "area": c.area, "contact": round(c.contact, 4),
                             "h": d.mask.shape[0], "w": d.mask.shape[1]})
    return bank, rows


_W: dict = {}


def _pool_init(masks, diams, srcs, mode, p, seed):
    _W.update(masks=masks, diams=diams, srcs=srcs, mode=mode, p=p, seed=seed)


def _pool_one(task):
    i, name, segs, h, w, r = task
    f = field_of(name)
    rng = np.random.default_rng([SALT_POOL, _W["seed"], i])
    specs, rows = image_pool(segs, h, w, r, _W["mode"], _W["p"], rng, _W["masks"].get(f, []),
                             _W["diams"].get(f, []), _W["srcs"].get(f, []), name)
    return i, specs, rows


def build_pools(tasks: list, bank: dict, mode: str, p: dict, seed: int, workers: int = 0):
    """Kho chỗ dán của từng ảnh. tasks: (chỉ số ảnh, tên, segs px khung, h, w, r).

    Mỗi ảnh có bộ số riêng seed từ (seed, chỉ số ảnh), nên kết quả không phụ
    thuộc số tiến trình. workers <= 1: chạy ngay trong tiến trình này.
    Trả về ({chỉ số ảnh: [Spec]}, các dòng nhật ký theo LOG_COLS).
    """
    args = ({f: [d.mask for d in ds] for f, ds in bank.items()},
            {f: [d.diam for d in ds] for f, ds in bank.items()},
            {f: [d.src for d in ds] for f, ds in bank.items()}, mode, p, seed)
    if workers <= 1:
        _pool_init(*args)
        res = [_pool_one(t) for t in tasks]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # spawn chứ không fork: lúc dựng dataset tiến trình chính đã giữ CUDA.
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                 initializer=_pool_init, initargs=args) as ex:
            res = list(ex.map(_pool_one, tasks, chunksize=4))
    pools, rows = {}, []
    for i, specs, rr in res:
        pools[i] = specs
        rows += rr
    return pools, rows
