# app/flower_classifier.py — Bộ phân loại pixel hoa / không hoa cho vòng nét vẽ.
#
# Kiểu ilastik: mỗi pixel mang một vector đặc trưng màu, cạnh, kết cấu ở nhiều
# thang (skimage.feature.multiscale_basic_features trên RGB), random forest học
# từ đúng những pixel người đã vẽ nét lên, rồi đoán cho mọi pixel còn lại của
# tán. Nét "Không hoa" vẽ lên thứ dễ nhầm của từng ruộng (lá vàng dưới nắng,
# chói, đất sáng) là cách dạy máy quy luật đó một lần cho cả vòng.
import time
from collections import OrderedDict

import numpy as np
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier

DEFAULTS = dict(n_trees=100, max_depth=12, sigma_min=0.7, sigma_max=5.0,
                num_sigma=4, max_samples=50000)


def features(rgb, sigma_min=0.7, sigma_max=5.0, num_sigma=4, **_):
    """(H, W, F) float32. RGB 0..255 vào; F = 3 kênh x (cường độ + cạnh +
    2 trị riêng kết cấu) x num_sigma — 48 với mặc định."""
    img = np.asarray(rgb, np.float32) / 255.0
    f = multiscale_basic_features(img, intensity=True, edges=True, texture=True,
                                  sigma_min=sigma_min, sigma_max=sigma_max,
                                  num_sigma=num_sigma, channel_axis=-1)
    return f.astype(np.float32, copy=False)


class FeatureCache:
    """Đặc trưng theo tán, giữ trong RAM tới `max_bytes` (bỏ cái cũ nhất trước).
    Một tán 450x450 px cỡ 39 MB."""

    def __init__(self, max_bytes=1 << 30):
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.size = 0

    def get(self, key, make):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        f = make()
        self.items[key] = f
        self.size += f.nbytes
        while self.size > self.max_bytes and len(self.items) > 1:
            _, old = self.items.popitem(last=False)
            self.size -= old.nbytes
        return f


def balanced_sample(X, y, max_samples, seed=0):
    """Tối đa max_samples/2 pixel mỗi lớp, chọn ngẫu nhiên có hạt giống."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        if len(idx) > max_samples // 2:
            idx = rng.choice(idx, max_samples // 2, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def fit(X, y, n_trees=100, max_depth=12, max_samples=50000, seed=0, **_):
    """Học random forest. Thiếu một trong hai lớp thì báo lỗi rõ ràng: học từ
    một lớp thì máy đoán đúng lớp đó cho mọi pixel."""
    y = np.asarray(y)
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("Cần nét của cả Hoa và Không hoa trong vòng này rồi mới học được.")
    X, y = balanced_sample(np.asarray(X, np.float32), y, max_samples, seed)
    t0 = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                 class_weight="balanced", n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    clf.fit_seconds_ = time.perf_counter() - t0
    clf.n_samples_ = int(len(y))
    return clf


def predict(clf, feats, inside=None):
    """Mask 0/1 (uint8) cho crop. Chỉ đoán pixel trong tán nếu có `inside`."""
    h, w, f = feats.shape
    out = np.zeros((h, w), np.uint8)
    flat = feats.reshape(-1, f)
    sel = np.ones(h * w, bool) if inside is None else inside.reshape(-1)
    if sel.any():
        out.reshape(-1)[sel] = (clf.predict_proba(flat[sel])[:, 1] >= 0.5).astype(np.uint8)
    return out
