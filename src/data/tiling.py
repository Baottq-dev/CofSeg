# Cắt ảnh drone thành tile vuông có overlap. Quét ĐỆ QUY thư mục.
# Tile mép được "dồn sát biên" (snap) để không tạo tile vụn toàn phần đệm.
from pathlib import Path
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window


def _positions(length, tile, step):
    # Toạ độ bắt đầu; luôn phủ hết ảnh, tile cuối dồn sát biên.
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def tile_image(path, out_dir, tile_size=1024, overlap=0.2,
               plot_id=None, name_prefix=None):
    # Cắt 1 ảnh -> list metadata tile (để ghép mask về toạ độ gốc).
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(tile_size * (1 - overlap)))
    stem = name_prefix or Path(path).stem
    metas = []
    with rasterio.open(path) as src:
        W, H = src.width, src.height
        for y in _positions(H, tile_size, step):
            for x in _positions(W, tile_size, step):
                w = min(tile_size, W - x)
                h = min(tile_size, H - y)
                arr = src.read(window=Window(x, y, w, h))  # (C, h, w)
                img = np.transpose(arr, (1, 2, 0))[..., :3]
                if h < tile_size or w < tile_size:  # chỉ pad khi ảnh < tile
                    pad = ((0, tile_size - h), (0, tile_size - w), (0, 0))
                    img = np.pad(img, pad, mode="reflect")
                name = f"{stem}_{x}_{y}.png"
                cv2.imwrite(str(out_dir / name), img[..., ::-1])  # RGB->BGR
                metas.append(dict(tile=name, path=str(out_dir / name),
                                  x=int(x), y=int(y), w=int(w), h=int(h),
                                  src=str(path), plot_id=plot_id))
    return metas


def tile_dir(images_dir, out_dir, tile_size=1024, overlap=0.2,
             exts=(".tif", ".tiff", ".jpg", ".jpeg", ".png")):
    # Quét đệ quy mọi thư mục con dưới images_dir.
    root = Path(images_dir)
    metas = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        # plot_id = thư mục cấp cao nhất (vd field_1); ảnh nằm ngay gốc -> "root".
        plot_id = rel.parts[0] if len(rel.parts) > 1 else "root"
        # tiền tố tên tile theo toàn bộ đường dẫn tương đối -> luôn duy nhất.
        prefix = "__".join(rel.with_suffix("").parts)
        metas += tile_image(str(p), out_dir, tile_size, overlap,
                            plot_id=plot_id, name_prefix=prefix)
    return metas


def save_manifest(metas, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metas, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def load_manifest(path):
    return json.load(open(path, encoding="utf-8"))