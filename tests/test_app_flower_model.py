"""Model segment hoa và hai script train / predict, chạy trọn một vòng nhỏ.

Bộ dữ liệu tổng hợp trong thư mục tạm (field_1 hai đường bay, field_2 một),
mỗi ảnh ba tán. Model chỉ chạy vài bước trên CPU với ô 32 px để kiểm đường
đi của dữ liệu, không phải huấn luyện thật. Không chạm dữ liệu thật.
"""

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app import flower_api as fa
from app import flower_data as fd
from app import flower_model as fm
from app import server

W, H = 96, 72
CA = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
CB = [50.0, 15.0, 85.0, 15.0, 85.0, 55.0, 50.0, 55.0]
CC = [0.0, 50.0, 20.0, 50.0, 20.0, 71.0, 0.0, 71.0]
FLIGHTS = {"field_1/10/1": 10, "field_1/10/2": 6, "field_2/10/1": 10}
NAMES = ["%s/DJI_20260301075%03d_%04d_D.jpg" % (f, i, i) for f, n in FLIGHTS.items() for i in range(1, n + 1)]
ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def root(tmp_path, monkeypatch):
    base = tmp_path / "images"
    rng = np.random.default_rng(2)
    for n in NAMES:
        p = base / n
        p.parent.mkdir(parents=True, exist_ok=True)
        img = rng.integers(0, 120, (H, W, 3), dtype=np.uint8)
        cv2.circle(img, (25, 25), 4, (240, 240, 240), -1)       # một "cụm hoa" trong tán A
        cv2.imwrite(str(p), img)
    monkeypatch.setattr(server, "BASE", str(base))
    monkeypatch.setattr(server, "ROOT", str(base))
    monkeypatch.setattr(server, "OUT_DIR", str(tmp_path / "labels"))
    monkeypatch.setattr(server, "EXPORT_DIR", str(tmp_path / "export"))
    run = tmp_path / "run_r2"
    cfg = tmp_path / "flower_rounds.yaml"
    cfg.write_text(
        "version: v1\ndata_dir: flower_v1\nrounds:\n"
        "  - {id: gold, kind: manual, pick: {crowns_per_field: 2, crowns_per_image: 1, skip_border: true, seed: 7}}\n"
        "  - {id: try, kind: scribble, pick: {mode: manual}}\n"
        "  - {id: r2, kind: review, model_run: '%s', pick: {crowns_per_field: 10, crowns_per_image: all,"
        " skip_border: true, buffer_frames: 1, seed: 13}, review: {fix_top: 0.3, check_random: 0.2}}\n"
        % run.as_posix(), encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", cfg)
    for cache in (fa._STORES, fa._BANK, fa._IMG, fa._SEG):
        cache.clear()
    monkeypatch.setattr(fa, "_CACHE", None)
    for n in NAMES:
        server.save(server.SaveReq(name=n, width=W, height=H, polygons=[CA, CB, CC]))
    fa.round_build(fa.RoundReq(round="gold"))
    return tmp_path


# ----------------------------------------------------------------------- val
def _items(names):
    return [dict(image=server._flat(n), crown=k, field=n.split("/", 1)[0]) for n in names for k in (0, 1)]


def test_val_is_cut_from_train_by_image_clusters():
    # field_1/10/1 khung 1..10, field_1/10/2 khung 1..6, field_2/10/1 khung 1..10;
    # lấy khung 1, 2, 6, 10 mỗi đường bay: (1, 2) dính nhau, 6 và 10 đứng riêng.
    pick = [n for n in NAMES if int(n[-10:-6]) in (1, 2, 6, 10)]
    items = _items(pick)
    train, val = fd.split_val(items, 0.3, buffer=3, seed=0)
    assert train and val and len(train) + len(val) == len(items)
    vimg, timg = {it["image"] for it in val}, {it["image"] for it in train}
    assert not vimg & timg
    for v in vimg:                                   # không ảnh val nào sát ảnh train cùng đường bay
        for t in timg:
            if fd.flight_of(v) == fd.flight_of(t):
                assert abs(fd.fr.frame_no(v) - fd.fr.frame_no(t)) > 3
    assert {it["field"] for it in val} == {"field_1", "field_2"}        # xen kẽ theo ruộng
    assert fd.split_val(items, 0.3, buffer=3, seed=0) == (train, val)   # cùng seed, cùng kết quả
    assert fd.split_val(items, 0.0) == (items, [])
    assert fd.split_val(_items(NAMES[:1]), 0.3) == (_items(NAMES[:1]), [])


# --------------------------------------------------------------------- model
def test_tiny_unet_and_the_masked_loss():
    m = fm.TinyUNet((4, 8, 16))
    x = torch.zeros(2, 3, 32, 32)
    assert m(x).shape == (2, 1, 32, 32)
    y = torch.full((2, 1, 32, 32), 255)
    assert float(fm.masked_loss(m(x), y)) == 0.0            # không pixel nào có nhãn
    y[:, :, :4, :4] = 1
    loss = fm.masked_loss(m(x), y)
    loss.backward()
    assert float(loss) > 0
    prob, dis = fm.predict_tiles(m, np.zeros((45, 70, 3), np.uint8), tile=32, overlap=8, tta=True)
    assert prob.shape == (45, 70) and dis.shape == (45, 70) and 0 <= prob.min() <= prob.max() <= 1


def test_train_and_predict_scripts_end_to_end(root, monkeypatch):
    st = fa._store()
    gold = st.load_round("gold")["items"][0]
    fa.stroke(fa.StrokeReq(round="gold", name=gold["name"], crown=gold["crown"], label=1, pts=[[25, 25]]))
    fa.finish(fa.FinishReq(round="gold", name=gold["name"], crown=gold["crown"]))
    far = [n for n in NAMES if n.startswith("field_2/")][-1]
    fa.stroke(fa.StrokeReq(round="try", name=far, crown=0, label=1, r=4, pts=[[25, 25]]))
    fa.stroke(fa.StrokeReq(round="try", name=far, crown=0, label=0, r=3, pts=[[12, 38], [38, 38]]))
    fa.finish(fa.FinishReq(round="try", name=far, crown=0))
    dirs = (root / "images", root / "labels", root / "flower_v1", fa.load_config()[0]["rounds"])
    monkeypatch.setattr(fd, "default_dirs", lambda repo=None: dirs)
    train = _script("flower_train")
    out = train.main(["--name", "t", "--runs", str(root / "runs"), "--epochs", "1", "--steps", "2",
                      "--batch", "2", "--tile", "32", "--ch", "4,8", "--device", "cpu"])
    for f in ("model.pt", "last.pt", "config.json", "log.csv", "metrics.json"):
        assert (out / f).exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["n_train_crowns"] == 1 and cfg["n_test_crowns"] == 1   # vàng không vào train
    assert cfg["n_val_crowns"] == 0                                    # một ảnh train: không tách val
    met = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert met["val"] is None and met["test"]["all"]["crowns"] == 1   # test chấm một lần lúc cuối
    assert fa.round_build(fa.RoundReq(round="r2"))["ok"]
    rows = _script("flower_predict").main(["--run", str(out), "--round", "r2", "--tile", "32", "--device", "cpu"])
    n_items = len(st.load_round("r2")["items"])
    assert len(rows) == n_items
    with open(out / "crowns.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == n_items
    p = cv2.imread(str(next((out / "pred").iterdir())), cv2.IMREAD_UNCHANGED)
    assert p.shape == (H, W) and p[2, 2] == 255             # ngoài các tán được tô: 255
