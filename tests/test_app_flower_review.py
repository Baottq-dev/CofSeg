"""Vòng duyệt (Chấp nhận / Sửa / chốt vòng) và xuất bộ nhãn hoa.

Bộ dữ liệu tổng hợp trong thư mục tạm (field_1 hai đường bay, field_2 một),
mỗi ảnh ba tán. Không chạm dữ liệu thật.
"""

import csv
import json

import cv2
import numpy as np
import pytest

from app import flower_api as fa
from app import flower_rounds as fr
from app import server

W, H = 96, 72
CA = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
CB = [50.0, 15.0, 85.0, 15.0, 85.0, 55.0, 50.0, 55.0]
CC = [0.0, 50.0, 20.0, 50.0, 20.0, 71.0, 0.0, 71.0]
FLIGHTS = {"field_1/10/1": 10, "field_1/10/2": 6, "field_2/10/1": 10}
NAMES = ["%s/DJI_20260301075%03d_%04d_D.jpg" % (f, i, i) for f, n in FLIGHTS.items() for i in range(1, n + 1)]


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


def _fake_predictions(root, entropy):
    """Dự đoán giả cho mọi tán của r2: nửa trái mỗi tán là hoa (254), nửa phải
    không hoa (0); entropy theo hàm `entropy(item)`."""
    run = root / "run_r2"
    (run / "pred").mkdir(parents=True, exist_ok=True)
    st = fa._store()
    items = st.load_round("r2")["items"]
    rows = []
    for flat in {it["image"] for it in items}:
        p = np.full((H, W), 255, np.uint8)
        for it in (x for x in items if x["image"] == flat):
            x0, y0, x1, y1 = it["bbox"]
            mid = (x0 + x1) // 2
            p[y0:y1, x0:mid] = 254
            p[y0:y1, mid:x1] = 0
            rows.append(dict(image=flat, crown=it["crown"], field=it["field"], cover=0.5,
                             entropy=entropy(it), tta_disagree=0.0))
        cv2.imwrite(str(run / "pred" / (flat.rsplit(".", 1)[0] + ".png")), p)
    with open(run / "crowns.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return items


# ------------------------------------------------------------- thứ tự duyệt
def test_review_queue_puts_the_least_sure_of_each_field_first():
    items = [dict(image="a%d" % i, crown=0, field=f) for f in ("f1", "f2") for i in range(10)]
    unc = {("a%d" % i, 0): i / 10 for i in range(10)}
    q = fr.review_queue(items, unc, fix_top=0.3, check_random=0.2, seed=1)
    fixes = [x for x in q if x[1] == "uncertain"]
    checks = [x for x in q if x[1] == "random"]
    assert len(fixes) == 6 and len(checks) == 4            # 30% và 20% của mỗi ruộng
    assert [x[0]["field"] for x in fixes[:2]] == ["f1", "f2"]   # xen kẽ ruộng
    assert {x[0]["image"] for x in fixes} == {"a9", "a8", "a7"}  # kém chắc nhất
    assert q.index(fixes[-1]) < q.index(checks[0])
    assert q == fr.review_queue(items, unc, fix_top=0.3, check_random=0.2, seed=1)


# ---------------------------------------------------------------- vòng duyệt
def test_review_accept_fix_and_close(root):
    assert fa.round_build(fa.RoundReq(round="r2"))["ok"]
    items = _fake_predictions(root, lambda it: 0.9 if it["crown"] == 0 else 0.1)
    q = fa.queue(round="r2", who="A")
    assert q["kind"] == "review" and q["items"][0]["reason"] == "uncertain"
    a, b = q["items"][0], q["items"][1]
    st = fa._store()
    # chưa bấm Sửa thì không vẽ được
    res = fa.stroke(fa.StrokeReq(round="r2", name=a["name"], crown=a["crown"], label=0, pts=[[20, 20]]))
    assert res.status_code == 400
    # Chấp nhận: mask = dự đoán (nửa trái là hoa)
    r = fa.review(fa.ReviewReq(round="r2", name=a["name"], crown=a["crown"], action="accept"))
    assert r["ok"] and r["state"] == "accepted" and r["cover"] == pytest.approx(0.5, abs=0.05)
    # Sửa rồi Xong: fixed
    fa.review(fa.ReviewReq(round="r2", name=b["name"], crown=b["crown"], action="reject"))
    assert fa.stroke(fa.StrokeReq(round="r2", name=b["name"], crown=b["crown"], label=0, r=30,
                                  pts=[[25, 25], [70, 35]]))["ok"]
    assert fa.finish(fa.FinishReq(round="r2", name=b["name"], crown=b["crown"]))["state"] == "fixed"
    q = fa.queue(round="r2", who="A")
    assert q["accepted"] == 1 and q["fixed"] == 1 and q["reject_rate"] == 0.5
    # Chốt vòng: phần còn lại thành auto, nguồn là lượt model
    c = fa.round_close(fa.RoundReq(round="r2"))
    assert c["ok"] and c["auto"] == len(items) - 2 and c["missing_prediction"] == 0
    rows = st.read_status()
    auto = [r for (img, k, rnd), r in rows.items() if rnd == "r2" and r["state"] == "auto"]
    assert len(auto) == len(items) - 2 and all(r["source"] == "model:run_r2" for r in auto)


# --------------------------------------------------------------- xuất bộ hoa
def test_export_keeps_gold_and_its_neighbours_out_of_train(root):
    st = fa._store()
    gold = st.load_round("gold")["items"]
    g = gold[0]
    # chốt một tán vàng
    fa.stroke(fa.StrokeReq(round="gold", name=g["name"], crown=g["crown"], label=1, pts=[[25, 25]]))
    fa.finish(fa.FinishReq(round="gold", name=g["name"], crown=g["crown"]))
    # một tán ở ảnh liền kề ảnh vàng (cùng đường bay, khung +1) và một tán ở xa,
    # cả hai đã chốt ở vòng chọn tay
    flight = g["name"].rsplit("/", 1)[0]
    seq = fr.frame_no(g["name"])
    near = next(n for n in NAMES if n.startswith(flight + "/") and abs(fr.frame_no(n) - seq) == 1)
    far = next(n for n in NAMES if n.startswith(flight + "/") and abs(fr.frame_no(n) - seq) > 3)
    for n in (near, far):
        fa.stroke(fa.StrokeReq(round="try", name=n, crown=0, label=1, pts=[[25, 25]]))
        fa.finish(fa.FinishReq(round="try", name=n, crown=0))
    res = fa.export(fa.ExportReq(name="ds"))
    assert res["ok"], res
    out = root / "export" / "ds"
    train = {p.name for p in (out / "images" / "train").iterdir()}
    assert server._flat(far) in train and server._flat(near) not in train and g["image"] not in train
    assert {p.name for p in (out / "images" / "gold").iterdir()} == {g["image"]}
    m = cv2.imread(str(out / "masks" / "train" / (server._flat(far).rsplit(".", 1)[0] + ".png")), cv2.IMREAD_UNCHANGED)
    assert m[25, 25] == 1 and m[60, 70] == 255            # tán B không chọn: 255
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"] == {"train": {"images": 1, "crowns": 1}, "gold": {"images": 1, "crowns": 1}}
