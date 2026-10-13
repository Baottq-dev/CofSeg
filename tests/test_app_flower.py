"""Thẻ Hoa của annotator: nét vẽ, mask, hoàn tác, trạng thái, Học & tô.

Ảnh tổng hợp nhỏ trong thư mục tạm: nền xanh, hai tán chồng nhau một phần,
đốm trắng làm "hoa" trong tán đầu. Nhãn tán ghi bằng chính /api/save của
server, nhãn hoa nằm ở <tmp>/flower_v1. Không chạm dữ liệu thật.
"""

import json
import os
import threading

import cv2
import numpy as np
import pytest

from app import flower_api as fa
from app import flower_store as fs
from app import server

W, H = 160, 120
NAME = "field_9/10/1/a.jpg"
# Tán 0 to; tán 1 nhỏ, chồng lên mép phải tán 0 (x 90..110).
C0 = [10.0, 10.0, 110.0, 10.0, 110.0, 110.0, 10.0, 110.0]
C1 = [90.0, 30.0, 150.0, 30.0, 150.0, 90.0, 90.0, 90.0]
DOTS = [(25 + 14 * i, 25 + 14 * j) for i in range(5) for j in range(6)]   # tâm đốm trong tán 0


def _image():
    rng = np.random.default_rng(0)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (60, 140, 60)                         # BGR: lá xanh
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    for x, y in DOTS:
        cv2.circle(img, (x, y), 3, (240, 240, 240), -1)
    return img


def _truth():
    m = np.zeros((H, W), np.uint8)
    for x, y in DOTS:
        cv2.circle(m, (x, y), 3, 1, -1)
    return m


@pytest.fixture
def root(tmp_path, monkeypatch):
    base = tmp_path / "images"
    p = base / NAME
    p.parent.mkdir(parents=True)
    cv2.imwrite(str(p), _image())
    monkeypatch.setattr(server, "BASE", str(base))
    monkeypatch.setattr(server, "ROOT", str(base))
    monkeypatch.setattr(server, "OUT_DIR", str(tmp_path / "labels"))
    cfg = tmp_path / "flower_rounds.yaml"
    # Đốm "hoa" tổng hợp chỉ ~29 px, phần máy tìm ra có khi dưới 20 px: tắt lọc đốm
    # nhỏ (mặc định 20 px, cỡ một bông thật) để các test học phân loại đo đúng việc học.
    cfg.write_text(
        "version: v1\ndata_dir: flower_v1\nrounds:\n"
        "  - {id: try, kind: scribble, pick: {mode: manual, fields: [field_9]},"
        " classifier: {n_trees: 30, max_depth: 10, sigma_min: 0.7, sigma_max: 4.0, num_sigma: 3, min_blob_px: 0}}\n"
        "  - {id: gold, kind: manual, pick: {mode: manual}}\n"
        "  - {id: other, kind: scribble, pick: {mode: manual, fields: [field_8]}}\n",
        encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", cfg)
    for cache in (fa._STORES, fa._BANK, fa._IMG):
        cache.clear()
    monkeypatch.setattr(fa, "_CACHE", None)
    server.save(server.SaveReq(name=NAME, width=W, height=H, polygons=[C0, C1]))
    return tmp_path


def _stroke(pts, label, crown=0, rnd="try", r=2):
    return fa.stroke(fa.StrokeReq(round=rnd, name=NAME, crown=crown, label=label, r=r, pts=pts))


def _mask(root):
    m = cv2.imread(str(root / "flower_v1" / "masks" / "field_9__10__1__a.png"), cv2.IMREAD_UNCHANGED)
    assert m is not None and m.shape == (H, W)
    return m


def test_a_stroke_is_cut_to_the_crown(root):
    # Nét đi từ trong tán 0 ra hẳn ngoài mọi tán (y = 115, dưới tán 0).
    res = _stroke([[50, 60], [50, 118]], 1, r=2)
    assert res["ok"] and res["n_ops"] == 1
    m = _mask(root)
    assert (m[60:110, 49:52] == 1).all()          # trong tán: hoa
    assert (m[112:, :] == 255).all()              # ngoài tán: không bao giờ có nhãn
    assert (m[:10, :] == 255).all()


def test_on_shared_area_the_latest_stroke_wins(root):
    _stroke([[100, 60]], 1, crown=0, r=4)         # vùng chung của hai tán
    assert _mask(root)[60, 100] == 1
    _stroke([[100, 60]], 0, crown=1, r=4)
    assert _mask(root)[60, 100] == 0
    _stroke([[100, 60]], 1, crown=0, r=4)
    assert _mask(root)[60, 100] == 1


def test_undo_drops_only_the_last_op_of_that_crown(root):
    _stroke([[30, 30], [60, 30]], 1, crown=0)
    _stroke([[120, 60], [140, 60]], 0, crown=1)
    before = _mask(root).copy()
    _stroke([[30, 80], [60, 80]], 1, crown=0)
    res = fa.undo(fa.CrownReq(round="try", name=NAME, crown=0))
    assert res["ok"] and res["n_ops"] == 1
    assert (_mask(root) == before).all()
    assert fa.undo(fa.CrownReq(round="try", name=NAME, crown=1))["n_ops"] == 0
    assert (_mask(root)[50:90, 112:150] == 255).all()   # tán 1 hết nét, không nền: 255


def test_human_beats_base_and_erase_falls_back_to_it(root):
    st = fa._store()
    flat = server._flat(NAME)
    base = np.full((H, W), 255, np.uint8)
    base[10:110, 10:110] = 1                      # máy nói cả tán 0 là hoa
    st.save_base(flat, base)
    _stroke([[40, 40], [80, 40]], 0, r=3)         # người sửa một dải thành không hoa
    m = _mask(root)
    assert m[40, 60] == 0 and m[70, 60] == 1
    _stroke([[60, 40]], "erase", r=3)             # tẩy giữa dải: về lại nền máy
    m = _mask(root)
    assert m[40, 60] == 1 and m[40, 45] == 0
    _stroke([[40, 70]], 255, r=3)                 # bỏ qua
    assert _mask(root)[70, 40] == 255


def test_status_csv_survives_concurrent_writes(root):
    st = fa._store()

    def work(t):
        for k in range(25):
            st.set_status("img_%d" % t, k, "try", state="doing", n_ops=str(k))
    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = st.read_status()
    assert len(rows) == 8 * 25
    assert all(r["state"] == "doing" for r in rows.values())


def test_one_image_belongs_to_one_round(root):
    assert _stroke([[30, 30]], 1)["ok"]
    res = _stroke([[30, 30]], 1, rnd="gold")
    assert res.status_code == 409


def test_a_round_only_takes_its_own_fields(root):
    res = _stroke([[30, 30]], 1, rnd="other")
    assert res.status_code == 400


def test_finishing_a_manual_crown_fills_the_rest_with_no_flower(root):
    _stroke([[30, 30]], 1, rnd="gold", r=2)
    res = fa.finish(fa.FinishReq(round="gold", name=NAME, crown=0, seconds=42))
    assert res["ok"] and res["state"] == "done"
    m = _mask(root)
    assert m[30, 30] == 1 and m[80, 80] == 0      # phần chưa tô thành không hoa
    row = fa._store().read_status()[(server._flat(NAME), 0, "gold")]
    assert row["state"] == "done" and float(row["seconds"]) == 42


def test_a_few_strokes_teach_the_classifier_the_white_dots(root):
    # Một chấm lên 6 đốm trắng (hoa) và ba dải lá xanh giữa các đốm (không hoa);
    # máy phải tô đúng 30 đốm của cả tán.
    for x, y in DOTS[:6]:
        _stroke([[x, y]], 1, r=1)
    for y in (32, 60, 88):
        _stroke([[15, y], [100, y]], 0, r=1)
    res = fa.classify(fa.CrownReq(round="try", name=NAME, crown=0))
    assert res["ok"], res
    m = _mask(root)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.min([np.hypot(xx - x, yy - y) for x, y in DOTS], axis=0)   # tới tâm đốm gần nhất
    inside = fs.crown_mask(C0, (0, 0, W, H))
    # Đốm bán kính 3 px: viền 1 px của nó đã là 25% diện tích, nên không chấm
    # bằng IoU (lệch viền là IoU rơi 0,1-0,2) mà kiểm hai thứ có nghĩa: lõi mọi
    # đốm được tô là hoa, và không pixel hoa nào nằm xa đốm quá 1,5 px.
    assert (m[inside & (d <= 2)] == 1).all()
    assert not (m[inside & (d > 4.5)] == 1).any()
    assert (root / "flower_v1" / "rounds" / "try.joblib").exists()


def test_flower_strokes_alone_are_enough(root):
    # Chỉ chấm Hoa lên 6 đốm, không nét Không hoa nào: pixel chưa tô trong tán làm
    # mẫu không hoa yếu, nên máy tìm ra cả 30 đốm chứ không tô kín cả tán. (Viền
    # đốm có thể hụt: chưa có nét Không hoa nào cho máy biết mép lá ở đâu.)
    for x, y in DOTS[:6]:
        _stroke([[x, y]], 1, r=1)
    res = fa.classify(fa.CrownReq(round="try", name=NAME, crown=0))
    assert res["ok"], res
    m = _mask(root)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.min([np.hypot(xx - x, yy - y) for x, y in DOTS], axis=0)
    inside = fs.crown_mask(C0, (0, 0, W, H))
    assert all(m[y, x] == 1 for x, y in DOTS)
    assert not (m[inside & (d > 4.5)] == 1).any()


def test_classify_on_a_crown_without_strokes_marks_it_in_progress(root):
    for x, y in DOTS[:6]:
        _stroke([[x, y]], 1, r=1)                 # nét chỉ ở tán 0
    res = fa.classify(fa.CrownReq(round="try", name=NAME, crown=1))
    assert res["ok"], res
    row = fa._store().read_status()[(server._flat(NAME), 1, "try")]
    assert row["state"] == "doing" and row["who"] == "A" and row["n_ops"] == "0"
    assert row["source"] == "classifier"


def test_learn_and_fill_drops_blobs_of_a_few_pixels(root, monkeypatch):
    from app import flower_classifier as fc
    assert fc.DEFAULTS["min_blob_px"] == fa.CLF_DEFAULTS["min_blob_px"] == 20
    m = np.zeros((40, 40), np.uint8)
    m[2:4, 2:4] = 1                               # 4 px: bỏ
    m[20:25, 20:25] = 1                           # 25 px: giữ
    m[0, 39] = 255                                # pixel bỏ qua giữ nguyên
    out = fc.drop_small(m, 10)
    assert (out[2:4, 2:4] == 0).all() and (out[20:25, 20:25] == 1).all() and out[0, 39] == 255
    assert (fc.drop_small(m, 0) == m).all()
    # qua API, ngưỡng của vòng (ở đây 10 px vì đốm tổng hợp nhỏ): không còn đốm
    # hoa nhỏ hơn trong phần máy tô
    cfg = fa.CFG_PATH.read_text(encoding="utf-8").replace("min_blob_px: 0}", "min_blob_px: 10}")
    (root / "blob10.yaml").write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", root / "blob10.yaml")
    monkeypatch.setattr(fa, "_CACHE", None)
    for x, y in DOTS[:6]:
        _stroke([[x, y]], 1, r=1)
    for y in (32, 60, 88):
        _stroke([[15, y], [100, y]], 0, r=1)
    assert fa.classify(fa.CrownReq(round="try", name=NAME, crown=0))["ok"]
    base = fa._store().load_base(server._flat(NAME), (H, W))
    n, lab, st, _ = cv2.connectedComponentsWithStats((base == 1).astype(np.uint8), connectivity=8)
    assert n > 1 and st[1:, cv2.CC_STAT_AREA].min() >= 10


def test_classify_needs_a_flower_stroke(root):
    _stroke([[15, 32], [100, 32]], 0)
    res = fa.classify(fa.CrownReq(round="try", name=NAME, crown=0))
    assert res.status_code == 400
    assert "Hoa" in json.loads(res.body)["error"]


def test_view_is_a_crop_with_its_position(root):
    _stroke([[30, 30], [60, 30]], 1)
    res = fa.view(name=NAME, crown=0, layer="mask")
    assert res.headers["X-Crop"] == "10,10,111,111"
    img = cv2.imdecode(np.frombuffer(res.body, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape == (101, 101, 4) and img[20, 30, 3] == 255


def test_flower_work_never_touches_the_crown_labels(root):
    jp = server._json_path(NAME)
    before = open(jp, "rb").read()
    _stroke([[30, 30]], 1)
    fa.undo(fa.CrownReq(round="try", name=NAME, crown=0))
    fa.finish(fa.FinishReq(round="try", name=NAME, crown=0))
    assert open(jp, "rb").read() == before
    crowns = fa.crowns(name=NAME, round="try")
    assert [c["crown"] for c in crowns["crowns"]] == [0, 1]
    assert crowns["crowns"][0]["state"] == "done"
    assert not os.path.exists(os.path.join(server.OUT_DIR, "flower_v1"))
