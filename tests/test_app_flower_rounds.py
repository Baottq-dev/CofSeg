"""Dựng vòng gán nhãn hoa, hàng đợi, gán đôi, cọ bám siêu điểm ảnh, đồng thuận.

Bộ dữ liệu tổng hợp trong thư mục tạm: field_1 có hai đường bay (10/1 tám
khung, 10/2 bốn khung, cùng số khung 0001…), field_2 một đường bay sáu khung.
Mỗi ảnh ba tán, tán thứ ba chạm mép ảnh. Không chạm dữ liệu thật.
"""

import cv2
import numpy as np
import pytest

from app import flower_api as fa
from app import flower_data as fd
from app import flower_rounds as fr
from app import server

W, H = 96, 72
CA = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
CB = [50.0, 15.0, 85.0, 15.0, 85.0, 55.0, 50.0, 55.0]
CC = [0.0, 50.0, 20.0, 50.0, 20.0, 71.0, 0.0, 71.0]          # chạm mép trái và dưới
FLIGHTS = {"field_1/10/1": 8, "field_1/10/2": 4, "field_2/10/1": 6}


def _name(flight, i):
    return "%s/DJI_20260301075%03d_%04d_D.jpg" % (flight, i, i)


NAMES = [_name(f, i) for f, n in FLIGHTS.items() for i in range(1, n + 1)]


@pytest.fixture
def root(tmp_path, monkeypatch):
    base = tmp_path / "images"
    rng = np.random.default_rng(1)
    for n in NAMES:
        p = base / n
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    monkeypatch.setattr(server, "BASE", str(base))
    monkeypatch.setattr(server, "ROOT", str(base))
    monkeypatch.setattr(server, "OUT_DIR", str(tmp_path / "labels"))
    cfg = tmp_path / "flower_rounds.yaml"
    cfg.write_text(
        "version: v1\ndata_dir: flower_v1\nrounds:\n"
        "  - {id: gold, kind: manual, pick: {crowns_per_field: 5, crowns_per_image: 3, skip_border: true, seed: 7},"
        " superpixel: {area_px: 20, compactness: 10}, double: {annotator: B, crowns: 2}}\n"
        "  - {id: r1, kind: scribble, pick: {crowns_per_field: 4, crowns_per_image: 2, skip_border: true,"
        " buffer_frames: 3, seed: 11}, classifier: {n_trees: 10, max_depth: 6, sigma_max: 2.0, num_sigma: 2}}\n"
        "  - {id: try, kind: scribble, pick: {mode: manual}}\n",
        encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", cfg)
    for cache in (fa._STORES, fa._BANK, fa._IMG, fa._SEG):
        cache.clear()
    monkeypatch.setattr(fa, "_CACHE", None)
    for n in NAMES:
        server.save(server.SaveReq(name=n, width=W, height=H, polygons=[CA, CB, CC]))
    return tmp_path


def _images():
    return fa._all_images()


# --------------------------------------------------------------- luật chọn tán
def test_build_is_repeatable_with_the_same_seed(root):
    rnd = fa._round("gold")
    a = fr.build(rnd, _images(), set(), set(), 0)
    b = fr.build(rnd, _images(), set(), set(), 0)
    assert a == b and len(a) == 10                      # 5 tán x 2 ruộng


def test_gold_spreads_over_flights_and_skips_border_crowns(root):
    items = fr.build(fa._round("gold"), _images(), set(), set(), 0)
    f1 = [it for it in items if it["field"] == "field_1"]
    assert len({it["flight"] for it in f1}) == 2        # hai đường bay của field_1
    assert len({it["image"] for it in f1}) == 3         # 3 + 2 tán: hai ảnh không đủ (mỗi ảnh 2 tán hợp lệ)
    assert all(it["crown"] != 2 for it in items)        # tán chạm mép không bao giờ được chọn


def test_training_round_keeps_its_distance_from_gold(root):
    imgs = _images()
    gold = fr.build(fa._round("gold"), imgs, set(), set(), 0)
    gflat = {it["image"] for it in gold}
    r1 = fr.build(fa._round("r1"), imgs, gflat, gflat, 3)
    assert not gflat & {it["image"] for it in r1}       # một ảnh một vòng
    gold_at = {}
    for im in imgs:
        if im["flat"] in gflat:
            gold_at.setdefault(im["flight"], []).append(im["seq"])
    for it in r1:                                       # không ảnh nào cách ảnh vàng ≤ 3 khung cùng đường bay
        assert all(abs(it["seq"] - s) > 3 for s in gold_at.get(it["flight"], []))


def test_images_per_field_takes_whole_images(root):
    rnd = {"pick": {"images_per_field": 2, "crowns_per_image": "all", "seed": 3}}
    items = fr.build(rnd, _images(), set(), set(), 0)
    for field in ("field_1", "field_2"):
        imgs = {it["image"] for it in items if it["field"] == field}
        assert len(imgs) == 2                           # đúng 2 ảnh mỗi ruộng
        for flat in imgs:                               # gán kín: cả 3 tán, kể cả tán chạm mép
            assert sorted(it["crown"] for it in items if it["image"] == flat) == [0, 1, 2]
    f1 = {it["flight"] for it in items if it["field"] == "field_1"}
    assert len(f1) == 2                                 # vẫn rải qua hai đường bay
    capped = fr.build({"pick": {"images_per_field": 2, "crowns_per_image": 1, "seed": 3}}, _images(), set(), set(), 0)
    assert len(capped) == 4                             # 2 ảnh x 1 tán x 2 ruộng


def test_buffer_only_counts_frames_in_the_same_flight():
    def im(flight, seq):
        return dict(name="%s/%d" % (flight, seq), flat="%s__%d" % (flight, seq), field="f", flight=flight,
                    seq=seq, size=(H, W), crowns=[CA])
    imgs = [im("f/1", 5), im("f/1", 9), im("f/2", 5)]
    rnd = {"pick": {"crowns_per_field": 10, "crowns_per_image": 1}}
    got = {it["image"] for it in fr.build(rnd, imgs, set(), {"f/1__5"}, 3)}
    assert got == {"f/1__9", "f/2__5"}                  # f/1 khung 5 bị chặn chính nó; f/2 khung 5 thì không


# -------------------------------------------------------------------- API vòng
def test_a_training_round_waits_for_the_gold_set(root):
    res = fa.round_build(fa.RoundReq(round="r1"))
    assert res.status_code == 400
    assert fa.round_build(fa.RoundReq(round="gold"))["n_items"] == 10
    r1 = fa.round_build(fa.RoundReq(round="r1"))
    assert r1["ok"] and r1["n_items"] > 0
    assert fa.round_build(fa.RoundReq(round="gold")).status_code == 409   # đã dựng


def test_a_scribble_round_with_role_test_is_the_scoring_side(root, monkeypatch):
    # Không bộ vàng: vòng test và vòng train đều vẽ nét; vòng có role: test giữ
    # vai bộ chấm (dựng trước, vòng train tránh ảnh quanh nó, script chấm trên nó).
    cfg = root / "rounds_test.yaml"
    cfg.write_text(
        "version: v1\ndata_dir: flower_v1\nrounds:\n"
        "  - {id: test, kind: scribble, role: test, pick: {images_per_field: 1, crowns_per_image: all, seed: 7}}\n"
        "  - {id: train, kind: scribble, pick: {images_per_field: 2, crowns_per_image: all, buffer_frames: 3, seed: 11}}\n",
        encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", cfg)
    monkeypatch.setattr(fa, "_CACHE", None)
    assert fa.round_build(fa.RoundReq(round="train")).status_code == 400   # test trước
    t = fa.round_build(fa.RoundReq(round="test"))
    assert t["by_field"] == {"field_1": {"images": 1, "crowns": 3}, "field_2": {"images": 1, "crowns": 3}}
    assert fa.round_build(fa.RoundReq(round="train"))["ok"]
    st = fa._store()
    test_items, train_items = st.load_round("test")["items"], st.load_round("train")["items"]
    test_at = {(it["flight"], it["seq"]) for it in test_items}
    for it in train_items:
        assert all(f != it["flight"] or abs(s - it["seq"]) > 3 for f, s in test_at)
    a, b = test_items[0], train_items[0]
    for rnd, it in (("test", a), ("train", b)):
        fa.stroke(fa.StrokeReq(round=rnd, name=it["name"], crown=it["crown"], label=1, pts=[[25, 25]]))
        fa.finish(fa.FinishReq(round=rnd, name=it["name"], crown=it["crown"]))
    rounds = fa.load_config()[0]["rounds"]
    labels = root / "labels"
    assert [(x["image"], x["crown"]) for x in fd.collect(st, rounds, labels, "test")] == [(a["image"], a["crown"])]
    assert [(x["image"], x["crown"]) for x in fd.collect(st, rounds, labels, "train")] == [(b["image"], b["crown"])]


def test_hand_picked_whole_images_for_test_and_train(root, monkeypatch):
    # Người gán tự chọn ảnh: thêm cả ảnh vào vòng; ảnh cách ảnh phía bên kia
    # ≤ 3 khung cùng đường bay thì không nhận (kể cả khi vẽ thẳng không bấm Thêm).
    cfg = root / "rounds_pick.yaml"
    cfg.write_text(
        "version: v1\ndata_dir: flower_v1\nrounds:\n"
        "  - {id: test, kind: scribble, role: test, pick: {mode: manual, whole_image: true}}\n"
        "  - {id: train, kind: scribble, pick: {mode: manual, whole_image: true, buffer_frames: 3}}\n",
        encoding="utf-8")
    monkeypatch.setattr(fa, "CFG_PATH", cfg)
    monkeypatch.setattr(fa, "_CACHE", None)
    st = fa._store()
    f1, f2 = "field_1/10/1", "field_1/10/2"

    def pick(rnd, name, action="add"):
        return fa.round_image(fa.PickReq(round=rnd, name=name, action=action))

    res = pick("test", _name(f1, 1))
    assert res["ok"] and res["crowns"] == 3 and res["images"] == 1          # mọi tán, kể cả tán chạm mép
    assert pick("train", _name(f1, 4)).status_code == 409                   # cách 3 khung: trùng cây
    assert pick("train", _name(f1, 5))["ok"]                                # cách 4 khung
    assert pick("train", _name(f2, 1))["ok"]                                # đường bay khác
    assert pick("test", _name(f1, 5)).status_code == 409                    # đã thuộc train
    res = fa.stroke(fa.StrokeReq(round="train", name=_name(f1, 2), crown=0, label=1, pts=[[25, 25]]))
    assert res.status_code == 409                                           # vẽ thẳng cũng bị chặn
    assert fa.stroke(fa.StrokeReq(round="train", name=_name(f1, 8), crown=0, label=1, pts=[[25, 25]]))["ok"]
    q = fa.queue(round="train", who="A")
    assert q["total"] == 9 and q["progress"]["field_1"]["images"] == 3     # vẽ một tán là nhận cả ảnh
    # bỏ ảnh chưa có nét: sạch khỏi vòng, kể cả dòng trạng thái do Học & tô để lại
    flat5 = server._flat(_name(f1, 5))
    st.set_status(flat5, 0, "train", state="doing", source="classifier")
    res = pick("train", _name(f1, 5), "remove")
    assert res["ok"] and not res["in_round"] and res["images"] == 2
    assert st.load_ops(flat5)["round"] is None
    assert not any(k[0] == flat5 for k in st.read_status())
    assert pick("train", _name(f1, 8), "remove").status_code == 409         # đã có nét


def test_role_must_be_test(root, tmp_path, monkeypatch):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("rounds:\n  - {id: x, kind: scribble, role: gold}\n", encoding="utf-8")
    _, errors = fa.load_config(cfg)
    assert errors and "role" in errors[0]


def test_queue_progress_and_membership(root):
    fa.round_build(fa.RoundReq(round="gold"))
    q = fa.queue(round="gold", who="A")
    assert q["total"] == 10 and q["done"] == 0
    it = q["items"][0]
    # ảnh ngoài vòng: từ chối
    inside = {x["image"] for x in q["items"]}
    other = next(n for n in NAMES if server._flat(n) not in inside)
    res = fa.stroke(fa.StrokeReq(round="gold", name=other, crown=0, label=1, pts=[[20, 20]]))
    assert res.status_code == 400
    assert fa.stroke(fa.StrokeReq(round="gold", name=it["name"], crown=it["crown"], label=1, pts=[[25, 25]]))["ok"]
    fa.finish(fa.FinishReq(round="gold", name=it["name"], crown=it["crown"], seconds=30))
    q = fa.queue(round="gold", who="A")
    assert q["done"] == 1 and q["median_seconds"] == 30
    assert q["progress"][it["field"]]["done"] == 1


def test_the_second_annotator_works_apart(root):
    fa.round_build(fa.RoundReq(round="gold"))
    qb = fa.queue(round="gold", who="B")
    assert qb["second"] and qb["total"] == 2
    assert {it["field"] for it in qb["items"]} == {"field_1", "field_2"}     # rải theo ruộng
    it = qb["items"][0]
    st = fa._store()
    # A tô hoa cả tán; B tô hoa nửa trái. Mask của A không đổi vì nét của B.
    x0, y0, x1, y1 = [10, 10, 41, 41] if it["crown"] == 0 else [50, 15, 86, 56]
    full = [[x, y] for y in range(y0, y1, 3) for x in (x0, x1)]
    for w, pts in (("A", full), ("B", [[x0 + 4, y] for y in range(y0, y1, 3)])):
        fa.stroke(fa.StrokeReq(round="gold", name=it["name"], crown=it["crown"], who=w, label=1, r=4, pts=pts))
        fa.finish(fa.FinishReq(round="gold", name=it["name"], crown=it["crown"], who=w))
    ma = st.load_mask(it["image"], (H, W))
    mb = st.load_mask(it["image"], (H, W), "B")
    inside = (slice(y0, y1 - 1), slice(x0, x1 - 1))
    assert (ma[inside] == 1).mean() > (mb[inside] == 1).mean()
    view_b = fa.crowns(name=it["name"], round="gold", who="B")["crowns"][it["crown"]]
    assert view_b["state"] == "done" and view_b["in_round"]
    ag = fa.agreement(round="gold")
    assert ag["all"]["crowns"] == 1 and 0 < ag["all"]["iou"] < 1
    # B không được đụng tán ngoài phần của mình
    others = [x for x in fa.queue(round="gold", who="A")["items"] if (x["image"], x["crown"]) not in
              {(y["image"], y["crown"]) for y in qb["items"]}]
    res = fa.stroke(fa.StrokeReq(round="gold", name=others[0]["name"], crown=others[0]["crown"], who="B",
                                 label=1, pts=[[20, 20]]))
    assert res.status_code == 400


def test_snap_paints_whole_superpixels(root):
    fa.round_build(fa.RoundReq(round="gold"))
    it = fa.queue(round="gold", who="A")["items"][0]
    st = fa._store()
    c = (25, 25) if it["crown"] == 0 else (67, 35)
    fa.stroke(fa.StrokeReq(round="gold", name=it["name"], crown=it["crown"], label=1, r=1, pts=[list(c)]))
    plain = int((st.load_mask(it["image"], (H, W)) == 1).sum())
    fa.undo(fa.CrownReq(round="gold", name=it["name"], crown=it["crown"]))
    fa.stroke(fa.StrokeReq(round="gold", name=it["name"], crown=it["crown"], label=1, r=1, snap=True, pts=[list(c)]))
    snapped = int((st.load_mask(it["image"], (H, W)) == 1).sum())
    assert plain == 5 and snapped > plain                # chấm 5 px bằng cả mảnh SLIC ~20 px


def test_agreement_counts_pixels_and_ignores_255():
    a = np.array([1, 1, 0, 0, 255], np.uint8)
    b = np.array([1, 0, 1, 0, 1], np.uint8)
    r = fr.agreement([(a, b)])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["iou"] == pytest.approx(1 / 3) and r["f1"] == pytest.approx(0.5)
    assert fr.agreement([])["iou"] is None
