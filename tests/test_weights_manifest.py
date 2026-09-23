"""Bản kê trọng số: configs/weights.yaml phải tự nhất quán, và đường tải/kiểm
sha phải từ chối file sai thay vì lặng lẽ dùng.

Không tải gì từ mạng: URL file:// trỏ vào file tạm là đủ để đi hết đường
tải -> .part -> kiểm sha -> đổi tên.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from canopyseg import weights as W

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_manifest_is_consistent():
    doc = W.load_manifest(ROOT / "configs" / "weights.yaml")
    files = doc["files"]
    assert set(W.select(doc, "all")) == set(files)
    main = {"yolo11s-seg.pt", "mask_rcnn_R_50_FPN_3x_coco.pkl",
            "maskformer2_R50_bs16_50ep_coco.pkl", "solov2_r50_fpn_3x_coco.pth"}
    assert main <= set(W.select(doc, "benchmark"))
    assert "sam2.1_hiera_large.pt" in W.select(doc, "annotator")
    assert len({s["url"] for s in files.values()}) == len(files)          # không hai tên một URL
    # Mọi config train trỏ vào weights/<file> thì file đó phải có trong bản kê.
    train_cfgs = list((ROOT / "configs" / "train").glob("*.yaml"))
    train_cfgs += ROOT.glob("members/*/configs/train/*.yaml")
    for cfg in train_cfgs:
        model = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("model")
        if isinstance(model, str) and model.startswith("weights/"):
            assert Path(model).name in files, f"{cfg.name}: {model} chưa khai trong weights.yaml"
    # URL trong bản kê phải khớp URL mà code của thành viên dùng; hai chỗ lệch
    # nhau thì máy lab tải một bản, trainer nạp một bản khác.
    for member, needle, name in (
            ("mask2former", "maskformer2_R50_bs16_50ep", "maskformer2_R50_bs16_50ep_coco.pkl"),
            ("solov2", "solov2_r50_fpn_3x_coco", "solov2_r50_fpn_3x_coco.pth")):
        code = "".join(p.read_text(encoding="utf-8")
                       for p in (ROOT / "benchmark" / member / "cofseg").rglob("*.py"))
        assert needle in code, f"{member}: không thấy checkpoint {needle} trong code"
        assert files[name]["url"].rsplit("/", 1)[-1] in code


def _manifest(tmp_path, url, sha, name="a.pt", extra=None):
    doc = {"dir": str(tmp_path / "w"),
           "groups": {"benchmark": [name]},
           "files": {name: {"url": url, "sha256": sha, **(extra or {})}}}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_manifest_validation_rejects_bad_entries(tmp_path):
    good = "0" * 64
    W.load_manifest(_manifest(tmp_path, "https://x/y.pt", good))
    with pytest.raises(ValueError, match="sha256"):
        W.load_manifest(_manifest(tmp_path, "https://x/y.pt", "abc"))
    with pytest.raises(ValueError, match="url"):
        W.load_manifest(_manifest(tmp_path, "ftp://x/y.pt", good))
    with pytest.raises(ValueError, match="tên"):
        W.load_manifest(_manifest(tmp_path, "https://x/y.pt", good, name="sub/a.pt"))
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"files": {"a.pt": {"url": "https://x", "sha256": good}},
                                 "groups": {"benchmark": ["b.pt"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="chưa khai"):
        W.load_manifest(p)
    doc = W.load_manifest(_manifest(tmp_path, "https://x/y.pt", good))
    with pytest.raises(ValueError, match="nhóm"):
        W.select(doc, "nope")
    with pytest.raises(ValueError, match="không có"):
        W.select(doc, only=["zzz"])
    assert W.select(doc, only=["a.pt"]) == ["a.pt"]


def test_download_checks_sha_and_refuses_to_overwrite(tmp_path):
    src = tmp_path / "src.pt"
    src.write_bytes(b"trong so" * 1000)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    url = src.resolve().as_uri()
    doc = W.load_manifest(_manifest(tmp_path, url, sha))
    spec = doc["files"]["a.pt"]
    root = Path(doc["dir"])

    seen = []
    r = W.download("a.pt", spec, root, progress=lambda d, t: seen.append((d, t)))
    assert r["status"] == "downloaded" and (root / "a.pt").read_bytes() == src.read_bytes()
    assert seen and not list(root.glob("*.part"))
    assert W.download("a.pt", spec, root)["status"] == "ok"        # lần hai: đã có, đúng sha
    assert W.local_for_url(url, doc) == root / "a.pt"
    assert W.local_for_url("https://elsewhere", doc) is None

    # File có sẵn nhưng sai nội dung: báo mismatch, không đè; --force thì tải lại.
    (root / "a.pt").write_bytes(b"hong")
    assert W.check(root / "a.pt", spec) == "mismatch"
    assert W.download("a.pt", spec, root)["status"] == "mismatch"
    assert (root / "a.pt").read_bytes() == b"hong"
    assert W.download("a.pt", spec, root, force=True)["status"] == "downloaded"

    # Nguồn trả về nội dung khác sha khai: xoá bản tải, không để file hỏng lại.
    bad = dict(spec, sha256="1" * 64)
    (root / "a.pt").unlink()
    assert W.download("a.pt", bad, root)["status"] == "bad_download"
    assert not (root / "a.pt").exists() and not list(root.glob("*.part"))
    assert W.check(root / "a.pt", spec) == "missing"
