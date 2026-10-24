"""Nối `--aug` vào dataloader train của ultralytics 8.4.143. Dataset val/test không đổi.

- AugDataset(YOLODataset) chèn hai phép vào chuỗi gốc của v8_transforms:
  PasteCrowns (pa1/ag4) đứng ĐẦU chuỗi, ngay sau khi ảnh được đọc, trước
  Mosaic/RandomPerspective; Transpose (ag1) đứng trước Format, sau hai phép
  lật. Kho tán nguồn và kho chỗ dán dựng một lần trong __init__ (aug_paste.py).
- make_trainer() trả về một SegmentationTrainer chỉ khác ở build_dataset():
  với mode train, build_yolo_dataset dựng AugDataset thay cho YOLODataset.
  Model, loss, optimizer, chọn checkpoint giữ nguyên.

Mỗi phép rút số từ bộ số RIÊNG (np.random.Generator), seed từ (seed của lượt
train, torch.initial_seed() của tiến trình). random, np.random và torch chung
không bị chạm, nên mọi phép gốc của chuỗi (lật, HSV, co giãn, thứ tự ảnh) rút
đúng các số như lượt không --aug cùng seed. Tiến trình nạp dữ liệu nào cũng có
torch.initial_seed() riêng và cố định theo seed, nên lượt chạy lặp lại được.

Nhật ký ghi vào `<run_dir>/aug/`: kho tán nguồn (donors.csv), kho chỗ dán
(pool.csv), mọi lần thử khi dựng kho (pool_log.csv), mỗi tán thật sự được dán
lúc train (pasted_<tiến trình>.csv) và tóm tắt (summary.json).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments

from . import aug as augmod
from . import aug_paste as ap

ULTRALYTICS_VERSION = "8.4.143"
#: Muối của bộ số lúc train, khác muối của kho (aug_paste.SALT_*).
SALT_PASTE, SALT_T = 4103, 4104
#: Cột của pasted_*.csv: một dòng mỗi tán được dán lúc train.
PASTED_COLS = ("im", "mode", "n_pool", "k", "n_chosen", "tgt", "donor", "rot", "flip", "x0", "y0",
               "c", "hid", "cstar", "vis_px")


class _Rng:
    """Bộ số riêng của một phép, tạo lại khi sang tiến trình khác (worker của DataLoader)."""

    def __init__(self, salt: int, seed: int):
        self.salt, self.seed, self._pid, self._g = salt, int(seed), None, None

    def __call__(self) -> np.random.Generator:
        pid = os.getpid()
        if pid != self._pid:
            import torch

            self._g = np.random.default_rng([self.salt, self.seed, torch.initial_seed() % 2**32])
            self._pid = pid
        return self._g


def _worker_tag() -> str:
    import torch.utils.data as tud

    info = tud.get_worker_info()
    return f"w{info.id}" if info is not None else "main"


class Transpose:
    """ag1: đổi trục x và y của ảnh và nhãn với xác suất p (ảnh vuông sau RandomPerspective)."""

    def __init__(self, p: float, seed: int):
        self.p = float(p)
        self.rng = _Rng(SALT_T, seed)

    def __call__(self, labels: dict) -> dict:
        if self.rng().random() >= self.p:
            return labels
        labels["img"] = np.ascontiguousarray(np.swapaxes(labels["img"], 0, 1))
        inst = labels.pop("instances")
        # xyxy, xywh hay ltwh đều đổi cặp (0, 1) và (2, 3): chuyển vị đổi vai x và y.
        boxes = inst.bboxes[:, [1, 0, 3, 2]].copy()
        segs = np.ascontiguousarray(inst.segments[..., ::-1])
        kpts = None
        if inst.keypoints is not None:
            kpts = inst.keypoints.copy()
            kpts[..., [0, 1]] = kpts[..., [1, 0]]
        inst.update(bboxes=boxes, segments=segs, keypoints=kpts)
        labels["instances"] = inst
        return labels


class PasteCrowns:
    """pa1/ag4: dán vài chỗ trong kho của ảnh vào ảnh vừa đọc, thêm nhãn phần nhìn thấy."""

    def __init__(self, dataset, mode: str, p: dict, seed: int):
        self.dataset, self.mode, self.p = dataset, mode, dict(p)
        self.rng = _Rng(SALT_PASTE, seed)
        self._log = None

    def _write(self, rows: list) -> None:
        log_dir = getattr(self.dataset, "LOG_DIR", None)
        if not log_dir or not rows:
            return
        if self._log is None or self._log[0] != os.getpid():
            path = Path(log_dir) / f"pasted_{_worker_tag()}_{os.getpid()}.csv"
            fh = open(path, "a", newline="", encoding="utf-8", buffering=1)
            w = csv.DictWriter(fh, PASTED_COLS)
            if fh.tell() == 0:
                w.writeheader()
            self._log = (os.getpid(), fh, w)
        self._log[2].writerows(rows)

    def __call__(self, labels: dict) -> dict:
        g = self.rng()
        if g.random() >= self.p["p"]:
            return labels
        pools = getattr(self.dataset, "aug_pools", None)
        i = getattr(self.dataset, "aug_index", {}).get(labels.get("im_file"))
        if pools is None or i is None:
            return labels
        specs = pools.get(i, [])
        k = int(g.integers(self.p["k_min"], self.p["k_max"] + 1))
        img = labels["img"]
        h, w = img.shape[:2]
        r = w / labels["ori_shape"][1]
        margin = ap.px(self.p["gap"], r) if self.mode == "ag4" else ap.px(ap.TOUCH, r) + 1
        chosen = ap.choose(specs, k, g, margin)
        name = Path(labels["im_file"]).name
        base = {"im": name, "mode": self.mode, "n_pool": len(specs), "k": k, "n_chosen": len(chosen)}
        if not chosen:
            self._write([base])
            return labels
        inst = labels["instances"]
        segs = inst.segments * np.array([w, h], np.float32) if inst.normalized else inst.segments.copy()
        out, polys, done = ap.apply_specs(img, segs, r, chosen, self.dataset.aug_bank.get(ap.field_of(name), []))
        self._write([{**base, "tgt": s.tgt, "donor": f"{s.donor[0]}:{s.donor[1]}", "rot": s.rot,
                      "flip": s.flip, "x0": s.x0, "y0": s.y0, "c": s.c, "hid": s.hid, "cstar": s.cstar,
                      "vis_px": s.vis_px} for s in done] or [base])
        if not polys:
            return labels
        labels["img"] = out
        add_instances(labels, polys, w, h)
        return labels


def add_instances(labels: dict, polys: list, w: int, h: int) -> None:
    """Thêm các đa giác (K, 2) px khung vào nhãn, lớp 0, cùng số điểm và cùng dạng với nhãn có sẵn."""
    inst = labels["instances"]
    n = inst.segments.shape[1] if inst.segments.ndim == 3 and inst.segments.shape[1] else 1000
    new = np.stack(resample_segments([q.astype(np.float32) for q in polys], n=n), 0).astype(np.float32)
    boxes = np.concatenate([new.min(1), new.max(1)], 1)
    fmt, norm = inst._bboxes.format, inst.normalized
    inst.convert_bbox("xyxy")
    inst.denormalize(w, h)
    both = Instances.concatenate([inst, Instances(boxes, new, None, bbox_format="xyxy", normalized=False)])
    both.convert_bbox(fmt)
    if norm:
        both.normalize(w, h)
    labels["instances"] = both
    cls = labels["cls"]
    labels["cls"] = np.concatenate([cls, np.zeros((len(polys), 1), cls.dtype)], 0)


class AugDataset(YOLODataset):
    """YOLODataset của tập train với --aug. AUG và LOG_DIR do make_trainer() gắn vào lớp con."""

    AUG: dict = {}
    LOG_DIR: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_pools, self.aug_bank, self.aug_index = None, {}, {}
        self.aug_summary: dict = {"use": list(self.AUG.get("use", []))}
        mode = augmod.paste_mode(self.AUG)
        if mode and self.augment:
            self._build_paste(mode)
        if self.LOG_DIR:
            Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
            (Path(self.LOG_DIR) / "summary.json").write_text(
                json.dumps(self.aug_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        if not self.augment:
            return t
        seed = self._aug_seed = int(getattr(hyp, "seed", 0) or 0)
        mode = augmod.paste_mode(self.AUG)
        if mode:
            t.insert(0, PasteCrowns(self, mode, self.AUG[mode], seed))
        if "ag1" in self.AUG.get("use", ()):
            t.insert(len(t.transforms) - 1, Transpose(self.AUG["ag1"]["p"], seed))  # trước Format
        return t

    # ------------------------------------------------------------------ kho
    def _frames(self) -> list:
        """(tên, đường dẫn, segs px khung, h, w, r) của mọi ảnh, segs lấy mẫu lại như update_labels_info."""
        out = []
        for lab in self.labels:
            h0, w0 = lab["shape"]
            h, w, r = ap.frame_size(int(h0), int(w0), int(self.imgsz))
            segs = [np.asarray(s, np.float32) for s in lab.get("segments", [])]
            if segs:
                longest = max(len(s) for s in segs)
                n = longest + 1 if longest > 1000 else 1000
                segs = np.stack(resample_segments([s.copy() for s in segs], n=n), 0) * np.array([w, h], np.float32)
            else:
                segs = np.zeros((0, 1000, 2), np.float32)
            out.append((Path(lab["im_file"]).name, lab["im_file"], segs, h, w, r))
        return out

    def _build_paste(self, mode: str) -> None:
        t0 = time.time()
        seed = getattr(self, "_aug_seed", 0)
        frames = self._frames()
        self.aug_index = {lab["im_file"]: i for i, lab in enumerate(self.labels)}
        threads = min(8, os.cpu_count() or 1)
        self.aug_bank, donor_rows = ap.build_bank(frames, seed, threads)
        t1 = time.time()
        workers = 0 if len(frames) < 32 else min(8, os.cpu_count() or 1)
        tasks = [(i, name, segs, h, w, r) for i, (name, _p, segs, h, w, r) in enumerate(frames)]
        self.aug_pools, rows = ap.build_pools(tasks, self.aug_bank, mode, self.AUG[mode], seed, workers)
        t2 = time.time()
        n_spec = [len(self.aug_pools.get(i, [])) for i in range(len(frames))]
        by_field: dict = {}
        for (name, *_), n in zip(frames, n_spec):
            f = by_field.setdefault(ap.field_of(name), {"images": 0, "with_spec": 0, "specs": 0})
            f["images"] += 1
            f["with_spec"] += n > 0
            f["specs"] += n
        why: dict = {}
        for rw in rows:
            why[rw.get("why") or "ok"] = why.get(rw.get("why") or "ok", 0) + 1
        self.aug_summary.update(
            mode=mode, params=self.AUG[mode], seed=seed, images=len(frames),
            images_with_spec=int(sum(n > 0 for n in n_spec)), specs=int(sum(n_spec)),
            by_field=by_field, bank={f: len(v) for f, v in self.aug_bank.items()}, attempts=why,
            seconds={"bank": round(t1 - t0, 1), "pool": round(t2 - t1, 1)}, workers=workers,
            constants={k: getattr(ap, k) for k in ("MIN_DIAM", "TOUCH", "STEP", "SIZE", "DONOR_CONTACT",
                                                    "SIGMA", "TRIES", "PAD", "NEAREST", "HOLE_PX", "KEEP_DIL",
                                                    "AG4_FRAG", "ROUNDS", "BANK_PER_FIELD")})
        if self.LOG_DIR:
            d = Path(self.LOG_DIR)
            d.mkdir(parents=True, exist_ok=True)
            _csv(d / "donors.csv", donor_rows, ("field", "j", "im", "crown", "diam", "area", "contact", "h", "w"))
            _csv(d / "pool_log.csv", rows, ap.LOG_COLS)
            spec_rows = [{"im": frames[i][0], "tgt": s.tgt, "donor": f"{s.donor[0]}:{s.donor[1]}", "rot": s.rot,
                          "flip": s.flip, "x0": s.x0, "y0": s.y0, "c": s.c, "hid": s.hid, "cstar": s.cstar,
                          "vis_px": s.vis_px, "box": " ".join(map(str, s.box))}
                         for i in sorted(self.aug_pools) for s in self.aug_pools[i]]
            _csv(d / "pool.csv", spec_rows, ("im", "tgt", "donor", "rot", "flip", "x0", "y0", "c", "hid",
                                             "cstar", "vis_px", "box"))


def _csv(path: Path, rows: list, cols) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_trainer(block: dict, run_dir: str | Path):
    """SegmentationTrainer dựng AugDataset cho tập train, mọi thứ khác như gốc."""
    import ultralytics
    import ultralytics.data.build as build
    from ultralytics.models.yolo.segment import SegmentationTrainer

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise SystemExit(
            f"--aug viết cho ultralytics {ULTRALYTICS_VERSION}, môi trường đang có {ultralytics.__version__}. "
            "Phép chèn dựa vào thứ tự của v8_transforms và build_yolo_dataset nên phải đối chiếu lại."
        )
    ds_cls = type("AugDataset", (AugDataset,), {"AUG": dict(block), "LOG_DIR": str(Path(run_dir) / "aug")})

    class AugSegTrainer(SegmentationTrainer):
        AUG = dict(block)

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode != "train":
                return super().build_dataset(img_path, mode, batch)
            # build_yolo_dataset đọc tên YOLODataset từ module của nó lúc gọi.
            old = build.YOLODataset
            build.YOLODataset = ds_cls
            try:
                return super().build_dataset(img_path, mode, batch)
            finally:
                build.YOLODataset = old

    return AugSegTrainer


def install_log(yolo_model, run_dir: str | Path) -> None:
    """In tóm tắt kho (nếu có) vào run.log khi train bắt đầu."""

    def on_train_start(trainer):
        ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        s = getattr(ds, "aug_summary", None)
        if not isinstance(ds, AugDataset) or s is None:
            raise RuntimeError("Dataset train không phải AugDataset: --aug không được nối vào.")
        line = f"Tăng cường thêm: dataset train là AugDataset, phép {'+'.join(s['use'])}"
        if "specs" in s:
            line += (f"; kho {s['mode']}: {s['specs']} chỗ dán trên {s['images_with_spec']}/{s['images']} ảnh, "
                     f"tán nguồn {s['bank']}, dựng trong {s['seconds']['bank'] + s['seconds']['pool']:.0f} s"
                     f" -> {Path(run_dir) / 'aug'}")
        print(line, flush=True)

    yolo_model.add_callback("on_train_start", on_train_start)
