"""Dataset train của phần chắt lọc mặt nạ: đích của thầy đi qua đúng các phép biến đổi của ảnh.

Đích thầy (teacher_masks.py) là một ảnh 4 kênh cỡ ảnh train cho mỗi ảnh: (mã tán 1, giá trị 1, mã tán 2, giá
trị 2), mã tán là số dòng trong file nhãn. Ultralytics 8.4.143 đã đưa `labels["semantic_mask"]` qua mọi phép
hình học của chuỗi train (RandomPerspective, LetterBox, RandomFlip, cả Mosaic/MixUp/CopyPaste) với nội suy gần
nhất và viền 255, nên ở đây chỉ cần đặt ảnh đích vào khoá đó; không sửa phép nào. Viền 255 của cv2 ở ảnh 4
kênh là (255, 0, 0, 0): mã "không có tán", giá trị 0.

Phần còn lại là giữ được tán nào là tán nào sau khi RandomPerspective lọc bớt tán ra ngoài khung và Format sắp
lại tán theo diện tích:

- update_labels_info gói mã tán vào cột lớp: lớp + CODE·(mã + 1). RandomPerspective lọc `cls` cùng với
  `instances`, nên mã đi theo tán. Tán do phép khác thêm vào (lớp 0, không mã) giải ra mã −1: không có đích.
- KDFormat (đứng cuối chuỗi, thay Format) giải mã trước khi Format dùng tới lớp (sem_masks), ghi lại thứ tự
  sắp theo diện tích của Format, rồi lấy mẫu ảnh đích về lưới mặt nạ (mask_ratio) ở tâm ô.

Batch có thêm ba khoá: `kd_map` (B, 4, H/r, W/r) uint8, `kd_ids` (mã tán của từng tán trong batch, cùng thứ
tự với `cls`/`masks`), `kd_has` (tán đó có đích thầy không). Dataset val giữ nguyên của ultralytics.

Với `--aug ag1`, make_dataset() ghép lớp này với AugDataset của benchmark (aug_train.py) thành một lớp. Phép
chuyển vị của ag1 không nằm trong ultralytics nên không biết `semantic_mask`; KDTranspose bọc nó để ảnh đích
đổi trục cùng lúc với ảnh. Mã tán không phải sửa: chuyển vị giữ nguyên thứ tự và số tán. Phép dán (pa1, ag4)
chưa ghép được, vì tán dán đè lên tán có sẵn mà ảnh đích không biết (blocks.check_kd_aug).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import polygons2masks_overlap

from .teachers import fold_info

#: Lớp và mã tán chung một số thực float32 (chính xác tới 2^24): mã + 1 nhân CODE, cộng lớp (< CODE).
CODE = 1024
NONE = 255
TARGETS_VERSION = 1


class Targets:
    """Đích thầy của một fold: <root>/<fold>/meta.json và maps/<ảnh>.png."""

    def __init__(self, root: str | Path, fold_root: str | Path, imgsz: int):
        info = fold_info(fold_root)
        self.dir = Path(root) / info["fold"]
        p = self.dir / "meta.json"
        if not p.exists():
            raise SystemExit(f"Không thấy {p}: dựng đích thầy cho fold {info['fold']} bằng distill/teacher_masks.py")
        self.meta = json.loads(p.read_text(encoding="utf-8"))
        bad = []
        if self.meta.get("version") != TARGETS_VERSION:
            bad.append(f"phiên bản {self.meta.get('version')} (cần {TARGETS_VERSION})")
        if self.meta.get("fold") != info["fold"]:
            bad.append(f"fold {self.meta.get('fold')!r}")
        if info.get("source_sha1") and self.meta.get("source_sha1") != info["source_sha1"]:
            bad.append("mã băm bản xuất khác fold.json (dữ liệu đã đổi sau khi dựng đích)")
        if int(self.meta.get("imgsz", -1)) != int(imgsz):
            bad.append(f"dựng ở imgsz {self.meta.get('imgsz')}, lượt train dùng {imgsz}")
        if bad:
            raise SystemExit(f"Đích thầy {self.dir} không dùng được: " + "; ".join(bad))
        self.images: dict = self.meta["images"]

    def describe(self) -> str:
        s = self.meta.get("summary", {})
        rate = ", ".join(f"{k} {100 * v:.1f}%" for k, v in s.get("match_rate", {}).items())
        return (f"đích thầy {self.dir.as_posix()}: chế độ {self.meta['mode']}, thầy {'+'.join(self.meta['teachers'])}, "
                f"{s.get('gt')} tán nhãn, tỉ lệ ghép {rate}")

    def load_map(self, stem: str, hw: tuple[int, int]) -> np.ndarray:
        h, w = int(hw[0]), int(hw[1])
        m = cv2.imread(str(self.dir / "maps" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Thiếu ảnh đích {self.dir / 'maps' / (stem + '.png')}")
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m


def align(boxes_label: np.ndarray, boxes_target: np.ndarray, tol: float = 1e-4) -> np.ndarray | None:
    """Mã tán (dòng trong file nhãn lúc dựng đích) của từng tán ultralytics đã đọc, so bằng hộp xywh chuẩn hoá.

    Thường là 0..n−1; khác khi ultralytics bỏ dòng trùng. None nếu không ghép được trọn vẹn.
    """
    a = np.asarray(boxes_label, np.float64).reshape(-1, 4)
    b = np.asarray(boxes_target, np.float64).reshape(-1, 4)
    if len(a) == len(b) and (len(a) == 0 or np.abs(a - b).max() <= tol):
        return np.arange(len(a), dtype=np.int64)
    if not len(b):
        return None
    d = np.abs(a[:, None, :] - b[None, :, :]).max(-1)
    perm = d.argmin(1)
    if (d[np.arange(len(a)), perm] > tol).any() or len(set(perm.tolist())) != len(perm):
        return None
    return perm.astype(np.int64)


class KDFormat(Format):
    """Format của ultralytics + giải mã tán và lấy mẫu ảnh đích về lưới mặt nạ."""

    def apply_instances(self, labels, params):
        cls = np.asarray(params["cls"])
        code = np.rint(cls.astype(np.float64).reshape(-1)).astype(np.int64)
        ids = code // CODE - 1
        params["cls"] = (code % CODE).astype(cls.dtype if cls.size else np.float32).reshape(-1, 1)
        self._order = None
        labels = super().apply_instances(labels, params)
        if self._order is not None:
            ids = ids[self._order]
        params["kd_ids"] = ids
        return labels

    def _format_segments(self, instances, cls, w, h):
        # Như Format._format_segments (ultralytics 8.4.143), giữ lại thứ tự sắp theo diện tích.
        if self.mask_overlap:
            masks, order = polygons2masks_overlap((h, w), instances.segments, downsample_ratio=self.mask_ratio)
            self._order = order
            return masks[None], instances[order], cls[order]
        return super()._format_segments(instances, cls, w, h)

    def apply_semantic(self, labels, params):
        m = labels.pop("semantic_mask", None)
        has = np.asarray(labels.pop("kd_has_orig", np.zeros(0, bool)), bool)
        h, w, r = params["h"], params["w"], self.mask_ratio
        if m is None:
            m = np.zeros((h, w, 4), np.uint8)
            m[..., 0] = m[..., 2] = NONE
        elif m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        sub = m[r // 2::r, r // 2::r][: h // r, : w // r]
        labels["kd_map"] = torch.from_numpy(np.ascontiguousarray(sub.transpose(2, 0, 1)))
        ids = np.asarray(params.get("kd_ids", np.zeros(0, np.int64)), np.int64)
        ok = (ids >= 0) & (ids < len(has))
        hv = np.zeros(len(ids), bool)
        hv[ok] = has[ids[ok]]
        labels["kd_ids"] = torch.from_numpy(ids)
        labels["kd_has"] = torch.from_numpy(hv)
        return labels


class KDDataset(YOLODataset):
    """YOLODataset của tập train có đích thầy. KD_TARGETS, KD_FOLD do make_dataset() gắn vào lớp con."""

    KD_TARGETS: str = ""
    KD_FOLD: str = ""
    format_class = KDFormat

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd = Targets(self.KD_TARGETS, self.KD_FOLD, self.imgsz)
        self.kd_perm: dict[str, np.ndarray] = {}
        self.kd_has: dict[str, np.ndarray] = {}
        bad = []
        for lab in self.labels:
            stem = Path(lab["im_file"]).stem
            rec = self.kd.images.get(stem)
            if rec is None:
                bad.append(f"{stem}: không có trong meta.json")
                continue
            perm = align(lab["bboxes"], rec["boxes"])
            if perm is None:
                bad.append(f"{stem}: {len(lab['bboxes'])} tán ở nhãn, {rec['n']} tán ở đích, hộp không khớp")
                continue
            self.kd_perm[lab["im_file"]] = perm
            self.kd_has[lab["im_file"]] = np.asarray(rec["has"], bool)
        if bad:
            raise SystemExit(f"Đích thầy {self.kd.dir} không khớp nhãn ở {len(bad)} ảnh, vd:\n  " + "\n  ".join(bad[:5]))

    def update_labels_info(self, label):
        perm = self.kd_perm[label["im_file"]]
        label["cls"] = label["cls"] + CODE * (perm[:, None] + 1).astype(label["cls"].dtype)
        label["kd_has_orig"] = self.kd_has[label["im_file"]]
        return super().update_labels_info(label)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        label["semantic_mask"] = self.kd.load_map(Path(label["im_file"]).stem, label["resized_shape"])
        return label

    @staticmethod
    def collate_fn(batch):
        kd = {k: [b.pop(k) for b in batch] for k in ("kd_map", "kd_ids", "kd_has")}
        out = YOLODataset.collate_fn(batch)
        out["kd_map"] = torch.stack(kd["kd_map"], 0)
        out["kd_ids"] = torch.cat(kd["kd_ids"], 0)
        out["kd_has"] = torch.cat(kd["kd_has"], 0)
        return out


class KDTranspose:
    """Bọc phép chuyển vị của --aug ag1: ảnh bị chuyển vị thì ảnh đích thầy cũng vậy.

    Phép gốc rút số từ bộ số riêng của nó và chỉ thay `labels["img"]` khi chuyển vị, nên so đối tượng ảnh trước
    và sau là biết phép có chạy không, mà không phải rút thêm số.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, labels: dict) -> dict:
        img = labels["img"]
        labels = self.inner(labels)
        m = labels.get("semantic_mask")
        if labels["img"] is not img and m is not None:
            labels["semantic_mask"] = np.ascontiguousarray(np.swapaxes(m, 0, 1))
        return labels


class _KDAug:
    """Phần ghép với AugDataset: bọc phép chuyển vị trong chuỗi biến đổi mà AugDataset vừa dựng."""

    def build_transforms(self, hyp=None):
        from cofseg.training.aug_train import Transpose

        t = super().build_transforms(hyp)
        t.transforms = [KDTranspose(x) if isinstance(x, Transpose) else x for x in t.transforms]
        return t


def make_dataset(targets: str | Path, fold_root: str | Path, aug: dict | None = None,
                 aug_log_dir: str | Path | None = None) -> type:
    """Lớp dataset train có đích thầy; có `aug` (khối của --aug) thì ghép thêm AugDataset của benchmark."""
    attrs = {"KD_TARGETS": str(targets), "KD_FOLD": str(fold_root)}
    if not aug:
        return type("KDDataset", (KDDataset,), attrs)
    from cofseg.training.aug_train import AugDataset

    from .blocks import check_kd_aug

    check_kd_aug({"targets": targets}, aug)
    log_dir = str(aug_log_dir) if aug_log_dir else None
    aug_cls = type("AugDataset", (AugDataset,), {"AUG": dict(aug), "LOG_DIR": log_dir})
    # Thứ tự lớp: KDDataset trước AugDataset, nên update_labels_info, get_image_and_label, format_class và
    # collate_fn là của KDDataset; build_transforms đi qua _KDAug rồi tới AugDataset (chèn phép của --aug).
    return type("KDAugDataset", (_KDAug, KDDataset, aug_cls), attrs)
