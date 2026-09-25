"""Sinh nhãn YOLO-seg từ bộ COCO đã có.

Vì sao chuyển đổi thay vì xuất lại từ app/: bộ COCO trong data/export đã được
nghiệm thu, và ảnh đã nằm sẵn ở images/<split>/. Xuất lại sẽ chép thêm 558 MB
ảnh y hệt. Ở đây chỉ GHI THÊM labels/<split>/ bên cạnh, không đụng vào
images/ hay annotations/.

Ultralytics tìm nhãn bằng cách thay "/images/" thành "/labels/" trong đường
dẫn ảnh, nên hai thư mục phải là anh em ruột — đó là lý do labels/ nằm trong
chính thư mục bộ dữ liệu chứ không phải ở runs/.

Định dạng một dòng: <class> x1 y1 x2 y2 ... xn yn, toạ độ chuẩn hoá [0, 1],
class đánh số TỪ 0 (lệ YOLO; COCO bên cạnh đánh từ 1, lệch nhau là cố ý).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .coco import CocoDataset

# Ultralytics đọc bằng float32 và ghi lại ở %.6g; 1e-6 là dưới mức đó nên
# sai lệch lớn hơn ngưỡng này là lỗi thật, không phải làm tròn.
ROUNDTRIP_TOL = 1e-5


def _label_lines(ds: CocoDataset, image, class_index: int) -> list[str]:
    lines = []
    for r in image.regions:
        p = r.normalized_polygon().reshape(-1)
        lines.append(str(class_index) + " " + " ".join(f"{v:.6f}" for v in p))
    return lines


def write_labels(
    root: str | Path,
    splits: list[str],
    class_index: int = 0,
    min_area: float = 0.0,
) -> dict:
    """Ghi labels/<split>/*.txt. Idempotent: chạy lại cho kết quả y hệt."""
    root = Path(root)
    stats: dict = {"splits": {}, "class_index": class_index}
    for split in splits:
        ds = CocoDataset(root, split, min_area=min_area)
        out_dir = root / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        n_lines = 0
        empty: list[str] = []
        for im in ds.image_list():
            lines = _label_lines(ds, im, class_index)
            n_lines += len(lines)
            if not lines:
                # Ảnh không có vùng nào vẫn phải có file rỗng: ultralytics coi
                # đó là "ảnh nền", khác hẳn với "thiếu nhãn" (nó sẽ cảnh báo).
                empty.append(im.file_name)
            # Stem của nhãn phải khớp stem của ảnh, kể cả ảnh có nhiều dấu chấm.
            (out_dir / (Path(im.file_name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        stats["splits"][split] = {
            "images": len(ds),
            "labels": n_lines,
            "empty_images": len(empty),
            "dropped": ds.dropped,
        }
    return stats


def write_data_yaml(
    root: str | Path,
    splits: dict[str, str],
    names: dict[int, str],
    out_file: str | Path | None = None,
) -> Path:
    """Ghi data.yaml. `splits` ánh xạ vai trò -> thư mục ảnh, vd
    {"train": "images/train", "val": "images/val"}.

    KHÔNG ghi khoá `path:`. Trước đây có, và nó là đường dẫn tuyệt đối của
    máy đang cắt fold — cắt ở Windows rồi train ở máy lab Linux thì
    `F:/CoffeeSeg/...` không còn là đường dẫn tuyệt đối nữa, nên ultralytics
    nối nó vào thư mục dataset của chính nó:

        check_det_dataset (8.4.143)
        path = Path(extract_dir or data.get("path")
                    or Path(data.get("yaml_file", "")).parent)
        if not path.exists() and not path.is_absolute():
            path = (DATASETS_DIR / path).resolve()

    và báo thiếu `<DATASETS_DIR>/F:/CoffeeSeg/.../images/val`. Thiếu khoá thì
    nhánh đầu rơi về thư mục chứa chính data.yaml — tức gốc fold, đúng thứ ta
    muốn, và đúng trên mọi máy. Bố cục này cố định: data.yaml luôn nằm cạnh
    images/ và labels/.
    """
    root = Path(root).resolve()
    doc: dict = {}
    doc.update(splits)
    doc["names"] = {int(k): v for k, v in names.items()}
    out = Path(out_file) if out_file else root / "data.yaml"
    out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return out


def verify_roundtrip(
    root: str | Path, splits: list[str], min_area: float = 0.0
) -> dict:
    """Đọc ngược labels/*.txt, quy về pixel, so với polygon trong COCO.

    Đây là phép kiểm bắt buộc: nhãn sai ở khâu này thì mọi con số huấn luyện
    về sau đều vô nghĩa mà không có dấu hiệu gì báo trước.
    """
    root = Path(root)
    report: dict = {"splits": {}, "ok": True}
    for split in splits:
        ds = CocoDataset(root, split, min_area=min_area)
        worst = 0.0
        n_checked = 0
        problems: list[str] = []
        for im in ds.image_list():
            lp = root / "labels" / split / (Path(im.file_name).stem + ".txt")
            if not lp.exists():
                problems.append(f"thiếu nhãn: {im.file_name}")
                continue
            lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(lines) != len(im.regions):
                problems.append(
                    f"{im.file_name}: {len(lines)} dòng nhãn vs {len(im.regions)} vùng COCO"
                )
                continue
            for line, reg in zip(lines, im.regions):
                vals = line.split()
                back = np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2)
                back *= (im.width, im.height)
                ref = np.clip(reg.polygon, 0, (im.width, im.height))
                if back.shape != ref.shape:
                    problems.append(f"{im.file_name}: lệch số đỉnh")
                    continue
                # Sai số tương đối theo kích thước ảnh: nhãn lưu ở dạng chuẩn
                # hoá %.6f nên ngưỡng cũng phải tính trên thang chuẩn hoá.
                d = np.abs(back - ref) / (im.width, im.height)
                worst = max(worst, float(d.max()))
                n_checked += 1
        ok = not problems and worst <= ROUNDTRIP_TOL
        report["splits"][split] = {
            "regions_checked": n_checked,
            "max_normalized_error": worst,
            "problems": problems[:10],
            "n_problems": len(problems),
            "ok": ok,
        }
        report["ok"] = report["ok"] and ok
    return report
