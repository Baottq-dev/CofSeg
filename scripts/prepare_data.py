"""Giải nén bản xuất chưa chia rồi cắt đủ sáu fold.

    python scripts/prepare_data.py all_v2.tar
    python scripts/prepare_data.py all_v2.tar --val configs/dataset/val_flight.yaml
    python scripts/prepare_data.py all_v2.tar --name all_v2 --folds f4 f2

all_v2.tar tạo ở máy nhà, từ gốc repo (Windows 10+ có sẵn tar):

    tar -cf all_v2.tar -C data/export all_v2

Bên trong là một thư mục `all_v2/` với images/, annotations/instances.json,
labels/, data.yaml — đúng thứ app/ xuất với split_by=none và format coco+yolo.
Bản xuất ĐÃ CHIA sẵn train/val/test thì không dùng được: fold phải cắt theo
ruộng, không theo cách chia của bản xuất.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "export"


def extract(tar_path: Path, dest: Path) -> None:
    """Giải nén, từ chối thành viên trỏ ra ngoài dest (tar từ máy khác)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise SystemExit(f"{tar_path.name}: thành viên trỏ ra ngoài thư mục đích: "
                                 f"{member.name}")
        tf.extractall(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tar", help="đường dẫn tới file .tar của bản xuất")
    ap.add_argument("--name", default="all_v2", help="tên thư mục bên trong tar")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="chỉ cắt các fold này; mặc định cắt hết")
    ap.add_argument("--val", default="configs/dataset/val_block.yaml",
                    help="cách cắt val: val_block.yaml hoặc val_flight.yaml")
    ap.add_argument("--out-root", default=None,
                    help="thư mục chứa fold; mặc định data/export/<tên cách chia>")
    ap.add_argument("--keep", action="store_true",
                    help="không giải nén lại nếu thư mục bản xuất đã có")
    a = ap.parse_args(argv)

    tar_path = Path(a.tar)
    if not tar_path.exists():
        raise SystemExit(f"Không thấy {tar_path}")
    export = EXPORT / a.name

    if export.exists() and a.keep:
        print(f"Đã có {export.relative_to(ROOT)}, bỏ qua bước giải nén (--keep)")
    else:
        print(f"Giải nén {tar_path} -> {EXPORT.relative_to(ROOT)}")
        extract(tar_path, EXPORT)

    ann = export / "annotations" / "instances.json"
    if not ann.exists():
        split = sorted((export / "annotations").glob("instances_*.json"))
        if split:
            raise SystemExit(
                f"{export.relative_to(ROOT)} là bản xuất ĐÃ CHIA sẵn "
                f"({', '.join(p.name for p in split)}).\n"
                "Fold cắt theo ruộng nên cần bản chưa chia: xuất lại từ app/ với "
                "split_by=none, format coco + yolo.")
        raise SystemExit(f"Không thấy {ann.relative_to(ROOT)} sau khi giải nén")
    if not (export / "labels").exists():
        print(f"{export.relative_to(ROOT)} không có labels/ — nhãn YOLO sẽ được sinh "
              "từ chính file COCO của từng fold.")

    out_root = Path(a.out_root) if a.out_root else EXPORT / Path(a.val).stem.replace("val_", "")
    base = [sys.executable, "scripts/make_fold.py",
            "--export", export.relative_to(ROOT).as_posix(),
            "--val", a.val, "--out-root", out_root.as_posix()]
    for args in ([["--all"]] if not a.folds else [["--fold", f] for f in a.folds]):
        print("  $", " ".join(base + args))
        subprocess.run(base + args, cwd=ROOT, check=True)

    made = sorted(d.name for d in out_root.glob("f*") if d.is_dir())
    print(f"\nFold đã có trong {out_root.relative_to(ROOT).as_posix()}: "
          f"{' '.join(made) or '(chưa có)'}")
    print("Tiếp: lệnh chạy từng model nằm trong benchmark/<model>/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
