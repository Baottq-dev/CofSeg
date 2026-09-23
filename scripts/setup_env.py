"""Dựng môi trường trên máy Linux có GPU (máy lab / máy thuê).

    python scripts/setup_env.py                  # chạy hết
    python scripts/setup_env.py --dry-run        # in ra sẽ làm gì, không chạy
    python scripts/setup_env.py --from mmdet     # chạy lại từ một bước
    python scripts/setup_env.py --only weights
    python scripts/setup_env.py --list

Từng bước một, dừng ngay khi một bước lỗi và in đúng lệnh đã chạy, để lần đầu
dựng trên máy mới còn biết hỏng ở đâu. Chạy TRONG env đã kích hoạt: script
không tự tạo conda env (kích hoạt env từ bên trong một tiến trình Python
không có tác dụng ra ngoài), nó kiểm tra và bảo bạn cần gõ gì.

Một env cho tất cả: torch 2.4.1+cu121 (mmcv chỉ có wheel dựng sẵn tới torch
2.4), ultralytics, detectron2 build từ source, mmcv/mmdet, submodule
Mask2Former với op MSDeformAttn biên dịch tại chỗ, rồi trọng số COCO.

CHƯA CHẠY THẬT trên máy Linux: lần đầu nên đi từng bước với --only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M2F_DIR = "benchmark/mask2former/Mask2Former"
TORCH = ["torch==2.4.1", "torchvision==0.19.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"]


class Step:
    def __init__(self, name: str, about: str, fn):
        self.name, self.about, self.fn = name, about, fn


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def py(*args: str) -> subprocess.CompletedProcess:
    return sh(sys.executable, *args)


def pip(*args: str) -> subprocess.CompletedProcess:
    return sh(sys.executable, "-m", "pip", *args)


# ---------------------------------------------------------------- các bước
def step_check() -> None:
    """Máy có đủ thứ để biên dịch không, và đang ở env nào."""
    print(f"  python {sys.version.split()[0]} tại {sys.executable}")
    if sys.version_info < (3, 12):
        raise SystemExit("Cần Python >= 3.12 (scipy/scikit-image ghim trong requirements.txt).\n"
                         "    conda create -y -n cofseg python=3.12 && conda activate cofseg")
    if not shutil.which("nvcc"):
        raise SystemExit("Thiếu nvcc (CUDA toolkit): detectron2, SAM 2 và MSDeformAttn "
                         "đều biên dịch CUDA lúc cài.")
    sh("nvcc", "--version")
    sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader", check=False)


def step_torch() -> None:
    """torch trước requirements: detectron2 và SAM 2 import torch lúc build."""
    pip("install", *TORCH)
    py("-c", "import torch; assert torch.cuda.is_available(), 'torch không thấy CUDA'; "
             "print('torch', torch.__version__, torch.cuda.get_device_name(0))")


def step_requirements() -> None:
    """--no-build-isolation vì detectron2/SAM 2 cần torch có sẵn trong env."""
    pip("install", "-r", "requirements.txt", "--no-build-isolation")
    pip("install", "-e", ".")
    py("-c", "import detectron2; print('detectron2', detectron2.__version__)")


def step_mmdet() -> None:
    """mmdet 3.3.0 khai mmcv < 2.2.0, nhưng wheel dựng sẵn cho torch 2.4 là
    2.2.0 và chạy được: nới đúng dòng kiểm phiên bản đó."""
    import re

    import mmdet

    p = Path(mmdet.__file__)
    s = p.read_text(encoding="utf-8")
    out = re.sub(r"mmcv_maximum_version = '2\.2\.0'", "mmcv_maximum_version = '2.3.0'", s)
    if out != s:
        p.write_text(out, encoding="utf-8")
        print(f"  {p}: mmcv_maximum_version -> 2.3.0")
    py("-c", "import mmcv, mmdet, mmengine; from mmcv.ops import nms; "
             "print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, "
             "'mmengine', mmengine.__version__)")


def step_mask2former() -> None:
    """Submodule đi theo git; chỉ op CUDA là phải biên dịch tại chỗ.

    make.sh của repo gốc gọi `setup.py install` mà setuptools mới đã bỏ, nên
    dùng pip build thẳng thư mục ops.
    """
    sh("git", "submodule", "update", "--init", M2F_DIR)
    sh("git", "-C", M2F_DIR, "log", "-1", "--format=Mask2Former @ %h (%ad)", "--date=short")
    pip("install", "--no-build-isolation", "--no-deps",
        f"{M2F_DIR}/mask2former/modeling/pixel_decoder/ops")
    py("-c", f"import sys; sys.path.insert(0, {M2F_DIR!r});"
             "from mask2former import add_maskformer2_config;"
             "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn;"
             "print('Mask2Former import OK')")


def step_weights() -> None:
    """Trọng số COCO theo configs/weights.yaml, kiểm sha256."""
    py("scripts/download_weights.py", "--group", "benchmark")


STEPS = [
    Step("check", "kiểm python, nvcc, GPU", step_check),
    Step("torch", "torch 2.4.1+cu121", step_torch),
    Step("requirements", "requirements.txt + gói repo + detectron2", step_requirements),
    Step("mmdet", "nới kiểm phiên bản mmcv, thử import", step_mmdet),
    Step("mask2former", "submodule + op MSDeformAttn", step_mask2former),
    Step("weights", "tải trọng số COCO", step_weights),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="liệt kê các bước rồi thoát")
    ap.add_argument("--only", default="", help="chỉ chạy các bước này, phẩy ngăn")
    ap.add_argument("--from", dest="start", default="", help="chạy từ bước này trở đi")
    ap.add_argument("--dry-run", action="store_true", help="in ra sẽ làm gì, không chạy")
    a = ap.parse_args(argv)

    names = [s.name for s in STEPS]
    if a.list:
        for s in STEPS:
            print(f"  {s.name:14s} {s.about}")
        return 0

    todo = list(STEPS)
    if a.start:
        if a.start not in names:
            raise SystemExit(f"không có bước {a.start!r}; có: {names}")
        todo = todo[names.index(a.start):]
    if a.only:
        want = {n.strip() for n in a.only.split(",") if n.strip()}
        bad = want - set(names)
        if bad:
            raise SystemExit(f"không có bước: {sorted(bad)}; có: {names}")
        todo = [s for s in todo if s.name in want]

    for i, s in enumerate(todo, 1):
        print(f"\n== [{i}/{len(todo)}] {s.name}: {s.about}", flush=True)
        if a.dry_run:
            continue
        try:
            s.fn()
        except subprocess.CalledProcessError as e:
            print(f"\n!! bước {s.name!r} lỗi (thoát {e.returncode}).")
            print(f"   Sửa xong chạy tiếp: python scripts/setup_env.py --from {s.name}")
            return 1
    if a.dry_run:
        print("\n(--dry-run: chưa chạy gì)")
        return 0
    print("\n== xong. Tiếp: python scripts/prepare_data.py all_v2.tar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
