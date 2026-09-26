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

MỘT env cho tất cả — cả annotator (app/) lẫn benchmark: torch 2.4.1+cu121
(mmcv chỉ có wheel dựng sẵn tới torch 2.4), ultralytics, detectron2 build từ
source, mmcv/mmdet, submodule Mask2Former với op MSDeformAttn biên dịch tại
chỗ, SAM 2.1, rồi trọng số COCO.

SAM 2 khai `torch>=2.5.1` nên không để chung requirements.txt được; bước
`sam2` cài nó với --no-deps. Máy chỉ chạy benchmark thì bỏ bước đó:
`--only check,torch,requirements,mmdet,mask2former,weights`.

CHƯA CHẠY THẬT trên máy Linux: lần đầu nên đi từng bước với --only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M2F_DIR = "benchmark/mask2former/upstream"
#: detectron2 ghim theo commit; build từ source, xem step_detectron2.
D2 = ("detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
      "@a2f4a8771ab77e8411c26b27f24f9489a28a2453")
#: SAM 2.1 ghim theo commit; cài riêng với --no-deps, xem step_sam2.
SAM2 = ("SAM-2 @ git+https://github.com/facebookresearch/sam2.git"
        "@2b90b9f5ceec907a1c18123530e92e794ad901a4")
#: CUDA của torch trong env này. Bị khoá ở 12.1 vì mmcv (SOLOv2) chỉ có wheel
#: dựng sẵn cho torch 2.4 / cu121 — index cu124 và torch2.5 đều không tồn tại.
#: Đổi ở đây thì phải đổi cả --find-links trong requirements.txt.
CUDA_TAG = "12.1"        # dùng để so với nvcc
CUDA_FULL = "12.1.1"     # tên nhãn kênh conda nvidia
TORCH = ["torch==2.4.1", "torchvision==0.19.1",
         "--index-url", f"https://download.pytorch.org/whl/cu{CUDA_TAG.replace('.', '')}"]


#: --skip-cuda-build: lúc CÀI thì không biên dịch nhân CUDA tự viết nào, để
#: khỏi cần nvcc khớp phiên bản với torch. KHÔNG liên quan tới lúc train —
#: train vẫn dùng GPU đầy đủ. Xem install_without_cuda_build().
SKIP_CUDA_BUILD = False


class Step:
    def __init__(self, name: str, about: str, fn):
        self.name, self.about, self.fn = name, about, fn


def install_without_cuda_build() -> dict:
    """Biến môi trường cho LỆNH PIP, để không gói nào biên dịch nhân CUDA.

    Chỉ áp cho tiến trình pip. Lúc train không ai đặt mấy biến này, nên GPU
    vẫn được dùng bình thường — cờ này không hề tắt GPU.

    CUDA_VISIBLE_DEVICES="" là cái đòn bẩy: setup.py của detectron2 chọn
    CUDAExtension khi `torch.cuda.is_available() and CUDA_HOME is not None`.
    Giấu nvcc khỏi PATH KHÔNG đủ — torch vẫn đoán ra /usr/local/cuda. Giấu GPU
    thì `is_available()` trả False và detectron2 tự dựng CppExtension, không
    báo lỗi. Biến này chỉ có tác dụng lúc CÀI; lúc train GPU vẫn bình thường.

    SAM2_BUILD_CUDA=0 bỏ op hậu xử lý của SAM 2 (setup.py của nó vốn đã cho
    phép build hỏng, đây chỉ là nói thẳng ra).
    """
    import os

    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SAM2_BUILD_CUDA": "0"}


def sh(*cmd: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, env=env)


def py(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return sh(sys.executable, *args, env=env)


def pip(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return sh(sys.executable, "-m", "pip", *args, env=env)


# ---------------------------------------------------------------- các bước
def nvcc_version(text: str) -> tuple[int, int] | None:
    """'... Cuda compilation tools, release 13.2, V13.2.78 ...' -> (13, 2)."""
    import re

    m = re.search(r"release\s+(\d+)\.(\d+)", text)
    return (int(m[1]), int(m[2])) if m else None


def step_check() -> None:
    """Máy có đủ thứ để biên dịch không, và đang ở env nào.

    Kiểm cả PHIÊN BẢN nvcc chứ không chỉ sự tồn tại: ba gói trong env này
    biên dịch op CUDA tại chỗ và link vào header của torch, nên nvcc lệch
    major với torch là torch từ chối build. Trên máy lab đây đúng là chuyện
    đã xảy ra — nvcc hệ thống 13.2 còn torch là cu121.
    """
    print(f"  python {sys.version.split()[0]} tại {sys.executable}")
    if sys.version_info < (3, 12):
        raise SystemExit("Cần Python >= 3.12 (scipy/scikit-image ghim trong requirements.txt).\n"
                         "    conda create -y -n cofseg python=3.12 && conda activate cofseg")
    if SKIP_CUDA_BUILD:
        print("  --skip-cuda-build: bỏ qua nvcc, không biên dịch nhân CUDA tự viết nào.")
        print("    TRAIN VẪN CHẠY TRÊN GPU — cờ này chỉ tác động lúc cài.")
        print("    Mask R-CNN, SOLOv2, YOLO: không đổi gì (không gọi op tự biên dịch).")
        print("    Mask2Former: MSDeformAttn dùng op PyTorch thường, vẫn trên GPU,")
        print("      đúng kết quả, chậm hơn ~1,3-1,8 lần.")
        sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
           "--format=csv,noheader", check=False)
        return
    if not shutil.which("nvcc"):
        raise SystemExit(
            "Thiếu nvcc (CUDA toolkit): detectron2, SAM 2 và MSDeformAttn đều biên dịch\n"
            f"CUDA lúc cài. Hoặc cài bản khớp torch vào chính env này:\n\n"
            f"    conda install -y -c nvidia/label/cuda-{CUDA_FULL} cuda-toolkit\n\n"
            "hoặc cài mà không biên dịch gì:\n\n"
            "    python scripts/setup_env.py --skip-cuda-build\n")

    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(out.stdout.strip())
    got = nvcc_version(out.stdout)
    want = tuple(int(x) for x in CUDA_TAG.split("."))
    if got and got[0] != want[0]:
        raise SystemExit(
            f"\nnvcc là CUDA {got[0]}.{got[1]} nhưng torch của env này dựng bằng CUDA {CUDA_TAG}.\n"
            "torch từ chối build op CUDA khi lệch major:\n\n"
            f"    RuntimeError: The detected CUDA version ({got[0]}.{got[1]}) mismatches the\n"
            f"    version that was used to compile PyTorch ({CUDA_TAG}).\n\n"
            f"Driver thì không sao — nó tương thích ngược, chạy binary cu{CUDA_TAG.replace('.', '')}\n"
            "bình thường. Chỉ khâu biên dịch cần toolkit khớp. Cài vào CHÍNH env này,\n"
            "nó sẽ che nvcc của hệ thống:\n\n"
            f"    conda install -y -c nvidia/label/cuda-{CUDA_FULL} cuda-toolkit\n"
            "    which nvcc && nvcc --version        # phải trỏ vào env và ra "
            f"{CUDA_TAG}\n\n"
            "Không muốn cài thêm gì thì bỏ hẳn phần biên dịch (train vẫn dùng GPU):\n\n"
            "    python scripts/setup_env.py --skip-cuda-build\n\n"
            f"Vì sao không nâng torch cho khớp CUDA {got[0]}: mmcv (SOLOv2) chỉ có wheel dựng\n"
            f"sẵn cho torch 2.4 / cu{CUDA_TAG.replace('.', '')}; index cho CUDA mới hơn không tồn tại.")
    if got and got != want:
        print(f"  (nvcc {got[0]}.{got[1]} vs torch cu{CUDA_TAG.replace('.', '')} — lệch minor, "
              "thường build được)")
    sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
       "--format=csv,noheader", check=False)


def step_torch() -> None:
    """torch trước requirements: detectron2 và SAM 2 import torch lúc build."""
    pip("install", *TORCH)
    py("-c", "import torch; assert torch.cuda.is_available(), 'torch không thấy CUDA'; "
             "print('torch', torch.__version__, torch.cuda.get_device_name(0))")


def step_requirements() -> None:
    """Thuần wheel dựng sẵn — không gói nào biên dịch, chạy được ở mọi máy."""
    pip("install", "-r", "requirements.txt")
    pip("install", "-e", ".")
    py("-c", "import torch, ultralytics, mmdet; print('torch', torch.__version__, "
             "'| ultralytics', ultralytics.__version__, '| mmdet', mmdet.__version__)")


def step_detectron2() -> None:
    """Mask R-CNN và Mask2Former. Gói duy nhất build từ source (~5-10 phút).

    --no-build-isolation vì setup.py của nó import torch, mà môi trường build
    cô lập của pip không có.

    Với --skip-cuda-build thì giấu GPU lúc cài: setup.py chọn CUDAExtension khi
    `torch.cuda.is_available() and CUDA_HOME is not None`, không thấy GPU thì
    nó tự dựng CppExtension và KHÔNG báo lỗi. Mask R-CNN không gọi op CUDA
    riêng nào của detectron2 — ROIAlign và NMS lấy của torchvision — nên vẫn
    train trên GPU như thường.
    """
    env = install_without_cuda_build() if SKIP_CUDA_BUILD else None
    if SKIP_CUDA_BUILD:
        print('  --skip-cuda-build: CUDA_VISIBLE_DEVICES="" lúc cài -> dựng CppExtension.')
        print("    Chỉ có tác dụng lúc CÀI; lúc train GPU vẫn thấy đủ.")
    pip("install", "--no-build-isolation", D2, env=env)
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
    if SKIP_CUDA_BUILD:
        # ms_deform_attn.py bọc lời gọi op trong try/except và rơi về
        # ms_deform_attn_core_pytorch, nên không biên dịch vẫn ra ĐÚNG kết quả,
        # chỉ chậm hơn. Bỏ hẳn bước build op.
        print("  --skip-cuda-build: không build op MSDeformAttn, dùng đường PyTorch.")
    else:
        pip("install", "--no-build-isolation", "--no-deps",
            f"{M2F_DIR}/mask2former/modeling/pixel_decoder/ops")
    py("-c", f"import sys; sys.path.insert(0, {M2F_DIR!r});"
             "from mask2former import add_maskformer2_config;"
             "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn;"
             "print('Mask2Former import OK')")


def step_sam2() -> None:
    """SAM 2.1 cho annotator — cài bỏ qua mốc torch của nó.

    setup.py của SAM 2 khai `torch>=2.5.1`, nhưng env này ở 2.4.1 vì mmcv chỉ
    có wheel dựng sẵn cho torch 2.4. Mốc đó vào từ commit 11/12/2024 để
    `torch.compile` TOÀN MODEL cho nhánh video; app/annotator.py chỉ dùng
    build_sam2 + SAM2ImagePredictor (nhánh ảnh) và không gọi torch.compile,
    nên --no-deps là an toàn. Hai gói SAM 2 thật sự cần (hydra-core, iopath)
    đã ghim trong requirements.txt.

    Bước này chỉ cần nếu máy có chạy annotator; benchmark không import sam2.
    """
    pip("install", "--no-deps", "--no-build-isolation", SAM2,
        env=install_without_cuda_build() if SKIP_CUDA_BUILD else None)
    py("-c", "from sam2.build_sam import build_sam2;"
             "from sam2.sam2_image_predictor import SAM2ImagePredictor;"
             "print('SAM 2 import OK')")


def step_weights() -> None:
    """Trọng số COCO theo configs/weights.yaml, kiểm sha256."""
    py("scripts/download_weights.py", "--group", "benchmark")


STEPS = [
    Step("check", "kiểm python, nvcc, GPU", step_check),
    Step("torch", "torch 2.4.1+cu121", step_torch),
    Step("requirements", "requirements.txt + gói repo (thuần wheel)", step_requirements),
    Step("detectron2", "build detectron2 từ source (Mask R-CNN, Mask2Former)", step_detectron2),
    Step("mmdet", "nới kiểm phiên bản mmcv, thử import", step_mmdet),
    Step("mask2former", "submodule + op MSDeformAttn", step_mask2former),
    Step("sam2", "SAM 2.1 cho annotator (bỏ qua mốc torch của nó)", step_sam2),
    Step("weights", "tải trọng số COCO", step_weights),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="liệt kê các bước rồi thoát")
    ap.add_argument("--only", default="", help="chỉ chạy các bước này, phẩy ngăn")
    ap.add_argument("--from", dest="start", default="", help="chạy từ bước này trở đi")
    ap.add_argument("--dry-run", action="store_true", help="in ra sẽ làm gì, không chạy")
    ap.add_argument("--skip-cuda-build", action="store_true",
                    help="lúc CÀI không biên dịch nhân CUDA tự viết nào, nên không cần "
                         "nvcc khớp phiên bản. TRAIN VẪN DÙNG GPU như thường; chỉ "
                         "Mask2Former chậm hơn ~1,3-1,8 lần, ba model kia không đổi")
    a = ap.parse_args(argv)

    global SKIP_CUDA_BUILD
    SKIP_CUDA_BUILD = a.skip_cuda_build

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
