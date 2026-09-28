"""Dựng môi trường trên máy Linux có GPU (máy lab / máy thuê).

    python scripts/setup_env.py --models yolo11,maskrcnn,mask2former
    python scripts/setup_env.py --models solov2      # env RIÊNG, xem bên dưới
    python scripts/setup_env.py --dry-run            # in ra sẽ làm gì, không chạy
    python scripts/setup_env.py --from mmdet         # chạy lại từ một bước
    python scripts/setup_env.py --list

Từng bước một, dừng ngay khi một bước lỗi và in đúng lệnh đã chạy, để lần đầu
dựng trên máy mới còn biết hỏng ở đâu. Chạy TRONG env đã kích hoạt: script
không tự tạo conda env (kích hoạt env từ bên trong một tiến trình Python
không có tác dụng ra ngoài), nó kiểm tra và bảo bạn cần gõ gì.

`--models` quyết định cài gói của model nào; rỗng thì chỉ app/ + canopyseg.
Mỗi thư mục benchmark khai gói riêng vì KHÔNG còn gộp chung được nữa:

    mmcv (SOLOv2)          chỉ có wheel tới torch 2.4 / cu121
    GPU đời Blackwell      đòi torch >= 2.7 (cu121 không có kernel sm_120)

Hai mốc đó không giao nhau. Trên GPU đời TRƯỚC Blackwell — 4090, L4, L40S,
A40, A6000, A100, H100 — thì cả bốn model vẫn cài chung một env được, vì mọi
file đều ghim torch 2.4.1+cu121. Chỉ khi chạy trên Blackwell mới buộc tách
SOLOv2 ra env riêng.

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
#: torch theo CUDA. MỘT chỗ ghim cho cả bốn model — bản torch phụ thuộc kiến
#: trúc GPU chứ không phụ thuộc kiến trúc mạng, nên nó không nằm trong file
#: requirements của từng model. Chọn bằng --cuda.
TORCH = {
    # sm_50..sm_90: T4, V100, RTX 20/30/40, L4, L40S, A40, A6000, A100, H100.
    # Tổ hợp DUY NHẤT mmcv còn phát hành wheel, nên SOLOv2 không phải build gì.
    "121": ["torch==2.4.1+cu121", "torchvision==0.19.1+cu121"],
    # sm_75..sm_120, gồm RTX 50xx / RTX PRO 6000 / B200. CUDA 13 bỏ sm_50-sm_70.
    # 2.11 vì đó là bản DUY NHẤT ta có bằng chứng của chính mình: detectron2
    # 0.6 build và import được với torch 2.11 + Python 3.13 trên molab. Cặp
    # hợp lệ khác trên cu130, nếu muốn đi cao hơn:
    #   2.9.0/0.24.0  2.9.1/0.24.1  2.10.0/0.25.0  2.12.0/0.27.0
    #   2.12.1/0.27.1 2.13.0/0.28.0 2.14.0/0.29.0
    # ĐỔI SANG ĐÂY LÀ mmcv MẤT WHEEL: phải --build-mmcv.
    "130": ["torch==2.11.0+cu130", "torchvision==0.26.0+cu130"],
}
#: cuXXX -> nhãn kênh conda nvidia, để thông báo lỗi đưa đúng lệnh cài.
CONDA_LABEL = {"12.1": "12.1.1", "13.0": "13.0.3"}

#: Bản CUDA đang chọn, đặt bởi --cuda. Mặc định CUDA 13: nó phủ Turing tới
#: Blackwell, còn cu121 không chạy được trên RTX 50xx.
CUDA = "130"


def torch_cuda_tag() -> str:
    """'121' -> '12.1'."""
    return f"{CUDA[:-1]}.{CUDA[-1]}"


#: --skip-cuda-build: lúc CÀI thì không biên dịch nhân CUDA tự viết nào, để
#: khỏi cần nvcc khớp phiên bản với torch. KHÔNG liên quan tới lúc train —
#: train vẫn dùng GPU đầy đủ. Xem install_without_cuda_build().
SKIP_CUDA_BUILD = False

#: Model được cài trong lượt này. Mặc định rỗng = chỉ app/ + canopyseg.
MODELS: list[str] = []

#: True = build mmcv từ nguồn thay vì lấy wheel. Bật mặc định vì --cuda mặc
#: định là 130, mà ở đó không có wheel nào. Xem step_mmcv.
BUILD_MMCV = True


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
    cuda_tag = torch_cuda_tag()        # theo --cuda
    cuda_full = CONDA_LABEL.get(cuda_tag, cuda_tag)
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
            "Thiếu nvcc. Cách nhanh nhất là bỏ hẳn phần biên dịch — train vẫn dùng GPU:\n\n"
            "    python scripts/setup_env.py --skip-cuda-build\n\n"
            "Muốn Mask2Former đủ tốc độ thì cần CẢ toolkit đầy đủ LẪN g++ <= 12:\n\n"
            f"    conda install -y -c nvidia/label/cuda-{cuda_full} cuda-toolkit\n"
            "    conda install -y -c conda-forge gxx_linux-64=12\n"
            "    export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\n\n"
            "(cuda-nvcc một mình KHÔNG đủ: header của torch cần cusparse.h, cublas_v2.h\n"
            " từ các gói lib*-dev mà chỉ metapackage cuda-toolkit mới kéo đủ.)\n")

    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print(out.stdout.strip())
    got = nvcc_version(out.stdout)
    want = tuple(int(x) for x in cuda_tag.split("."))
    if got and got[0] != want[0]:
        raise SystemExit(
            f"\nnvcc là CUDA {got[0]}.{got[1]} nhưng torch của env này dựng bằng CUDA {cuda_tag}.\n"
            "torch từ chối build op CUDA khi lệch major:\n\n"
            f"    RuntimeError: The detected CUDA version ({got[0]}.{got[1]}) mismatches the\n"
            f"    version that was used to compile PyTorch ({cuda_tag}).\n\n"
            f"Driver thì không sao — nó tương thích ngược, chạy binary cu{cuda_tag.replace('.', '')}\n"
            "bình thường. Chỉ khâu biên dịch cần toolkit khớp. Cài vào CHÍNH env này,\n"
            "nó sẽ che nvcc của hệ thống:\n\n"
            f"    conda install -y -c nvidia/label/cuda-{cuda_full} cuda-toolkit\n"
            "    which nvcc && nvcc --version        # phải trỏ vào env và ra "
            f"{cuda_tag}\n\n"
            "Cách nhanh nhất: bỏ hẳn phần biên dịch. Train vẫn dùng GPU, chỉ\n"
            "Mask2Former chậm hơn:\n\n"
            "    python scripts/setup_env.py --skip-cuda-build\n\n"
            "Muốn biên dịch thật thì cần cả toolkit ĐẦY ĐỦ lẫn g++ <= 12 —\n"
            "cuda-nvcc một mình không đủ (thiếu cusparse.h, cublas_v2.h):\n\n"
            f"Vì sao không nâng torch cho khớp CUDA {got[0]}: mmcv (SOLOv2) chỉ có wheel dựng\n"
            f"sẵn cho torch 2.4 / cu{cuda_tag.replace('.', '')}; index cho CUDA mới hơn không tồn tại.")
    if got and got != want:
        print(f"  (nvcc {got[0]}.{got[1]} vs torch cu{cuda_tag.replace('.', '')} — lệch minor, "
              "thường build được)")
    sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
       "--format=csv,noheader", check=False)


def step_torch() -> None:
    """torch trước requirements: detectron2 và SAM 2 import torch lúc build.

    Phiên bản ghim ở TORCH đầu file, MỘT chỗ cho cả bốn model — torch phụ
    thuộc GPU chứ không phụ thuộc model, nên nó không nằm trong file
    requirements của từng model. Chọn bằng --cuda 121 (mặc định) hoặc 130.
    """
    pip("install", *TORCH[CUDA], "--index-url",
        f"https://download.pytorch.org/whl/cu{CUDA}")
    py("-c", "import torch; assert torch.cuda.is_available(), 'torch không thấy CUDA'; "
             "print('torch', torch.__version__, torch.cuda.get_device_name(0))")


#: Mỗi thư mục benchmark khai gói riêng của model đó. Không file nào ghim
#: torch — nó cài trước bằng bước `torch`, nên cài bốn file vào MỘT env hay
#: vào BỐN env riêng đều ra cùng một bản torch.
BENCH_REQS = {
    "yolo11": "benchmark/yolo11/requirements.txt",
    "solov2": "benchmark/solov2/requirements.txt",
    "maskrcnn": "benchmark/maskrcnn/requirements.txt",
    "mask2former": "benchmark/mask2former/requirements.txt",
}


def step_requirements() -> None:
    """app/ + canopyseg, rồi gói của các model được chọn qua --models.

    Thuần wheel dựng sẵn — không gói nào biên dịch, chạy được ở mọi máy.
    """
    pip("install", "-r", "requirements.txt")
    pip("install", "-e", ".")
    for name in MODELS:
        print(f"  gói riêng của {name}")
        pip("install", "-r", BENCH_REQS[name])
    # KHÔNG import mmdet ở đây: mmdet 3.3.0 tự chặn khi thấy mmcv 2.2.0, và
    # chỗ nới dòng kiểm đó là bước `mmdet` NGAY SAU. Kiểm sớm một bước là báo
    # lỗi cho một thứ chưa tới lượt được sửa.
    py("-c", "import torch; print('torch', torch.__version__)")


def step_detectron2() -> None:
    """Mask R-CNN và Mask2Former. Gói duy nhất build từ source (~5-10 phút).

    --no-build-isolation vì setup.py của nó import torch, mà môi trường build
    cô lập của pip không có.

    Trên máy không biên dịch CUDA được, --skip-cuda-build né được cả hai lỗi
    hay gặp: nvcc không được gọi nên luật "g++ <= 12" của nó không áp, và
    không file .cu nào được dịch nên không cần cusparse.h/cublas_v2.h.

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
    # Kiểm cả model_zoo chứ không chỉ `import detectron2`: model_zoo là chỗ
    # duy nhất còn `import pkg_resources`, và `import detectron2` KHÔNG kéo
    # nó theo. Vì vậy bước này từng báo xanh còn lần train đầu mới chết, sau
    # khi đã nạp xong dữ liệu. Nếu dòng này gãy vì pkg_resources thì env có
    # setuptools >= 82: `pip install "setuptools<82"` (đã ghim trong
    # requirements.txt, chỉ env cài từ trước mới thiếu).
    py("-c", "import detectron2; from detectron2 import model_zoo; "
             "print('detectron2', detectron2.__version__)")


#: Chỉ mục wheel mmcv. Chỉ tồn tại cho cu118/cu121 tới torch 2.4 — dò trực
#: tiếp thì mọi tổ hợp cu124/cu128/cu130 và torch2.5+ đều trả 404.
MMCV_INDEX = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html"
MMCV_REPO = "https://github.com/open-mmlab/mmcv.git"
MMCV_TAG = "v2.2.0"


def gpu_arch() -> tuple[int, int] | None:
    """(major, minor) của GPU 0, hoặc None nếu không hỏi được."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import torch;print('%d %d' % torch.cuda.get_device_capability(0))"],
        capture_output=True, text=True)
    try:
        a, b = out.stdout.split()
        return int(a), int(b)
    except ValueError:
        return None


def step_mmcv() -> None:
    """Wheel dựng sẵn hay build từ nguồn — tuỳ MÁY, nên không để trong requirements.

    Cùng lý do detectron2 không nằm trong file của Mask R-CNN: một dòng
    requirements không rẽ nhánh theo GPU được.

    Blackwell (sm_120) không có đường wheel: torch 2.4/cu121 là tổ hợp duy
    nhất mmcv còn phát hành, mà nó ra đời TRƯỚC Blackwell nên không có kernel
    cho kiến trúc đó. Bước này dừng và nói ra thay vì cài một thứ chắc chắn
    chết ở lời gọi kernel đầu tiên.
    """
    arch = gpu_arch()
    if arch is not None:
        print(f"  GPU sm_{arch[0]}{arch[1]}")
    blackwell = arch is not None and arch[0] >= 10
    if blackwell and not BUILD_MMCV:
        raise SystemExit(
            f"\nGPU là sm_{arch[0]}{arch[1]} (Blackwell) — mmcv KHÔNG có wheel chạy được.\n\n"
            "torch 2.4/cu121 là tổ hợp duy nhất OpenMMLab còn phát hành, và nó ra\n"
            "trước Blackwell nên thiếu kernel sm_120. Cài vào thì chết ở lời gọi\n"
            "kernel đầu tiên: 'no kernel image is available for execution'.\n\n"
            "Hai lối đi:\n\n"
            "  1. Đổi sang GPU đời trước Blackwell (A100, L40S, A6000, 4090...)\n"
            "     — mmcv dùng wheel, không biên dịch gì.\n\n"
            "  2. Cài lại torch với --cuda 130 (CUDA 13) rồi build:\n"
            "         python scripts/setup_env.py --only mmcv --build-mmcv\n"
            "     Mất 20-120 phút. Có người báo làm được ở torch 2.7 + CUDA 12.8\n"
            "     (mmcv#3327), nhưng upstream đứng yên từ 04/2024 nên không ai\n"
            "     bảo đảm. Nhớ lấy mmengine từ git: torch >= 2.6 đổi mặc định\n"
            "     torch.load(weights_only=True) và bản 0.10.7 trên PyPI chưa vá.")
    if not BUILD_MMCV:
        pip("install", "mmcv==2.2.0", "-f", MMCV_INDEX)
    else:
        import os

        src = ROOT / "build" / "mmcv"
        if not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            sh("git", "clone", "--branch", MMCV_TAG, "--depth", "1", MMCV_REPO, str(src))
        # FORCE_CUDA: setup.py của mmcv bật op CUDA khi
        # `torch.cuda.is_available() or os.getenv('FORCE_CUDA') == '1'`, nên
        # cờ này cho phép biên dịch trên máy không có GPU — tách được câu hỏi
        # "compile nổi không" khỏi việc phải thuê đúng card.
        # TORCH_CUDA_ARCH_LIST: mặc định torch build cho 6-7 kiến trúc, mỗi
        # file .cu compile lại từng ấy lần. Ghim đúng cái cần là nhanh hơn
        # nhiều lần.
        extra = {"FORCE_CUDA": "1", "MMCV_WITH_OPS": "1",
                 "MAX_JOBS": os.environ.get("MAX_JOBS", "4")}
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and arch is not None:
            extra["TORCH_CUDA_ARCH_LIST"] = f"{arch[0]}.{arch[1]}"
        print("  build mmcv %s: %s" % (MMCV_TAG, extra))
        pip("install", "--no-build-isolation", "-e", str(src),
            env={**os.environ, **extra})
    py("-c", "import mmcv; from mmcv.ops import nms; print('mmcv', mmcv.__version__)")


def step_mmdet() -> None:
    """mmdet 3.3.0 khai mmcv < 2.2.0, nhưng wheel dựng sẵn cho torch 2.4 là
    2.2.0 và chạy được: nới đúng dòng kiểm phiên bản đó.

    Tìm file bằng find_spec chứ KHÔNG `import mmdet`: chính dòng assert ta sắp
    gỡ nằm trong __init__.py, nên import nó là gãy trước khi kịp sửa.
    """
    import importlib.util
    import re

    spec = importlib.util.find_spec("mmdet")
    if spec is None or not spec.origin:
        raise SystemExit("Không thấy gói mmdet. Chạy lại: "
                         "python scripts/setup_env.py --from requirements")
    p = Path(spec.origin)
    s = p.read_text(encoding="utf-8")
    out = re.sub(r"mmcv_maximum_version = '2\.2\.0'", "mmcv_maximum_version = '2.3.0'", s)
    if out != s:
        p.write_text(out, encoding="utf-8")
        print(f"  {p}: mmcv_maximum_version -> 2.3.0")
    py("-c", "import mmcv, mmdet, mmengine; from mmcv.ops import nms; "
             "print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, "
             "'mmengine', mmengine.__version__)")


def unguard_msdeformattn() -> None:
    """Cho phép nạp MSDeformAttn khi op CUDA chưa biên dịch.

    ms_deform_attn_func.py NÉM LỖI ngay lúc import nếu thiếu op:

        raise ModuleNotFoundError("Please compile MultiScaleDeformableAttention…")

    Trong khi ms_deform_attn.py lại đã bọc lời gọi op trong try/except và rơi
    về ms_deform_attn_core_pytorch — đường lùi có sẵn nhưng không bao giờ tới
    lượt, vì import chết trước. Đổi dòng raise thành `MSDA = None` là đường
    lùi đó chạy.

    Đã thử: forward và backward đều ra đúng hình, gradient hữu hạn và khác 0.
    Chậm hơn vì không có nhân gộp sẵn, nhưng vẫn trên GPU và vẫn đúng kết quả.
    """
    f = ROOT / M2F_DIR / "mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py"
    s = f.read_text(encoding="utf-8")
    if "MSDA = None" in s:
        print(f"  {f.name}: đã vá từ trước, bỏ qua")
        return
    old = "    raise ModuleNotFoundError(info_string)"
    if old not in s:
        raise SystemExit(
            f"Không thấy dòng cần vá trong {f}.\n"
            "Submodule Mask2Former có thể đã đổi. Hoặc build op thật:\n"
            "    python scripts/setup_env.py --only mask2former   (bỏ --skip-cuda-build)")
    f.write_text(s.replace(old, "    MSDA = None", 1), encoding="utf-8")
    print(f"  {f.name}: raise -> MSDA = None, dùng đường PyTorch")
    print("    (hoàn tác: git -C %s checkout .)" % M2F_DIR)


def step_mask2former() -> None:
    """Submodule đi theo git; chỉ op CUDA là phải biên dịch tại chỗ.

    make.sh của repo gốc gọi `setup.py install` mà setuptools mới đã bỏ, nên
    dùng pip build thẳng thư mục ops. Với --skip-cuda-build thì không build gì,
    chỉ nới chốt chặn lúc import (xem unguard_msdeformattn).
    """
    sh("git", "submodule", "update", "--init", M2F_DIR)
    sh("git", "-C", M2F_DIR, "log", "-1", "--format=Mask2Former @ %h (%ad)", "--date=short")
    if SKIP_CUDA_BUILD:
        unguard_msdeformattn()
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
    Step("torch", "torch theo --cuda (mặc định 121)", step_torch),
    Step("requirements", "app/ + canopyseg + gói của --models (thuần wheel)", step_requirements),
    Step("detectron2", "build detectron2 từ source (Mask R-CNN, Mask2Former)", step_detectron2),
    Step("mmcv", "mmcv: wheel dựng sẵn, hoặc --build-mmcv để build từ nguồn", step_mmcv),
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
    ap.add_argument("--models", default="",
                    help="model cần cài, phẩy ngăn: " + ",".join(BENCH_REQS)
                         + ". Rỗng = chỉ app/ + canopyseg. LƯU Ý: solov2 khoá ở "
                           "torch 2.4/cu121 nên KHÔNG cài chung env với ba model "
                           "kia được nếu GPU là đời Blackwell")
    ap.add_argument("--build-mmcv", action="store_true",
                    help="build mmcv từ nguồn thay vì lấy wheel. Bắt buộc trên GPU "
                         "Blackwell (sm_120): wheel duy nhất của OpenMMLab là "
                         "torch 2.4/cu121, ra đời trước kiến trúc đó. Mất 20-120 phút")
    ap.add_argument("--cuda", default="130", choices=sorted(TORCH),
                    help="chỉ mục CUDA của torch. 130 (mặc định) = CUDA 13, phủ "
                         "Turing tới Blackwell, mmcv phải --build-mmcv. "
                         "121 = chỉ GPU đời trước Blackwell, nhưng mmcv có wheel")
    a = ap.parse_args(argv)

    global SKIP_CUDA_BUILD, MODELS, BUILD_MMCV, CUDA
    CUDA = a.cuda
    BUILD_MMCV = a.build_mmcv or a.cuda != "121"
    SKIP_CUDA_BUILD = a.skip_cuda_build
    MODELS = [n.strip() for n in a.models.split(",") if n.strip()]
    bad = set(MODELS) - set(BENCH_REQS)
    if bad:
        raise SystemExit(f"không có model: {sorted(bad)}; có: {sorted(BENCH_REQS)}")

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
