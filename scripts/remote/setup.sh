#!/usr/bin/env bash
# Dựng môi trường trên máy thuê Linux (Ubuntu, driver CUDA 12.x, có nvcc).
# Chạy từ gốc repo:  bash scripts/remote/setup.sh
#
# Một env "cofseg" cho tất cả: torch 2.5.1+cu121 như máy nhà, ultralytics
# (YOLO11), detectron2 build từ source (Mask R-CNN, Cascade), repo Mask2Former
# clone vào third_party/ và op MSDeformAttn biên dịch tại chỗ. Tách env chỉ
# khi pip báo xung đột — tới giờ chưa thấy.
#
# CHƯA CHẠY THẬT trên máy thuê: kiểm từng bước bằng mắt lần đầu.
set -euo pipefail

ENV_NAME="${ENV_NAME:-cofseg}"
PY_VER="${PY_VER:-3.10}"
# Ghim commit sau lần khói đầu tiên để lần sau dựng lại y hệt.
D2_REF="${D2_REF:-main}"
M2F_REF="${M2F_REF:-main}"
M2F_DIR="third_party/Mask2Former"

cd "$(dirname "$0")/../.."
echo "== repo: $(pwd)"
command -v nvcc >/dev/null || { echo "Thiếu nvcc (CUDA toolkit): detectron2 và MSDeformAttn cần biên dịch CUDA."; exit 1; }
nvcc --version | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- conda env -------------------------------------------------------------
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
python -V

# ---- torch trước, đúng index cu121, rồi mới requirements (ghim +cu121) ------
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; assert torch.cuda.is_available(), 'torch không thấy CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# ---- detectron2 (build từ source, ~5-10 phút) --------------------------------
pip install "git+https://github.com/facebookresearch/detectron2.git@${D2_REF}"
python -c "import detectron2; print('detectron2', detectron2.__version__)"

# ---- Mask2Former: repo + phụ thuộc + op CUDA ---------------------------------
mkdir -p third_party
if [ ! -d "$M2F_DIR/.git" ]; then
  git clone https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR"
fi
git -C "$M2F_DIR" checkout -q "$M2F_REF"
pip install timm scipy shapely h5py scikit-image cython
( cd "$M2F_DIR/mask2former/modeling/pixel_decoder/ops" && sh make.sh )
python - <<'PY'
import sys; sys.path.insert(0, "third_party/Mask2Former")
from mask2former import add_maskformer2_config  # noqa
from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn  # noqa: op đã biên dịch
print("Mask2Former import OK")
PY

# ---- trọng số ----------------------------------------------------------------
mkdir -p weights
python - <<'PY'
from ultralytics import YOLO
YOLO("weights/yolo11s-seg.pt")           # tự tải nếu thiếu
print("yolo11s-seg.pt OK")
PY
# Checkpoint COCO của detectron2 / Mask2Former tự tải về cache lúc train
# (MODEL.WEIGHTS là URL). Kéo trước để lần train đầu không chờ mạng:
python - <<'PY'
from detectron2 import model_zoo
from detectron2.utils.file_io import PathManager
from canopyseg.models.detectron2 import ZOO
for arch, (cfg, url) in ZOO.items():
    if cfg is None:
        continue
    url = url or model_zoo.get_checkpoint_url(cfg)
    print(arch, "->", PathManager.get_local_path(url))
PY

echo "== xong. Tiếp: bash scripts/remote/prepare_data.sh all_v2.tar"
