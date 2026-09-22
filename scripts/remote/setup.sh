#!/usr/bin/env bash
# Dựng môi trường trên máy lab / máy thuê Linux (Ubuntu, driver CUDA 12.x, có nvcc).
# Chạy từ gốc repo:  bash scripts/remote/setup.sh
#
# Một env cho tất cả: torch 2.4.1+cu121 (mmcv chỉ có wheel tới 2.4), ultralytics
# (YOLO11), detectron2 build từ source (Mask R-CNN, Cascade, Mask2Former),
# mmcv/mmdet (SOLOv2), submodule Mask2Former trong third_party/ với op
# MSDeformAttn biên dịch tại chỗ. Mọi phiên bản ghim trong requirements.txt; dòng chỉ-Linux
# ở đó pip tự chọn theo hệ điều hành.
#
# CHƯA CHẠY THẬT trên máy Linux: kiểm từng khối bằng mắt lần đầu.
set -euo pipefail

ENV_NAME="${ENV_NAME:-cofseg}"
PY_VER="${PY_VER:-3.12}"        # scipy/scikit-image ghim trong requirements cần >= 3.12
M2F_DIR="third_party/Mask2Former"    # submodule, commit ghim trong .gitmodules/index

cd "$(dirname "$0")/../.."
echo "== repo: $(pwd)"
command -v nvcc >/dev/null || { echo "Thiếu nvcc (CUDA toolkit): detectron2, SAM 2 và MSDeformAttn cần biên dịch CUDA."; exit 1; }
nvcc --version | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- conda env -------------------------------------------------------------
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
python -V

# ---- torch trước (detectron2/SAM 2 import torch lúc build), rồi requirements --
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; assert torch.cuda.is_available(), 'torch không thấy CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
pip install -r requirements.txt --no-build-isolation
pip install -e .
python -c "import detectron2; print('detectron2', detectron2.__version__)"

# ---- mmdet 3.3.0 khai mmcv < 2.2.0; wheel dựng sẵn cho torch 2.4 là 2.2.0 và
# chạy được. Nới đúng một dòng kiểm phiên bản (cách được dùng rộng rãi).
python - <<'PY'
import re, mmdet, pathlib
p = pathlib.Path(mmdet.__file__)
s = p.read_text(encoding="utf-8")
s2 = re.sub(r"mmcv_maximum_version = '2\.2\.0'", "mmcv_maximum_version = '2.3.0'", s)
if s2 != s:
    p.write_text(s2, encoding="utf-8"); print("mmdet/__init__.py: mmcv_maximum_version -> 2.3.0")
PY
python -c "import mmcv, mmdet, mmengine; from mmcv.ops import nms; print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmengine', mmengine.__version__)"

# ---- Mask2Former: submodule + op CUDA ----------------------------------------
# Repo con đi theo git; chỉ cần init nếu clone không có --recurse-submodules.
git submodule update --init "$M2F_DIR"
git -C "$M2F_DIR" log -1 --format="Mask2Former @ %h (%ad)" --date=short
# make.sh gốc gọi `setup.py install` (setuptools mới đã bỏ); pip build tại chỗ.
pip install --no-build-isolation --no-deps "$M2F_DIR/mask2former/modeling/pixel_decoder/ops"
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
