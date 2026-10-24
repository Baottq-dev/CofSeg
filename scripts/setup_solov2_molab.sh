#!/bin/bash
set -e

echo "=========================================================="
echo " [1/6] KHỞI TẠO MÔI TRƯỜNG PYTHON 3.12 VỚI UV"
echo "=========================================================="
uv python install 3.12
uv venv /venv/cofseg --python 3.12
source /venv/cofseg/bin/activate
uv pip install --python /venv/cofseg/bin/python pip

echo "=========================================================="
echo " [2/6] CÀI ĐẶT PYTORCH 2.11 CU130, CUDA TOOLKIT VÀ NINJA"
echo "=========================================================="
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install "cuda-toolkit[nvcc,nvvm,crt,cccl]==13.0.2" ninja

echo "=========================================================="
echo " [3/6] CẤU HÌNH BIẾN MÔI TRƯỜNG & VÁ LỖI CCCL (<nv/target>)"
echo "=========================================================="
# 1. Xác định CUDA_HOME chuẩn
CUDA_HOME=$(python -c "import sysconfig,os; p=os.path.join(sysconfig.get_paths()['purelib'],'nvidia','cu13'); print(p if os.path.isdir(p) else '/usr/local/lib/python3.13/site-packages/nvidia/cu13')")
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")

# 2. Tạo symlink libcudart.so cho linker
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so" 2>/dev/null || true

# 3. VÁ TRIỆT ĐỂ LỖI <nv/target>:
# cuda_fp16.h của CUDA 13 include <nv/target>, nhưng CCCL được đặt trong package riêng (cuda_cccl).
# Ta copy/link toàn bộ thư mục include của cuda_cccl trực tiếp vào $CUDA_HOME/include để nvcc luôn tìm thấy.
CCCL_DIRS=$(python -c "
import glob
dirs = glob.glob('/venv/cofseg/lib/python*/site-packages/nvidia/cuda_cccl/include') + \
       glob.glob('/usr/local/lib/python*/site-packages/nvidia/cuda_cccl/include')
print(' '.join(dirs))
")

for d in $CCCL_DIRS; do
    if [ -d "$d" ]; then
        echo ">>> Tìm thấy CCCL include tại: $d -> đồng bộ vào $CUDA_HOME/include..."
        cp -rn "$d"/* "$CUDA_HOME/include/" 2>/dev/null || true
    fi
done

export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"

# 4. Ghi vĩnh viễn cấu hình vào /root/.bashrc và /venv/cofseg/bin/activate
# Giúp mở terminal mới không bao giờ bị mất biến môi trường
cat << ENV_EOF >> /root/.bashrc
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include\${CPATH:+:\$CPATH}"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
source /venv/cofseg/bin/activate 2>/dev/null || true
ENV_EOF

cat << ENV_EOF >> /venv/cofseg/bin/activate
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include\${CPATH:+:\$CPATH}"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
ENV_EOF

echo "=========================================================="
echo " [4/6] CÀI REQUIREMENTS SOLOV2 & VÁ MMDETECTION"
echo "=========================================================="
cd /marimo/CofSeg 2>/dev/null || cd /workspace/CofSeg 2>/dev/null || cd CofSeg
pip install -r benchmark/solov2/requirements.txt
pip install "git+https://github.com/open-mmlab/mmengine"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"

echo "=========================================================="
echo " [5/6] BIÊN DỊCH MMCV 2.2.0 TỪ NGUỒN"
echo "=========================================================="
rm -rf /root/mmcv
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git /root/mmcv
cd /root/mmcv
rm -rf build mmcv/_ext*.so

FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=4 pip install --no-build-isolation -e .

echo "=========================================================="
echo " [6/6] KIỂM TRA MÔI TRƯỜNG GPU & MMCV"
echo "=========================================================="
python -c "
import inspect, os, torch, torchvision, mmcv, mmdet
import mmengine, mmengine.runner.checkpoint as ckpt
from torchvision.ops import nms as tv_nms
from mmcv.ops import nms as mmcv_nms
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s = torch.tensor([0.9,0.8]).cuda()
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms CUDA', tv_nms(b,s,0.5).tolist())
print('mmcv       ', mmcv.__version__, os.path.dirname(mmcv.__file__))
print('             nms CUDA', mmcv_nms(b,s,0.5)[1].tolist())
print('mmdet      ', mmdet.__version__, '| mmengine', mmengine.__version__,
      '| vá:', 'weights_only' in inspect.getsource(ckpt))"

echo "=========================================================="
echo " >>> HOÀN TẤT CÀI ĐẶT SOLOV2 TRÊN MOLAB THÀNH CÔNG 100%! <<<"
echo "=========================================================="
