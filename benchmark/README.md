# benchmark/ — bốn model, mỗi model một thư mục độc lập

Mỗi model của bảng so sánh nằm trọn trong một thư mục, với **code, config,
cách chạy và kết quả riêng**. Không thư mục nào import thư mục nào, nên sửa
một thư mục không làm thay đổi ba thư mục kia.

## Bốn model

| Thư mục | Model | Framework | Vai trò trong bảng |
|---|---|---|---|
| `maskrcnn/` | Mask R-CNN R50-FPN | detectron2 | **mốc số 0**: mọi model khác báo Δ% mAP so với nó |
| `solov2/` | SOLOv2 R50-FPN | mmdetection | box-free: lưới + kernel động, không box |
| `yolo/` | YOLO-Seg (v8 / 11 / 26) | ultralytics | một giai đoạn, thời gian thực |
| `mask2former/` | Mask2Former R50 | detectron2 + repo gốc | query / transformer, mask toàn ảnh |

Mask R-CNN, SOLOv2 và Mask2Former dùng chung backbone ResNet-50, nên chênh
lệch giữa ba model này đến từ cơ chế chứ không từ backbone.

## Một thư mục có gì

```
benchmark/<model>/
├─ README.md              model gì, vì sao chọn, cách dựng env, cách chạy
├─ requirements.txt       gói Python riêng của model (không ghim torch)
├─ train.py               huấn luyện, nạp cofseg/ của thư mục này
├─ evaluate.py            chấm, nạp cofseg/ của thư mục này
├─ cofseg/                BẢN SAO LÕI của riêng thư mục này
│   ├─ datasets/          đọc fold COCO
│   ├─ metrics/           mask AP, Boundary AP/IoU, sai số diện tích
│   ├─ evaluation/        vòng chấm
│   ├─ models/            wrapper suy luận của model này
│   └─ training/          trainer của model này
├─ configs/train/*.yaml   cấu hình huấn luyện
├─ configs/eval/*.yaml    cấu hình chấm
├─ tests/                 test của thư mục
├─ runs/                  (không vào git) kết quả train/eval
├─ results/               file kết quả nhỏ, ĐƯỢC commit
└─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
```

## Dựng môi trường

Linux, GPU NVIDIA, **torch `2.11.0+cu130`**, một env cho cả bốn model. CUDA
toolkit **không cài bằng conda**: dùng toolkit của máy nếu máy có bản 13.x,
còn không thì cài trình biên dịch vào thẳng env bằng pip. Không thay đổi gì
ngoài env.

Chỉ chạy một model thì theo mục *Dựng môi trường riêng* trong README của thư
mục đó, ngắn hơn.

### Hai máy đã dựng, khác nhau ở đúng một bước

| | máy 1 | máy 2 |
|---|---|---|
| GPU | RTX 4090 — `sm_89` | RTX 5090 — `sm_120` |
| `nvcc` của máy | **13.2**, dùng được | **12.8**, lệch major nên bỏ qua |
| driver | chạy được cu130 | 580.105.08 |
| toolkit lấy ở đâu | **Nhánh A** — `/usr/local/cuda` của máy | **Nhánh B** — pip vào env |
| kiến trúc (bước 7 tự hỏi card) | `8.9` | `12.0` |

Chỉ **bước 3** rẽ nhánh, mười bước còn lại giống hệt nhau.

Máy 2 **có** toolkit 12.8 đầy đủ ở `/usr/local/cuda` nhưng **không dùng
được**: 12.8 lệch major với `cu130` và torch chặn thẳng. Trên máy như vậy,
đừng dựa vào `which nvcc`.

### Vì sao một bản torch phủ được cả hai

Chốt phiên bản của torch nằm ở `torch/utils/cpp_extension.py:543-546`:

```python
if cuda_ver.major != torch_cuda_version.major:
    raise RuntimeError(CUDA_MISMATCH_MESSAGE, ...)   # lệch MAJOR -> dừng
logger.warning(CUDA_MISMATCH_WARN, ...)              # lệch MINOR -> chỉ cảnh báo
```

`torch 2.11.0+cu130` có `torch.version.cuda = 13.0`, nên:

- toolkit 13.0 (nhánh B): khớp hoàn toàn, không cảnh báo
- toolkit 13.2 (nhánh A): lệch minor, **chỉ cảnh báo**, vẫn build
- toolkit 12.8: lệch major, **dừng**; đây là lý do máy 2 phải đi nhánh B

Torch 2.11.0 cũng là bản của bản dựng `cu128` trước đó, nên mọi thứ đã kiểm
trên 2.11 vẫn còn giá trị. Chỉ mục `cu132` bắt đầu từ torch 2.12.0, tức phải
nâng torch mà không được thêm gì.

CUDA 13 bỏ `sm_50`–`sm_70` (Maxwell / Pascal / Volta): GTX 10xx và V100 không
chạy được bản dựng này.

---

### 0. Kiểm máy

```bash
nvcc --version | tail -2
g++ --version | head -1
nvidia-smi --query-gpu=name,driver_version --format=csv
ls -d /usr/local/cuda* 2>/dev/null
```

| | yêu cầu | nếu không đạt |
|---|---|---|
| driver | **≥ 580.65.06** (mốc tối thiểu của CUDA 13.0) | không chạy được CUDA 13 |
| `g++` | ≤ **15.x** (CUDA 13 nhận tới GCC 15) | cài `gxx` cũ hơn vào env |
| `nvcc` | 13.x → **nhánh A** | 12.x, hoặc không có → **nhánh B** |

`g++ 13.3` của Ubuntu 24.04 nằm trong khoảng cho phép.

### 1. Env

Env conda:

```bash
conda create -y -n cofseg python=3.12 && conda activate cofseg
```

Hoặc venv, nếu máy không có conda:

```bash
python3.12 -m venv /venv/cofseg && . /venv/cofseg/bin/activate
```

Một số ảnh máy thuê có sẵn một venv và dấu nhắc mặc định là `(main)`. **Tạo
env riêng**, không cài đè lên venv đó.

### 2. torch

```bash
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

Kiểm bằng cách **gọi op trên GPU**, không chỉ `import`: torchvision lệch bản
dựng vẫn import được rồi mới hỏng lúc gọi.

```bash
python -c "import torch, torchvision; from torchvision.ops import nms; b=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]).cuda(); s=torch.tensor([0.9,0.8]).cuda(); print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, '| nms', nms(b,s,0.5).tolist()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Phải in `2.11.0+cu130 13.0`, tên card, và `(8, 9)` trên 4090 / `(12, 0)` trên
5090.

Bước này đứng **trước** bước toolkit có chủ đích: torch `cu130` tự kéo theo
`cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2`,
tức phần **thư viện và header** của toolkit đã có sẵn trong env. Nhánh B chỉ
phải bù phần **trình biên dịch**.

### 3. `CUDA_HOME` — bước duy nhất rẽ nhánh

#### Nhánh A — máy có nvcc 13.x

```bash
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
echo "$CUDA_HOME"
ls $CUDA_HOME/bin/nvcc $CUDA_HOME/include/cusparse.h $CUDA_HOME/include/cublas_v2.h
```

Không tải gì, không sửa gì ngoài env.

#### Nhánh B — máy có nvcc 12.x hoặc không có nvcc

Cài bốn phần còn thiếu; chúng nằm trong **cùng cây** `nvidia/cu13/` với các
gói torch đã cài ở bước 2:

```bash
pip install "cuda-toolkit[nvcc,nvvm,crt,cccl]==13.0.2"

export CUDA_HOME=$(python -c "import sysconfig,os;print(os.path.join(sysconfig.get_paths()['purelib'],'nvidia','cu13'))")
ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so

ls $CUDA_HOME/bin/nvcc $CUDA_HOME/nvvm/bin/cicc \
   $CUDA_HOME/include/cusparse.h $CUDA_HOME/include/cublas_v2.h $CUDA_HOME/lib/libcudart.so
$CUDA_HOME/bin/nvcc --version | tail -2
```

Cây wheel có đủ `bin/ include/ lib/ nvvm/`, đúng cấu trúc của một
`CUDA_HOME`. Hai điểm cần biết:

- torch tìm thư viện ở `lib64`, còn wheel đặt ở `lib`. Không sao, vì
  `cpp_extension.py:1685-1690` tự lùi về `lib` khi không có `lib64`.
- wheel chỉ có `libcudart.so.13`, **thiếu** symlink `libcudart.so` mà
  `-lcudart` cần lúc liên kết, nên mới có dòng `ln -sf`.

Đường này chạy được **cả trong venv**, nơi không dùng được `conda install`.
Nó chỉ có cho CUDA 13: wheel `nvidia-cuda-nvcc-cu12` **không chứa `nvcc`**,
chỉ có `ptxas`, nên không thể cài toolkit 12.8 vào env bằng pip.

#### Cả hai nhánh: giữ biến qua mọi shell

Các lượt train chạy từ nhiều phiên shell khác nhau; thiếu biến này là build
hỏng giữa chừng.

```bash
# conda
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export CUDA_HOME=$CUDA_HOME" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh

# venv
echo "export CUDA_HOME=$CUDA_HOME" >> /venv/cofseg/bin/activate
```

### 4. Vá `libnvrtc-builtins` — Mask R-CNN và Mask2Former

Wheel PyTorch đóng gói thiếu một liên kết: `auditwheel` đổi tên file thành
`libnvrtc-builtins-<băm>.so.13.0`, trong khi `libnvrtc.so.13` vẫn tìm tên
gốc `libnvrtc-builtins.so.13.0`. Thiếu nó thì **Mask R-CNN và Mask2Former
dừng**, SOLOv2 và YOLO thì không: chỉ hai model này gọi op cần NVRTC biên
dịch tại chỗ.

Xem đang có gì trước. **Bước này chỉ cần khi tên file có băm**:

```bash
P=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
ls -l "$P"/nvidia/cu13/lib/libnvrtc*
```

- Có `libnvrtc-builtins.so.13.0` **không băm**: wheel đóng gói đúng, **bỏ qua
  bước 4**.
- Có `libnvrtc-builtins-<băm>.so.13.0`: chạy lệnh dưới.
- Không có gì, hoặc `ls` báo không có thư mục: torch chưa cài, quay lại bước 2.

```bash
P=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
S=$(find "$P/nvidia" -name "libnvrtc-builtins-*.so.13.*" | head -1)
if [ -z "$S" ]; then echo "khong tim thay ban co bam — bo qua buoc 4"; else
  ln -sfn "$(basename "$S")" "$(dirname "$S")/libnvrtc-builtins.so.${S##*.so.}"
  ls -l "$(dirname "$S")"/libnvrtc-builtins*
fi
```

Chỉ là một symlink trong env, `rm` là gỡ. Câu `if` cần thiết: không tìm thấy
file thì `$S` rỗng, và `ln` sẽ tạo symlink ở `/`.

Kiểm bằng đúng phép đã hỏng — `prod` trên GPU, phép mà
`detectron2/structures/boxes.py:330` gọi:

```bash
python -c "import torch; wh=torch.rand(8,16,2).cuda(); print('nvrtc OK', float(wh.prod(dim=2).sum()))"
```

In ra số là xong. Nếu báo `failed to open libnvrtc-builtins.so.13.0` thì thêm
thư mục thư viện vào `LD_LIBRARY_PATH`, và ghi dòng này vào file activate như
ở bước 3:

```bash
export LD_LIBRARY_PATH="$P/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

### 5. Gói Python của bốn model

```bash
pip install -r benchmark/requirements.txt
pip check
```

Không cần `pip install -e .`: mỗi thư mục model tự chứa bản `cofseg/` riêng.

Nếu env dùng chung với việc khác, `pip check` có thể báo vài gói không liên
quan (`jupytext`, `paramiko`…). Bỏ qua được, miễn không dòng nào nhắc `torch`,
`mmcv`, `mmdet`, `detectron2`.

### 6. detectron2 — Mask R-CNN và Mask2Former

```bash
pip install --no-build-isolation --no-cache-dir \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

python -c "import detectron2; from detectron2 import _C, model_zoo; print(detectron2.__version__, 'ext OK')"
```

- `--no-build-isolation` vì `setup.py` của detectron2 `import torch`.
- **`--no-cache-dir`** vì commit ghim là URL bất biến, nên pip **dùng lại
  wheel đã dựng lần trước**. Cài lại sau khi đổi torch mà thiếu cờ này thì pip
  báo "Successfully installed" nhưng lỗi cũ vẫn còn. Build thật chạy **5–10
  phút** và in nhiều dòng `nvcc`; xong trong vài giây nghĩa là pip vẫn dùng
  bản cũ.
- Dòng kiểm phải gọi **cả `_C` lẫn `model_zoo`**: `_C` là phần biên dịch, còn
  `model_zoo` là chỗ duy nhất còn `import pkg_resources` (setuptools 82 đã bỏ
  module này, nên `requirements.txt` ghim `setuptools<82`).

### 7. mmcv — SOLOv2

```bash
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch;print('%d.%d' % torch.cuda.get_device_capability(0))")
echo "arch = $TORCH_CUDA_ARCH_LIST"          # 8.9 trên 4090, 12.0 trên 5090

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
rm -rf build mmcv/_ext*.so
FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=4 pip install --no-build-isolation -e .
cd -
```

`TORCH_CUDA_ARCH_LIST` lấy thẳng từ card thay vì viết tay. Viết tay thì chép
lệnh từ máy này sang máy kia sẽ dựng sai kiến trúc, và lỗi chỉ hiện ở lời gọi
kernel đầu tiên, **sau** 20–120 phút build:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

`rm -rf build mmcv/_ext*.so` cần thiết: setuptools dùng lại file `.o` đã biên
dịch, nên build lại mà không dọn thì kiến trúc **không đổi** dù đã sửa biến.

Clone **ra ngoài repo**: thư mục `mmcv/` trong gốc repo sẽ che gói đã cài, và
mọi lệnh chạy từ gốc repo báo `module 'mmcv' has no attribute '__version__'`.

Đây là bước lâu nhất, 20–120 phút. pip **ẩn output build** khi thành công,
nên không thấy dòng `nvcc` nào không có nghĩa là build bị bỏ qua. Kiểm bằng
file:

```bash
ls -l ~/mmcv/mmcv/_ext*.so
python -c "import os, torch, mmcv; from mmcv.ops import nms; b=torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s=torch.tensor([0.9,0.8]).cuda(); print(os.path.dirname(mmcv.__file__)); print('mmcv nms CUDA', nms(b,s,0.5)[1].tolist())"
```

`_ext.so` cỡ **~26 MB** là bản có nhân CUDA; vài MB là bản chỉ có CPU, phải
build lại. Đường dẫn in ra phải là `~/mmcv/mmcv`.

### 8. mmengine và mmdet — SOLOv2

```bash
pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
grep -n "mmcv_maximum_version" "$MMDET"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
```

Phải chạy **sau bước 7**, vì cài mmcv kéo bản mmengine trên PyPI về đè. Dòng
kiểm phải in `True`: bản PyPI hỏng trên torch ≥ 2.6 (ném `UnpicklingError`
lúc nạp trọng số COCO), và số phiên bản không phân biệt được hai bản, nên phải
kiểm bằng mã nguồn.

`sed` nới chốt phiên bản của mmdet: nó khai `mmcv < 2.2.0`, trong khi 2.2.0 là
bản mới nhất hiện có. `grep` trước để không vá hai lần.

### 9. MSDeformAttn — Mask2Former

```bash
git submodule update --init benchmark/mask2former/upstream
git -C benchmark/mask2former/upstream checkout .

sed -i "s/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g" \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

`git checkout .` đứng trước để xoá mọi bản vá cũ trong repo con. Ví dụ, một
bản vá đổi `raise` trong `ms_deform_attn_func.py` thành `MSDA = None` để mở
đường lùi CPU sẽ làm op hỏng mà không báo lỗi, và Mask2Former chạy chậm hẳn.

`sed` vá hai dòng của repo gốc; thiếu nó nvcc dừng ngay.
`AT_DISPATCH_FLOATING_TYPES` nhận `c10::ScalarType`, còn `value.type()` trả
`at::DeprecatedTypeProperties`. Torch 2.11 đã gỡ phép chuyển ngầm giữa hai
kiểu, nên dòng 69 và 139 báo `no suitable conversion function`.
`value.scalar_type()` là đúng thứ mà thông điệp deprecated của torch chỉ tới:
cùng giá trị, không đổi hành vi.

Build thẳng thư mục `ops`, không chạy `make.sh` của repo gốc: script đó gọi
`setup.py install`, mà setuptools mới đã bỏ lệnh này.

Dòng kiểm cuối đi qua `mask2former/__init__` → `instance_evaluation` →
`fast_eval_api` → `from detectron2 import _C`, nên nó **đồng thời** xác nhận
detectron2 còn khớp torch.

### 10. Kiểm cả env

```bash
python -c "
import inspect, os, torch, torchvision, ultralytics, detectron2, mmcv, mmdet
import mmengine, mmengine.runner.checkpoint as ckpt
from torchvision.ops import nms as tv_nms
from mmcv.ops import nms as mmcv_nms
from detectron2 import model_zoo, _C
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s = torch.tensor([0.9,0.8]).cuda()
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms CUDA', tv_nms(b,s,0.5).tolist())
print('detectron2 ', detectron2.__version__, '| _C + model_zoo ok')
print('mmcv       ', mmcv.__version__, os.path.dirname(mmcv.__file__))
print('             nms CUDA', mmcv_nms(b,s,0.5)[1].tolist())
print('mmdet      ', mmdet.__version__, '| mmengine', mmengine.__version__,
      '| vá:', 'weights_only' in inspect.getsource(ckpt))
print('ultralytics', ultralytics.__version__)
print('nvrtc      ', float(torch.rand(8,16,2).cuda().prod(dim=2).sum()))"

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Đủ tám dòng, có `vá: True` và `MSDeformAttn OK` là env dùng được cho cả bốn
model. **Mọi phép kiểm đều chạy trên GPU**: kiểm bằng tensor CPU thì vẫn qua
cả khi nhân CUDA hỏng.

### 11. Chạy khói bốn model

Cần fold đã cắt. Chưa có thì cắt trước:

```bash
ls data/export/field/f1/annotations/ || python scripts/make_fold.py \
  --export data/export/dataset_v1 --val configs/dataset/val_field.yaml \
  --all --out-root data/export/field
```

Dò cấu hình và VRAM rồi dừng, vài giây mỗi model:

```bash
python benchmark/maskrcnn/train.py     --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml     --data data/export/field/f1 --probe
python benchmark/solov2/train.py       --config benchmark/solov2/configs/train/solov2_r50_mm.yaml         --data data/export/field/f1 --probe
python benchmark/mask2former/train.py  --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/field/f1 --probe
python benchmark/yolo/train.py         --config benchmark/yolo/configs/train/yolo11s.yaml                 --data data/export/field/f1 --probe
```

Qua hết thì chạy khói thật, vài phút mỗi lượt:

```bash
python benchmark/maskrcnn/train.py     --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml     --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/solov2/train.py       --config benchmark/solov2/configs/train/solov2_r50_mm.yaml         --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/mask2former/train.py  --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/yolo/train.py         --config benchmark/yolo/configs/train/yolo11s.yaml                 --data data/export/field/f1 --fraction 0.05 --epochs 1
```

YOLO không có `--limit`; lượt khói dùng `--fraction` của ultralytics (tỉ lệ
tập train).

Máy hai card thì chọn card bằng `CUDA_VISIBLE_DEVICES=1` đặt trước lệnh. Bên
trong tiến trình card đó thành `cuda:0`, nên log in `GPU: 0` là đúng.

`--probe` báo `suggested_batch` lớn hơn 16 thì **giữ 16**: batch phải giống
nhau giữa các model (trừ Mask2Former, xem mục *Siêu tham số*), nếu không chênh
lệch batch sẽ lẫn vào cột kiến trúc của bảng kết quả.

### Dùng lại env cũ — phải build lại mọi phần mở rộng

Nếu env đã có torch bản khác (vd `2.4.1+cu121` hay `2.11.0+cu128`), mọi gói có
phần biên dịch **phải build lại**. Wheel torch `manylinux_2_28` dùng ABI
`std::string` mới (`__cxx11`), bản build cũ dùng ABI cũ (`Ss`), và lỗi có
dạng:

```
ImportError: detectron2/_C...so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
```

Chữ ký hàm đó không đổi giữa torch 2.4 và 2.11; thứ đổi là cách mã hoá
`std::string`. Không vá được, chỉ build lại được:

```bash
pip uninstall -y detectron2 mmcv MultiScaleDeformableAttention
pip cache remove detectron2
```

rồi làm lại bước 6, 7 và 9.

### Bảng lỗi đã gặp

| lỗi | nguyên nhân | sửa |
|---|---|---|
| `nvcc not found at '<env>/bin/nvcc'` | `CUDA_HOME` trỏ vào env mà env không có toolkit | bước 3 |
| `detected CUDA version (13.2) mismatches ... (13.0)` | chỉ là **cảnh báo** lệch minor | bỏ qua |
| `RuntimeError: The detected CUDA version (12.8) mismatches ... (13.0)` | dùng nhầm toolkit 12.8 của máy | bước 3 **nhánh B** |
| `nvrtc: error: failed to open libnvrtc-builtins.so.13.0` | wheel torch đặt tên file có băm | bước 4 |
| `no kernel image is available for execution on the device` | build sai kiến trúc (vd `sm_89` cho card `sm_120`) | dọn `build/` rồi build lại, bước 7 |
| `cannot find -lcudart` | wheel thiếu symlink `libcudart.so` | bước 3 nhánh B |
| `undefined symbol: ...parseSchemaOrNameERKSsb` | phần mở rộng build cho torch cũ | *Dùng lại env cũ* |
| detectron2 cài trong vài giây rồi vẫn lỗi cũ | pip lấy wheel trong cache | `--no-cache-dir` |
| `module 'mmcv' has no attribute '__version__'` | có thư mục `mmcv/` trong gốc repo | clone ra `~/mmcv` |
| `MMCV==2.2.0 is used but incompatible` | chốt phiên bản của mmdet | bước 8 |
| `UnpicklingError` lúc nạp trọng số COCO | mmengine bản PyPI | bước 8 |
| `No module named 'pkg_resources'` lúc train | setuptools ≥ 82 | `pip install "setuptools<82"` |
| `ERROR: This cross-compiler package contains no program ...` | `gcc_linux-64` hỏng trong conda base | không còn gặp: toolkit không cài bằng conda |

## Siêu tham số, và số liệu chọn ra chúng

Mặc định của bốn model đặt theo chính bộ dữ liệu này, không theo recipe COCO
và không theo card 8 GB. Đo trên 850 ảnh / 12138 vùng của
`data/export/dataset_v1`:

| | |
|---|---|
| ảnh | 2560 x 1440, đồng nhất cả bộ |
| tán, cạnh tương đương | trung vị **324 px**, p5 127 px, p95 620 px |
| vùng mỗi ảnh | trung vị 13, trung bình 14.3, tối đa **48** |
| ảnh nền | 2 / 850 |

Ba số phải giống nhau ở cả bốn model, vì chúng ảnh hưởng kết quả mà không
thuộc về kiến trúc:

| | giá trị | vì sao |
|---|---|---|
| `imgsz` | **1024** | tán trung vị còn 129 px. Chạy 1536 là nhìn tán to hơn 1.5 lần, và chênh lệch đó sẽ bị ghi vào cột kiến trúc |
| `batch` | **16** | đo trên RTX 5090: 12.7 GB ở imgsz 1024. Mask2Former **8**: 16 tràn VRAM vì nó giữ 100 query cộng attention toàn ảnh. `lr` để trống nên trainer nhân `batch/16`, lịch học vẫn đúng tỉ lệ recipe |
| `epochs` | **100** | Mask2Former **50**: lượt chạy thử 100 epoch cho thấy epoch tốt nhất luôn rơi trước mốc 50, phần sau chỉ tốn giờ máy. Chênh lệch có chủ đích, ghi trong config của nó |

`warmup_iters` tính theo **phần của lịch** (`0.03`), cùng quy ước với
`lr_steps`: dưới 1 là phần, từ 1 trở lên là số vòng tuyệt đối. Lý do: 500 ảnh
ở batch 16 là 32 vòng/epoch, 50 epoch là 1600 vòng, nên 200 vòng cố định thành
**12.5% lịch**, trong khi recipe COCO warmup chưa tới 1%. Lượt khói 28/09 cho
thấy đúng điều đó: hết 3 epoch, lr vẫn chưa lên tới giá trị đã đặt.

`max_det` 100 và `num_queries` 100 giữ nguyên: ảnh dày nhất có 48 tán, nên cả
hai đều dư gấp đôi.

### Toàn bộ tham số, và nguồn của chúng

Nguyên tắc: **giữ recipe của tác giả**, chỉ đổi những gì bộ dữ liệu này bắt
phải đổi, và đã đổi thì đổi cho cả bốn. Cột cuối ghi nguồn của giá trị.

#### Lịch học

| | Mask R-CNN | SOLOv2 | Mask2Former | nguồn |
|---|---|---|---|---|
| optimizer | SGD | SGD | AdamW | recipe |
| `lr` @ batch 16 | 0.02 | 0.01 | 1e-4 | recipe |
| `weight_decay` | 1e-4 | 1e-4 | 0.05 | recipe |
| `momentum` | 0.9 | 0.9 | (AdamW không dùng) | recipe |
| clip gradient | — | norm 35 | full_model 0.01 | recipe |
| backbone lr x | — | — | 0.1 | recipe |
| `lr_steps` | **0.7 / 0.9** | **0.7 / 0.9** | **0.7 / 0.9** | **đặt cho bộ này** |
| gốc là | 0.778 / 0.926 | 0.75 / 0.917 | 0.889 / 0.963 | |
| `warmup_iters` | **0.03** | **0.03** | **0.03** | **đặt cho bộ này** |
| gốc là | 1000 vòng | 500 vòng | 10 vòng @factor 1.0 | |

`lr` để trống trong config, trainer nhân `batch/16`, đúng luật scale tuyến
tính mà cả ba recipe đều dùng. Truyền `--lr` tay là tắt phép nhân đó.

Hai dòng **đặt cho bộ này** có chung một lý do: lịch ở đây ngắn hơn COCO
khoảng 170 lần (1600 vòng so với 270000). Giữ `warmup 1000 vòng` thì warmup
dài hơn cả lượt chạy; giữ đúng tỉ lệ gốc (0.37%) thì còn 6 vòng, quá ngắn để
ổn định. 3% (48 vòng, gần 1.5 epoch) nằm giữa. `lr_steps` 0.7/0.9 giảm lr sớm
hơn recipe một chút, để lịch ngắn vẫn có đủ thời gian ở lr thấp.

#### Suy luận và chấm

| | giá trị | gốc | ghi chú |
|---|---|---|---|
| `val_conf` | **0.05** cả bốn | 0.05 / 0.1 / không có / 0.001 | bốn recipe bốn kiểu; xem dưới |
| `max_det` | 100 | 100 / 100 / 100 / 300 | ảnh dày nhất 48 tán, dư gấp đôi |
| `num_queries` | 100 | 100 | chỉ Mask2Former |
| NMS IoU | 0.5 (d2) / matrix NMS (SOLOv2) / 0.7 (YOLO) | recipe | cơ chế khác nhau, không đưa về một giá trị được |
| `mask_thr` | 0.5 | 0.5 | SOLOv2 |

Ngưỡng điểm gốc của bốn recipe: Mask R-CNN `SCORE_THRESH_TEST=0.05`, SOLOv2
`score_thr=0.1`, YOLO `conf=0.001` khi val, còn Mask2Former **không có ngưỡng
nào**: `instance_inference` lấy `topk(100)` trên 100 query rồi giữ hết. Cả bốn
được đưa về 0.05 vì `precision`, `false_positive` và sai số đếm trong
`per_region.csv` đếm mọi dự đoán lọt qua ngưỡng; để nguyên thì Mask2Former ra
100 dự đoán trên ảnh có 14 tán.

Hệ quả: AP của COCO không phạt dự đoán điểm thấp, nên cắt ở 0.05 làm AP
**thấp hơn** so với ngưỡng 0.001 mà nhiều báo cáo YOLO dùng. Bốn model chịu
cùng một mức nên so sánh nội bộ vẫn đúng; so với số in trong bài báo thì
không.

#### Tăng cường — chỗ duy nhất không đưa về giống nhau được

| | Mask R-CNN | SOLOv2 | Mask2Former | YOLO |
|---|:---:|:---:|:---:|:---:|
| lật ngang / dọc | 0.5 / 0.5 | 0.5 / 0.5 | LSJ tự lật ngang | 0.5 / 0.5 |
| xoay 90° | ✓ | — | — | — |
| co giãn | — | — | **LSJ 0.1–2.0** | 0.5 |
| dịch | — | — | crop ô vuông | 0.1 |
| màu (HSV) | ✓ | ✓ | ✓ | ✓ |
| mosaic | — | — | — | **tắt** |

Hai ô đáng chú ý. **LSJ** là recipe gốc của Mask2Former, nên được giữ.
**mosaic tắt** là lệch khỏi recipe YOLO có chủ đích: mosaic ghép 4 ảnh thành 1
nên thu nhỏ tán, đi ngược mục tiêu giữ độ phân giải đường biên.

**Màu có ở cả bốn**, cùng ba giá trị `hsv_h 0.015 / hsv_s 0.7 / hsv_v 0.4`.
Đây là mặc định của ultralytics; trước đây chỉ YOLO có. Bộ dữ liệu bay trong 3
ngày, 07h–16h, và độ sáng chênh nhau thật: đo trên 850 ảnh thì V trải 99–175
(1.76 lần), S trải 78–180 (2.32 lần). Ruộng tối nhất (field_1, V trung bình
115) là test của fold f1, nên độ bền với ánh sáng ảnh hưởng thẳng lên bảng. Một
model có tăng cường màu còn ba model không thì chênh lệch đó bị ghi vào cột
kiến trúc.

Ba framework không có transform giống hệt nhau, nên chỗ lệch phải nói rõ:
detectron2 (Mask R-CNN, Mask2Former) không có phép đổi **sắc**, nên `hsv_h`
khai ra mà không được áp; mmdet (SOLOv2) chỉ có `PhotoMetricDistortion`, phép
này **cộng** delta độ sáng thay vì **nhân**, quy đổi qua mức xám 128 nên
`hsv_v 0.4` thành ±51, và phần tương phản bị khoá ở 1.0 vì ultralytics không
đổi tương phản.

`hsv_s 0.7` cho biên `×[0.3, 1.7]` = 5.67 lần, rộng gấp 2.4 lần mọi thứ bộ dữ
liệu chứa. Giá trị này được giữ để bốn model khớp nhau; ảnh hưởng riêng của nó
chỉ tách ra được bằng một lượt ablation tắt màu.

`degrees` và `copy_paste` của YOLO để 0: hai tăng cường này **chỉ YOLO có**,
và chúng không phải mặc định của ultralytics. HSV, `scale`, `translate` thì
giữ vì đó là mặc định của chính recipe YOLO, cùng lý do giữ LSJ cho
Mask2Former.

#### Dữ liệu và vận hành

| | giá trị | vì sao |
|---|---|---|
| `min_area` | 50 px² | đo được: chỉ loại **1 vùng** trên 12138 (một nhãn 1 px²) |
| ảnh nền | giữ trong train | `filter_empty_gt=False` cả ba framework; chỉ có 2/850 ảnh nên ảnh hưởng nhỏ, nhưng bốn model phải dùng cùng một tập |
| `amp` | bật cả bốn | Mask2Former có sẵn trong recipe; ba model kia bật thêm, lượt khói SOLOv2 cho thấy loss giảm đều |
| `val_batch` | theo batch train | mặc định của detectron2 và mmdet là 1; đo được val tốn gấp đôi train vì thế |
| `workers` | 0 trên Windows, 8 trên Linux (Mask2Former 2) | mỗi worker Windows nạp lại torch, paging file mặc định không đủ → `WinError 1455` |
| `seed` | 0 | cả bốn |

### Hai trần độ phân giải không nằm ở imgsz

Dự án lấy đường biên làm trọng tâm, nên cần biết hai giới hạn này trước khi
đọc bảng kết quả:

| | trần | ở imgsz 1024 |
|---|---|---|
| Mask R-CNN | đầu mask **28x28** rồi phóng lên bbox | tán trung vị 129 px → **4.6 px mỗi ô** |
| YOLO | lưới prototype `imgsz/4` = 256 | **10 px ảnh gốc** mỗi ô |

Cả hai đều nâng được: `--mask_resolution 28` cho R-CNN (thành 56x56, còn 2.3
px/ô), `--mask_ratio 2` cho YOLO (proto 512, còn 5 px). Mặc định giữ nguyên
recipe gốc để mốc số 0 không bị đổi ngầm; nâng chúng là một thí nghiệm riêng,
ngoài bảng chính. SOLOv2 và Mask2Former không có trần kiểu này: cả hai sinh
mask ở stride 4 của toàn ảnh.

### Ngưỡng diện tích của COCO gần như vô nghĩa ở đây

97.4% số vùng rơi vào nhóm **large** theo thang COCO (small `<32²`, medium
`<96²`, tính trên ảnh gốc): chỉ 91 vùng small và 230 vùng medium trên 12138.

Vì vậy trong `cocoeval.txt`, ba dòng `AP_small` / `AP_medium` / `AP_large`
không chia bộ này thành ba nhóm có ý nghĩa: `AP_large` gần như chính là `AP`.
Muốn phân theo cỡ tán thì dùng `per_region.csv`, nơi mỗi vùng có `gt_area` và
`gt_side` thật.

## Chạy

Cắt fold một lần, từ gốc repo. Sáu lượt luôn giống nhau (mỗi ruộng làm test
một lần, các ruộng còn lại để train); ba cách chia khác nhau ở chỗ val lấy từ
đâu. Benchmark chạy **cả ba cách**, nên cắt cả ba bộ fold:

```bash
# bộ 1: khối ảnh cuối mỗi đường bay, trượt theo lượt, đệm theo đồ thị chồng lấn
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block

# bộ 2: trọn hai đường bay cuối (field_1/10/4 + field_2/10/2), cố định
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# bộ 3: trọn một ruộng làm val, xoay vòng — val là ruộng kế tiếp ruộng test
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_field.yaml --all --out-root data/export/field

# bảng so ba cách trên cả sáu lượt
python scripts/compare_val_splits.py
```

Ra 18 thư mục fold. Ảnh được hardlink nên gần như không tốn thêm đĩa.

Ba bộ khác nhau ở **khoảng cách giữa val và train**:

| bộ | val nằm ở đâu | train | val TB | đệm TB |
|---|---|---|---|---|
| `block` | cùng đường bay với train | 5 ruộng | 105 ảnh | 34 ảnh |
| `flight` | khác đường bay, cùng ruộng | 5 ruộng | 110 ảnh | 33 ảnh |
| `field` | **khác ruộng**, đúng điều kiện của test | 4 ruộng | 142 ảnh | 0 ảnh |

Chỉ bộ `field` chọn checkpoint bằng thước đo giống thước sẽ chấm nó. Hai bộ
kia chọn bằng ảnh cùng mảnh đất với train, tức lạc quan hơn test; nếu `field`
cho số thấp hơn thì chênh lệch đó chính là phần lạc quan, và đo được.

Bộ `field` mất một ruộng trong train nhưng **gần như không mất ảnh** (567 so
với 570 của `block`): nó không cần khoảng đệm, vì hai ruộng là hai mảnh đất
tách rời. Điểm yếu của nó là val dao động 48..280 ảnh và 6.9..25.1 vùng/ảnh
tuỳ lượt, trong khi `block` giữ val trong khoảng 12.7..15.9 vùng/ảnh.

**Mỗi model chạy 18 lượt**: 6 fold x 3 bộ; bốn model là 72 lượt. Lệnh trong
README của từng model viết sẵn cho một bộ và một fold; đổi tên bộ (`block`,
`flight`, `field`) hoặc fold (`f1`..`f6`) là ra các lượt còn lại. Số của ba bộ
là ba thí nghiệm khác nhau: so trong cùng một bộ, không so chéo.

Thiếu giờ máy thì cắt đủ 18 thư mục nhưng chạy bộ thứ ba trên hai lượt sàng
`f4` và `f2` (khai sẵn ở `screening` trong `folds.yaml`) trước.

Lệnh train và chấm của từng model nằm trong README của thư mục đó. Không có
script chạy cả bốn: mỗi model một framework và một lịch học riêng.

Ba quy ước phải giữ, vì bước gộp bảng dựa vào chúng:

| | |
|---|---|
| `--runs benchmark/<model>/runs` | kết quả nằm trong thư mục của model, không lẫn với model khác |
| `--name <model>` | chỉ tên model, KHÔNG kèm fold |
| `preds/<model>_<bộ>_<fold>.json` | file dự đoán test, đặt ở `preds/` gốc để đối chiếu giữa các model |

**Vì sao `--name` không kèm fold.** Fold và bộ fold đọc được từ đường dẫn dữ
liệu đã chạy, nên không cần gõ lại, và không nên gõ lại: gõ
`--name maskrcnn_f4` mà trỏ vào `data/export/block/f5` thì tên sai, còn đường
dẫn thì không. Bảng tổng hợp lấy theo đường dẫn và báo ra nếu hai chỗ mâu
thuẫn.

Tên thư mục lần chạy được ghép tự động, không lặp:

```
runs/train/2026-09-25_091500_maskrcnn_block-f4_i1024b4e50/
                             └ model ┘ └bộ-fold┘ └tham số┘
```

**Tên file trong `preds/` và `results/` thì PHẢI có bộ fold**, vì những file
đó đặt tên bằng tay: ba bộ cùng đánh số f1..f6 nên `preds/maskrcnn_f4.json`
của bộ này sẽ đè file của bộ kia.

Gộp kết quả bốn model thành bảng model × ruộng:

```bash
# cả ba bộ trong một bảng, có cột "bộ fold"
python scripts/summarize_folds.py --eval benchmark/*/runs/eval

# chỉ một bộ
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block
```

Nếu hai lần chấm trùng nhau (cùng model, cùng bộ, cùng fold) thì bảng giữ lần
mới nhất và **báo ra**, không đè mà không báo.

## Màn hình khi train, và log đầy đủ nằm đâu

Cả bốn model in ra cùng một dạng, để log của bốn lượt train đọc được cạnh
nhau:

```
Dữ liệu: train 470 ảnh / 7101 vùng | val 85 ảnh / 1078 vùng | test 280 ảnh / 3720 vùng (field_1)
maskrcnn: 3 epoch x 30 iteration, batch 16, imgsz 1024, lr 0.02, max_iter 90
epoch 1/3   iter 30/90   loss 2.627   lr 2.00e-02   9.3G   mAP50-95   3.76   mAP50  13.52   0:44   * tốt nhất
epoch 2/3   iter 60/90   loss 1.987   lr 2.00e-02   9.5G   mAP50-95  12.81   mAP50  33.70   0:43   * tốt nhất
epoch 3/3   iter 90/90   loss 1.800   lr 2.00e-02   9.5G   mAP50-95  27.61   mAP50  55.79   0:43   * tốt nhất

Chấm test bằng best.pth trên 280 ảnh ...

----------------------------------------------------------------
maskrcnn · 2026-09-24_144830_maskrcnn-r50-d2_block-f1_i1024b16e3
----------------------------------------------------------------
  epoch tốt nhất   3/3
  val              mAP50-95 27.61   mAP50 55.79
  test (field_1)   mAP50-95 24.10   mAP50 51.30
  thời gian        train 2:12
  trọng số         weights/best.pth
----------------------------------------------------------------
```

Trong lúc một epoch đang chạy có thanh tiến trình `tqdm` kèm loss. Thanh tự
tắt khi stdout không phải terminal (`nohup`, `> log.txt`); khi đó chỉ còn các
dòng epoch.

### Tên chỉ số

Cả bốn model dùng tên chỉ số theo cách gọi của YOLO. Cùng một đại lượng, mỗi
framework một tên:

| In ra | detectron2 | mmdet | Nghĩa |
|---|---|---|---|
| `mAP50-95` | `AP` | `segm_mAP` | AP trung bình trên IoU 0,50 → 0,95 |
| `mAP50` | `AP50` | `segm_mAP_50` | AP tại IoU 0,50 |

mmengine trả thang 0–1, hai framework kia trả 0–100; màn hình luôn in thang
**0–100**. `results.csv` và `summary.json` giữ nguyên thang gốc của từng
framework.

### Log đầy đủ

Màn hình chỉ **nâng ngưỡng lên WARNING**; trên đĩa không mất dòng nào:

| File | Nội dung |
|---|---|
| `<run>/run.log` | bản sao đúng màn hình, kể cả thanh tiến trình |
| `<run>/d2/log.txt` | detectron2: config đầy đủ, kiến trúc model, từng iteration, bảng COCO |
| `<run>/mmdet/<timestamp>/<timestamp>.log` | mmengine: tương tự |
| `<run>/results.csv` | một dòng mỗi lần ghi số liệu, có cột epoch |
| `<run>/warnings.log` | cảnh báo của thư viện, mỗi nội dung một lần |

Cảnh báo từ `logging` (`WARNING`) vẫn hiện ra màn hình, ví dụ dòng
`Skip loading parameter 'cls_score' ... (81, 1024) vs (2, 1024)`: đầu phân
loại COCO 81 lớp được khởi tạo lại cho 1 lớp, đúng như mong muốn.

Cảnh báo từ module `warnings` thì ghi vào `<run>/warnings.log`, mỗi nội dung
đúng một lần, và khối tổng kết cho biết có bao nhiêu loại. Chúng vốn bắn ra
**mỗi iteration** (detectron2 gọi `torch.cuda.amp.autocast` đã bị khai tử), và
vì chúng ra stderr còn thanh tiến trình vẽ ra stdout, mỗi lần bắn là một lần
thanh bị cắt đôi.

Cơ chế gộp sẵn của Python không đủ: nó đánh dấu "cảnh báo này đã hiện" trong
một registry mà CPython **xoá sạch** mỗi khi danh sách bộ lọc đổi phiên bản, và
`warnings.catch_warnings()` lúc thoát luôn làm việc đó. Trong vòng lặp train có
thư viện gọi nó ở mỗi mẫu:

| | dòng cảnh báo / 30 iteration |
|---|---|
| không làm gì | 60 |
| bộ lọc `"once"`, không ai gọi `catch_warnings` | 2 |
| bộ lọc `"once"`, có `catch_warnings` mỗi iteration | **60** |
| chuyển vào file (cách đang dùng) | **0** trên màn hình, 2 trong file |

Muốn xem đúng kiểu mặc định của framework thì thêm `--verbose true`. Đo trên
một lượt Mask R-CNN 3 epoch: **1474 dòng** khi bật, **khoảng 30 dòng** khi
tắt; 1050 dòng trong số đó là dump config và kiến trúc model, riêng kiến trúc
bị in hai lần.

## Phần chấm điểm phải giống nhau ở cả bốn thư mục

Bốn bản sao `cofseg/` xuất phát giống hệt nhau. Sửa phần **chấm điểm**
(`metrics/`, `evaluation/`, `datasets/`) ở một thư mục thì model đó được đo
bằng một thước khác, và cột "Δ% mAP so với Mask R-CNN" mất nghĩa.

Sửa trainer hay wrapper model không ảnh hưởng phép đo. Sửa phần chấm điểm thì
phải sửa giống nhau ở cả bốn thư mục, hoặc ghi rõ khác biệt khi trình bày
bảng.

Những thứ thật sự dùng chung: `data/` (ảnh + nhãn), `weights/` (trọng số
COCO), `scripts/make_fold.py` (cắt fold — mọi model phải dùng CÙNG bộ fold),
`scripts/summarize_folds.py` (gộp bảng).

## Quy ước

- **Gói của model nào khai trong thư mục model đó**, ở
  `benchmark/<model>/requirements.txt`. `requirements.txt` ở gốc repo chỉ lo
  `app/` và `canopyseg/`. **torch không khai ở đó**: bản torch phụ thuộc máy
  chứ không phụ thuộc model, nên cài riêng ở bước 2 của *Dựng môi trường*.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`.
  Kết quả cần chia sẻ thì chép file nhỏ vào `benchmark/<model>/results/`.
- Chạy test của thư mục trước khi commit:
  `cd benchmark/<model> && python -m pytest tests`
