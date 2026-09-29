# benchmark/ — mỗi model một thư mục độc lập, mỗi thư mục một người

Bốn model của bảng so sánh, mỗi model nằm trọn trong một thư mục: **code
riêng, config riêng, cách chạy riêng, kết quả riêng**. Không thư mục nào
import thư mục nào, không ai phải chờ ai để sửa phần của mình.

## Phân công

| Thư mục | Người phụ trách | GitHub | Model | Vai trò trong bảng |
|---|---|---|---|---|
| `maskrcnn/` | VanNguyen | @vnguyen123 | Mask R-CNN R50-FPN | **mốc số 0** — mọi model khác báo Δ% mAP so với nó |
| `solov2/` | PhuongQuynh | @Phquynh2312 | SOLOv2 R50-FPN | box-free: lưới + kernel động, không box |
| `yolo11/` | QuangBao | @Baottq-dev | YOLOv11-Seg | một giai đoạn, thời gian thực |
| `mask2former/` | AnhVu | @tranphuocanhvu2103 | Mask2Former R50 | query / transformer, mask toàn ảnh |

Ba model dùng chung backbone ResNet-50 (Mask R-CNN, SOLOv2, Mask2Former) nên
chênh lệch giữa chúng là do cơ chế, không do backbone.

## Một thư mục có gì

```
benchmark/<model>/
├─ README.md              model gì, vì sao chọn, cách chạy, trạng thái
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
├─ tests/                 test cho phần của mình
├─ runs/                  (không vào git) kết quả train/eval
├─ results/               file kết quả nhỏ, ĐƯỢC commit
└─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
```

Sửa gì trong thư mục mình cũng được — kể cả `cofseg/`. Không ai bị ảnh hưởng.

## Dựng môi trường

Bốn bước, luôn theo thứ tự: **xem nvcc → torch → gói của model → gói phải build
từ nguồn**. Thứ tự bắt buộc vì `setup.py` của detectron2 và Mask2Former
`import torch` ngay lúc build, và vì torch phải khớp major với nvcc sẵn có.

```bash
conda create -y -n cofseg python=3.12 && conda activate cofseg
```

### Bước 0 — hỏi máy đang có nvcc bản nào

```bash
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Bước này không bỏ được, và bỏ nó là hỏng ở Bước 3 chứ không phải ở đây. Ba gói
ở Bước 3 đều biên dịch CUDA, mà `torch/utils/cpp_extension.py` so **major** của
nvcc với major của `torch.version.cuda`: lệch major thì `raise`, lệch minor chỉ
`warn`.

```
RuntimeError: The detected CUDA version (12.8) mismatches the version
that was used to compile PyTorch (13.0)
```

Đúng lỗi này gặp trên máy Vast RTX 5090 ngày 28/09/2026: ảnh máy có nvcc 12.8,
torch cài khi đó là `cu130`, detectron2 gãy ngay trong `build_ext`. mmcv và op
MSDeformAttn của Mask2Former sẽ chết ở đúng chỗ đó.

Toàn dự án đi **một bản torch duy nhất**, dựng bằng **CUDA 12.8**. Nvcc mới là
thứ điều chỉnh cho khớp, không phải torch:

| nvcc trên máy | làm gì |
|---|---|
| **12.x** (phần lớn ảnh máy thuê cho 5090) | không phải làm gì, đi thẳng Bước 1 |
| 13.x, hoặc chưa có nvcc | cài `cuda-toolkit=12.8.1` vào env ở mục *Cài mmcv* — nó che nvcc của hệ thống |

### Bước 1 — torch

Bản torch phụ thuộc **kiến trúc GPU và nvcc của máy**, không phụ thuộc model.
Vì vậy nó không nằm trong file requirements nào: ghim ở đó thì bốn file sẽ trôi
ra xa nhau rồi không cài chung một env được nữa.

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

Kiểm ngay, đừng đợi — và kiểm cả con số CUDA, không chỉ số phiên bản:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Vì sao CUDA 12.8

12.8 là bản **đầu tiên** có `sm_120`, tức bản đầu tiên chạy được Blackwell. Nó
phủ `sm_50`–`sm_120`: T4, V100, RTX 20/30/40, L4, L40S, A40, A6000, A100,
H100, **RTX 5090**, RTX PRO 6000, B200 — không card nào trong kế hoạch rơi ra
ngoài, kể cả máy lab đời cũ.

Từng đặt mặc định là `cu130`, nay bỏ. CUDA 13 **không mua được gì** cho dự án
này mà lại đắt hơn:

| | cu128 | cu130 |
|---|---|---|
| kiến trúc | `sm_50`–`sm_120` | `sm_75`–`sm_120`, **bỏ** Maxwell/Pascal/Volta |
| nvcc của ảnh máy thuê | khớp sẵn | lệch major, phải tải toolkit ~3 GB |
| trần host compiler | GCC 14 | GCC 15 — thừa, Ubuntu 22.04 chỉ có GCC 11 |
| torch bản cao hơn | tới 2.10 | tới 2.14 — thừa, ta ghim 2.11 |
| mmcv | không có wheel, build từ nguồn | không có wheel, build từ nguồn |

Ô cuối là chỗ đáng nhớ: đi `cu130` **không** đỡ được lần biên dịch mmcv nào,
nên nó không có ưu thế nào để bù cho việc lệch nvcc.

Cặp torch/torchvision hợp lệ khác trên `cu128`, nếu muốn đi thấp hơn:
`2.7.0/0.22.0`, `2.7.1/0.22.1`, `2.8.0/0.23.0`, `2.9.0/0.24.0`, `2.9.1/0.24.1`,
`2.10.0/0.25.0`.

Chọn **2.11** vì đó là bản duy nhất ta có bằng chứng của chính mình —
detectron2 0.6 (phát hành 2021) build và import được với torch 2.11 + Python
3.13. Xuống 2.7 thì chưa ai trong nhóm thử.

### Vì sao không dùng cu121

`cu121` là chỉ mục duy nhất mmcv còn có wheel dựng sẵn, nên nó cài nhanh hơn.
Nhưng torch 2.4/cu121 ra đời **trước** Blackwell và không có kernel `sm_120`.
Trên RTX 5090 nó chết ngay lời gọi kernel đầu tiên:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Và đường cứu PTX JIT cũng hỏng trên Blackwell (thiếu `libnvptxcompiler.so`).

Đổi lại, `cu128` nghĩa là **mmcv phải build từ nguồn** — OpenMMLab chỉ phát
hành `cu118` và `cu121`, tới torch 2.4; dò trực tiếp thì mọi tổ hợp
`cu124`/`cu128`/`cu130` và `torch2.5+` đều trả 404. Mục *Cài mmcv* nói đủ.

Nếu chắc chắn **không bao giờ** đụng tới card Blackwell thì `cu121` vẫn dùng
được và đỡ được một lần biên dịch:

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Chọn cách này thì mọi bước dưới đây có nhánh riêng, đánh dấu *(cu121)*.

### Bước 2 — gói của model

Cả bốn vào **một env**:

```bash
pip install -r benchmark/requirements.txt
```

Hoặc **bốn env riêng**, mỗi env một lệnh:

```bash
pip install -r benchmark/yolo11/requirements.txt
pip install -r benchmark/solov2/requirements.txt
pip install -r benchmark/maskrcnn/requirements.txt
pip install -r benchmark/mask2former/requirements.txt
```

Chạy benchmark **không cần** `pip install -e .`: mỗi thư mục tự chứa bản
`cofseg/` riêng, và `train.py` tự thêm thư mục của nó vào `sys.path`.

### Bước 3 — ba gói build từ nguồn

Không gói nào trong ba gói này cài bằng một dòng requirements được: cách cài
phụ thuộc vào máy lúc cài, không phụ thuộc phiên bản.

| gói | cho | mục |
|---|---|---|
| `detectron2` | Mask R-CNN, Mask2Former | ngay dưới |
| `mmcv` | SOLOv2 | *Cài mmcv* |
| `MSDeformAttn` | Mask2Former | *Biên dịch op CUDA cho Mask2Former* |

**detectron2** (~5–10 phút):

```bash
pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

`--no-build-isolation` vì `setup.py` của nó `import torch`, mà môi trường build
cô lập của pip không có torch.

Máy **không biên dịch CUDA được** (thiếu nvcc, hoặc nvcc lệch major với torch)
thì giấu GPU lúc cài:

```bash
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

`setup.py` của detectron2 chọn `CUDAExtension` khi
`torch.cuda.is_available() and CUDA_HOME is not None`; không thấy GPU thì nó
lặng lẽ dựng `CppExtension`. Với Mask R-CNN **không mất gì** — ROIAlign và NMS
lấy của torchvision. Biến này chỉ có tác dụng lúc **cài**; lúc train GPU vẫn
dùng bình thường.

Kiểm **cả `model_zoo`**, không chỉ `import detectron2`:

```bash
python -c "import detectron2; from detectron2 import model_zoo; print(detectron2.__version__)"
```

`model_zoo` là chỗ duy nhất còn `import pkg_resources`, mà `import detectron2`
không kéo nó theo. Bỏ qua dòng này thì bước cài báo xanh rồi lần train đầu mới
chết, sau khi đã nạp xong dữ liệu.

## Cài mmcv

Trên `cu128` mmcv **không có wheel** — OpenMMLab dừng ở `cu118` và `cu121`, tới
torch 2.4. Phải build từ nguồn. Mất 20–120 phút, làm một lần cho mỗi env.

### 1. Toolkit và biến môi trường

**Máy đã có nvcc 12.x** (phần lớn ảnh máy thuê) thì bỏ qua phần cài, chỉ trỏ
`CUDA_HOME` vào toolkit sẵn có:

```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
python -c "import torch; print(torch.version.cuda)" && nvcc --version | tail -2
```

**Máy có nvcc 13.x, hoặc chưa có nvcc** thì cài toolkit 12.8 vào chính env —
nó che nvcc của hệ thống, không đụng gì tới phần còn lại của máy:

```bash
conda install -y -c nvidia cuda-toolkit=12.8.1
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
```

`CPATH` và `LIBRARY_PATH` **không bỏ được**: gói conda để header ở
`targets/x86_64-linux/`, chỉ đặt `CUDA_HOME` thì nvcc không thấy `cusparse.h`.

nvcc 12.8 nhận host compiler tới **GCC 14**, cao hơn `g++` mà Ubuntu 22.04 mang
sẵn (11), nên không phải ghim `g++`. Chỉ đường *(cu121)* mới phải, vì nvcc 12.1
từ chối g++ mới hơn 12.

### 2. Build

```bash
# torch PHẢI có mặt trước. Không có là mmcv lặng lẽ dựng bản rỗng.
python -c "import torch; print(torch.__version__, torch.version.cuda)"

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 \
  pip install --no-build-isolation -e .

# KIỂM BẮT BUỘC: op có thật hay không. Không cần GPU.
python -c "from mmcv.ops import nms; print('op CUDA có thật')"
cd ..
```

#### Nếu thiếu torch, mmcv **không báo lỗi**

Đây là cái bẫy nguy hiểm nhất của cả mục này. `setup.py` của mmcv bắt luôn
`ModuleNotFoundError` của torch rồi đi tiếp:

```python
EXT_TYPE = ''
try:
    import torch
    ...
    EXT_TYPE = 'pytorch'
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')
```

`EXT_TYPE` rỗng thì `get_extensions()` trả về danh sách rỗng, và **cả
`MMCV_WITH_OPS=1` lẫn `FORCE_CUDA=1` đều vô hiệu** — hai biến đó chỉ được
đọc sau khi `EXT_TYPE` đã được đặt. Kết quả: pip in `Successfully installed
mmcv-2.2.0`, build xong trong vài giây thay vì 20–120 phút, và trong env có
một gói mmcv **không có file `_ext`**. Lần train SOLOv2 đầu tiên mới chết,
sau khi đã nạp xong dữ liệu.

Hai dấu hiệu nhận ra ngay, không cần chờ:

| | build thật | build rỗng |
|---|---|---|
| thời gian | 20–120 phút | vài giây |
| màn hình | hàng trăm dòng `nvcc ... -c ... .cu` | không dòng nvcc nào |

Dính rồi thì gỡ ra làm lại, đừng build đè:

```bash
pip uninstall -y mmcv
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
rm -rf mmcv/build
```

rồi chạy lại khối lệnh trên.

Dùng `.dev_scripts/check_installation.py` để kiểm thì **cần đúng GPU**, vì
nó gọi op thật. Dòng `from mmcv.ops import nms` ở trên chỉ nạp file `_ext`
nên chạy được cả trên máy không có card — và nó bắt đúng lỗi hay gặp.

Ba biến, mỗi cái một việc:

| biến | vì sao |
|---|---|
| `FORCE_CUDA=1` | `setup.py` của mmcv bật op CUDA khi `torch.cuda.is_available() or FORCE_CUDA == '1'`. Cờ này cho phép **biên dịch trên máy không có GPU** |
| `MMCV_WITH_OPS=1` | không có thì nó dựng bản không op, `from mmcv.ops import nms` sẽ gãy |
| `TORCH_CUDA_ARCH_LIST` | mặc định torch build cho 6–7 kiến trúc, mỗi file `.cu` compile lại từng ấy lần. Ghim đúng cái cần là nhanh hơn nhiều lần |

`TORCH_CUDA_ARCH_LIST` đặt theo card: `12.0` cho RTX 5090 / RTX PRO 6000,
`9.0` cho H100, `8.9` cho L40S / RTX 4090, `8.0` cho A100. Nhiều card thì ngăn
bằng dấu chấm phẩy: `"8.9;12.0"`.

`MAX_JOBS` đổi số luồng — nvcc ngốn RAM, máy ít RAM thì hạ xuống 1–2.

### 3. mmengine phải lấy từ git

Trên torch ≥ 2.6 thì bản trên PyPI **hỏng**. torch 2.6 đổi mặc định
`torch.load(weights_only=True)`, còn checkpoint mmdet chứa dict `meta` với
object tuỳ ý → `UnpicklingError` ngay lúc nạp trọng số COCO của SOLOv2.

| | ngày |
|---|---|
| mmengine 0.10.7 (bản mới nhất trên PyPI) | 2025-03-04 |
| PR *Load checkpoint with weights_only=False* | 2025-10-25 |

Bản phát hành ra trước bản vá bảy tháng, và không có bản nào sau nó.

**Bước này phải chạy SAU bước 2, và không bỏ được.** mmcv khai
`mmengine>=0.3.0`, nên lúc cài mmcv pip tự kéo đúng bản 0.10.7 hỏng từ PyPI về:

```
Successfully installed addict-2.4.0 ... mmcv-2.2.0 mmengine-0.10.7 ...
```

Thấy dòng đó là biết bản hỏng đã nằm trong env. Đè lên bằng bản git:

```bash
pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import mmengine; print(mmengine.__version__, mmengine.__file__)"
```

Cài đúng thì `__file__` trỏ vào `site-packages/mmengine/`, còn số phiên bản in
ra vẫn có thể là `0.10.7` — bản trên `main` chưa tăng số sau lần phát hành
cuối. Số phiên bản KHÔNG phân biệt được hai bản; muốn chắc thì xem có bản vá
chưa:

```bash
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"
```

### 4. Nới chốt phiên bản trong mmdet

mmdet 3.3.0 khai `mmcv < 2.2.0`, nhưng 2.2.0 là bản mới nhất tồn tại:

```
AssertionError: MMCV==2.2.0 is used but incompatible.
Please install mmcv>=2.0.0rc4, <2.2.0.
```

```bash
MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
python -c "import mmdet, mmcv, mmengine; print(mmdet.__version__, mmcv.__version__)"
```

Dùng `find_spec` chứ đừng `import mmdet` để tìm file — chính dòng assert cần
sửa nằm trong `__init__.py`, import vào là chết trước khi kịp sửa.

### Thử compile mà không cần đúng card

Nhờ `FORCE_CUDA=1`, câu hỏi *"mmcv có compile nổi với torch mới không"* trả lời
được trên **bất kỳ máy Linux nào có nvcc** — không phải thuê 5090 trước:

```bash
docker run --rm -it nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv && FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 \
  pip install --no-build-isolation -e .
```

| | cần gì | rủi ro |
|---|---|---|
| **compile được không** | máy Linux bất kỳ có nvcc ≥ 12.8 | **cao** — đây là chỗ C++ gãy |
| **chạy được không** | đúng GPU `sm_120` | thấp, nếu compile đã qua |

`.dev_scripts/check_installation.py` gọi op thật nên cần đúng card. Muốn chạy
được cả trên máy test thì build hai kiến trúc: `TORCH_CUDA_ARCH_LIST="8.9;12.0"`.

### Không có ai bảo trì

mmcv 2.2.0 ra 24/04/2024, mmdet 3.3.0 ra 05/01/2024, commit gần đây trên `main`
chỉ là NPU (Ascend) với MUSA — không đụng CUDA. Gãy thì tự sửa. Có người báo
build được ở torch 2.7 + CUDA 12.8 và ghi lại công thức:
[mmcv#3327](https://github.com/open-mmlab/mmcv/issues/3327).

### *(cu121)* Đường wheel

Chỉ dùng được với torch 2.4.1+cu121, và khi đó SOLOv2 **không chạy trên
Blackwell**:

```bash
pip install mmengine==0.10.7 mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"
```

Rồi vẫn phải nới chốt phiên bản ở bước 4 trên. `mmengine` bản PyPI dùng được
vì torch 2.4 chưa đổi mặc định `weights_only`.

## Biên dịch op CUDA cho Mask2Former

```bash
git submodule update --init benchmark/mask2former/upstream
```

Đây là model **duy nhất** mà việc bỏ biên dịch CUDA phải trả giá. Thiếu kernel
MSDeformAttn thì mỗi forward rơi xuống đường Python, và đường đó hiện vật hoá
một tensor mà kernel CUDA không bao giờ dựng:

```
torch.stack(sampling_value_list, dim=-2)   # (N*M, D, Lq, L, P)

imgsz 1024, batch  4  ->  0.98 GiB mỗi lớp  ->  5.9 GiB  (6 lớp encoder)
imgsz 1024, batch 16  ->  3.94 GiB mỗi lớp  -> 23.6 GiB  -> OOM trên card 24 GB
```

### Biên dịch

Toolkit và các biến môi trường đã xong ở mục *Cài mmcv*, đi nhánh nào cũng vậy
— kể cả nhánh trỏ `CUDA_HOME` vào nvcc sẵn có của máy thuê. Không có gì thêm
phải chuẩn bị:

```bash
pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops
```

Dùng pip build thẳng thư mục `ops` chứ đừng chạy `make.sh` của repo gốc: nó gọi
`setup.py install`, thứ setuptools mới đã bỏ.

*(cu121)* Đường đó cần thêm một thứ mà CUDA 13 không cần — **nvcc 12.1 từ chối
host compiler mới hơn g++ 12**:

```bash
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
conda install -y -c conda-forge gxx_linux-64=12
```

Đây là một lý do nữa để đi CUDA 12.8: nvcc 12.8 nhận host compiler tới GCC 14,
không phải hạ cấp `g++` như đường cu121.

### Không biên dịch được

Phải nới một chốt chặn, nếu không thì `import` chết trước khi kịp dùng đường
lùi. `ms_deform_attn_func.py` ném lỗi ngay lúc import:

```python
raise ModuleNotFoundError("Please compile MultiScaleDeformableAttention…")
```

trong khi `ms_deform_attn.py` đã bọc lời gọi op trong `try/except` và rơi về
`ms_deform_attn_core_pytorch` — đường lùi có sẵn nhưng không bao giờ tới lượt:

```bash
sed -i "s/    raise ModuleNotFoundError(info_string)/    MSDA = None/" \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py
```

Kiểm:

```bash
PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former import add_maskformer2_config; \
   from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('OK')"
```

Mask R-CNN thì không cần bước này: detectron2 lấy ROIAlign và NMS của
torchvision.

## Chạy

Cắt fold một lần cho cả nhóm, từ gốc repo. Sáu lượt luôn giống nhau (mỗi ruộng
làm test một lần, các ruộng còn lại train); khác nhau ở chỗ cắt val ra sao.
Nhóm chạy **cả ba cách**, nên cắt cả ba bộ fold:

```bash
# bộ 1: khối ảnh cuối mỗi đường bay, trượt theo lượt, đệm theo đồ thị chồng lấn
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block

# bộ 2: trọn hai đường bay cuối (field_1/10/4 + field_2/10/2), cố định
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# bộ 3: trọn một ruộng làm val, xoay vòng — val là ruộng kế tiếp ruộng test
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_field.yaml --all --out-root data/export/field

# bảng so ba cách trên cả sáu lượt (để viết báo cáo, không phải để chọn một)
python scripts/compare_val_splits.py
```

Ra 18 thư mục fold. Ảnh được hardlink nên gần như không tốn thêm đĩa.

Ba bộ khác nhau ở **khoảng cách giữa val và train**, và đó là điều đáng viết
trong báo cáo:

| bộ | val nằm ở đâu | train | val TB | đệm TB |
|---|---|---|---|---|
| `block` | cùng đường bay với train | 5 ruộng | 105 ảnh | 34 ảnh |
| `flight` | khác đường bay, cùng ruộng | 5 ruộng | 110 ảnh | 33 ảnh |
| `field` | **khác ruộng** — đúng điều kiện của test | 4 ruộng | 142 ảnh | 0 ảnh |

Chỉ bộ `field` chọn checkpoint bằng thước đo giống thước sẽ chấm nó. Hai bộ
kia chọn bằng ảnh cùng mảnh đất với train, tức lạc quan hơn test; nếu `field`
cho số thấp hơn thì chênh lệch đó chính là phần lạc quan, đo được.

Bộ `field` mất một ruộng trong train nhưng **gần như không mất ảnh** (567 so
với 570 của `block`): nó không phải trả khoảng đệm nào, vì hai ruộng là hai
mảnh đất không dính nhau. Chỗ nó yếu là val nhảy 48..280 ảnh và 6.9..25.1
vùng/ảnh tuỳ lượt, trong khi `block` giữ val quanh 12.7..15.9.

**Mỗi model chạy 18 lượt**: 6 fold x 3 bộ. Cả nhóm là 72 lượt. Lệnh trong
README của từng model viết sẵn cho `data/export/block/f1`; đổi `block` thành
`flight` hoặc `field`, hoặc `f1` thành `f2`..`f6`, là ra các lượt còn lại. Số
của ba bộ là ba thí nghiệm khác nhau — so trong cùng một bộ, đừng so chéo.

Thiếu giờ máy thì cắt đủ 18 thư mục nhưng chạy bộ thứ ba trên hai lượt sàng
`f4` và `f2` (khai sẵn ở `screening` trong `folds.yaml`) trước, rồi quyết.

Sau đó **mỗi người chạy model của mình** — lệnh cụ thể nằm trong README của
từng thư mục. Không có script chạy cả bốn: mỗi model một framework, một lịch,
một người chịu trách nhiệm.

Ba quy ước phải giữ, vì bước gộp bảng cuối dựa vào chúng:

| | |
|---|---|
| `--runs benchmark/<model>/runs` | kết quả rơi vào thư mục của mình, không lẫn của người khác |
| `--name <model>` | chỉ tên model, KHÔNG kèm fold |
| `preds/<model>_<bộ>_<fold>.json` | file dự đoán test, đặt ở `preds/` gốc để cả nhóm đối chiếu |

**Vì sao `--name` không kèm fold.** Fold và bộ fold đọc được từ đường dẫn dữ
liệu đã chạy, nên không cần gõ lại — và không nên gõ lại: gõ
`--name maskrcnn_f4` mà trỏ vào `data/export/block/f5` thì tên nói dối, còn
đường dẫn thì không. Bảng tổng hợp lấy theo đường dẫn và báo ra nếu hai chỗ
mâu thuẫn.

Thư mục lần chạy tự ghép đủ, không lặp:

```
runs/train/2026-09-25_091500_maskrcnn_block-f4_i1024b4e50/
                             └ model ┘ └bộ-fold┘ └tham số┘
```

**Tên file trong `preds/` và `results/` thì PHẢI có bộ fold**, vì đó là file
thường do người gõ đặt tên: ba bộ cùng đánh số f1..f6 nên
`preds/maskrcnn_f4.json` của bộ này sẽ đè của bộ kia.

Gộp kết quả bốn người thành bảng model × ruộng:

```bash
# cả ba bộ trong một bảng, có cột "bộ fold"
python scripts/summarize_folds.py --eval benchmark/*/runs/eval

# chỉ một bộ
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block
```

Nếu hai lần chấm trùng (cùng model, cùng bộ, cùng fold) thì bảng giữ lần mới
nhất và **báo ra** — không im lặng đè.

## Màn hình khi train, và log đầy đủ nằm đâu

Cả bốn model in ra cùng một dạng, để bốn lượt train đọc được cạnh nhau:

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
tắt khi stdout không phải terminal (`nohup`, `> log.txt`), lúc đó chỉ còn các
dòng epoch.

### Tên chỉ số

Lấy theo cách gọi của YOLO cho **cả bốn** model. Cùng một đại lượng, mỗi khung
một tên:

| In ra | detectron2 | mmdet | Nghĩa |
|---|---|---|---|
| `mAP50-95` | `AP` | `segm_mAP` | AP trung bình trên IoU 0,50 → 0,95 |
| `mAP50` | `AP50` | `segm_mAP_50` | AP tại IoU 0,50 |

mmengine trả thang 0–1, ba khung kia trả 0–100; màn hình luôn là **0–100**.
`results.csv` và `summary.json` giữ nguyên thang gốc của từng khung.

### Log đầy đủ vẫn còn nguyên

Màn hình chỉ bị **nâng ngưỡng lên WARNING**, không mất dòng nào trên đĩa:

| File | Nội dung |
|---|---|
| `<run>/run.log` | bản sao đúng màn hình, kể cả thanh tiến trình |
| `<run>/d2/log.txt` | detectron2: config đầy đủ, kiến trúc model, từng iteration, bảng COCO |
| `<run>/mmdet/<timestamp>/<timestamp>.log` | mmengine: tương tự |
| `<run>/results.csv` | một dòng mỗi lần ghi số liệu, có cột epoch |
| `<run>/warnings.log` | cảnh báo của thư viện, mỗi nội dung một lần |

Cảnh báo từ `logging` (`WARNING`) vẫn hiện ra màn hình — ví dụ dòng
`Skip loading parameter 'cls_score' ... (81, 1024) vs (2, 1024)`, là dấu hiệu
đầu phân loại COCO 81 lớp được khởi tạo lại cho 1 lớp, đúng như mong muốn.

Cảnh báo từ module `warnings` thì đi vào `<run>/warnings.log`, mỗi nội dung
đúng một lần, và khối tổng kết cho biết có bao nhiêu loại. Chúng vốn bắn ra
**mỗi iteration** — detectron2 gọi `torch.cuda.amp.autocast` đã bị khai tử —
và vì chúng ra stderr còn thanh tiến trình vẽ ra stdout nên mỗi lần bắn là
một lần thanh bị cắt đôi.

Cơ chế gộp sẵn của Python không cứu được: nó đánh dấu "cảnh báo này hiện rồi"
trong một registry mà CPython **xoá sạch** mỗi khi danh sách bộ lọc đổi phiên
bản, và `warnings.catch_warnings()` lúc thoát luôn làm việc đó. Trong vòng lặp
train có thư viện gọi nó ở mỗi mẫu:

| | dòng cảnh báo / 30 iteration |
|---|---|
| không làm gì | 60 |
| bộ lọc `"once"`, không ai gọi `catch_warnings` | 2 |
| bộ lọc `"once"`, có `catch_warnings` mỗi iteration | **60** — thua |
| chuyển vào file (cách đang dùng) | **0** trên màn hình, 2 trong file |

Muốn xem đúng kiểu mặc định của khung thì thêm `--verbose true`. Đo trên một
lượt Mask R-CNN 3 epoch: **1474 dòng** khi bật, **khoảng 30 dòng** khi tắt —
1050 dòng trong số đó là dump config và kiến trúc model, riêng kiến trúc bị in
hai lần.

## Cái giá của việc tách rời

Bốn bản sao `cofseg/` xuất phát giống hệt nhau. Nếu một người sửa phần **chấm
điểm** (`metrics/`, `evaluation/`, `datasets/`) thì model của người đó đo bằng
một cái thước khác, và cột "Δ% mAP so với Mask R-CNN" không còn nghĩa.

Sửa trainer hay model wrapper thì thoải mái — đó là phần của bạn. Sửa phần
chấm điểm thì **phải báo nhóm** để ba người kia sửa theo, hoặc ghi rõ khác
biệt khi trình bày bảng.

Dùng chung thật sự chỉ còn: `data/` (ảnh + nhãn), `weights/` (trọng số COCO),
`scripts/make_fold.py` (cắt fold — mọi người phải dùng CÙNG bộ fold),
`scripts/summarize_folds.py` (gộp bảng).

## Quy ước làm chung

- **Gói của model nào khai trong thư mục model đó** — `benchmark/<model>/requirements.txt`.
  Cần thư viện mới thì thêm vào file của mình rồi báo nhóm, không `pip install`
  riêng rồi quên ghi. `requirements.txt` ở gốc chỉ lo `app/` và `canopyseg/`.
  **torch thì KHÔNG khai ở đó** — nó là lựa chọn của máy, không phải của
  model, nên cài riêng ở bước 1; xem mục *Dựng môi trường* ở trên.
- **Mỗi người một nhánh**, gộp vào `main` bằng merge hoặc rebase.
  **Không dùng "Squash and merge"**: squash gộp nhiều commit thành một và làm
  mất author của từng commit, tức mất dấu vết phân công.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
  Commit trong thư mục nào thì mang tên người phụ trách thư mục đó.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`.
  Kết quả muốn chia sẻ thì chép file nhỏ vào `benchmark/<model>/results/`.
- Chạy test của thư mục mình trước khi commit:
  `cd benchmark/<model> && python -m pytest tests`

## Commit đúng tên khi dùng chung một máy

Tạo `.authors.local` ở gốc repo — file này **không vào git** (tên và email
thật của bốn người không cần nằm trên GitHub), nên mỗi máy tự tạo một lần. Nội
dung hỏi trong nhóm; mẫu:

```bash
_as() { local n="$1" e="$2"; shift 2
  GIT_AUTHOR_NAME="$n" GIT_AUTHOR_EMAIL="$e" \
  GIT_COMMITTER_NAME="$n" GIT_COMMITTER_EMAIL="$e" "$@"; }
as_quynh() { _as "PhuongQuynh" "<email của Quỳnh>" "$@"; }
```

Dùng:

```bash
source .authors.local
as_quynh git commit -m "solov2: raise the score threshold for the f5 run"
git log -1 --format="A: %an <%ae>%nC: %cn <%ce>"    # kiểm lại
```

Đếm theo người: `git shortlog -sne`. Ai muốn gộp hai tên của cùng một người
(`Baottq-dev` và `QuangBao` chung một email) thì tự tạo `.mailmap` ở gốc repo
— git đọc nó từ thư mục làm việc, không cần được track.

Chỉ commit dưới tên một người khi người đó **thật sự làm phần đó**.
