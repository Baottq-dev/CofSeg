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

Một đường duy nhất cho cả bốn model: **Linux, GPU NVIDIA, một env conda, CUDA
12.8 cài vào chính env đó**. Tám bước, đọc từ trên xuống, mỗi bước có lệnh kiểm
ngay sau nó. Không nhánh — không "nếu máy đã có nvcc thì...", không "nếu không
biên dịch được thì...". Máy nào cũng dựng ra đúng một env như nhau, và đó là
điều kiện để bốn người so số với nhau.

Hỏng ở đâu thì tra mục *Khi hỏng* bên dưới; phần này chỉ có đường đi đúng.

Mất ~15 phút cho các bước pip, cộng 20–120 phút cho riêng mmcv ở bước 5.

Thứ tự **không đảo được**, vì ba lý do có thật:

- **toolkit trước torch**, để biết chắc nvcc và torch cùng major ngay từ đầu;
- **torch trước mọi gói build từ nguồn** — `setup.py` của detectron2, mmcv và
  MSDeformAttn đều `import torch` ngay lúc build;
- **mmcv trước mmengine**, vì mmcv khai `mmengine>=0.3.0` nên nó kéo bản PyPI
  về và đè lên bản git nếu làm ngược.

Env này cài cả bốn model. Ai chỉ chạy một model thì theo mục *Dựng môi trường
riêng* trong README của thư mục đó — ngắn hơn, bỏ được những bước model đó
không cần.

### 1. Env và CUDA toolkit

```bash
conda create -y -n cofseg python=3.12
conda activate cofseg
conda install -y -c nvidia cuda-toolkit=12.8.1

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh <<'EOF'
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
EOF
conda activate cofseg

nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Toolkit cài **vào env**, không vào máy: nó che nvcc của hệ thống trong lúc env
bật, tắt env là mọi thứ như cũ. Nhờ vậy bước này giống nhau trên mọi ảnh máy
thuê — không phải hỏi máy đang có nvcc bản nào rồi rẽ hai đường.

`nvcc --version` phải in ra **12.8**. Ra số khác nghĩa là `which nvcc` vẫn trỏ
vào nvcc hệ thống: kiểm `echo $CUDA_HOME` trước khi đi tiếp, vì lệch major với
torch thì ba lần build ở bước 4, 5, 7 đều gãy.

`CPATH` và `LIBRARY_PATH` không bỏ được: gói conda để header ở
`targets/x86_64-linux/`, chỉ đặt `CUDA_HOME` thì nvcc không thấy `cusparse.h`.
Viết chúng vào `activate.d/` chứ đừng `export` tay, để mở terminal mới hay tmux
mới cũng có sẵn — build mmcv chạy hàng chục phút, rất dễ rơi vào phiên khác.

nvcc 12.8 nhận host compiler tới **GCC 14**, cao hơn `g++` mà Ubuntu 22.04 mang
sẵn (11), nên không phải ghim `g++`.

### 2. torch

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

Torch không nằm trong file requirements nào: bản của nó phụ thuộc kiến trúc GPU
chứ không phụ thuộc model, ghim vào bốn file là cách chắc chắn để bốn file trôi
ra xa nhau rồi không cài chung một env được nữa.

Kiểm — **gọi op thật**, đừng dừng ở `import`:

```bash
python -c "
import torch, torchvision
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ra `[0]` và tên card là xong. Torchvision có phần mở rộng C++ link thẳng vào
`libtorch`, nên một bản dựng cho CUDA khác sẽ `import` trót lọt rồi mới gãy lúc
gọi op — mà detectron2 lấy ROIAlign và NMS của torchvision, tức Mask R-CNN dựa
hẳn vào đó. Vì vậy cài cả hai **trong cùng một lệnh**, đừng cài rời.

Danh sách cài sẽ có dòng `cuda-toolkit-12.8.1` của pip:

```
Installing collected packages: cuda-toolkit, torch
Successfully installed cuda-toolkit-12.8.1 torch-2.11.0+cu128
```

torch 2.11 đổi cách đóng gói: thay vì hàng chục wheel `nvidia-*-cu12` rời, nó
khai một metapackage `cuda-toolkit[cublas,cudart,cufft,...]==12.8.1`. Danh sách
extras **không có `nvcc`** — đây chỉ là thư viện runtime để torch chạy, nó
không thay được bước 1.

### 3. Gói Python của bốn model

```bash
pip install -r benchmark/requirements.txt
```

File này `-r` sang bốn file con, mỗi model một file, và không file nào ghim
torch. Chạy benchmark **không cần** `pip install -e .`: mỗi thư mục tự chứa bản
`cofseg/` riêng, và `train.py` tự thêm thư mục của nó vào `sys.path`.

### 4. detectron2 — cho Mask R-CNN và Mask2Former

```bash
pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

python -c "import detectron2; from detectron2 import model_zoo; print(detectron2.__version__)"
```

~5–10 phút. `--no-build-isolation` vì `setup.py` của nó `import torch`, mà môi
trường build cô lập của pip không có torch.

Kiểm phải gọi **cả `model_zoo`**: đó là chỗ duy nhất còn `import pkg_resources`,
mà `import detectron2` không kéo nó theo. Bỏ dòng đó thì bước cài báo xanh rồi
lần train đầu mới chết, sau khi đã nạp xong dữ liệu.

### 5. mmcv — cho SOLOv2

Trên `cu128` mmcv **không có wheel**: OpenMMLab dừng ở `cu118` và `cu121`, tới
torch 2.4. Build từ nguồn, 20–120 phút, làm một lần cho mỗi env.

```bash
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 \
  pip install --no-build-isolation -e .
cd ..

python -c "from mmcv.ops import nms; print('op CUDA có thật')"
```

Màn hình phải chạy **hàng trăm dòng `nvcc ... -c ... .cu`** trong hàng chục
phút. Xong trong vài giây là hỏng, xem *Khi hỏng*.

| biến | việc của nó |
|---|---|
| `FORCE_CUDA=1` | bật op CUDA cả khi máy đang build không thấy GPU |
| `MMCV_WITH_OPS=1` | không có thì mmcv dựng bản không op |
| `TORCH_CUDA_ARCH_LIST` | ghim đúng kiến trúc cần; bỏ trống là mỗi file `.cu` compile lại 6–7 lần |
| `MAX_JOBS` | số luồng; nvcc ngốn RAM, máy ít RAM thì hạ xuống 1–2 |

`TORCH_CUDA_ARCH_LIST` đặt theo card: `12.0` cho RTX 5090 / RTX PRO 6000, `9.0`
cho H100, `8.9` cho L40S / RTX 4090, `8.0` cho A100. Nhiều card thì ngăn bằng
dấu chấm phẩy: `"8.9;12.0"`.

Dòng `from mmcv.ops import nms` chỉ nạp file `_ext` nên chạy được cả trên máy
không có card, và nó bắt đúng lỗi hay gặp.
`.dev_scripts/check_installation.py` gọi op thật nên cần đúng GPU — không thay
được dòng trên.

### 6. mmengine và mmdet — cho SOLOv2

Bước này **phải sau bước 5**: mmcv khai `mmengine>=0.3.0`, nên lúc cài mmcv pip
đã kéo bản PyPI về rồi, mà bản PyPI hỏng trên torch ≥ 2.6.

```bash
pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
python -c "import mmdet, mmcv, mmengine; print(mmdet.__version__, mmcv.__version__)"
```

Dòng kiểm phải in `True`. Số phiên bản **không** phân biệt được hai bản mmengine
— nhánh `main` chưa tăng số sau lần phát hành cuối (0.10.7, ra 04/03/2025),
trong khi bản vá `weights_only` merge 25/10/2025. Phải xem bằng mã nguồn.

torch 2.6 đổi mặc định `torch.load(weights_only=True)`, còn checkpoint mmdet
chứa dict `meta` với object tuỳ ý, nên bản chưa vá ném `UnpicklingError` ngay
lúc nạp trọng số COCO của SOLOv2.

`sed` nới chốt phiên bản của mmdet: mmdet 3.3.0 khai `mmcv < 2.2.0` trong khi
2.2.0 là bản mới nhất tồn tại. Dùng `find_spec` chứ đừng `import mmdet` để tìm
file — chính dòng assert cần sửa nằm trong `__init__.py`, import vào là chết
trước khi kịp sửa.

### 7. MSDeformAttn — cho Mask2Former

```bash
git submodule update --init benchmark/mask2former/upstream
pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Toolkit và biến môi trường đã xong ở bước 1, không phải chuẩn bị gì thêm.

Dùng pip build thẳng thư mục `ops` chứ đừng chạy `make.sh` của repo gốc: nó gọi
`setup.py install`, thứ setuptools mới đã bỏ.

Đây là op **duy nhất** trong cả dự án mà thiếu nó thì vẫn chạy được, nên nó dễ
bị bỏ qua nhất — và cái giá không nằm ở tốc độ mà ở VRAM. Mã gốc bọc lời gọi op
trong `try/except` trần rồi rơi xuống `ms_deform_attn_core_pytorch`, đường đó
hiện vật hoá một tensor mà kernel CUDA không bao giờ dựng:

```
torch.stack(sampling_value_list, dim=-2)   # (N*M, D, Lq, L, P)

imgsz 1024, batch  4  ->  0.98 GiB mỗi lớp  ->  5.9 GiB  (6 lớp encoder)
imgsz 1024, batch 16  ->  3.94 GiB mỗi lớp  -> 23.6 GiB  -> OOM trên card 24 GB
```

### 8. Kiểm cả env một lượt

Chạy trước khi đặt lịch train dài. Mọi dòng đều **gọi op**, không dòng nào dừng
ở `import` — hai kiểu hỏng nguy hiểm nhất đều import trót lọt:

```bash
python -c "
import inspect, torch, torchvision, ultralytics, detectron2, mmcv, mmdet
import mmengine.runner.checkpoint as ckpt
from torchvision.ops import nms as tv_nms
from mmcv.ops import nms as mmcv_nms
from detectron2 import model_zoo
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, tv_nms(b, s, 0.5).tolist())
print('detectron2 ', detectron2.__version__)
print('mmcv       ', mmcv.__version__, mmcv_nms(b, s, 0.5)[1].tolist())
print('mmdet      ', mmdet.__version__, '| mmengine vá:', 'weights_only' in inspect.getsource(ckpt))
print('ultralytics', ultralytics.__version__)"

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Tám dòng in ra, `mmengine vá: True`, và `MSDeformAttn OK` là env đủ cho cả bốn
model. Sau đó chạy một lượt khói của model mình trước khi đặt lịch dài — lệnh
nằm trong README của từng thư mục.

### Vì sao đúng những bản này

| | chọn | vì sao |
|---|---|---|
| CUDA | **12.8** | bản **đầu tiên** có `sm_120` (Blackwell) mà vẫn phủ từ `sm_50`: T4, V100, RTX 20/30/40, L4, L40S, A40, A6000, A100, H100, **RTX 5090**, RTX PRO 6000, B200 — không card nào trong kế hoạch rơi ra ngoài |
| torch | **2.11.0** | bản duy nhất nhóm có bằng chứng của chính mình: detectron2 0.6 (phát hành 2021) build và import được với nó |
| mmcv | **v2.2.0, từ nguồn** | bản mới nhất tồn tại; trên `cu128` không có wheel nào |
| mmengine | **từ git** | bản PyPI cuối ra trước bản vá `weights_only` bảy tháng |
| detectron2 | **commit `a2f4a87`** | repo không có bản phát hành mới; ghim commit để bốn máy cùng một bản |

Hai lựa chọn đã cân và **đã loại**, ghi lại để khỏi cân lại:

| | vì sao loại |
|---|---|
| `cu121` | chỉ mục duy nhất mmcv còn wheel dựng sẵn, nên cài nhanh hơn. Nhưng torch 2.4/cu121 ra đời **trước** Blackwell: trên RTX 5090 nó chết ngay lời gọi kernel đầu tiên (`no kernel image is available for execution on the device`), và đường cứu PTX JIT cũng hỏng (thiếu `libnvptxcompiler.so`) |
| `cu130` | phủ ít kiến trúc hơn (`sm_75` trở lên, bỏ Maxwell/Pascal/Volta), lệch major với nvcc của phần lớn ảnh máy thuê, trần host compiler GCC 15 là thừa, torch tới 2.14 cũng thừa — và **vẫn** phải build mmcv từ nguồn, nên không đỡ được lần biên dịch nào |

Cặp torch/torchvision hợp lệ khác trên `cu128`, nếu buộc phải xuống thấp hơn:
`2.7.0/0.22.0`, `2.7.1/0.22.1`, `2.8.0/0.23.0`, `2.9.0/0.24.0`, `2.9.1/0.24.1`,
`2.10.0/0.25.0`.

### Phần OpenMMLab không còn ai bảo trì

mmcv 2.2.0 ra 24/04/2024, mmdet 3.3.0 ra 05/01/2024; commit gần đây trên `main`
chỉ là NPU (Ascend) với MUSA, không đụng CUDA. Gãy thì tự sửa. Có người báo
build được ở torch 2.7 + CUDA 12.8 và ghi lại công thức:
[mmcv#3327](https://github.com/open-mmlab/mmcv/issues/3327).

## Khi hỏng

Mục này tách khỏi phần hướng dẫn có chủ đích: đường cài chỉ có một, đọc từ trên
xuống là xong. Xuống đây khi có gì đó không như trên.

### Ba chỗ báo xanh rồi mới chết

Nguy hơn lỗi thường, vì bước cài in `Successfully installed` còn cái chết thì
đến ở lần train đầu — sau khi đã nạp xong dữ liệu, thường là sau khi đã đặt
lịch chạy qua đêm trên máy tính tiền theo giờ.

| | dấu hiệu ngay lúc cài | chết ở đâu |
|---|---|---|
| mmcv dựng lúc chưa có torch | bước 5 xong trong **vài giây**, không một dòng `nvcc` | `ModuleNotFoundError: No module named 'mmcv._ext'` |
| mmengine ở lại bản PyPI | dòng `... mmcv-2.2.0 mmengine-0.10.7 ...` của bước 5, rồi bỏ bước 6 | `UnpicklingError` lúc nạp trọng số COCO |
| MSDeformAttn không biên dịch | bước 7 báo lỗi nhưng train vẫn chạy | không chết — **OOM**, hoặc chậm mà không ai biết vì sao |

**mmcv dựng bản rỗng** là cái bẫy nguy hiểm nhất. `setup.py` của mmcv bắt luôn
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

`EXT_TYPE` rỗng thì `get_extensions()` trả danh sách rỗng, và **cả
`MMCV_WITH_OPS=1` lẫn `FORCE_CUDA=1` đều vô hiệu** — hai biến đó chỉ được đọc
sau khi `EXT_TYPE` đã đặt. pip vẫn in `Successfully installed mmcv-2.2.0`, và
trong env có một gói mmcv không có file `_ext`.

Gỡ ra làm lại, đừng build đè:

```bash
pip uninstall -y mmcv
rm -rf mmcv/build
```

rồi quay về bước 2 kiểm torch có thật trong env, và chạy lại bước 5.

### Tra theo dòng báo lỗi

| dòng báo | nguyên nhân | làm gì |
|---|---|---|
| `RuntimeError: The detected CUDA version (12.8) mismatches the version that was used to compile PyTorch (13.0)` | nvcc và torch lệch **major**. `torch/utils/cpp_extension.py` `raise` khi lệch major, chỉ `warn` khi lệch minor | bước 1 chưa chạy hoặc `CUDA_HOME` trỏ nhầm |
| `ModuleNotFoundError: No module named 'mmcv._ext'` | mmcv dựng bản rỗng | xem ngay trên |
| `_pickle.UnpicklingError` lúc nạp checkpoint | mmengine còn bản PyPI | bước 6 |
| `AssertionError: MMCV==2.2.0 is used but incompatible` | chốt phiên bản của mmdet | `sed` ở bước 6 |
| `undefined symbol` / `Couldn't load custom C++ ops` khi gọi `torchvision.ops` | torch và torchvision lệch bản dựng | cài lại **cả hai trong một lệnh**, bước 2 |
| `fatal error: cusparse.h: No such file or directory` | thiếu `CPATH` | bước 1 |
| `CUDA error: no kernel image is available for execution on the device` | wheel không có kernel cho card này (`cu121` trên Blackwell) | bước 2 |
| `ModuleNotFoundError: Please compile MultiScaleDeformableAttention` | op của Mask2Former chưa biên dịch | bước 7 |
| `pkg_resources` không tìm thấy lúc train detectron2 | `setuptools` quá mới | hai file requirements của maskrcnn/mask2former đã ghim sẵn; kiểm `pip show setuptools` |

### Máy không biên dịch CUDA được

Không phải đường chuẩn. Chỉ dùng khi bước 1 không xong được — máy không cho cài
toolkit, hoặc chỉ có CPU — mà vẫn cần chạy thử đường code.

detectron2 dựng được bản CPU: `setup.py` chọn `CUDAExtension` khi
`torch.cuda.is_available() and CUDA_HOME is not None`, giấu GPU đi thì nó lặng
lẽ dựng `CppExtension`.

```bash
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

Với **Mask R-CNN không mất gì**, vì ROIAlign và NMS đều lấy của torchvision.
Biến này chỉ có tác dụng lúc **cài**; lúc train GPU vẫn dùng bình thường.

Mask2Former chạy được nhưng phải nới một chốt: `ms_deform_attn_func.py` ném lỗi
ngay lúc import, trong khi `ms_deform_attn.py` đã bọc sẵn lời gọi op trong
`try/except` và có đường lùi Python — đường lùi có sẵn mà không bao giờ tới
lượt.

```bash
sed -i "s/    raise ModuleNotFoundError(info_string)/    MSDA = None/" \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py
```

Rồi hạ `--batch` xuống 2–4 và xác nhận bằng `--probe` trước khi đặt lịch chạy
dài, vì đường lùi tốn 3.94 GiB mỗi lớp ở imgsz 1024 batch 16.

SOLOv2 **không có đường này**: mmcv không có bản CPU dùng được, buộc phải biên
dịch.

### Thử mmcv compile được không, trước khi thuê máy

Nhờ `FORCE_CUDA=1`, câu hỏi *"mmcv có compile nổi với torch mới không"* trả lời
được trên bất kỳ máy Linux nào có nvcc — không phải thuê 5090 trước:

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

Muốn chạy được cả trên máy test lẫn máy thuê thì build hai kiến trúc:
`TORCH_CUDA_ARCH_LIST="8.9;12.0"`.

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
