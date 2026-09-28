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

Ba bước, luôn theo thứ tự: **torch → gói của model → gói phải build từ nguồn**.
Thứ tự bắt buộc vì `setup.py` của detectron2 và Mask2Former `import torch` ngay
lúc build.

```bash
conda create -y -n cofseg python=3.12 && conda activate cofseg
```

### Bước 1 — torch

Bản torch phụ thuộc **kiến trúc GPU**, không phụ thuộc model. Vì vậy nó không
nằm trong file requirements nào: ghim ở đó thì bốn file sẽ trôi ra xa nhau rồi
không cài chung một env được nữa. Chọn một dòng:

```bash
# GPU sm_50..sm_90 — T4, V100, RTX 20/30/40, L4, L40S, A40, A6000, A100, H100
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# GPU sm_75..sm_120 — thêm RTX 50xx, RTX PRO 6000, B200. CUDA 13 bỏ sm_50..sm_70.
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

Kiểm ngay, đừng đợi:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Chọn dòng nào thì quyết định luôn mmcv cài kiểu gì:

| torch | GPU | mmcv |
|---|---|---|
| 2.4.1+cu121 | `sm_50`–`sm_90` | **wheel dựng sẵn**, vài giây |
| 2.11.0+cu130 | `sm_75`–`sm_120`, gồm **RTX 5090** | **build từ nguồn**, 20–120 phút |

Ranh giới không phải "Blackwell hay không" mà là **mmcv có wheel hay không**.
OpenMMLab chỉ phát hành `cu118` và `cu121`, tới torch 2.4 — dò trực tiếp thì
mọi tổ hợp `cu124`/`cu128`/`cu130` và `torch2.5+` đều trả 404. Mà torch
2.4/cu121 ra đời **trước** Blackwell nên không có kernel `sm_120`: trên RTX
5090 nó chết ngay lời gọi kernel đầu tiên.

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Cặp torch/torchvision hợp lệ khác trên `cu130`, nếu muốn đi cao hơn 2.11:
`2.9.0/0.24.0`, `2.9.1/0.24.1`, `2.10.0/0.25.0`, `2.12.0/0.27.0`,
`2.12.1/0.27.1`, `2.13.0/0.28.0`, `2.14.0/0.29.0`. Chọn 2.11 vì đó là bản duy
nhất ta có bằng chứng của chính mình: detectron2 0.6 build và import được với
torch 2.11 + Python 3.13.

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

### Có wheel (torch 2.4/cu121)

```bash
pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"
```

Rồi nới một dòng kiểm phiên bản trong mmdet. mmdet 3.3.0 khai `mmcv < 2.2.0`,
nhưng 2.2.0 lại đúng là bản duy nhất có wheel cho torch 2.4:

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

### Không có wheel (CUDA 12.8 / 13.x, RTX 5090)

```bash
conda install -y -c nvidia cuda-toolkit=13.0.3
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 \
  pip install --no-build-isolation -e .
python .dev_scripts/check_installation.py
```

Ba biến, mỗi cái một việc:

| biến | vì sao |
|---|---|
| `FORCE_CUDA=1` | `setup.py` của mmcv bật op CUDA khi `torch.cuda.is_available() or FORCE_CUDA == '1'`. Cờ này cho phép **biên dịch trên máy không có GPU** |
| `MMCV_WITH_OPS=1` | không có thì nó dựng bản không op, `from mmcv.ops import nms` sẽ gãy |
| `TORCH_CUDA_ARCH_LIST` | mặc định torch build cho 6–7 kiến trúc, mỗi file `.cu` compile lại từng ấy lần. Ghim đúng cái cần là nhanh hơn nhiều lần |

`MAX_JOBS` đổi số luồng (nvcc ngốn RAM, máy ít RAM thì hạ xuống 1–2). Tổng
20–120 phút.

`export CPATH`/`LIBRARY_PATH` **không bỏ được**: gói conda để header ở
`targets/x86_64-linux/`, chỉ đặt `CUDA_HOME` thì nvcc không thấy `cusparse.h`.

### Thử compile mà không cần đúng card

Nhờ `FORCE_CUDA=1`, câu hỏi *"mmcv có compile nổi với torch mới không"* trả lời
được trên **bất kỳ máy Linux nào có nvcc** — không phải thuê 5090 trước:

```bash
docker run --rm -it nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
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

### Hai cái bẫy còn lại

1. **mmengine trên PyPI hỏng với torch ≥ 2.6.** torch 2.6 đổi mặc định
   `torch.load(weights_only=True)`, còn checkpoint mmdet chứa dict `meta` với
   object tuỳ ý → `UnpicklingError` lúc nạp trọng số COCO. mmengine 0.10.7 ra
   **04/03/2025**, bản vá merge **25/10/2025** — bản phát hành ra trước bản vá
   bảy tháng:

   ```bash
   pip install "git+https://github.com/open-mmlab/mmengine"
   ```

2. **Không có ai bảo trì.** mmcv 2.2.0 ra 24/04/2024, mmdet 3.3.0 ra
   05/01/2024, commit gần đây trên `main` chỉ là NPU (Ascend) với MUSA — không
   đụng CUDA. Gãy thì tự sửa. Có người báo build được ở torch 2.7 + CUDA 12.8
   và ghi lại công thức: [mmcv#3327](https://github.com/open-mmlab/mmcv/issues/3327).

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

### Có biên dịch (nên làm)

Cần **ba** thứ khớp nhau; thiếu một là gãy sau vài phút:

| | torch cu121 | torch cu130 |
|---|---|---|
| toolkit đầy đủ, không phải mỗi nvcc | `conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit` | `conda install -c nvidia cuda-toolkit=13.0.3` |
| host compiler | **`conda install -c conda-forge gxx_linux-64=12`** — nvcc 12.1 từ chối g++ mới hơn 12 | không cần: CUDA 13 nhận tới GCC 15 |
| nvcc cùng major với torch | tự khớp sau dòng trên | tự khớp sau dòng trên |

```bash
pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops
```

Dùng pip build thẳng thư mục `ops` chứ đừng chạy `make.sh` của repo gốc: nó gọi
`setup.py install`, thứ setuptools mới đã bỏ.

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
làm test một lần, train năm ruộng còn lại); khác nhau ở chỗ cắt val ra sao.
Nhóm chạy **cả hai cách**, nên cắt cả hai bộ fold:

```bash
# bộ 1: khối ảnh cuối mỗi đường bay, trượt theo lượt, đệm theo đồ thị chồng lấn
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block

# bộ 2: trọn hai đường bay cuối (field_1/10/4 + field_2/10/2), cố định
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# bảng so hai cách trên cả sáu lượt (để viết báo cáo, không phải để chọn một)
python scripts/compare_val_splits.py
```

Ra 12 thư mục fold. Ảnh được hardlink nên gần như không tốn thêm đĩa.

**Mỗi model chạy 12 lượt**: 6 fold x 2 bộ. Cả nhóm là 48 lượt. Lệnh trong
README của từng model viết sẵn cho `data/export/block/f1`; đổi `block` thành
`flight` hoặc `f1` thành `f2`..`f6` là ra các lượt còn lại. Số của hai bộ là
hai thí nghiệm khác nhau — so trong cùng một bộ, đừng so chéo.

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
thường do người gõ đặt tên: hai bộ cùng đánh số f1..f6 nên
`preds/maskrcnn_f4.json` của bộ này sẽ đè của bộ kia.

Gộp kết quả bốn người thành bảng model × ruộng:

```bash
# cả hai bộ trong một bảng, có cột "bộ fold"
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
