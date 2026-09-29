Hướng dẫn chạy 

## Phân công

Đồ án 4 thành viên, mỗi người phụ trách một model; khung benchmark dùng chung
để các con số so được với nhau.

| Thư mục | Người phụ trách | Model | Vai trò |
|---|---|---|---|
| `benchmark/maskrcnn/` | VanNguyen | Mask R-CNN R50-FPN | mốc số 0 của bảng |
| `benchmark/solov2/` | PhuongQuynh | SOLOv2 R50-FPN | box-free |
| `benchmark/yolo11/` | QuangBao | YOLOv11-Seg | một giai đoạn, real-time |
| `benchmark/mask2former/` | AnhVu | Mask2Former R50 | query / transformer |

Mỗi thư mục `benchmark/` chứa TRỌN model của một người: code (bản sao lõi
riêng trong `cofseg/`), config, script chạy, test, kết quả. Không thư mục nào
import thư mục nào. Dùng chung chỉ còn dữ liệu (`data/`), trọng số
(`weights/`), việc cắt fold và một env Python.

Cái giá: bốn bản sao phần chấm điểm, ai sửa thì số của người đó không còn so
được với ba người kia, nên sửa phần chấm phải báo nhóm. Bố cục, quy ước
nhánh/commit và cách commit đúng author khi dùng chung một máy: xem
`benchmark/README.md`.

## 0. Yêu cầu

- GPU NVIDIA có CUDA (khuyến nghị ≥ 8 GB VRAM cho SAM2 hiera-large khi gán nhãn; benchmark cần 24 GB). Không có GPU vẫn chạy được annotator nhưng rất chậm (đặt `sam.device: cpu`).
- Python 3.12 (các gói ghim trong `requirements.txt` chỉ có wheel cho ≥ 3.12), ~25 GB đĩa trống cho weights + dữ liệu.
- Windows chạy được annotator và **YOLOv11-Seg**. Ba model còn lại (Mask R-CNN, Mask2Former trên detectron2; SOLOv2 trên mmdet) chỉ chạy trên Linux — bản Mask R-CNN torchvision từng chạy ở nhà đã bị gỡ.

## 1. Lấy mã

```
git clone --recurse-submodules https://github.com/Baottq-dev/CofSeg.git CoffeeSeg
cd CoffeeSeg
# đã clone rồi mà thiếu benchmark/mask2former/upstream:
git submodule update --init
```

`benchmark/mask2former/upstream` là submodule (repo gốc của Mask2Former). `requirements.txt` ở gốc lo `app/` và `canopyseg/`; mỗi model benchmark khai gói riêng trong thư mục của nó.

## 2. Cài môi trường

```
conda create -y -n cofseg python=3.12 && conda activate cofseg
```

**Annotator (`app/`) và `canopyseg/`:**

```
pip install -r requirements.txt
pip install -e .
```

SAM 2.1 cài riêng với `--no-deps`, vì `setup.py` của nó khai `torch>=2.5.1` —
để nguyên mốc đó thì pip từ chối cài chung với torch 2.4.1 mà mmcv bắt buộc:

```
pip install --no-deps --no-build-isolation \
  "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
```

Bỏ qua mốc đó an toàn: nó xuất hiện ở commit 11/12/2024 để `torch.compile` toàn
model cho nhánh video. `app/annotator.py` chỉ dùng nhánh ảnh và không gọi
`torch.compile` ở đâu.

**Bốn model benchmark** — torch cài trước, rồi gói của model, rồi ba gói build
từ nguồn. Hướng dẫn đầy đủ, kể cả cách build `mmcv` cho GPU Blackwell:
[`benchmark/README.md`](benchmark/README.md) mục *Dựng môi trường*.

```
# 0. hỏi máy đang có nvcc bản nào — torch phải khớp MAJOR với nó
nvcc --version | tail -2

# 1. torch — một bản duy nhất, dựng bằng CUDA 12.8
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 2. gói của model — cả bốn vào một env
pip install -r benchmark/requirements.txt

# 3. ba gói build từ nguồn: detectron2, mmcv, MSDeformAttn
```

**Bước 0 không bỏ được.** Ba gói ở bước 3 đều biên dịch CUDA, và torch từ
chối build khi major của nvcc lệch major của `torch.version.cuda` — báo lỗi
ở `build_ext`, bằng một câu không hề nhắc tới torch:

```
RuntimeError: The detected CUDA version (12.8) mismatches the version
that was used to compile PyTorch (13.0)
```

nvcc **12.x** thì không phải làm gì. nvcc **13.x hoặc chưa có** thì cài
`cuda-toolkit=12.8.1` vào chính env — nó che nvcc của hệ thống. Điều chỉnh
nvcc chứ đừng đổi torch: torch là thứ cả nhóm phải giống nhau.

Không file requirements nào ghim torch: bản torch phụ thuộc **kiến trúc GPU**
chứ không phụ thuộc model. Ghim trong từng file là cách chắc chắn để bốn file
trôi ra xa nhau rồi không cài chung một env được nữa.

| torch | GPU | mmcv |
|---|---|---|
| **2.11.0+cu128** (mặc định) | `sm_50`–`sm_120`: T4, V100, RTX 20/30/40, L4, L40S, A40, A6000, A100, H100, **RTX 5090**, RTX PRO 6000, B200 | **build từ nguồn**, 20–120 phút |
| 2.4.1+cu121 | `sm_50`–`sm_90` — **không có Blackwell** | wheel dựng sẵn, vài giây |

CUDA 12.8 là bản **đầu tiên** có `sm_120`, tức bản đầu tiên chạy được
Blackwell, và nó vẫn giữ cả Maxwell/Pascal/Volta — không card nào trong kế
hoạch rơi ra ngoài. `cu121` cài nhanh hơn vì mmcv có wheel ở đó, nhưng torch
2.4/cu121 ra đời **trước** Blackwell nên thiếu kernel `sm_120`: trên RTX 5090
nó chết ngay lời gọi kernel đầu tiên.

`cu130` từng là mặc định, nay bỏ: nó phủ ít kiến trúc hơn (`sm_75` trở lên),
lệch major với nvcc của phần lớn ảnh máy thuê, và **vẫn** phải biên dịch mmcv
như cu128 — không đổi lại được gì.

Chạy benchmark **không cần** `pip install -e .`: mỗi thư mục tự chứa bản
`cofseg/` riêng và `train.py` tự thêm thư mục của nó vào `sys.path`.

### Kiểm lại sau khi cài

Hai gói phải kiểm bằng cách **gọi op**, không phải bằng `import`: torchvision
và mmcv đều có phần mở rộng C++, và cả hai đều có kiểu hỏng mà `import` vẫn
trót lọt.

```
python -c "
import torch, torchvision, ultralytics, detectron2, mmdet, mmcv
from mmcv.ops import nms as mmcv_nms
from torchvision.ops import nms as tv_nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print('torch', torch.__version__, torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms ->', tv_nms(b, s, 0.5).tolist())
print('detectron2', detectron2.__version__, '| mmdet', mmdet.__version__,
      '| mmcv', mmcv.__version__, '| ultralytics', ultralytics.__version__)
"
```

| gãy ở dòng nào | nghĩa là |
|---|---|
| `tv_nms(...)` ném `undefined symbol` | torchvision dựng cho bản CUDA khác torch — cài lại **cả hai** cùng một lệnh |
| `from mmcv.ops import` ném `ModuleNotFoundError: mmcv._ext` | mmcv build lúc chưa có torch nên không có op nào; xem *Cài mmcv* |
| `torch.version.cuda` khác major của `nvcc --version` | ba gói build từ nguồn sẽ gãy ở `build_ext` |

Bốn model đã cài được trên máy lab (2 × RTX 4090). **Chưa train thật lần nào**
— khói một lượt mỗi model trước khi chạy cả sáu fold.

## 3. Tải trọng số

```
python scripts/download_weights.py --group annotator    # SAM 2.1 hiera-L cho gán nhãn
python scripts/download_weights.py                      # nhóm benchmark (4 model chính + Cascade)
python scripts/download_weights.py --check --group all  # kiểm sha256 những gì đã có
```

Danh sách, URL và sha256 nằm trong `configs/weights.yaml`; file về `weights/` (ngoài git).

## 4. Dữ liệu

Giải nén thư mục iachim_dataset_export/ vào data/ <br>
Cấu trúc thư mục data sẽ là
```
data/
├─iachim_dataset_export/
    ├─data_compressed/
    ├─predictions/
└─masks/
```

## 5. Chạy annotator

```
python -m app.server -r      
```

Cho nhiều người trong mạng truy cập (máy lab):

```
python -m uvicorn app.server:app --host 0.0.0.0 --port 1801
```

## 6. Chạy benchmark

```
# 1. xem nhật ký bay và đồ thị chồng lấn — bằng chứng cho cách chia val
python scripts/inspect_flights.py --export data/export/dataset_v1
python scripts/export_overlap_edges.py

# 2. cắt fold. Sáu lượt như nhau (mỗi ruộng làm test một lần), khác nhau ở
#    cách cắt val. Nhóm chạy CẢ BA cách nên cắt cả ba bộ: val cùng đường bay
#    với train, val khác đường bay, val khác hẳn ruộng.
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_field.yaml --all --out-root data/export/field

# 3. bảng so ba cách chia trên cả sáu lượt (số liệu cho báo cáo)
python scripts/compare_val_splits.py

# trên máy Linux, gộp bước giải nén và cắt fold làm một
python scripts/prepare_data.py all_v2.tar --val configs/dataset/val_block.yaml

# 4. mỗi người chạy model của mình: lệnh trong benchmark/<model>/README.md
#    18 lượt mỗi model (6 fold x 3 bộ), cả nhóm 72 lượt

# 5. bảng model x ruộng từ kết quả của cả bốn thư mục
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block

# 6. gói kết quả mang về
python scripts/pack_results.py
```

Ai phụ trách model nào, ba quy ước đặt tên khi chạy, và cách gộp bảng:
`benchmark/README.md`.
