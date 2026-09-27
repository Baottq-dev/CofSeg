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

`benchmark/mask2former/upstream` là submodule (repo gốc của Mask2Former), `requirements.txt` ghim mọi thư viện — dòng chỉ-Linux có marker `sys_platform`, pip tự bỏ qua trên Windows.

## 2. Cài môi trường

**Một env** cho cả annotator (`app/`) lẫn benchmark, chạy được cả bốn model.

```
conda create -y -n cofseg python=3.12 && conda activate cofseg
python scripts/setup_env.py --skip-cuda-build
```

Tám bước, dừng đúng chỗ lỗi và in cách chạy tiếp. `--list` xem các bước,
`--from <tên bước>` chạy tiếp từ một bước, `--only <tên bước>` chạy đúng một.

### `--skip-cuda-build` là gì

Lúc **cài** thì không biên dịch nhân CUDA tự viết nào, nên không cần `nvcc`
khớp phiên bản với torch, không cần CUDA toolkit, không cần g++ đúng đời. Lúc
**train** vẫn dùng GPU đầy đủ — cờ này chỉ đi vào các lệnh pip.

| | |
|---|---|
| Mask R-CNN, SOLOv2, YOLOv11-Seg | không đổi gì |
| Mask2Former | chậm hơn, **đúng kết quả** — MSDeformAttn chạy bằng op PyTorch thường, vẫn trên GPU |

Bỏ cờ này đi thì Mask2Former đủ tốc độ, nhưng máy phải có **cả ba**: CUDA
toolkit đầy đủ trong env, `g++ ≤ 12`, và `nvcc` cùng major với torch. Máy lab
hiện không thoả (nvcc 13.2, g++ > 12). Chi tiết và cách đổi ý sau:
`docs/reports/cai_moi_truong_may_lab_2026-09-27.md`.

### Một file cho app/, một file cho mỗi model

`requirements.txt` ở gốc chỉ lo **annotator (`app/`) và `canopyseg/`**. Mỗi
model benchmark khai gói riêng trong thư mục của nó:

```
pip install -r requirements.txt            # app/ + canopyseg
pip install -e .

pip install -r benchmark/requirements.txt  # cả bốn model, MỘT env
```

Muốn **bốn env riêng** thì cài từng file, mỗi env một lệnh:

```
pip install -r benchmark/yolo11/requirements.txt
pip install -r benchmark/solov2/requirements.txt
pip install -r benchmark/maskrcnn/requirements.txt
pip install -r benchmark/mask2former/requirements.txt
```

Hai cách đều ra **cùng một bản torch**, vì cả bốn file cùng `-r ../base.txt`
và `base.txt` `-r torch.txt`:

```
benchmark/torch.txt        torch + torchvision   <- ĐỔI Ở ĐÂY, một chỗ
benchmark/base.txt         -r torch.txt + numpy/cv2/pycocotools/yaml/tqdm
benchmark/<model>/requirements.txt   -r ../base.txt + gói riêng của model
```

torch là lựa chọn của **máy** (kiến trúc GPU), không phải của model — nên nó
không nằm trong file của từng model. `benchmark/torch.txt` có hai khối, bỏ
chú thích một khối:

| khối | torch | GPU | mmcv |
|---|---|---|---|
| **A** (mặc định) | 2.4.1+cu121 | `sm_50`–`sm_90`: 4090, L4, L40S, A40, A6000, A100, H100 | **wheel dựng sẵn** |
| **B** | 2.11.0+cu130 | `sm_75`–`sm_120`, gồm RTX 50xx, RTX PRO 6000, B200 | **phải build từ nguồn** |

Ranh giới thật sự không phải "Blackwell hay không", mà là **mmcv có wheel hay
phải build**: OpenMMLab chỉ phát hành cu118/cu121 tới torch 2.4, và torch
2.4/cu121 ra đời trước Blackwell nên không có kernel `sm_120`.

Chạy benchmark **không cần** `pip install -e .`: mỗi thư mục tự chứa bản
`cofseg/` riêng và `train.py` tự thêm thư mục của nó vào `sys.path`.

Mọi gói trong các file trên là **wheel dựng sẵn** — không gói nào biên dịch.
Hai gói build từ source nằm ngoài, vì chúng phụ thuộc vào môi trường lúc cài
chứ không chỉ vào phiên bản:

| Gói | Cho | Cài bằng |
|---|---|---|
| `detectron2` | Mask R-CNN, Mask2Former | `setup_env.py --only detectron2` |
| `SAM 2.1` | annotator (`app/`) | `setup_env.py --only sam2` |

### Kiểm lại sau khi cài

```
python -c "
import torch, ultralytics, detectron2, mmdet, mmcv
from mmcv.ops import nms
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))
print('detectron2', detectron2.__version__, '| mmdet', mmdet.__version__,
      '| mmcv', mmcv.__version__, '| ultralytics', ultralytics.__version__)
"
```

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

# 2. cắt fold. Sáu lượt như nhau (mỗi ruộng làm test một lần, train năm ruộng
#    còn lại), khác nhau ở cách cắt val. Nhóm chạy CẢ HAI cách nên cắt cả hai bộ.
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# 3. bảng so hai cách chia trên cả sáu lượt (số liệu cho báo cáo)
python scripts/compare_val_splits.py

# trên máy Linux, gộp bước giải nén và cắt fold làm một
python scripts/prepare_data.py all_v2.tar --val configs/dataset/val_block.yaml

# 4. mỗi người chạy model của mình: lệnh trong benchmark/<model>/README.md
#    12 lượt mỗi model (6 fold x 2 bộ), cả nhóm 48 lượt

# 5. bảng model x ruộng từ kết quả của cả bốn thư mục
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block

# 6. gói kết quả mang về
python scripts/pack_results.py
```

Ai phụ trách model nào, ba quy ước đặt tên khi chạy, và cách gộp bảng:
`benchmark/README.md`.
