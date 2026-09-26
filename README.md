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

Một env cho cả annotator (`app/`) lẫn benchmark.

`requirements.txt` **thuần wheel dựng sẵn** — không gói nào biên dịch, nên cài
được trên mọi máy mà không cần `nvcc`, CUDA toolkit hay trình biên dịch C++:

```
conda create -n coffee python=3.12 -y
conda activate coffee
pip install -r requirements.txt
pip install -e .
```

Đủ để chạy **YOLOv11-Seg** và **SOLOv2**. Hai model còn lại cần `detectron2`,
và annotator cần SAM 2 — cả hai build từ source nên nằm ngoài file đó:

| Gói | Cho | Cài bằng |
|---|---|---|
| `detectron2` | Mask R-CNN, Mask2Former | `setup_env.py --only detectron2` |
| `SAM 2.1` | annotator (`app/`) | `setup_env.py --only sam2` |

Muốn đủ bốn model thì chạy thẳng `python scripts/setup_env.py`, nó làm cả hai
bước đó theo đúng thứ tự.

Máy lab / máy thuê (Linux) — tám bước, dừng đúng chỗ lỗi; hỏng bước nào thì
`--from <tên bước>` chạy tiếp từ đó (`--list` xem các bước):

```
conda create -y -n cofseg python=3.12 && conda activate cofseg
python scripts/setup_env.py
```

**Máy lab có nvcc lệch phiên bản với torch** (ví dụ nvcc 13.2, torch cu121)
thì có hai đường, chọn một:

```
# A. không cài thêm gì. Lúc CÀI không biên dịch nhân CUDA nào;
#    lúc TRAIN vẫn dùng GPU đầy đủ.
python scripts/setup_env.py --skip-cuda-build

# B. cài nvcc khớp vào CHÍNH env này (không đụng CUDA của máy, không cần sudo)
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
python scripts/setup_env.py
```

| | A — `--skip-cuda-build` | B — cài toolkit vào env |
|---|---|---|
| Cài thêm | không | ~2–3 GB trong env của bạn |
| **Train trên GPU** | **có** | **có** |
| Mask R-CNN, SOLOv2, YOLO | như thường | như thường |
| Mask2Former | chậm hơn ~1,3–1,8 lần | đủ tốc độ |
| Kết quả | **giống nhau** | |

`--skip-cuda-build` chỉ tác động lúc **cài**: nó bảo pip đừng biên dịch nhân
CUDA tự viết của detectron2 và Mask2Former. Lúc **train**, torch vẫn là bản
cu121 và GPU vẫn chạy đầy tải. Mask2Former chậm hơn không phải vì rơi xuống
CPU — tensor vẫn trên GPU — mà vì phép attention đa tỉ lệ được ghép từ nhiều
`grid_sample` rời thay cho một nhân gộp sẵn.

Đường A dựa vào hai đường lùi có sẵn trong chính mã nguồn: `setup.py` của
detectron2 tự dựng `CppExtension` khi không thấy GPU lúc cài (Mask R-CNN
không dùng op CUDA riêng nào của nó — ROIAlign và NMS lấy của torchvision),
còn `ms_deform_attn.py` của Mask2Former bọc lời gọi op trong `try/except` và
rơi về bản thuần PyTorch.

Không nâng torch cho khớp CUDA 13 được: mmcv (SOLOv2) chỉ có wheel dựng sẵn
cho torch 2.4 / cu121 — index `cu124` và `torch2.5` của OpenMMLab đều không
tồn tại. Driver thì không sao, nó tương thích ngược.

**Trên Linux đừng chạy `pip install -r requirements.txt` một mình** — nó sẽ
dừng ở:

```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'detectron2' when getting requirements to build wheel
```

detectron2 và SAM 2 biên dịch op CUDA lúc cài và `setup.py` của họ import
torch, mà pip dựng gói trong môi trường cô lập không có torch — ở thời điểm
đó torch trong `requirements.txt` cũng chưa kịp cài. Torch phải đi bằng một
lệnh riêng, trước:

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
pip install -e .
```

`setup_env.py` làm đúng thứ tự đó và kiểm `nvcc` trước, nên dùng nó thì không
phải nhớ.

Mask2Former, Cascade (detectron2) và SOLOv2 (mmdet) **chưa chạy thật trên
Linux lần nào** — máy phát triển là Windows. Lần đầu nên đi từng bước và khói
một fold trước khi chạy cả sáu.

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
