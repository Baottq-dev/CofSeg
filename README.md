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

Cái giá: bốn bản sao phần chấm điểm, lệch nhau là bảng so sánh mất nghĩa —
`python benchmark/check_copies.py` kiểm điều đó. Bố cục, quy ước nhánh/commit
và cách commit đúng author khi dùng chung một máy: xem `benchmark/README.md`.

## 0. Yêu cầu

- GPU NVIDIA có CUDA (khuyến nghị ≥ 8 GB VRAM cho SAM2 hiera-large khi gán nhãn; benchmark cần 24 GB). Không có GPU vẫn chạy được annotator nhưng rất chậm (đặt `sam.device: cpu`).
- Python 3.12 (các gói ghim trong `requirements.txt` chỉ có wheel cho ≥ 3.12), ~25 GB đĩa trống cho weights + dữ liệu.
- Windows chạy được annotator và YOLO/Mask R-CNN (torchvision); Mask2Former, Cascade (detectron2) và SOLOv2 (mmdet) chỉ chạy trên Linux.

## 1. Lấy mã

```
git clone --recurse-submodules https://github.com/Baottq-dev/CofSeg.git CoffeeSeg
cd CoffeeSeg
# đã clone rồi mà thiếu third_party/Mask2Former:
git submodule update --init
```

`third_party/` là submodule (Mask2Former), `requirements.txt` ghim mọi thư viện — dòng chỉ-Linux có marker `sys_platform`, pip tự bỏ qua trên Windows.

## 2. Cài môi trường

Máy nhà (Windows):

```
conda create -n coffee python=3.12 -y
conda activate coffee
pip install -r requirements.txt
pip install -e .
```

Máy lab / máy thuê (Linux, có `nvcc`) — sáu bước, dừng đúng chỗ lỗi:

```
python scripts/setup_env.py
```

Chi tiết và cách chạy benchmark: `CHAY_MAY_LINUX.md`.

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
# cắt 6 fold từ một bản xuất chưa chia
python scripts/make_fold.py --export data/export/all_v2 --all

# cả bốn model trên một fold
python benchmark/run.py f4 --smoke      # kiểm đường chạy trước
python benchmark/run.py f4

# một người chạy model của mình
python benchmark/run.py f4 --only yolo11

# bốn bản chấm điểm có còn giống nhau không (số có so được không)
python benchmark/check_copies.py

# bảng model x ruộng từ kết quả của cả bốn thư mục
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
```

Chi tiết cho máy lab / máy thuê: `CHAY_MAY_LINUX.md`.
