# CoffeeSeg

Phân đoạn thực thể từng cây (tán) cà phê vối từ ảnh UAV nhìn thẳng xuống, và
gán nhãn mật độ hoa trên từng tán. Trọng tâm là chất lượng đường biên của mặt
nạ, nhất là ở chỗ các tán chạm nhau.

- **Dữ liệu:** 850 ảnh 2560×1440, 12.137 tán, 6 ruộng ở Ia Chim (Kon Tum). Dữ
  liệu chưa công bố nên repo không kèm theo.
- **Gán nhãn:** công cụ web tự xây (`app/`), dùng SAM 2 đề xuất mặt nạ từng tán
  từ một cú click hoặc một box, người gán nhãn sửa lại.
- **Benchmark:** bốn model segmentation (Mask R-CNN, SOLOv2, YOLO-Seg,
  Mask2Former), chấm theo kiểu giữ lại một ruộng làm test (6 fold).

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `app/` | Công cụ gán nhãn: backend FastAPI + SAM 2 (`server.py`, `annotator.py`), giao diện web (`static/`), gán nhãn mật độ hoa (`flower*.py`), xem dự đoán của model chồng lên ảnh (`predictions.py`) |
| `canopyseg/` | Thư viện dùng chung: đọc cấu hình, dữ liệu COCO/YOLO, cắt fold, chấm theo fold, đo chồng lấn giữa các ảnh, quản lý trọng số, ghi log lượt chạy |
| `benchmark/` | Bốn model của bảng so sánh, mỗi model một thư mục độc lập. Cài đặt và chạy: [`benchmark/README.md`](benchmark/README.md) |
| `configs/` | `config.yaml` (đường dẫn dữ liệu, SAM), `weights.yaml` (bản kê trọng số), `dataset/` (cách cắt fold), `flower_rounds.yaml` (các vòng gán nhãn hoa) |
| `scripts/` | Chuẩn bị dữ liệu, kiểm tra ảnh và chuyến bay, gộp kết quả, tải trọng số, huấn luyện model hoa |
| `tests/` | Test (pytest) |
| `data/`, `weights/`, `runs/`, `docs/` | Dữ liệu, trọng số, kết quả chạy và tài liệu nội bộ; **không vào git** |

## Cài đặt công cụ gán nhãn (`app/`)

Phần này chỉ dựng env cho `app/`. Benchmark dùng env riêng (xem mục *Benchmark*).

### 0. Yêu cầu

- GPU NVIDIA có CUDA (khuyến nghị ≥ 12 GB VRAM cho SAM 2 hiera-large). Không có
  GPU vẫn chạy nhưng rất chậm (đặt `sam.device: cpu` trong `configs/config.yaml`).
- Python 3.12, khoảng 20 GB đĩa trống cho trọng số và ảnh.
- Khuyến nghị Linux/WSL2: rasterio và pycocotools dễ cài hơn Windows thuần.

### 1. Cài môi trường

```bash
conda create -n CoffeeSegApp python=3.12 -y
conda activate CoffeeSegApp
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
pip install scikit-learn==1.9.1
pip install segmentation-models-pytorch==0.5.0
pip install pytest==9.1.1
pip install -e .
```

`requirements.txt` ghim torch với hậu tố `+cu121`, nên phải cài torch từ index
trên **trước**; thiếu index thì pip báo lỗi ngay thay vì âm thầm kéo bản CPU về.
`scikit-learn` dùng cho bộ phân loại pixel hoa, `segmentation-models-pytorch`
cho model segment hoa, `pytest` cho test.

### 2. Tải trọng số SAM 2

```bash
python scripts/download_weights.py --group annotator
```

Lệnh này tải `sam2.1_hiera_large.pt` (857 MB) về `weights/` theo
`configs/weights.yaml` và kiểm sha256. Muốn đủ bốn cỡ SAM 2.1 (large, base_plus,
small, tiny) thì dùng `--group sam_all`, hoặc tải tay từ
https://github.com/facebookresearch/sam2 về `weights/`. Cỡ nào thì
`sam.checkpoint` và `sam.model_cfg` trong `configs/config.yaml` phải khớp.

### 3. Dữ liệu

Giải nén `iachim_dataset_export/` vào `data/`. Cấu trúc:

```
data/
├─ iachim_dataset_export/
│   ├─ data_compressed/     ảnh gốc, theo field_*/...
│   └─ predictions/
└─ masks/                   nhãn người sửa lưu ở masks/corrected/
```

Đường dẫn do `configs/config.yaml` quy định (`data.images_dir`, `data.masks_dir`).

### 4. Chạy

```bash
python -m app.server
```

Mở http://localhost:8000. Thêm `-r` để backend tự nạp lại khi sửa mã. Muốn mở
thẳng một ruộng thì đặt `ANNOT_DIR` trước khi chạy, ví dụ
`export ANNOT_DIR=data/iachim_dataset_export/data_compressed/field_1`
(PowerShell: `$env:ANNOT_DIR="data/iachim_dataset_export/data_compressed/field_1"`).

### 5. Kiểm tra

```bash
pytest
```

Test dựng dữ liệu tại chỗ, không cần `data/` hay `weights/`. Marker `gpu`,
`slow`, `weights` (khai trong `pyproject.toml`) để lọc, ví dụ
`pytest -m "not slow"`.

## Các phần liên quan

**Trọng số.** Không vào git; `configs/weights.yaml` kê tên file, URL gốc và
sha256. `python scripts/download_weights.py --check` kiểm các file đã có, và
`--group` chọn nhóm cần tải (`annotator`, `sam_all`, `benchmark`, `backup`,
`all`).

**Gán nhãn mật độ hoa.** Tích hợp sẵn trong `app/`: đếm pixel hoa theo ngưỡng
màu, tô nét để bộ phân loại pixel học theo từng vòng, hoặc dùng model segment
hoa (U-Net) huấn luyện bằng `scripts/flower_train.py` và dự đoán bằng
`scripts/flower_predict.py`. Các vòng gán nhãn khai trong
`configs/flower_rounds.yaml`.

**Dữ liệu cho benchmark.** Benchmark đọc các thư mục fold (mỗi ruộng làm test
một lần). Cắt fold từ bản xuất bằng `scripts/make_fold.py` hoặc
`scripts/prepare_data.py`; bố cục thư mục fold ghi ở
[`benchmark/README.md`](benchmark/README.md) mục *Chạy*.

**Các script còn lại** trong `scripts/`:

| Script | Làm gì |
|---|---|
| `inspect_flights.py`, `measure_overlap.py`, `export_overlap_edges.py` | Xem nhật ký chuyến bay, đo chồng lấn giữa các ảnh, xuất đồ thị chồng lấn sang `configs/dataset/` |
| `compare_val_splits.py` | So các cách chia tập val trên cả sáu lượt |
| `summarize_folds.py`, `pack_results.py` | Gộp kết quả các lượt chấm thành bảng model × ruộng; gói kết quả để mang về máy khác |
| `07_data_inventory.py`, `05_viz_flower_hsv.py`, `06_superpixel_flower.py` | Phân tích riêng: kiểm kê ảnh theo chuyến bay, trực quan hoá đếm hoa HSV, thử phát hiện hoa bằng superpixel |

## Benchmark

Bốn model chạy trong **env riêng** (torch 2.11 + CUDA 13, dựng trên molab),
khác env của `app/` ở trên (torch 2.5.1 + CUDA 12.1); đừng cài chung. Cách cài,
cách chạy, siêu tham số và nơi lưu kết quả nằm ở
[`benchmark/README.md`](benchmark/README.md); mỗi model có README riêng trong
thư mục của nó (`maskrcnn/`, `solov2/`, `yolo/`, `mask2former/`).

## Tài liệu

`docs/` chứa báo cáo, kế hoạch và phân tích của dự án. Thư mục này nằm trong
`.gitignore` nên không có trên GitHub; mỗi thư mục con có `README.md` riêng ghi
mục đích và file nên đọc trước.
