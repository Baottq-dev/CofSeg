# CoffeeSeg

Phân đoạn thực thể từng tán cà phê vối (Robusta) từ ảnh UAV nhìn thẳng xuống, làm nền cho việc ước lượng mật độ hoa trên từng tán. Repo gồm:
- công cụ gán nhãn có SAM 2;
- benchmark tám mô hình phân đoạn thực thể theo giao thức giữ lại một ruộng;
- một mô hình nhẹ (khoảng 20% GFLOPs của YOLO26s-seg) huấn luyện bằng chắt lọc tri thức từ ba mô hình thầy.

- **Dữ liệu:** 850 ảnh RGB 2560×1440 chụp ở độ cao khoảng 10 m trên 6 ruộng ở Ia Chim (Kon Tum), 12.138 tán vẽ bằng polygon và được người kiểm (12.137 tán được chấm sau khi bỏ tán dưới 50 px²). Dữ liệu chưa công bố nên repo không kèm theo.
- **Giao thức đánh giá:** giữ lại một ruộng (6 fold). Ruộng thử không xuất hiện trong lúc train. Ruộng val dùng để chọn checkpoint cũng là một ruộng chưa thấy.
- **Chỉ số chính:** mask mAP@50:95 kiểu COCO, lấy trung bình không trọng số trên 6 ruộng thử. mAP@75 được đọc kèm.

## Mục lục

- [Kết quả](#kết-quả)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Cài đặt](#cài-đặt)
- [Dữ liệu](#dữ-liệu)
- [Huấn luyện và đánh giá](#huấn-luyện-và-đánh-giá)
- [Công cụ gán nhãn](#công-cụ-gán-nhãn)
- [Kiểm thử](#kiểm-thử)
- [Tái lập](#tái-lập)
- [Ghi công](#ghi-công)

## Kết quả

Mọi số dưới đây là **trung bình 3 seed (0, 1, 2) ± độ lệch chuẩn giữa 3 seed**, mỗi seed là trung bình 6 ruộng thử. Tham số và GFLOPs tính cho mô hình suy luận một lớp, ảnh vào 1024×576. GFLOPs = 2 × MACs, đo bằng `thop` trên đồ thị đã fuse.

### Benchmark

| Mô hình | Họ | Tham số (triệu) | GFLOPs | mask mAP@50:95 | mask mAP@75 |
|---|---|---:|---:|---:|---:|
| YOLOv8s-seg | Một giai đoạn | 11,8 | 57,5 | 64,85 ± 0,29 | 77,77 ± 0,55 |
| YOLO11s-seg | Một giai đoạn | 10,1 | 47,5 | 64,14 ± 0,31 | 77,32 ± 0,68 |
| YOLO26n-seg | Một giai đoạn | 2,69 | 13,15 | 64,65 ± 0,26 | 77,78 ± 0,13 |
| **YOLO26s-seg** (mô hình gốc) | Một giai đoạn | 10,37 | 49,54 | 65,89 ± 1,19 | 79,30 ± 1,60 |
| YOLO26l-seg | Một giai đoạn | 27,90 | 201,51 | 65,78 ± 0,71 | 79,31 ± 0,86 |
| Mask R-CNN R50-FPN | Hai giai đoạn | 43,9 | — | 63,99 ± 0,17 | 76,73 ± 0,33 |
| SOLOv2 R50-FPN | Không dùng hộp | 46,2 | — | **66,54 ± 0,26** | 79,14 ± 0,25 |
| Mask2Former R50 | Dựa trên truy vấn | 44,0 | — | 64,84 ± 0,25 | 76,60 ± 0,36 |

"—": chưa đo GFLOPs của ba mô hình dùng backbone ResNet-50.

**Vì sao chọn YOLO26s-seg làm mô hình gốc:**
- Lúc chọn, ở seed 0, YOLO26s-seg đạt 66,82 so với 66,72 của SOLOv2, với số tham số ít hơn 4,4 lần.
- Tính trung bình 3 seed thì SOLOv2 cao hơn 0,65 điểm. Mức chênh này nhỏ hơn độ lệch chuẩn giữa các seed của YOLO26s-seg (1,19).

### Mô hình nhẹ với chắt lọc tri thức (`distill/`)

**Mô hình trò** là YOLO26-seg độ rộng 0,25 với ba thay đổi ở đầu:
- lớp 3×3 thứ hai của Proto chạy ở stride 8;
- bỏ nhánh dự đoán ở P3;
- 16 prototype thay vì 32.

**Chắt lọc** học từ ba thầy khác họ (YOLO26s-seg, SOLOv2, Mask2Former):
- đích mặt nạ là trung bình mặt nạ của các thầy, trộn với nhãn;
- chắt lọc đặc trưng (PKD) từ YOLO26s-seg.

| Cấu hình | Tham số (triệu) | GFLOPs | mask mAP@50:95 | mask mAP@75 |
|---|---:|---:|---:|---:|
| YOLO26s-seg (mô hình gốc) | 10,37 | 49,54 | 65,89 ± 1,19 | 79,30 ± 1,60 |
| YOLO26s-seg + 8 phép xoay/lật | 10,37 | 49,54 | 66,86 ± 0,35 | 80,32 ± 0,24 |
| YOLO26n-seg | 2,69 | 13,15 | 64,65 ± 0,26 | 77,78 ± 0,13 |
| Trò, không chắt lọc | 2,57 | 9,95 | 63,48 ± 1,20 | 76,39 ± 1,25 |
| Trò + chắt lọc ba thầy | 2,57 | 9,95 | 65,87 ± 0,20 | 79,21 ± 0,14 |
| **Trò + chắt lọc + 8 phép xoay/lật** (cấu hình cuối) | **2,57** | **9,95** | 66,63 ± 0,22 | 80,00 ± 0,47 |

**Cách đọc bảng:**
- Cấu hình cuối dùng khoảng 20% GFLOPs và 25% tham số của YOLO26s-seg.
- So với YOLO26s-seg: mAP@50:95 chênh +0,74, nhỏ hơn độ lệch chuẩn giữa seed của mô hình gốc (1,19), nên đọc là **ngang**, không phải hơn.
- So với YOLO26s-seg train cùng 8 phép xoay/lật: thấp hơn 0,23.
- Chắt lọc tăng 2,39 điểm so với trò không chắt lọc, lợi chủ yếu ở độ khít mặt nạ (mAP@75, mAP@90).

### Toàn bộ kết quả: `results/`

[`results/ket_qua_tat_ca_luot_chay.xlsx`](results/ket_qua_tat_ca_luot_chay.xlsx) có mọi cấu hình đã chạy, mỗi cấu hình một sheet:
- 8 mô hình benchmark;
- 17 biến thể của YOLO26s-seg (loss, kiến trúc, tăng cường);
- 10 cấu hình của trò và chắt lọc, gồm các bước ablation và các cách chọn thầy.

Sheet "Mục lục" tóm tắt mỗi cấu hình một dòng.

**Bảng trong mỗi sheet:**
- **Cột:** `Ruộng, mAP@50:95, mAP@50, mAP@75, mAP@90, IoU, BIoU, P, R`.
- **Mỗi seed một bảng.** Dòng "Cả 6 ruộng" là trung bình không trọng số.
- **Cấu hình đủ 3 seed** có thêm bảng "trung bình ± độ lệch chuẩn 3 seed". Độ lệch chuẩn là độ lệch chuẩn mẫu (chia n − 1).
- **Số theo ruộng** lấy từ `metrics.json` của từng lượt chấm. Dòng tổng và bảng 3 seed là công thức Excel.

**Định nghĩa chỉ số:**
- mAP tính theo kiểu COCO (101 điểm recall), đơn vị %.
- IoU và BIoU (IoU dải biên, độ rộng dải bằng 2% cạnh của tán) tính trên các tán khớp.
- P và R lấy tại ngưỡng điểm cho F1 lớn nhất, đúng cách Ultralytics báo P, R cạnh mAP. Khớp tán ở IoU 0,5. Số nằm ở khoá `summary.pr_best_f1` của `metrics.json` (khoá đó ghi cả ngưỡng `conf` và F1).
- Hai khoá `summary.precision` và `summary.recall` của `metrics.json` là P, R ở ngưỡng chấm 0,05. Ngưỡng này để giữ đủ dự đoán cho đường precision–recall của mAP, nên hai số đó không dùng để báo cáo.

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `app/` | Công cụ gán nhãn: backend FastAPI + SAM 2 (`server.py`, `annotator.py`), giao diện web (`static/`), gán nhãn mật độ hoa (`flower*.py`), xem dự đoán của model chồng lên ảnh (`predictions.py`) |
| `canopyseg/` | Thư viện dùng chung: đọc cấu hình, dữ liệu COCO/YOLO, cắt fold, chấm theo fold, đo chồng lấn giữa các ảnh, quản lý trọng số, ghi log lượt chạy |
| `benchmark/` | Các mô hình của bảng so sánh, mỗi framework một thư mục độc lập (`yolo/`, `maskrcnn/`, `solov2/`, `mask2former/`) với code, config, test và README riêng. Xem [`benchmark/README.md`](benchmark/README.md) |
| `distill/` | Mô hình trò, chắt lọc tri thức, giám sát mặt nạ ngoài hộp. Dùng lại trainer, bộ đọc dữ liệu và bộ chấm của `benchmark/yolo`, nên số so thẳng được với benchmark. Xem [`distill/README.md`](distill/README.md) |
| `results/` | File kết quả đã gộp (mục [Kết quả](#toàn-bộ-kết-quả-results)) |
| `configs/` | `config.yaml` (đường dẫn dữ liệu, SAM), `weights.yaml` (bản kê trọng số), `dataset/` (cách cắt fold), `flower_rounds.yaml` (các vòng gán nhãn hoa) |
| `scripts/` | Chuẩn bị dữ liệu, kiểm tra ảnh và chuyến bay, gộp kết quả, tải trọng số, huấn luyện model hoa |
| `tests/` | Test (pytest) của phần dùng chung: `app/`, `canopyseg/`, `scripts/`, config và README của benchmark |
| `data/`, `weights/`, `runs/`, `docs/` | Dữ liệu, trọng số, kết quả chạy và tài liệu nội bộ; **không vào git** |

## Cài đặt

Repo dùng hai môi trường tách riêng, đừng cài chung:

| Môi trường | Dùng cho | torch |
|---|---|---|
| `CoffeeSegApp` | Công cụ gán nhãn `app/` | 2.5.1 + CUDA 12.1 |
| `cofseg` | `benchmark/` và `distill/` (dựng trên molab) | 2.11.0 + CUDA 13.0 |

### Môi trường gán nhãn (`app/`)

Yêu cầu:
- GPU NVIDIA có CUDA, khuyến nghị ≥ 12 GB VRAM cho SAM 2 hiera-large. Không có GPU vẫn chạy nhưng rất chậm (đặt `sam.device: cpu` trong `configs/config.yaml`).
- Python 3.12, khoảng 20 GB đĩa trống cho trọng số và ảnh.
- Khuyến nghị Linux/WSL2, vì rasterio và pycocotools dễ cài hơn Windows thuần.

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

**Ghi chú:**
- `requirements.txt` ghim torch với hậu tố `+cu121`, nên phải cài torch từ index ở trên **trước**. Thiếu index thì pip báo lỗi ngay, thay vì âm thầm kéo bản CPU về.
- `scikit-learn` dùng cho bộ phân loại pixel hoa, `segmentation-models-pytorch` cho model segment hoa, `pytest` cho test.

### Môi trường benchmark và chắt lọc (`benchmark/`, `distill/`)

Cách dựng env `cofseg` nằm ở [`benchmark/README.md`](benchmark/README.md), mục *Dựng môi trường*: torch 2.11.0+cu130, CUDA toolkit, detectron2, MMDetection và phần cài riêng của từng mô hình. `distill/` dùng chung env này, chỉ đòi đúng Ultralytics 8.4.143.

## Dữ liệu

Bản xuất của bộ dữ liệu đặt trong `data/`:

```
data/
├─ iachim_dataset_export/
│   └─ data_compressed/     ảnh 2560×1440, theo field_*/...
├─ masks/
│   └─ corrected/           nhãn đã người kiểm, mỗi ảnh một file JSON
└─ export/field/f1 … f6     thư mục fold giữ lại một ruộng
```

- Đường dẫn ảnh và nhãn do `configs/config.yaml` quy định (`data.images_dir`, `data.masks_dir`).
- Thư mục fold cắt bằng `scripts/make_fold.py` hoặc `scripts/prepare_data.py`. Bố cục ghi ở [`benchmark/README.md`](benchmark/README.md), mục *Chạy*.

## Huấn luyện và đánh giá

### Benchmark

Mỗi mô hình có lệnh train và chấm trong README của thư mục mình. Lệnh viết cho một fold; đổi `f1` thành `f2` … `f6` để chạy các fold còn lại. Gộp các lượt chấm thành bảng mô hình × ruộng:

```bash
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
```

### Trò và chắt lọc

Chạy từ gốc repo, env `cofseg`, cho từng fold. Ví dụ với fold 1, seed 0:

```bash
# 1. Ba thầy (lượt train seed 0 của benchmark cùng fold) suy luận trên ảnh train
python distill/teacher_infer.py --data data/export/field/f1 --weights-from runs/train_2 --runs runs/distill_teachers

# 2. Dựng đích mặt nạ: trung bình mặt nạ của các thầy khớp với từng tán nhãn
python distill/teacher_masks.py --data data/export/field/f1 --evals runs/distill_teachers/eval --out runs/distill_teachers/targets/mean3

# 3. Train trò với chắt lọc và 8 phép xoay/lật (cấu hình cuối)
python distill/train.py --config distill/configs/train/tro-025-kd.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0 --aug ag1

# 4. Chấm trên ruộng thử
python distill/evaluate.py --config distill/configs/eval/tro-025-kd-ag1.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0
```

- Trò không chắt lọc dùng `distill/configs/train/tro-025.yaml` và `distill/configs/eval/tro-025.yaml`.
- Thêm `--print-config` để in tham số cuối cùng mà không train.
- Phải chấm trò bằng `distill/evaluate.py`, vì checkpoint của trò chứa lớp của gói `distill`.

Chi tiết các khối config (`student:`, `kd:`, `outside_box:`) và các dòng ablation: [`distill/README.md`](distill/README.md).

### Trọng số

Trọng số không vào git. `configs/weights.yaml` kê tên file, URL gốc và sha256 của các checkpoint dùng chung.
- `python scripts/download_weights.py --check` kiểm các file đã có.
- `--group` chọn nhóm cần tải: `annotator`, `sam_all`, `benchmark`, `backup`, `all`.

## Công cụ gán nhãn

```bash
python scripts/download_weights.py --group annotator   # sam2.1_hiera_large.pt, 857 MB, kiểm sha256
python -m app.server
```

Mở http://localhost:8000.
- Thêm `-r` để backend tự nạp lại khi sửa mã.
- Muốn mở thẳng một ruộng thì đặt `ANNOT_DIR` trước khi chạy, ví dụ `export ANNOT_DIR=data/iachim_dataset_export/data_compressed/field_1` (PowerShell: `$env:ANNOT_DIR="data/iachim_dataset_export/data_compressed/field_1"`).
- Muốn đủ bốn cỡ SAM 2.1 thì dùng `--group sam_all`. Cỡ nào thì `sam.checkpoint` và `sam.model_cfg` trong `configs/config.yaml` phải khớp cỡ đó.

**Gán nhãn mật độ hoa** tích hợp sẵn trong `app/`, có ba cách:
- đếm pixel hoa theo ngưỡng màu;
- tô nét để bộ phân loại pixel học theo từng vòng;
- dùng model segment hoa (U-Net): huấn luyện bằng `scripts/flower_train.py`, dự đoán bằng `scripts/flower_predict.py`.

Các vòng gán nhãn khai trong `configs/flower_rounds.yaml`.

## Kiểm thử

```bash
pytest                       # tests/
python -m pytest distill/tests
```

- Test dựng dữ liệu tại chỗ, không cần `data/` hay `weights/`.
- Marker `gpu`, `slow`, `weights` (khai trong `pyproject.toml`) dùng để lọc, ví dụ `pytest -m "not slow"`.
- Mỗi thư mục trong `benchmark/` có test riêng.

## Tái lập

- **Seed:** mọi cấu hình chính chạy 3 seed (0, 1, 2) × 6 fold.
- **Phần cứng:** benchmark seed 0 và bốn biến thể đầu chạy trên máy thuê có GPU RTX 5090; các lượt còn lại chạy trên molab (NVIDIA RTX PRO 6000 Blackwell). Tổng cộng 436 lượt train, 205,8 giờ GPU.
- **Phần mềm:** PyTorch 2.11.0 (CUDA 13.0), Ultralytics 8.4.143, detectron2 0.6, MMDetection 3.3.0.
- **Nhật ký mỗi lượt chạy:** phiên bản phần mềm, GPU, commit và mã băm của bộ dữ liệu, để mọi con số truy được về đúng mã và dữ liệu đã sinh ra nó.
- **Chấm:**
  - mọi mô hình đi qua cùng một bộ chấm ở độ phân giải gốc 2560×1440, ngưỡng điểm 0,05, tối đa 100 dự đoán mỗi ảnh;
  - các YOLO và trò dùng NMS 0,7;
  - checkpoint chấm là `best.pt`, chọn theo mask AP trên ruộng val;
  - P, R, F1 tại ngưỡng cho F1 lớn nhất do bộ chấm ghi vào `summary.pr_best_f1`. Các lượt chấm trước ngày 14/11/2026 được thêm khối này bằng lệnh dưới đây. Lệnh chỉ đọc lại `per_region.csv` của lượt chấm, không chạy mô hình, và không đổi khoá nào khác:

    ```bash
    python benchmark/yolo/evaluate.py --resummarize runs/<thư mục>/eval/<lượt chấm>
    ```

## Tài liệu

`docs/` chứa báo cáo, kế hoạch và phân tích của dự án. Thư mục này nằm trong `.gitignore` nên không có trên GitHub.

## Ghi công

Repo dựa trên các dự án mã nguồn mở sau; giấy phép của từng dự án áp dụng cho phần mã tương ứng:
- [SAM 2](https://github.com/facebookresearch/sam2);
- [Ultralytics](https://github.com/ultralytics/ultralytics);
- [detectron2](https://github.com/facebookresearch/detectron2);
- [MMDetection](https://github.com/open-mmlab/mmdetection);
- [Mask2Former](https://github.com/facebookresearch/Mask2Former).
