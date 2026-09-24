# Mask R-CNN R50-FPN — mốc số 0

**Người phụ trách:** VanNguyen (@vnguyen123)
**Framework:** detectron2 (`trainer: detectron2`, `model.arch: maskrcnn`)

## Vai trò

Cột mốc số 0 của bảng benchmark: mọi model khác báo cáo **Δ% mAP so với model
này** trên cùng một ruộng. Vì vậy hai thứ phải giữ nghiêm:

- Đổi cấu hình của model này (imgsz, số epoch, augmentation) là đổi mốc của cả
  bốn model → phải báo nhóm trước khi đổi.
- Chạy đủ 6 fold trước các model khác nếu được, để họ có mốc mà so.

## Vì sao chọn

- Kiến trúc gốc mà Cascade, PointRend, Mask Transfiner đều sửa từ nó — lấy làm
  mốc thì chênh lệch trong bảng đọc được là "đổi khối nào được bao nhiêu".
- Mask 28×28 cho mỗi ROI; tán ~366 px thì mỗi ô ≈ 13 px, biên thô — đây là
  cận dưới của trục biên trong bảng.
- Xử lý từng ROI riêng, không biết cây bên cạnh → tán chạm nhau hay tràn mask.

## Cách chạy

Từ **gốc repo** (để `data/` và `weights/` dùng chung). Đặt `--runs` vào thư mục
này, và `--name` chỉ là **tên model** — bảng tổng hợp đọc fold và bộ fold từ
đường dẫn dữ liệu chứ không từ tên, nên gõ lại fold vào tên chỉ tạo ra một bản
thứ hai có thể sai lệch.

`--data` nhận **thư mục fold**. Viết thẳng đường dẫn ra, đừng đặt biến shell:
nhìn lệnh là biết ngay đang chạy bộ nào, fold nào. Cùng một cú pháp cho cả bốn
model, dù bên trong ba model đọc `data.root` còn ultralytics đọc `data.yaml`.

Lệnh dưới đây chạy bộ **block**, fold **f1**. Đổi lượt khác thì sửa `block`
hoặc `f1` — có ba chỗ trong khối lệnh, sửa hết cả ba. Hai bộ fold cắt ra bằng
`scripts/make_fold.py`, xem `benchmark/README.md`.

```bash
# 1. huấn luyện (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --runs benchmark/maskrcnn/runs --name maskrcnn --workers 8

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
RUN=$(ls -td benchmark/maskrcnn/runs/train/*_maskrcnn_block-f1_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/maskrcnn_block_f1.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/maskrcnn/evaluate.py --config benchmark/maskrcnn/configs/eval/_coco.yaml --file preds/maskrcnn_block_f1.json --data data/export/block/f1 --split test --runs benchmark/maskrcnn/runs --name maskrcnn

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/maskrcnn/runs/eval/*_maskrcnn_block-f1_* | head -1)
cp "$EV/metrics.json"   benchmark/maskrcnn/results/maskrcnn_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/maskrcnn/results/maskrcnn_block_f1_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/maskrcnn && python -m pytest tests
```

## Trong thư mục này

| | |
|---|---|
| `cofseg/` | bản sao lõi của riêng thư mục: đọc dữ liệu, chỉ số, vòng chấm, trainer + model của model này |
| `configs/train/`, `configs/eval/` | cấu hình huấn luyện và chấm |
| `train.py`, `evaluate.py` | bản riêng, nạp `cofseg/` của thư mục này |
| `tests/` | test cho phần của mình — chạy trước khi commit |
| `runs/` | kết quả train/eval (không vào git) |
| `results/` | file kết quả nhỏ, được commit |
| `notes/experiments.md` | nhật ký thí nghiệm |

Sửa gì trong đây cũng được, kể cả `cofseg/`. Riêng phần chấm điểm
(`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`) mà sửa thì số
không còn so được với ba model kia. Sửa thì báo nhóm.

## Trạng thái

- **Đã chạy thật trên máy lab** (24/09/2026): một lượt khói 3 epoch trên
  `block/f1`, đi hết đường train → chấm val mỗi epoch → chấm test trên
  field_1. Val mAP50-95 lên 3,76 → 12,81 → 27,61 và vẫn đang dốc, tức là
  đường chạy thông chứ chưa phải kết quả.

  ```bash
  python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --imgsz 1024 --batch 16 --epochs 3 --workers 4
  ```

  Một điểm phải theo dõi khi chạy đủ epoch: `APs` = 0,000 ở cả ba epoch và
  `APm` chỉ 1,51 — gần như toàn bộ điểm đến từ tán `large`, tán nhỏ chưa bắt
  được cái nào.

- Bản torchvision chạy ở nhà trên Windows đã bị gỡ khỏi repo; giờ chỉ còn
  đường detectron2, nên phần này của bảng phải chờ máy Linux.
