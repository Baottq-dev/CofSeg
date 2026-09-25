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
này và `--name` theo đúng dạng `<model>_<fold>` — bảng tổng hợp đọc tên đó để
biết model nào chấm trên ruộng nào.

Lệnh dưới đây chạy trên bộ fold cắt theo cách val **block**
(`data/export/block/`). Nhóm đang cân nhắc hai cách chia val; đổi sang cách
kia chỉ cần `--set data.root=data/export/flight/$FOLD`. Cả hai bộ fold cắt ra
bằng `scripts/make_fold.py`, xem `benchmark/README.md`.

```bash
SET=block        # bộ fold: block hoặc flight
FOLD=f4

# 1. huấn luyện (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --set data.root=data/export/$SET/$FOLD --runs benchmark/maskrcnn/runs --name maskrcnn-$FOLD

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
#    Tên file có cả $SET: hai bộ fold cùng đặt tên f1..f6, thiếu nó là đè nhau.
RUN=$(ls -td benchmark/maskrcnn/runs/train/*_maskrcnn-${FOLD}_${SET}-${FOLD}_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/maskrcnn_${SET}_$FOLD.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/maskrcnn/evaluate.py --config benchmark/maskrcnn/configs/eval/_coco.yaml --file preds/maskrcnn_${SET}_$FOLD.json --set data.root=data/export/$SET/$FOLD --split test --runs benchmark/maskrcnn/runs --name maskrcnn_$FOLD

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/maskrcnn/runs/eval/*_maskrcnn_${FOLD}_${SET}-${FOLD}_* | head -1)
cp "$EV/metrics.json"   benchmark/maskrcnn/results/maskrcnn_${SET}_${FOLD}_metrics.json
cp "$EV/per_region.csv" benchmark/maskrcnn/results/maskrcnn_${SET}_${FOLD}_per_region.csv

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

- Đường detectron2 **chưa chạy thật lần nào** — mới có code + config, chờ máy
  Linux có detectron2. Việc đầu tiên: `run_fold.sh f4 --smoke --only maskrcnn`.
- Có sẵn bản torchvision (`configs/train/maskrcnn_r50.yaml`) chạy được ở nhà
  trên Windows, dùng để thử nhanh; **không** dùng số của nó cho bảng.
