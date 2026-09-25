# YOLOv11-Seg — họ một giai đoạn, thời gian thực

**Người phụ trách:** QuangBao (@Baottq-dev)
**Framework:** ultralytics (`trainer: yolo`)

## Vai trò

Đại diện nhóm real-time. Model này trả lời câu hỏi: **đưa model lên drone bay
trực tiếp thì độ chính xác giảm bao nhiêu** — so ms/ảnh và mAP với mốc 0.

## Vì sao chọn

- Model duy nhất trong bảng chạy được thời gian thực trên phần cứng nhúng.
  Cỡ **s** để so ngang với các model kia; cần số cho drone thật thì chạy thêm
  cỡ **n**.
- Mask từ 32 prototype ở stride 4 (~10 px một ô ở ảnh gốc) rồi cắt theo box:
  mịn hơn ROI 28×28 nhưng tán chạm nhau dễ dính mép cây bên — đo trên field_5.
- v11 là bản hiện hành, v8 là bản các bài trước hay dùng, cùng cơ chế mask;
  chạy cả hai nếu còn giờ (`yolo26s` cũng có sẵn config).

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

# 1. huấn luyện (thêm --epochs 1 --fraction 0.05 để khói)
python benchmark/yolo11/train.py --config benchmark/yolo11/configs/train/yolo11s.yaml --set data.yaml=data/export/$SET/$FOLD/data.yaml --runs benchmark/yolo11/runs --name yolo11s

# 2. chấm bằng trọng số tốt nhất
RUN=$(ls -td benchmark/yolo11/runs/train/*_yolo11s_${SET}-${FOLD}_* | head -1)
python benchmark/yolo11/evaluate.py --config benchmark/yolo11/configs/eval/yolo11s.yaml --set model.weights="$RUN/ultralytics/weights/best.pt" --set data.root=data/export/$SET/$FOLD --split test --runs benchmark/yolo11/runs --name yolo11s

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/yolo11/runs/eval/*_yolo11s_${SET}-${FOLD}_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/yolo11s_${SET}_$FOLD.json
cp "$EV/metrics.json"   benchmark/yolo11/results/yolo11s_${SET}_${FOLD}_metrics.json
cp "$EV/per_region.csv" benchmark/yolo11/results/yolo11s_${SET}_${FOLD}_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/yolo11 && python -m pytest tests
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

- Đường chạy **đã thử ở nhà**: probe (autobatch 13 → khuyến nghị 5 ở 1024),
  train 1 epoch trên 5 % fold f4, chấm 6 ảnh test ra `predictions.json`.
  Là model duy nhất trong bốn model đã đi hết đường train → eval.
- Chạy được trên Windows, không cần chờ máy Linux.
