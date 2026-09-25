# SOLOv2 R50-FPN — họ box-free

**Người phụ trách:** PhuongQuynh (@Phquynh2312)
**Framework:** mmdetection (`trainer: mmdet`, `model.arch: solov2`)

## Vai trò

Đại diện cho hướng **bỏ bounding box**: chia ảnh thành lưới, ô nào chứa tâm
vật thể thì ô đó sinh kernel động, tích chập lên đặc trưng mask stride 4 của
toàn ảnh. Không RPN, không ROI, không cắt mask theo box.

## Vì sao chọn

- Cách tìm vật thể khác hẳn Mask R-CNN (ROI) và YOLO (một giai đoạn cắt theo
  box) — bảng cần có mặt hướng này để đủ các nhóm.
- Mask không bị box cắt, nên chỗ hai tán chạm nhau không dính mép cây bên như
  YOLO. Đây là điểm cần soi trên field_5.
- Điểm yếu cần soi: mỗi ô lưới chỉ đại diện được một vật thể — hai tán cùng cỡ
  chạm nhau mà tâm rơi vào cùng ô thì mất một cây.
- Cùng backbone R50 với Mask R-CNN và Mask2Former.

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
python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --set data.root=data/export/$SET/$FOLD --runs benchmark/solov2/runs --name solov2-$FOLD

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
#    Tên file có cả $SET: hai bộ fold cùng đặt tên f1..f6, thiếu nó là đè nhau.
RUN=$(ls -td benchmark/solov2/runs/train/*_solov2-${FOLD}_${SET}-${FOLD}_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/solov2_${SET}_$FOLD.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/_coco.yaml --file preds/solov2_${SET}_$FOLD.json --set data.root=data/export/$SET/$FOLD --split test --runs benchmark/solov2/runs --name solov2_$FOLD

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/solov2/runs/eval/*_solov2_${FOLD}_${SET}-${FOLD}_* | head -1)
cp "$EV/metrics.json"   benchmark/solov2/results/solov2_${SET}_${FOLD}_metrics.json
cp "$EV/per_region.csv" benchmark/solov2/results/solov2_${SET}_${FOLD}_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/solov2 && python -m pytest tests
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

- **Chưa chạy thật lần nào.** mmcv không cài được trên Windows nên ở nhà chỉ
  kiểm được phần dựng config. Việc đầu tiên trên máy Linux:
  `run_fold.sh f4 --smoke --only solov2`.
- Khác biệt có chủ đích so với hai model detectron2: **không có xoay 90°**
  (mmdet không có transform sẵn cho mask + box), chỉ lật ngang/dọc/chéo. Nhớ
  ghi chú khi đọc bảng.
- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  `scripts/setup_env.py` nới dòng kiểm đó.
