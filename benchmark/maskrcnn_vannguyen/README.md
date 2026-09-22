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

Từ **gốc repo**:

```bash
bash benchmark/maskrcnn_vannguyen/run.sh f4 --smoke     # vài iteration, kiểm đường chạy TRƯỚC
bash benchmark/maskrcnn_vannguyen/run.sh f4             # một fold đầy đủ
bash benchmark/maskrcnn_vannguyen/run.sh f4 --epochs 30 --batch 2

# test của thư mục này
cd benchmark/maskrcnn_vannguyen && python -m pytest tests
```

`run.sh` để lại `preds/maskrcnn_f4.json` ở gốc và các file kết quả trong
`benchmark/maskrcnn_vannguyen/results/`.

## Trong thư mục này

| | |
|---|---|
| `cofseg/` | bản sao lõi của riêng thư mục: đọc dữ liệu, chỉ số, vòng chấm, trainer + model của model này |
| `configs/train/`, `configs/eval/` | cấu hình huấn luyện và chấm |
| `scripts/train.py`, `scripts/evaluate.py` | bản riêng, nạp `cofseg/` của thư mục này |
| `run.sh` | train một fold, chấm, chép kết quả |
| `tests/` | test cho phần của mình — chạy trước khi commit |
| `runs/` | kết quả train/eval (không vào git) |
| `results/` | file kết quả nhỏ, được commit |
| `notes/experiments.md` | nhật ký thí nghiệm |

Sửa gì trong đây cũng được, kể cả `cofseg/`. Riêng phần chấm điểm
(`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`) mà sửa thì số
không còn so được với ba model kia — `python benchmark/check_copies.py` sẽ báo.

## Trạng thái

- Đường detectron2 **chưa chạy thật lần nào** — mới có code + config, chờ máy
  Linux có detectron2. Việc đầu tiên: `run_fold.sh f4 --smoke --only maskrcnn`.
- Có sẵn bản torchvision (`configs/train/maskrcnn_r50.yaml`) chạy được ở nhà
  trên Windows, dùng để thử nhanh; **không** dùng số của nó cho bảng.
