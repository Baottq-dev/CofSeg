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

```bash
# khói trên máy Linux (bắt buộc làm trước lần chạy thật)
bash scripts/remote/run_fold.sh f4 --smoke --only maskrcnn

# một fold đầy đủ
python scripts/train.py --config members/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --set data.root=data/export/f4 --workers 8

# chấm lại ở nhà từ predictions.json mang về
python scripts/evaluate.py --config configs/eval/coco_predictions.yaml --file preds/maskrcnn_f4.json --set data.root=data/export/f4 --split test
```

Recipe dùng chung ở `configs/train/_base_d2.yaml` (thuộc cả nhóm — sửa là đổi
số của cả Cascade lẫn Mask2Former). Cascade: `--only cascade`.

## Trong thư mục này

| | |
|---|---|
| `configs/train/`, `configs/eval/` | config của model |
| `notes/` | nhật ký thí nghiệm: chạy gì, ra số gì, nhận xét |
| `results/` | file kết quả nhỏ chép từ `runs/` (runs/ không vào git) |
| `plugin.py` | (tuỳ chọn) code riêng của model này |

## Trạng thái

- Đường detectron2 **chưa chạy thật lần nào** — mới có code + config, chờ máy
  Linux có detectron2. Việc đầu tiên: `run_fold.sh f4 --smoke --only maskrcnn`.
- Có sẵn bản torchvision (`configs/train/maskrcnn_r50.yaml`) chạy được ở nhà
  trên Windows, dùng để thử nhanh; **không** dùng số của nó cho bảng.
