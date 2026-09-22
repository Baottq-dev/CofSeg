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

```bash
python scripts/train.py --config members/yolo11/configs/train/yolo11s.yaml --set data.yaml=data/export/f4/data.yaml --imgsz 1024 --batch 4

python scripts/evaluate.py --config members/yolo11/configs/eval/yolo11s.yaml --set model.weights=runs/train/<...>/ultralytics/weights/best.pt --set data.root=data/export/f4 --split test --imgsz 1024
```

Hai bản còn lại của họ YOLO (`yolo26s`, `yolov8s`) kế thừa cùng khối train,
chỉ khác trọng số.

## Trong thư mục này

| | |
|---|---|
| `configs/train/`, `configs/eval/` | config của model |
| `notes/` | nhật ký thí nghiệm: chạy gì, ra số gì, nhận xét |
| `results/` | file kết quả nhỏ chép từ `runs/` (runs/ không vào git) |
| `plugin.py` | (tuỳ chọn) code riêng của model này |

## Trạng thái

- Đường chạy **đã thử ở nhà**: probe (autobatch 13 → khuyến nghị 5 ở 1024),
  train 1 epoch trên 5 % fold f4, chấm 6 ảnh test ra `predictions.json`.
  Là model duy nhất trong bốn model đã đi hết đường train → eval.
- Chạy được trên Windows, không cần chờ máy Linux.
