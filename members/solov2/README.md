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

```bash
bash scripts/remote/run_fold.sh f4 --smoke --only solov2

python scripts/train.py --config members/solov2/configs/train/solov2_r50_mm.yaml --set data.root=data/export/f4 --workers 8

python scripts/evaluate.py --config configs/eval/coco_predictions.yaml --file preds/solov2_f4.json --set data.root=data/export/f4 --split test
```

`members/solov2/configs/mmdet/solov2_r50_coffee.py` là phần ghi đè lên config zoo của mmdet
(một lớp, ngưỡng test); trainer nạp config zoo trong gói mmdet rồi gộp file này.

## Trong thư mục này

| | |
|---|---|
| `configs/train/`, `configs/eval/`, `configs/mmdet/` | config của model + phần ghi đè config mmdet |
| `notes/` | nhật ký thí nghiệm: chạy gì, ra số gì, nhận xét |
| `results/` | file kết quả nhỏ chép từ `runs/` (runs/ không vào git) |
| `plugin.py` | (tuỳ chọn) code riêng của model này |

## Trạng thái

- **Chưa chạy thật lần nào.** mmcv không cài được trên Windows nên ở nhà chỉ
  kiểm được phần dựng config. Việc đầu tiên trên máy Linux:
  `run_fold.sh f4 --smoke --only solov2`.
- Khác biệt có chủ đích so với hai model detectron2: **không có xoay 90°**
  (mmdet không có transform sẵn cho mask + box), chỉ lật ngang/dọc/chéo. Nhớ
  ghi chú khi đọc bảng.
- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  `scripts/remote/setup.sh` nới dòng kiểm đó.
