# Mask2Former R50 — họ query / transformer

**Người phụ trách:** AnhVu (@tranphuocanhvu2103)
**Framework:** detectron2 + repo Mask2Former (`trainer: detectron2`,
`model.arch: mask2former`, `model.repo: third_party/Mask2Former`)

## Vai trò

Đại diện hướng **query-based**: N query học được, mỗi query nhận một vật thể
qua masked attention, mask sinh ở stride 4 trên toàn ảnh, không box, không NMS.
Đây cũng là khối thô của T2 — chạy riêng nó chính là ablation "T2 bỏ khối tinh
chỉnh biên".

## Vì sao chọn

- Cách tìm vật thể thứ ba, khác hẳn ROI và một-giai-đoạn.
- Mask không bị box hay ROI cắt → kỳ vọng bám đúng đường chia giữa hai tán
  chạm nhau; cần số để biết bỏ box đi thì biên tốt lên bao nhiêu.
- Điểm yếu đã biết của query với vật thể dày, cùng lớp, chạm nhau: hai query
  cùng nhận một tán, hoặc sót tán. Soi trên field_5.
- Cùng backbone R50 với Mask R-CNN và SOLOv2 → chênh lệch là do cơ chế.
- Số query phải lớn hơn số tán tối đa một ảnh (48); để 100.

## Cách chạy

```bash
bash scripts/remote/run_fold.sh f4 --smoke --only mask2former

python scripts/train.py --config members/mask2former/configs/train/mask2former_r50_d2.yaml --set data.root=data/export/f4 --workers 8

python scripts/evaluate.py --config configs/eval/coco_predictions.yaml --file preds/mask2former_f4.json --set data.root=data/export/f4 --split test
```

Recipe dùng chung ở `configs/train/_base_d2.yaml` (thuộc cả nhóm); file của
model chỉ đặt arch, repo, 100 epoch và số query.

## Trong thư mục này

| | |
|---|---|
| `configs/train/`, `configs/eval/` | config của model |
| `notes/` | nhật ký thí nghiệm: chạy gì, ra số gì, nhận xét |
| `results/` | file kết quả nhỏ chép từ `runs/` (runs/ không vào git) |
| `plugin.py` | (tuỳ chọn) code riêng của model này |

## Trạng thái

- **Chưa chạy thật lần nào.** Cần máy Linux: detectron2 build từ source + repo
  Mask2Former (submodule `third_party/Mask2Former`) + op `MSDeformAttn` biên
  dịch tại chỗ — `scripts/remote/setup.sh` làm hết. Việc đầu tiên:
  `run_fold.sh f4 --smoke --only mask2former`.
- Tốn giờ nhất trong bốn model: recipe 100 epoch, ước ~3 h một fold trên 4090.
  Chạy sau khi ba model kia đã xong một fold để không chiếm GPU quá lâu.
- Giữ nguyên tăng cường LSJ của recipe gốc (mapper trong `train_net.py` của
  repo), không áp flipud/rot90 như hai model R-CNN.
