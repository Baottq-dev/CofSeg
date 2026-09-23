# Mask2Former R50 — họ query / transformer

**Người phụ trách:** AnhVu (@tranphuocanhvu2103)
**Framework:** detectron2 + repo Mask2Former (`trainer: detectron2`,
`model.arch: mask2former`, `model.repo: benchmark/mask2former/Mask2Former`)

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

Từ **gốc repo**:

```bash
python benchmark/run.py f4 --only mask2former --smoke    # vài iteration, kiểm đường chạy TRƯỚC
python benchmark/run.py f4 --only mask2former            # một fold đầy đủ
python benchmark/run.py f4 --only mask2former --epochs 30 --batch 2

# test của thư mục này
cd benchmark/mask2former && python -m pytest tests
```

Để lại `preds/mask2former_f4.json` ở gốc và các file kết quả trong
`benchmark/mask2former/results/`. Gọi thẳng cũng được:
`python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/... --runs benchmark/mask2former/runs`

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
không còn so được với ba model kia — `python benchmark/check_copies.py` sẽ báo.

## Mask2Former/ — repo gốc, submodule

| | |
|---|---|
| Nguồn | https://github.com/facebookresearch/Mask2Former |
| Commit ghim | `9b0651c` (20/05/2022, bản cuối của repo) |
| License | MIT |
| Dùng làm gì | `cofseg/models/detectron2.py` nạp `train_net.py` và gói `mask2former` làm module; op `MSDeformAttn` biên dịch tại chỗ bằng `scripts/setup_env.py` |

Repo con đi theo git: `git clone --recurse-submodules`, hoặc
`git submodule update --init benchmark/mask2former/Mask2Former` nếu đã clone rồi.

**Không sửa mã bên trong repo con.** Cần vá thì vá bằng code trong `cofseg/`
(như hàm `load_mask2former`). Đổi commit: `git -C benchmark/mask2former/Mask2Former
checkout <commit>` rồi commit ở repo ngoài — git ghi lại commit mới của repo con.

## Trạng thái

- **Chưa chạy thật lần nào.** Cần máy Linux: detectron2 build từ source + repo
  Mask2Former (submodule `benchmark/mask2former/Mask2Former`) + op `MSDeformAttn` biên
  dịch tại chỗ — `scripts/setup_env.py` làm hết. Việc đầu tiên:
  `run_fold.sh f4 --smoke --only mask2former`.
- Tốn giờ nhất trong bốn model: recipe 100 epoch, ước ~3 h một fold trên 4090.
  Chạy sau khi ba model kia đã xong một fold để không chiếm GPU quá lâu.
- Giữ nguyên tăng cường LSJ của recipe gốc (mapper trong `train_net.py` của
  repo), không áp flipud/rot90 như hai model R-CNN.
