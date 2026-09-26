# Mask2Former R50 — họ query / transformer

**Người phụ trách:** AnhVu (@tranphuocanhvu2103)
**Framework:** detectron2 + repo Mask2Former (`trainer: detectron2`,
`model.arch: mask2former`, `model.repo: benchmark/mask2former/upstream`)

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

Từ **gốc repo** (để `data/` và `weights/` dùng chung). Đặt `--runs` vào thư mục
này và `--name` theo đúng dạng `<model>_<fold>` — bảng tổng hợp đọc tên đó để
biết model nào chấm trên ruộng nào.

`--data` nhận **thư mục fold**, viết thẳng ra cũng được —
`--data data/export/flight/f1` — nhìn lệnh là biết đang chạy bộ nào, fold nào.
Cùng một cú pháp cho cả bốn model, dù bên trong ba model đọc `data.root` còn
ultralytics đọc `data.yaml`. Hai bộ fold cắt ra bằng `scripts/make_fold.py`,
xem `benchmark/README.md`.

```bash
SET=block        # bộ fold: block hoặc flight
FOLD=f4

# 1. huấn luyện (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/$SET/$FOLD --runs benchmark/mask2former/runs --name mask2former

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
RUN=$(ls -td benchmark/mask2former/runs/train/*_mask2former_${SET}-${FOLD}_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/mask2former_${SET}_$FOLD.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/mask2former/evaluate.py --config benchmark/mask2former/configs/eval/_coco.yaml --file preds/mask2former_${SET}_$FOLD.json --data data/export/$SET/$FOLD --split test --runs benchmark/mask2former/runs --name mask2former

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/mask2former/runs/eval/*_mask2former_${SET}-${FOLD}_* | head -1)
cp "$EV/metrics.json"   benchmark/mask2former/results/mask2former_${SET}_${FOLD}_metrics.json
cp "$EV/per_region.csv" benchmark/mask2former/results/mask2former_${SET}_${FOLD}_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/mask2former && python -m pytest tests
```

## Trong thư mục này

| | |
|---|---|
| `upstream/` | repo gốc facebookresearch/Mask2Former, submodule ghim commit |
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

## upstream/ — repo gốc của Mask2Former (submodule)

| | |
|---|---|
| Nguồn | https://github.com/facebookresearch/Mask2Former |
| Commit ghim | `9b0651c` (20/05/2022, bản cuối của repo) |
| License | MIT |
| Dùng làm gì | `cofseg/models/detectron2.py` nạp `train_net.py` và gói `mask2former` làm module; op `MSDeformAttn` biên dịch tại chỗ bằng `scripts/setup_env.py` |

Repo con đi theo git: `git clone --recurse-submodules`, hoặc
`git submodule update --init benchmark/mask2former/upstream` nếu đã clone rồi.
Đặt tên `upstream/` chứ không phải `Mask2Former/` vì trùng tên thư mục cha,
chỉ khác hoa thường — trên Windows và macOS nhìn như một.

**Không sửa mã bên trong repo con.** Cần vá thì vá bằng code trong `cofseg/`
(như hàm `load_mask2former`). Đổi commit: `git -C benchmark/mask2former/upstream
checkout <commit>` rồi commit ở repo ngoài — git ghi lại commit mới của repo con.

## Trạng thái

- **Chưa chạy thật lần nào.** Cần máy Linux: detectron2 build từ source + repo
  Mask2Former (submodule `benchmark/mask2former/upstream`) + op `MSDeformAttn` biên
  dịch tại chỗ — `scripts/setup_env.py` làm hết. Việc đầu tiên:
  `run_fold.sh f4 --smoke --only mask2former`.
- Tốn giờ nhất trong bốn model: recipe 100 epoch, ước ~3 h một fold trên 4090.
  Chạy sau khi ba model kia đã xong một fold để không chiếm GPU quá lâu.
- Giữ nguyên tăng cường LSJ của recipe gốc (mapper trong `train_net.py` của
  repo), không áp flipud/rot90 như hai model R-CNN.
