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

## Dựng môi trường riêng

Phần này dành cho trường hợp chạy **một mình model này**, env riêng, không dùng
env chung của `benchmark/requirements.txt`. Lệnh ở đây đủ để chép chạy từ đầu
đến cuối; phần *vì sao* (vì sao CUDA 12.8, vì sao mmcv không có wheel, vì sao
mmcv dựng bản rỗng mà không báo lỗi) nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Model **nặng cài nhất**: cần detectron2, cần repo con `upstream/`, và cần biên
dịch op CUDA `MSDeformAttn` tại chỗ.

```bash
conda create -y -n cofseg-mask2former python=3.12 && conda activate cofseg-mask2former
nvcc --version | tail -2
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/mask2former/requirements.txt
pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

# repo con, nếu lúc clone chưa kèm --recurse-submodules
git submodule update --init benchmark/mask2former/upstream

# op CUDA của riêng model này
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
pip install --no-build-isolation --no-deps benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops
```

`nvcc --version` phải ra **12.x**. Lệch major với `torch.version.cuda` là gói
biên dịch từ nguồn gãy ở `build_ext`, bằng một câu không hề nhắc tới torch.
Máy có nvcc 13.x thì cài `cuda-toolkit=12.8.1` vào chính env — xem
`benchmark/README.md` mục *Cài mmcv*.

Dùng pip build thẳng thư mục `ops` chứ đừng chạy `make.sh` của repo gốc: nó gọi
`setup.py install`, thứ setuptools mới đã bỏ.

Kiểm:

```bash
python -c "
import torch, torchvision, detectron2
from detectron2 import model_zoo
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print('detectron2', detectron2.__version__)"
PYTHONPATH=benchmark/mask2former/upstream python -c "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

**Bỏ bước biên dịch op thì vẫn chạy, nhưng tốn gấp nhiều lần VRAM.** Mã gốc bọc
lời gọi op trong `try/except` trần rồi rơi xuống `ms_deform_attn_core_pytorch`,
và đường lùi đó hiện vật hoá một tensor mà kernel CUDA không bao giờ dựng: 3.94
GiB mỗi lớp ở imgsz 1024 batch 16, nhân 6 lớp encoder. Xem mục *Trạng thái* bên
dưới. Buộc phải bỏ thì hạ `--batch` xuống 2–4 và xác nhận bằng `--probe` trước
khi đặt lịch chạy dài.

Torchvision có phần mở rộng C++ link vào `libtorch`: bản dựng cho CUDA khác sẽ
`import` trót lọt rồi gãy lúc gọi op. Vì vậy cài torch và torchvision **trong
cùng một lệnh**, đừng cài rời.

## Cách chạy

Từ **gốc repo** (để `data/` và `weights/` dùng chung). Đặt `--runs` vào thư mục
này, và `--name` chỉ là **tên model** — bảng tổng hợp đọc fold và bộ fold từ
đường dẫn dữ liệu chứ không từ tên, nên gõ lại fold vào tên chỉ tạo ra một bản
thứ hai có thể sai lệch.

`--data` nhận **thư mục fold**. Viết thẳng đường dẫn ra, đừng đặt biến shell:
nhìn lệnh là biết ngay đang chạy bộ nào, fold nào. Cùng một cú pháp cho cả bốn
model, dù bên trong ba model đọc `data.root` còn ultralytics đọc `data.yaml`.

Lệnh dưới đây chạy bộ **block**, fold **f1**. Đổi lượt khác thì sửa `block`
hoặc `f1` — có ba chỗ trong khối lệnh, sửa hết cả ba. Hai bộ fold cắt ra bằng
`scripts/make_fold.py`, xem `benchmark/README.md`.

```bash
# 1. huấn luyện (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/block/f1 --runs benchmark/mask2former/runs --name mask2former --workers 8

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
RUN=$(ls -td benchmark/mask2former/runs/train/*_mask2former_block-f1_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/mask2former_block_f1.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/mask2former/evaluate.py --config benchmark/mask2former/configs/eval/_coco.yaml --file preds/mask2former_block_f1.json --data data/export/block/f1 --split test --runs benchmark/mask2former/runs --name mask2former

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/mask2former/runs/eval/*_mask2former_block-f1_* | head -1)
cp "$EV/metrics.json"   benchmark/mask2former/results/mask2former_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/mask2former/results/mask2former_block_f1_per_region.csv

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
| Dùng làm gì | `cofseg/models/detectron2.py` nạp `train_net.py` và gói `mask2former` làm module; op `MSDeformAttn` biên dịch tại chỗ — xem `benchmark/README.md` mục *Biên dịch op CUDA cho Mask2Former* |

Repo con đi theo git: `git clone --recurse-submodules`, hoặc
`git submodule update --init benchmark/mask2former/upstream` nếu đã clone rồi.
Đặt tên `upstream/` chứ không phải `Mask2Former/` vì trùng tên thư mục cha,
chỉ khác hoa thường — trên Windows và macOS nhìn như một.

**Không sửa mã bên trong repo con.** Cần vá thì vá bằng code trong `cofseg/`
(như hàm `load_mask2former`). Đổi commit: `git -C benchmark/mask2former/upstream
checkout <commit>` rồi commit ở repo ngoài — git ghi lại commit mới của repo con.

## Trạng thái

- **24/09: chạy thật lần đầu trên máy lab, OOM ngay iteration 1 ở `--batch 16`.**
  Không phải sự cố ngẫu nhiên, và biết trước được. Hai nguyên nhân cộng lại:

  1. `--skip-cuda-build` nên op `MSDeformAttn` không có kernel CUDA. Mã gốc
     bọc lời gọi trong `try/except` trần: mỗi forward ném `AttributeError:
     'NoneType' object has no attribute 'ms_deform_attn_forward'` rồi rơi
     xuống `ms_deform_attn_core_pytorch`. Đường dự phòng đó **hiện vật hoá**
     một tensor mà kernel CUDA không bao giờ dựng:

     ```
     torch.stack(sampling_value_list, dim=-2)   # (N*M, D, Lq, L, P)
     16*8 x 32 x 21504 x 3*4 x 4 byte = 3.94 GiB
     ```

     Đúng con số trong `Tried to allocate 3.94 GiB`. Encoder có **6 lớp** và
     autograd giữ lại cho backward, nên riêng khoản này ~23.6 GiB. Chạy một
     mình trên 4090 24 GB cũng không đủ. (`Lq = 128² + 64² + 32² = 21504` ở ô
     LSJ 1024x1024; MSDeformAttn chạy fp32 vì `@autocast(enabled=False)`.)

  2. Recipe gốc là **batch 16 trên 8 GPU**, tức 2 ảnh mỗi GPU. Dồn cả 16 vào
     một card là gấp 8 lần bộ nhớ mỗi GPU so với recipe.

  Nên **batch 2–4** cho model này trên một card, và xác nhận bằng `--probe`
  trước khi đặt lịch chạy dài. Batch khác ba model kia thì phải ghi vào bảng
  kết quả, vì nó ảnh hưởng tới cả tốc độ lẫn chất lượng.

- **Mapper LSJ của họ gãy trên ảnh nền**, đã vá bằng `empty_safe_mapper` trong
  `cofseg/models/detectron2.py` (không sửa repo con). Dòng gãy:

  ```python
  instances = utils.annotations_to_instances(annos, image_shape)
  instances.gt_boxes = instances.gt_masks.get_bounding_boxes()   # <- gãy
  ...
  if hasattr(instances, 'gt_masks'):                             # <- có canh
  ```

  `annotations_to_instances` chỉ gắn `gt_masks` khi `len(annos)` > 0. Dòng canh
  ở dưới cho thấy tác giả biết trường đó có thể vắng, chỉ sót một chỗ. Upstream
  không lộ vì recipe COCO của họ để `FILTER_EMPTY_ANNOTATIONS: True`; ta cố ý
  để `False` để giữ ảnh nền. block/f1 có **2 ảnh nền trên 470 ảnh train** (val
  và test không có) — 0.4% đủ giết lượt chạy ở iteration 43/235.

  Chọn vá mapper chứ không bật `FILTER_EMPTY_ANNOTATIONS` cho riêng model này:
  bốn model phải ăn cùng một tập dữ liệu thì bảng so sánh mới có nghĩa.
  **Chưa chạy thật trên GPU**; đường mất mát khi cả batch đều là ảnh nền mới
  chỉ đọc mã, chưa đo.

  ```bash
  python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/block/f1 --imgsz 1024 --batch 4 --probe
  ```
- Tốn giờ nhất trong bốn model: recipe **100 epoch**, gấp đôi Mask R-CNN và
  SOLOv2, ước ~2–3 h một fold trên 4090 — riêng nó chiếm khoảng 59 % tổng giờ
  máy của cả bảng. Chạy sau khi ba model kia đã xong một fold để không chiếm
  GPU quá lâu. Nếu cần cắt giờ, `--epochs 50` đưa nó về ngang hai model kia và
  bảng cũng công bằng hơn khi so bốn model.
- Giữ nguyên tăng cường LSJ của recipe gốc (mapper trong `train_net.py` của
  repo), không áp flipud/rot90 như hai model R-CNN.
