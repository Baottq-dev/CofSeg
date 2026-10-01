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

## Dựng môi trường riêng

Env riêng cho một mình model này, không dùng `benchmark/requirements.txt` của
cả nhóm. Chép chạy từ trên xuống; phần chung nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Cần detectron2; không cần mmcv và không có op CUDA nào của riêng nó —
ROIAlign và NMS lấy của torchvision.

```bash
conda create -y -n cofseg-maskrcnn python=3.12 && conda activate cofseg-maskrcnn
conda install -y -c nvidia cuda-toolkit=12.8.1
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/maskrcnn/requirements.txt
pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

python -c "
import torch, torchvision, detectron2
from detectron2 import model_zoo
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print('detectron2', detectron2.__version__)"
```

`--no-build-isolation` vì `setup.py` của detectron2 `import torch`. Kiểm phải
gọi cả `model_zoo` — đó là chỗ duy nhất còn `import pkg_resources`, nên bỏ nó
thì bước cài báo xanh rồi lần train đầu mới chết.

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
# 1. huấn luyện — chỉ train + val, không đụng tới split test
#    (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --runs benchmark/maskrcnn/runs --name maskrcnn --workers 8

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/maskrcnn/runs/train/*_maskrcnn_block-f1_* | head -1)
python benchmark/maskrcnn/evaluate.py --config benchmark/maskrcnn/configs/eval/maskrcnn_r50_d2.yaml --set model.weights="$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/maskrcnn/runs --name maskrcnn

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/maskrcnn/runs/eval/*_maskrcnn_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/maskrcnn_block_f1.json
cp "$EV/metrics.json"   benchmark/maskrcnn/results/maskrcnn_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/maskrcnn/results/maskrcnn_block_f1_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/maskrcnn && python -m pytest tests
```

### Lệnh đầy đủ

Khối trên là lệnh hằng ngày; khối này liệt kê **mọi tham số** để khi cần chỉnh
thì khỏi đi tra. Giá trị ghi ra chính là mặc định, nên lệnh này cho kết quả y
hệt lệnh ngắn ở trên.

`--data`, `--runs`, `--name` là của `train.py`; phần còn lại đi thẳng vào
trainer, gõ sai tên thì nó chặn và gợi ý tên gần đúng. `--list-params` in đủ
danh sách, `--print-config` in config đã gộp mà không chạy gì, `--probe` chỉ
dò VRAM rồi thoát.

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs --name maskrcnn \
  --imgsz 1024 --batch 16 --epochs 50 \
  --lr 0.02 --weight_decay 1e-4 --momentum 0.9 \
  --lr_steps "[0.7,0.9]" --lr_gamma 0.1 --warmup_iters 0.03 --amp true \
  --fliplr 0.5 --flipud 0.5 --rot90 true \
  --mask_resolution 14 \
  --keep_ckpts 1 \
  --val_every 1 --val_conf 0.05 --val_batch 16 --max_det 100 \
  --workers 8 --seed 0 --log_every 20
```

`--keep_ckpts 1` giữ đúng một checkpoint định kỳ trong `d2/`. Mặc định của
detectron2 là giữ **hết** — một file mỗi epoch, ~350 MB mỗi file — nên ~17 GB
cho một lượt 50 epoch. Một là đủ: thứ cứu lượt chạy bị ngắt giữa chừng là
checkpoint định kỳ gần nhất, vì `model_final.pth` (và `weights/last.pth` chép
ra từ nó) chỉ có khi train chạy hết.

Ba giá trị trong lệnh là **hiệu dụng**, còn config để trống cho trainer tự
tính: `--lr 0.02` là `0.02 x batch/16`, `--weight_decay 1e-4` là mặc định của
SGD trong recipe, `--val_batch 16` là "theo batch train". Bỏ cả ba đi thì kết
quả y hệt, nhưng đổi `--batch` sẽ không phải tính lại. `--workers 8` là giá
trị cho máy Linux; trên Windows trainer tự đặt 0 vì paging file.

Bỏ `--lr` đi thì lr tự tính theo batch (`0.02 x batch/16`); truyền tay là
tắt phép tự tính đó. `--lr_steps` phải có nháy vì giá trị đọc bằng YAML.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/maskrcnn/evaluate.py \
  --config benchmark/maskrcnn/configs/eval/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/maskrcnn/runs --name maskrcnn \
  --set model.weights=benchmark/maskrcnn/runs/train/<...>/weights/best.pth \
  --set model.arch=maskrcnn --set model.conf=0.05 --set model.max_det=100 \
  --set eval.iou_thr=0.5 --set eval.band_ratio=0.02 \
  --set eval.dilation_ratio=0.02 --set eval.nsd_tau=2.0
```

`--set model.*` đi vào khối `model:` của config chấm, `--set eval.*` vào khối
`eval:`. Bốn khoá `eval:` là định nghĩa của phép đo, đổi chúng là số không so
được với ba model kia nữa: `iou_thr` ngưỡng ghép cặp, `band_ratio` bề rộng
vành biên theo cỡ tán, `dilation_ratio` bề rộng vành của Boundary AP (2%
đường chéo ảnh, đúng bài báo), `nsd_tau` dung sai của NSD.

### Đổi backbone

R50 là mặc định vì ba model dùng chung nó, nhờ vậy chênh lệch giữa chúng quy
về cơ chế. Đổi backbone ở **một** model là mất tính chất đó — đổi thì đổi cả
ba, hoặc báo cáo riêng như thí nghiệm phụ.

```bash
# R101
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs --name maskrcnn-r101 \
  --set model.config_file=COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

# X101-32x8d
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs --name maskrcnn-x101 \
  --set model.config_file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
```

Trọng số COCO đi theo config: `base_cfg` gọi `model_zoo.get_checkpoint_url`
cho đúng file đó, nên không phải khai `model.weights`. Đặt `--name` khác đi,
không thì hai backbone rơi vào tên thư mục giống nhau.

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

- **Đã chạy thật trên máy lab** (24/09/2026): một lượt khói 3 epoch trên
  `block/f1`, đi hết đường train → chấm val mỗi epoch → chấm test trên
  field_1. Val mAP50-95 lên 3,76 → 12,81 → 27,61 và vẫn đang dốc, tức là
  đường chạy thông chứ chưa phải kết quả.

  ```bash
  python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --imgsz 1024 --batch 16 --epochs 3 --workers 4
  ```

  Một điểm phải theo dõi khi chạy đủ epoch: `APs` = 0,000 ở cả ba epoch và
  `APm` chỉ 1,51 — gần như toàn bộ điểm đến từ tán `large`, tán nhỏ chưa bắt
  được cái nào.

- Bản torchvision chạy ở nhà trên Windows đã bị gỡ khỏi repo; giờ chỉ còn
  đường detectron2, nên phần này của bảng phải chờ máy Linux.
