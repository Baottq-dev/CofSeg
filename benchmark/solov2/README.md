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

## Dựng môi trường riêng

Env riêng cho một mình model này, không dùng `benchmark/requirements.txt` của
cả nhóm. Chép chạy từ trên xuống; phần chung nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Lâu cài nhất: `mmcv` không có wheel cho CUDA 12.8 nên phải build từ nguồn,
mất 20–120 phút.

```bash
conda create -y -n cofseg-solov2 python=3.12 && conda activate cofseg-solov2
conda install -y -c nvidia cuda-toolkit=12.8.1
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/solov2/requirements.txt

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 pip install --no-build-isolation -e .
cd -
python -c "from mmcv.ops import nms; print('op CUDA co that')"

pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"

python -c "
import torch, torchvision, mmdet, mmcv
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print('mmdet', mmdet.__version__, '| mmcv', mmcv.__version__)"
```

Clone mmcv **ra ngoài repo**: thư mục `mmcv/` nằm trong gốc repo sẽ che gói đã
cài, và mọi lệnh chạy từ gốc repo báo `module 'mmcv' has no attribute
'__version__'`.

Ba dòng kiểm ở trên không bỏ được, vì cả ba kiểu hỏng đều báo xanh lúc cài:
mmcv dựng bản rỗng nếu chưa có torch, mmengine bản PyPI ném `UnpicklingError`
lúc nạp trọng số COCO, và mmdet chốt `mmcv < 2.2.0` trong khi 2.2.0 là bản mới
nhất tồn tại. `TORCH_CUDA_ARCH_LIST` đặt theo card: `12.0` cho RTX 5090, `8.9`
cho RTX 4090 / L40S, `8.0` cho A100, `9.0` cho H100.

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
python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --data data/export/block/f1 --runs benchmark/solov2/runs --name solov2 --workers 8

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/solov2/runs/train/*_solov2_block-f1_* | head -1)
python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml --set model.weights="$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/solov2/runs --name solov2

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/solov2/runs/eval/*_solov2_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/solov2_block_f1.json
cp "$EV/metrics.json"   benchmark/solov2/results/solov2_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/solov2/results/solov2_block_f1_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/solov2 && python -m pytest tests
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
python benchmark/solov2/train.py \
  --config benchmark/solov2/configs/train/solov2_r50_mm.yaml \
  --data data/export/field/f1 --runs benchmark/solov2/runs --name solov2 \
  --imgsz 1024 --batch 16 --epochs 50 \
  --lr 0.01 --weight_decay 1e-4 --momentum 0.9 \
  --lr_steps "[0.7,0.9]" --lr_gamma 0.1 --warmup_iters 0.03 --amp true \
  --fliplr 0.5 --flipud 0.5 \
  --val_every 1 --val_conf 0.05 --val_batch 16 --max_det 100 \
  --workers 8 --seed 0 --log_every 20
```

Không có `--rot90`: mmdet không có transform xoay cho mask + box. Đây là chênh
lệch có chủ đích so với hai model detectron2, nhớ ghi chú khi đọc bảng.

Ba giá trị trong lệnh là **hiệu dụng**, còn config để trống cho trainer tự
tính: `--lr 0.01` là `0.01 x batch/16`, `--weight_decay 1e-4` theo recipe, và
`--val_batch 16` là "theo batch train". `--workers 8` là giá trị cho máy
Linux; trên Windows trainer tự đặt 0 vì paging file.

Bỏ `--lr` đi thì lr tự tính theo batch (`0.01 x batch/16`); truyền tay là
tắt phép tự tính đó. `--lr_steps` phải có nháy vì giá trị đọc bằng YAML.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/solov2/evaluate.py \
  --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/solov2/runs --name solov2 \
  --set model.weights=benchmark/solov2/runs/train/<...>/weights/best.pth \
  --set model.conf=0.05 --set model.max_det=100 \
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

Model này vướng hơn hai model detectron2: config zoo ghim trong `ZOO` của
`cofseg/training/mmdet.py`, còn `model.config` là file **ghi đè**. Nên cách
gọn là thêm một arch vào `ZOO`:

```python
    "solov2_r101_dcn": dict(
        config="solov2/solov2_r101-dcn_fpn_ms-3x_coco.py",
        checkpoint="https://download.openmmlab.com/mmdetection/v2.0/solov2/"
                   "solov2_r101_dcn_fpn_3x_coco/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth",
        overrides="configs/mmdet/solov2_r50_coffee.py",   # dùng lại: chỉ đặt num_classes
        lr=0.01,
    ),
```

rồi `--set model.arch=solov2_r101_dcn`.

Bốn backbone OpenMMLab có phát hành **trọng số COCO**, tra từ
`configs/solov2/metafile.yml` của mmdet 3.3.0:

| arch | config | ghi chú |
|---|---|---|
| R50 | `solov2_r50_fpn_ms-3x_coco.py` | đang dùng |
| R101-DCN | `solov2_r101-dcn_fpn_ms-3x_coco.py` | deformable conv, nặng hơn |
| X101-DCN | `solov2_x101-dcn_fpn_ms-3x_coco.py` | nặng nhất |
| R18 / R50 light | `solov2-light_*_fpn_ms-3x_coco.py` | nhẹ, mask stride thô hơn |

Có `solov2_r101_fpn_ms-3x_coco.py` (R101 **không** DCN) nhưng metafile không
kèm trọng số COCO nào cho nó — chọn bản đó là khởi đầu từ ImageNet, tức không
so được với ba model kia vốn đều bắt đầu từ COCO.

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
  kiểm được phần dựng config. Việc đầu tiên trên máy Linux, một lượt khói:

  ```bash
  python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --data data/export/block/f1 --set data.limit=16 --epochs 1
  ```
- Khác biệt có chủ đích so với hai model detectron2: **không có xoay 90°**
  (mmdet không có transform sẵn cho mask + box), chỉ lật ngang/dọc/chéo. Nhớ
  ghi chú khi đọc bảng.
- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  Nới dòng kiểm đó bằng `sed`; xem `benchmark/README.md` mục *Dựng môi
  trường*, bước 6.
