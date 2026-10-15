# SOLOv2 R50-FPN — họ box-free

**Framework:** mmdetection (`trainer: mmdet`, `model.arch: solov2`) · chỉ chạy
trên Linux

## Vai trò

Đại diện cho hướng **bỏ bounding box**: chia ảnh thành lưới, ô nào chứa tâm
vật thể thì ô đó sinh kernel động, tích chập lên đặc trưng mask stride 4 của
toàn ảnh. Không RPN, không ROI, không cắt mask theo box.

## Vì sao chọn

- Cách tìm vật thể khác hẳn Mask R-CNN (ROI) và YOLO (một giai đoạn, cắt mask
  theo box), nên bảng có đủ các nhóm phương pháp.
- Mask không bị box cắt, nên ở chỗ hai tán chạm nhau mask không bị cắt thẳng
  theo cạnh box như YOLO.
- Điểm yếu đã biết: mỗi ô lưới chỉ đại diện được một vật thể. Hai tán cùng cỡ
  chạm nhau mà tâm rơi vào cùng ô thì mất một cây.
- Cùng backbone R50 với Mask R-CNN và Mask2Former.

## Dựng môi trường riêng

Env riêng chỉ cho model này, không dùng `benchmark/requirements.txt` chung.
Chạy từ trên xuống, từ gốc repo. Số bước trong ngoặc khớp với mục *Dựng môi
trường* của `benchmark/README.md`, nơi có giải thích từng bước và bảng lỗi.

Model này cần **mmcv build từ nguồn** (không có wheel cho CUDA 13, mất 20–120
phút), mmengine bản git và mmdet đã nới chốt phiên bản. Không cần detectron2,
và không cần bản vá `libnvrtc-builtins` (bước 4).

Kiểm máy trước (bước 0): driver **≥ 580.65.06**, `g++` ≤ 15.x. `nvcc 13.x` thì
đi nhánh A ở bước 3; `nvcc 12.x` hoặc không có `nvcc` thì đi nhánh B.

```bash
nvcc --version | tail -2
g++ --version | head -1
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Env và torch (bước 1–2):

```bash
conda create -y -n cofseg-solov2 python=3.12 && conda activate cofseg-solov2
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

`CUDA_HOME` (bước 3), **chọn một trong hai nhánh**:

```bash
# nhánh A — máy có nvcc 13.x
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
```

```bash
# nhánh B — máy có nvcc 12.x hoặc không có nvcc
pip install "cuda-toolkit[nvcc,nvvm,crt,cccl]==13.0.2"
export CUDA_HOME=$(python -c "import sysconfig,os;print(os.path.join(sysconfig.get_paths()['purelib'],'nvidia','cu13'))")
ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
```

rồi giữ biến cho mọi shell sau:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export CUDA_HOME=$CUDA_HOME" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
$CUDA_HOME/bin/nvcc --version | tail -2
```

Gói Python (bước 5):

```bash
pip install -r benchmark/solov2/requirements.txt
```

mmcv (bước 7):

```bash
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch;print('%d.%d' % torch.cuda.get_device_capability(0))")
echo "arch = $TORCH_CUDA_ARCH_LIST"          # 8.9 trên 4090, 12.0 trên 5090

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
rm -rf build mmcv/_ext*.so
FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=4 pip install --no-build-isolation -e .
cd -
ls -l ~/mmcv/mmcv/_ext*.so
```

- `TORCH_CUDA_ARCH_LIST` lấy từ card. Build sai kiến trúc thì lỗi `no kernel
  image is available` chỉ hiện ở lời gọi kernel đầu tiên, sau khi đã build
  xong.
- `rm -rf build mmcv/_ext*.so` vì setuptools dùng lại file `.o` cũ; không dọn
  thì đổi biến kiến trúc cũng không có tác dụng.
- Clone **ra ngoài repo**: thư mục `mmcv/` trong gốc repo che gói đã cài, và
  mọi lệnh chạy từ gốc repo báo `module 'mmcv' has no attribute
  '__version__'`.
- pip ẩn output build khi thành công. `_ext.so` cỡ **~26 MB** là bản có nhân
  CUDA; vài MB là bản chỉ có CPU, phải build lại.

mmengine và mmdet (bước 8), phải chạy **sau** mmcv vì cài mmcv kéo bản
mmengine trên PyPI về đè:

```bash
pip install "git+https://github.com/open-mmlab/mmengine"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
grep -n "mmcv_maximum_version" "$MMDET"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
```

mmengine bản PyPI ném `UnpicklingError` lúc nạp trọng số COCO trên torch ≥ 2.6,
và số phiên bản không phân biệt được hai bản, nên phải lấy từ git và kiểm
bằng mã nguồn. `sed` nới chốt của mmdet: nó khai `mmcv < 2.2.0`, trong khi
2.2.0 là bản mới nhất hiện có; `grep` trước để không vá hai lần.

Kiểm cả env, mọi phép đều chạy trên GPU:

```bash
python -c "
import inspect, os, torch, torchvision, mmcv, mmdet
import mmengine, mmengine.runner.checkpoint as ckpt
from torchvision.ops import nms as tv_nms
from mmcv.ops import nms as mmcv_nms
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s = torch.tensor([0.9,0.8]).cuda()
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms CUDA', tv_nms(b,s,0.5).tolist())
print('mmcv       ', mmcv.__version__, os.path.dirname(mmcv.__file__))
print('             nms CUDA', mmcv_nms(b,s,0.5)[1].tolist())
print('mmdet      ', mmdet.__version__, '| mmengine', mmengine.__version__,
      '| vá:', 'weights_only' in inspect.getsource(ckpt))"
```

Phải in đủ năm dòng, dòng đầu có `2.11.0+cu130 13.0`, đường dẫn mmcv là
`~/mmcv/mmcv`, và dòng cuối có `vá: True`.

## Cách chạy

Từ **gốc repo** (để dùng chung `data/` và `weights/`). Đặt `--runs` vào thư
mục này, và `--name` chỉ là **tên model**: bảng tổng hợp đọc fold và bộ fold từ
đường dẫn dữ liệu chứ không từ tên, nên gõ lại fold vào tên chỉ tạo ra một bản
thứ hai có thể sai lệch.

`--data` nhận **thư mục fold**. Viết thẳng đường dẫn ra, không đặt biến shell:
nhìn lệnh là biết đang chạy bộ nào, fold nào. Cả bốn model dùng cùng cú pháp
này.

Lệnh dưới đây chạy bộ **block**, fold **f1**. Đổi lượt khác thì sửa `block`
hoặc `f1` ở cả ba chỗ trong khối lệnh. Ba bộ fold (`block`, `flight`, `field`)
cắt bằng `scripts/make_fold.py`, xem `benchmark/README.md` mục *Chạy*.

```bash
# 1. huấn luyện — chỉ train + val, không đụng tới split test
#    (khói: --limit 64 --epochs 3 --warmup_iters 30. warmup_iters dưới 1 là
#     PHẦN của lịch (0.03), lượt ngắn co nó về 1 vòng; truyền số vòng tuyệt đối.)
python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --data data/export/block/f1 --runs benchmark/solov2/runs --name solov2 --workers 8

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/solov2/runs/train/*_solov2_block-f1_* | head -1)
python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml --weights "$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/solov2/runs --name solov2

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/solov2/runs/eval/*_solov2_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/solov2_block_f1.json
cp "$EV/metrics.json"   benchmark/solov2/results/solov2_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/solov2/results/solov2_block_f1_per_region.csv

# test của thư mục này
cd benchmark/solov2 && python -m pytest tests
```

### Lệnh đầy đủ

Khối trên là lệnh thường dùng; khối này liệt kê **mọi tham số**. Giá trị ghi
ra chính là mặc định, nên lệnh này cho kết quả y hệt lệnh ngắn.

`--data`, `--runs`, `--name` là của `train.py`; phần còn lại đi thẳng vào
trainer, gõ sai tên thì trainer chặn và gợi ý tên gần đúng. `--list-params` in
đủ danh sách, `--print-config` in config đã gộp mà không chạy gì, `--probe`
chỉ dò VRAM rồi thoát.

```bash
python benchmark/solov2/train.py \
  --config benchmark/solov2/configs/train/solov2_r50_mm.yaml \
  --data data/export/field/f1 --runs benchmark/solov2/runs --name solov2 \
  --imgsz 1024 --batch 16 --epochs 100 \
  --lr 0.01 --weight_decay 1e-4 --momentum 0.9 \
  --lr_steps "[0.7,0.9]" --lr_gamma 0.1 --warmup_iters 0.03 --amp true \
  --fliplr 0.5 --flipud 0.5 \
  --val_every 1 --val_conf 0.05 --val_batch 16 --max_det 100 \
  --workers 8 --seed 0 --log_every 20
```

Không có `--rot90`: mmdet không có transform xoay sẵn cho cả mask lẫn box, nên
model này chỉ lật. Đây là khác biệt có chủ đích so với hai model detectron2,
cần ghi chú khi đọc bảng. Lật ngang và dọc vẫn độc lập như ở các model kia:
`RandomFlip` của mmdet nhận ba hướng loại trừ nhau, nên trainer ghép chúng
thành ngang `p(1-q)`, dọc `(1-p)q`, chéo `pq`.

Ba giá trị trong lệnh là **hiệu dụng**, còn config để trống cho trainer tự
tính: `--lr 0.01` là `0.01 x batch/16`, `--weight_decay 1e-4` theo recipe,
`--val_batch 16` là "theo batch train". `--workers 8` là giá trị cho Linux;
trên Windows trainer tự đặt 0 vì paging file.

Truyền `--lr` tay là tắt phép tự tính theo batch. `--lr_steps` phải có nháy vì
giá trị đọc bằng YAML.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/solov2/evaluate.py \
  --config benchmark/solov2/configs/eval/solov2_r50_mm.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/solov2/runs --name solov2 \
  --weights benchmark/solov2/runs/train/<...>/weights/best.pth \
  --conf 0.05 --max_det 100 \
  --iou-thr 0.5 --band-ratio 0.02 \
  --dilation-ratio 0.02 --nsd-tau 2.0
```

Mọi tham số ở đây là **cờ thật**, không qua `--set`: `--weights` là trọng số
của lần train; `--conf`/`--max_det` (và các khoá khác của lớp model) đi vào
khối `model:`; `--iou-thr`/`--band-ratio`/`--dilation-ratio`/`--nsd-tau` đi
vào khối `eval:`. Bốn khoá `eval:` là định nghĩa của phép đo, đổi chúng thì số
không còn so được với các model kia: `iou_thr` ngưỡng ghép cặp, `band_ratio`
bề rộng vành biên theo cỡ tán, `dilation_ratio` bề rộng vành của Boundary AP
(2% đường chéo ảnh, đúng bài báo), `nsd_tau` dung sai của NSD.

### Đổi backbone

`--backbone <tên>` đổi backbone ngay trên dòng lệnh; `--list-backbones` in
bảng lựa chọn kèm mask AP trên COCO rồi thoát:

```bash
python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --list-backbones
```

| `--backbone` | mask AP COCO | ghi chú |
|---|---|---|
| `r50` | 37,5 | mặc định, cùng backbone với Mask R-CNN và Mask2Former |
| `r101-dcn` | 41,2 | R101 **cộng** deformable conv: hai thay đổi cùng lúc |
| `x101-dcn` | 42,4 | nặng nhất, chạy `--probe` trước |
| `light-r18` | 29,7 | bản nhẹ, mask stride thô hơn |
| `light-r50` | 33,7 | bản nhẹ, mask stride thô hơn |

Số lấy từ `configs/solov2/metafile.yml` của mmdet 3.3.0. File ghi đè
`configs/mmdet/solov2_r50_coffee.py` chỉ đặt `num_classes` và ngưỡng test, nên
dùng chung được cho mọi backbone.

```bash
python benchmark/solov2/train.py \
  --config benchmark/solov2/configs/train/solov2_r50_mm.yaml \
  --data data/export/field/f1 --runs benchmark/solov2/runs \
  --backbone r101-dcn --workers 8
```

Không cần `--name`: thiếu nó thì tên thư mục run lấy theo backbone
(`solov2-r101-dcn`), nên hai backbone không rơi vào cùng một thư mục.

**SOLOv2 không có R101 thường.** Có config `solov2_r101_fpn_ms-3x_coco.py`
nhưng metafile không kèm trọng số COCO nào cho nó; chọn bản đó là khởi đầu từ
ImageNet, không so được với các model kia vốn đều bắt đầu từ COCO. Muốn lên
R101 thì phải nhận thêm deformable conv làm biến thứ hai, và ghi rõ điều đó
khi báo cáo.

**Đọc trước khi đổi.** Trên bộ này AP50 đã cao, phần mất điểm nằm ở IoU cao,
tức chất lượng đường biên. Backbone sâu hơn chủ yếu tăng AP50, phần còn ít dư
địa nhất, và thêm tham số vào một bộ ~450 ảnh train thì model bão hoà sớm hơn.

Đổi backbone ở **một** model cũng làm mất tính chất "ba model cùng R50 nên
chênh lệch quy về cơ chế". Đổi đồng bộ cả bốn thì không làm được: YOLO không
có backbone để đổi, còn SOLOv2 không có trọng số COCO cho R101 thường. Vì vậy
phần này nằm **ngoài bảng chính**, báo cáo riêng như thí nghiệm phụ.

## Trong thư mục này

| | |
|---|---|
| `cofseg/` | bản sao lõi của riêng thư mục: đọc dữ liệu, chỉ số, vòng chấm, trainer + model của model này |
| `configs/train/`, `configs/eval/` | cấu hình huấn luyện và chấm |
| `configs/mmdet/solov2_r50_coffee.py` | file ghi đè config mmdet: `num_classes` và ngưỡng test |
| `train.py`, `evaluate.py` | bản riêng, nạp `cofseg/` của thư mục này |
| `tests/` | test của thư mục |
| `runs/` | kết quả train/eval (không vào git) |
| `results/` | file kết quả nhỏ, được commit |
| `notes/experiments.md` | nhật ký thí nghiệm |

Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`)
phải giống hệt ba thư mục model kia; sửa ở đây mà không sửa ở đó thì số không
còn so được.

## Ghi chú

- Chỉ chạy trên Linux: mmcv không cài được trên Windows.
- Tăng cường hình học khác hai model detectron2 ở đúng một chỗ: không có xoay
  90° (xem *Lệnh đầy đủ*).
- Tăng cường màu đi qua `PhotoMetricDistortion` của mmdet, phép này cộng
  delta độ sáng thay vì nhân; cách quy đổi `hsv_v` nằm ở `benchmark/README.md`
  mục *Tăng cường*.
