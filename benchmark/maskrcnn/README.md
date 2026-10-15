# Mask R-CNN R50-FPN — mốc số 0

**Framework:** detectron2 (`trainer: detectron2`, `model.arch: maskrcnn`) ·
chỉ chạy trên Linux

## Vai trò

Cột mốc số 0 của bảng benchmark: mọi model khác báo cáo **Δ% mAP so với model
này** trên cùng một ruộng. Vì vậy đổi cấu hình của model này (imgsz, số
epoch, augmentation) là đổi mốc chung của cả bảng, và mọi Δ% của các model
kia phải tính lại.

## Vì sao chọn

- Kiến trúc gốc mà Cascade, PointRend, Mask Transfiner đều sửa từ đó. Lấy làm
  mốc thì chênh lệch trong bảng đọc được là "đổi khối nào được bao nhiêu".
- Mask 28×28 cho mỗi ROI rồi phóng lên box: tán trung vị 324 px ảnh gốc thì
  mỗi ô ≈ 12 px, biên thô. Đây là cận dưới của trục chất lượng biên trong
  bảng.
- Xử lý từng ROI riêng, không biết cây bên cạnh, nên ở chỗ tán chạm nhau mask
  dễ tràn sang cây bên.

## Dựng môi trường riêng

Env riêng chỉ cho model này, không dùng `benchmark/requirements.txt` chung.
Chạy từ trên xuống, từ gốc repo. Số bước trong ngoặc khớp với mục *Dựng môi
trường* của `benchmark/README.md`, nơi có giải thích từng bước và bảng lỗi.

Model này cần detectron2 (build từ nguồn, nên cần `nvcc`) và bản vá
`libnvrtc-builtins`. Không cần mmcv, và không có op CUDA riêng: ROIAlign và
NMS lấy của torchvision.

Kiểm máy trước (bước 0): driver **≥ 580.65.06**, `g++` ≤ 15.x. `nvcc 13.x` thì
đi nhánh A ở bước 3; `nvcc 12.x` hoặc không có `nvcc` thì đi nhánh B.

```bash
nvcc --version | tail -2
g++ --version | head -1
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Env và torch (bước 1–2):

```bash
conda create -y -n cofseg-maskrcnn python=3.12 && conda activate cofseg-maskrcnn
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

Vá `libnvrtc-builtins` (bước 4). Thiếu bước này thì model dừng ngay ở phép
`prod` trên GPU mà `detectron2/structures/boxes.py:330` gọi. Lệnh tự bỏ qua
khi wheel đã đặt tên đúng:

```bash
P=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
S=$(find "$P/nvidia" -name "libnvrtc-builtins-*.so.13.*" | head -1)
if [ -z "$S" ]; then echo "khong tim thay ban co bam — bo qua buoc 4"; else
  ln -sfn "$(basename "$S")" "$(dirname "$S")/libnvrtc-builtins.so.${S##*.so.}"
  ls -l "$(dirname "$S")"/libnvrtc-builtins*
fi
```

Gói Python và detectron2 (bước 5–6):

```bash
pip install -r benchmark/maskrcnn/requirements.txt
pip install --no-build-isolation --no-cache-dir \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

- `--no-build-isolation` vì `setup.py` của detectron2 `import torch`.
- `--no-cache-dir` để pip không dùng lại wheel đã build cho torch cũ. Build
  thật mất 5–10 phút; xong trong vài giây nghĩa là pip vẫn lấy bản cũ.

Kiểm cả env, mọi phép đều chạy trên GPU:

```bash
python -c "
import torch, torchvision, detectron2
from torchvision.ops import nms
from detectron2 import _C, model_zoo
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s = torch.tensor([0.9,0.8]).cuda()
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms CUDA', nms(b,s,0.5).tolist())
print('detectron2 ', detectron2.__version__, '| _C + model_zoo ok')
print('nvrtc      ', float(torch.rand(8,16,2).cuda().prod(dim=2).sum()))"
```

Phải in đủ bốn dòng, dòng đầu có `2.11.0+cu130 13.0`. Dòng kiểm phải gọi cả
`_C` (phần biên dịch) lẫn `model_zoo`: `model_zoo` là chỗ duy nhất còn
`import pkg_resources`, nên thiếu `setuptools<82` thì bước cài vẫn báo thành
công và lần train đầu mới dừng.

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
#     PHẦN của lịch (0.03), nên lượt ngắn co nó về 1 vòng và lr 0.02 làm RPN
#     nổ ngay — FloatingPointError ở vòng 13. Truyền số vòng tuyệt đối.)
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --runs benchmark/maskrcnn/runs --name maskrcnn --workers 8

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/maskrcnn/runs/train/*_maskrcnn_block-f1_* | head -1)
python benchmark/maskrcnn/evaluate.py --config benchmark/maskrcnn/configs/eval/maskrcnn_r50_d2.yaml --weights "$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/maskrcnn/runs --name maskrcnn

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/maskrcnn/runs/eval/*_maskrcnn_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/maskrcnn_block_f1.json
cp "$EV/metrics.json"   benchmark/maskrcnn/results/maskrcnn_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/maskrcnn/results/maskrcnn_block_f1_per_region.csv

# test của thư mục này
cd benchmark/maskrcnn && python -m pytest tests
```

### Lệnh đầy đủ

Khối trên là lệnh thường dùng; khối này liệt kê **mọi tham số**. Giá trị ghi
ra chính là mặc định, nên lệnh này cho kết quả y hệt lệnh ngắn.

`--data`, `--runs`, `--name` là của `train.py`; phần còn lại đi thẳng vào
trainer, gõ sai tên thì trainer chặn và gợi ý tên gần đúng. `--list-params` in
đủ danh sách, `--print-config` in config đã gộp mà không chạy gì, `--probe`
chỉ dò VRAM rồi thoát.

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs --name maskrcnn \
  --imgsz 1024 --batch 16 --epochs 100 \
  --lr 0.02 --weight_decay 1e-4 --momentum 0.9 \
  --lr_steps "[0.7,0.9]" --lr_gamma 0.1 --warmup_iters 0.03 --amp true \
  --fliplr 0.5 --flipud 0.5 --rot90 true \
  --mask_resolution 14 \
  --keep_ckpts 1 \
  --val_every 1 --val_conf 0.05 --val_batch 16 --max_det 100 \
  --workers 8 --seed 0 --log_every 20
```

`--keep_ckpts 1` giữ đúng một checkpoint định kỳ trong `d2/`. Mặc định của
detectron2 là giữ **hết**, mỗi epoch một file ~350 MB, tức ~17 GB cho một lượt
50 epoch. Một file là đủ: thứ cứu lượt chạy bị ngắt giữa chừng là checkpoint
định kỳ gần nhất, vì `model_final.pth` (và `weights/last.pth` chép ra từ nó)
chỉ có khi train chạy hết.

Ba giá trị trong lệnh là **hiệu dụng**, còn config để trống cho trainer tự
tính: `--lr 0.02` là `0.02 x batch/16`, `--weight_decay 1e-4` là mặc định của
SGD trong recipe, `--val_batch 16` là "theo batch train". Bỏ cả ba thì kết quả
y hệt, và đổi `--batch` không phải tính lại. `--workers 8` là giá trị cho
Linux; trên Windows trainer tự đặt 0 vì paging file.

Truyền `--lr` tay là tắt phép tự tính theo batch. `--lr_steps` phải có nháy vì
giá trị đọc bằng YAML.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/maskrcnn/evaluate.py \
  --config benchmark/maskrcnn/configs/eval/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/maskrcnn/runs --name maskrcnn \
  --weights benchmark/maskrcnn/runs/train/<...>/weights/best.pth \
  --arch maskrcnn --conf 0.05 --max_det 100 \
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
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --list-backbones
```

| `--backbone` | mask AP COCO | Train Mem | s/iter | ước lượng ở đây |
|---|---|---|---|---|
| `r50` | 37,2 | 3,4 GB | 0,261 | **12,7 GB · 26 phút/fold (đo thật)** |
| `r50-dconv` | 38,5 | 3,5 GB | 0,349 | ~13 GB · ~35 phút |
| `r50-gn` | 38,6 | 5,6 GB | 0,309 | ~21 GB · ~31 phút |
| `r101` | 38,6 | 4,6 GB | 0,340 | ~17 GB · ~34 phút |
| `x101` | 39,5 | 7,2 GB | 0,690 | ~27 GB · ~70 phút |

Ba cột giữa lấy từ `MODEL_ZOO.md` của detectron2 (COCO val2017). Cột cuối suy
từ tỉ lệ so với R50 nhân số đo thật của lượt R50, nên chỉ là **ước lượng**;
chạy `--probe` trước khi đặt lịch.

#### ViTDet — backbone transformer

| `--backbone` | mask AP COCO | tham số | ghi chú |
|---|---|---|---|
| `vit-b` | 45,9 | 86 M | bản nhỏ nhất, khả năng vừa một card cao nhất |
| `vit-l` | 49,2 | 304 M | cần batch rất nhỏ |
| `vit-h` | 50,2 | 632 M | ghi cho đủ bảng, khó chạy trên một card |

Đây là **Mask R-CNN với backbone ViT** (Li et al. 2022), không phải kiến trúc
khác, nên nó nằm cùng bảng với `r50`. Chênh lệch với `r50` trên COCO là +8,7
điểm, lớn hơn hẳn mọi backbone CNN ở bảng trên (`x101` chỉ +2,3).

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --backbone vit-b --batch 4 --keep_ckpts 1 --workers 8
```

**Bốn thứ lượt ViT tự đổi, phải ghi rõ khi báo cáo:**

| | `r50` | `vit-b` | vì sao |
|---|---|---|---|
| optimizer | SGD 0,02 | **AdamW 1e-4** | ViT tiền huấn luyện MAE; SGD 0,02 phá đặc trưng ngay vài trăm vòng đầu |
| `weight_decay` | 1e-4 | **0,1** | recipe AdamW của detectron2 |
| lr theo độ sâu | không | **layer-wise decay 0,7** | recipe ViTDet (`vit-l` 0,8, `vit-h` 0,9) |
| kênh ảnh | BGR | **RGB** | trọng số MAE dùng mean/std ImageNet RGB |

Lịch học thì **không** đổi: `max_iter`, `lr_steps`, `warmup_iters` và số epoch
giữ nguyên, nên lượt ViT vẫn cùng ngân sách với các model kia.

`--batch 4` trong lệnh trên **chưa được đo**. Recipe gốc của ViTDet là batch
64 trên 64 GPU, tức 1 ảnh/GPU, nên `--batch 16` gần như chắc chắn hết VRAM.
Chạy `--probe` trước. Vì batch khác 16, lượt này **không vào bảng chính**; nó
là thí nghiệm phụ, báo cáo riêng.

Model của bản ViT dựng bằng `instantiate` từ chính LazyConfig của detectron2
chứ không dựng tay: tên tham số trong checkpoint COCO do cách dựng quyết định.
Sau khi nạp, `summary.json` ghi mục `weights_loaded` cho biết khoá nào không
vào được model. Đầu phân loại lệch hình là cố ý (80 lớp → 1); backbone lệch
hình thì không, và khi đó có một dòng cảnh báo in ra màn hình.

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --backbone r101 --keep_ckpts 1 --workers 8
```

Không cần `--name`: thiếu nó thì tên thư mục run lấy theo backbone
(`maskrcnn-r101`), nên hai backbone không rơi vào cùng một thư mục.

Trọng số COCO đi theo config đã chọn: `base_cfg` lấy `get_checkpoint_url` của
**chính** file đó.

Cascade Mask R-CNN là *kiến trúc* khác chứ không phải backbone khác, nên có
config riêng và bảng backbone riêng (`r50`, `x152`):

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/cascade_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --keep_ckpts 1 --workers 8
```

`new_baselines` LSJ (mask AP 40,3–42,5) chưa có trong bảng. Chúng cũng là
LazyConfig như ViTDet nên đường dựng model đã có, nhưng phần còn lại của
recipe (SyncBN trong backbone và FPN, box head 4conv1fc) nằm trong cùng file
config đó; thêm chúng là thêm một hàng vào `BACKBONES`. Một điểm đáng chú ý từ
bảng đó: R50 + LSJ 100 epoch (40,3) hơn X101 3x (39,5), tức khác biệt đến từ
recipe chứ không từ backbone.

Cascade + ViTDet thì chưa có: config của nó nằm trong
`projects/ViTDet/configs/`, không được đóng gói theo package detectron2 nên
cần đường dẫn tới bản checkout.

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
| `configs/train/`, `configs/eval/` | cấu hình huấn luyện và chấm (`maskrcnn_r50_d2`, `cascade_r50_d2`) |
| `train.py`, `evaluate.py` | bản riêng, nạp `cofseg/` của thư mục này |
| `tests/` | test của thư mục |
| `runs/` | kết quả train/eval (không vào git) |
| `results/` | file kết quả nhỏ, được commit |
| `notes/experiments.md` | nhật ký thí nghiệm |

Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`)
phải giống hệt ba thư mục model kia; sửa ở đây mà không sửa ở đó thì số không
còn so được.

## Ghi chú

- Chỉ chạy trên Linux: repo chỉ còn đường detectron2, bản torchvision chạy
  trên Windows đã gỡ.
- Lượt khói ngắn phải truyền `--warmup_iters` bằng số vòng tuyệt đối (vd
  `30`), xem chú thích trong khối *Cách chạy*.
