# Mask2Former R50 — họ query / transformer

**Framework:** detectron2 + repo Mask2Former (`trainer: detectron2`,
`model.arch: mask2former`, `model.repo: benchmark/mask2former/upstream`) · chỉ
chạy trên Linux

## Vai trò

Đại diện hướng **query-based**: N query học được, mỗi query nhận một vật thể
qua masked attention; mask sinh ở stride 4 trên toàn ảnh, không box, không NMS.

## Vì sao chọn

- Cách tìm vật thể thứ ba, khác hẳn ROI và một giai đoạn.
- Mask không bị box hay ROI cắt, nên có thể bám đúng đường chia giữa hai tán
  chạm nhau; bảng cho biết bỏ box thì biên thay đổi bao nhiêu.
- Điểm yếu đã biết của query với vật thể dày, cùng lớp, chạm nhau: hai query
  cùng nhận một tán, hoặc sót tán.
- Cùng backbone R50 với Mask R-CNN và SOLOv2, nên chênh lệch là do cơ chế.
- Số query phải lớn hơn số tán tối đa trong một ảnh (48); đặt 100.

## Dựng môi trường riêng

Env riêng chỉ cho model này, không dùng `benchmark/requirements.txt` chung.
Chạy từ trên xuống, từ gốc repo. Số bước trong ngoặc khớp với mục *Dựng môi
trường* của `benchmark/README.md`, nơi có giải thích từng bước và bảng lỗi.

Model nặng cài nhất: cần detectron2, bản vá `libnvrtc-builtins`, repo con
`upstream/`, và biên dịch op CUDA `MSDeformAttn` tại chỗ. Không cần mmcv.

Kiểm máy trước (bước 0): driver **≥ 580.65.06**, `g++` ≤ 15.x. `nvcc 13.x` thì
đi nhánh A ở bước 3; `nvcc 12.x` hoặc không có `nvcc` thì đi nhánh B.

```bash
nvcc --version | tail -2
g++ --version | head -1
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Env và torch (bước 1–2):

```bash
conda create -y -n cofseg-mask2former python=3.12 && conda activate cofseg-mask2former
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

Vá `libnvrtc-builtins` (bước 4). Thiếu bước này thì detectron2 dừng ở phép
`prod` trên GPU. Lệnh tự bỏ qua khi wheel đã đặt tên đúng:

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
pip install -r benchmark/mask2former/requirements.txt
pip install --no-build-isolation --no-cache-dir \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
```

`--no-cache-dir` để pip không dùng lại wheel đã build cho torch cũ. Build thật
mất 5–10 phút; xong trong vài giây nghĩa là pip vẫn lấy bản cũ.

MSDeformAttn (bước 9):

```bash
git submodule update --init benchmark/mask2former/upstream
git -C benchmark/mask2former/upstream checkout .

sed -i "s/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g" \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops
```

- `git checkout .` xoá mọi bản vá cũ trong repo con trước khi vá lại.
- `sed` vá hai dòng của repo gốc; thiếu nó nvcc dừng ngay.
  `AT_DISPATCH_FLOATING_TYPES` nhận `c10::ScalarType`, còn `value.type()` trả
  `at::DeprecatedTypeProperties`. Torch cũ cho chuyển ngầm giữa hai kiểu,
  torch 2.11 đã gỡ, nên `src/cuda/ms_deform_attn_cuda.cu` dòng 69 và 139 báo
  `no suitable conversion function`. `value.scalar_type()` là đúng thứ mà
  thông điệp deprecated của torch chỉ tới: cùng giá trị, không đổi hành vi.
  Chỉ hai dòng đó hỏng: `THC/THCAtomics.cuh` torch vẫn giữ shim,
  `.data<scalar_t>()` (15 chỗ) và `AT_ASSERTM` (26 chỗ) vẫn biên dịch được.
- Build thẳng thư mục `ops`, không chạy `make.sh` của repo gốc: script đó gọi
  `setup.py install`, mà setuptools mới đã bỏ lệnh này.
- Bỏ hẳn bước này thì model **không chạy được**:
  `ops/functions/ms_deform_attn_func.py` ném `ModuleNotFoundError` ngay lúc
  import. Đường lùi PyTorch ở tầng trên (`ops/modules/ms_deform_attn.py` bọc
  lời gọi op trong `try/except`) không bao giờ tới lượt vì import dừng trước;
  có tới lượt thì cũng tốn 3.94 GiB VRAM mỗi lớp encoder (xem *Ghi chú*).

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

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Phải in đủ bốn dòng, dòng đầu có `2.11.0+cu130 13.0`, rồi `MSDeformAttn OK`.
Dòng kiểm cuối đi qua `mask2former/__init__` → `instance_evaluation` →
`fast_eval_api` → `from detectron2 import _C`, nên nó đồng thời xác nhận
detectron2 còn khớp torch.

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
python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/block/f1 --runs benchmark/mask2former/runs --name mask2former --workers 2

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/mask2former/runs/train/*_mask2former_block-f1_* | head -1)
python benchmark/mask2former/evaluate.py --config benchmark/mask2former/configs/eval/mask2former_r50_d2.yaml --weights "$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/mask2former/runs --name mask2former

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/mask2former/runs/eval/*_mask2former_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/mask2former_block_f1.json
cp "$EV/metrics.json"   benchmark/mask2former/results/mask2former_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/mask2former/results/mask2former_block_f1_per_region.csv

# test của thư mục này
cd benchmark/mask2former && python -m pytest tests
```

### Lệnh đầy đủ

Khối trên là lệnh thường dùng; khối này liệt kê **mọi tham số**. Giá trị ghi
ra chính là mặc định, nên lệnh này cho kết quả y hệt lệnh ngắn.

`--data`, `--runs`, `--name` là của `train.py`; phần còn lại đi thẳng vào
trainer, gõ sai tên thì trainer chặn và gợi ý tên gần đúng. `--list-params` in
đủ danh sách, `--print-config` in config đã gộp mà không chạy gì, `--probe`
chỉ dò VRAM rồi thoát.

```bash
python benchmark/mask2former/train.py \
  --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/mask2former/runs --name mask2former \
  --imgsz 1024 --batch 8 --epochs 50 \
  --lr 1e-4 --weight_decay 0.05 --momentum 0.9 \
  --lr_steps "[0.7,0.9]" --lr_gamma 0.1 --warmup_iters 0.03 --amp true \
  --num_queries 100 \
  --keep_ckpts 1 \
  --val_every 2 --val_conf 0.05 --val_batch 8 --max_det 100 \
  --workers 2 --seed 0 --log_every 20
```

`--keep_ckpts 1` giữ đúng một checkpoint định kỳ trong `d2/`. Mặc định của
detectron2 là giữ **hết**, mỗi epoch một file ~530 MB (AdamW lưu hai moment),
tức ~53 GB cho một lượt 100 epoch. Một file là đủ: thứ cứu lượt chạy bị ngắt
giữa chừng là checkpoint định kỳ gần nhất, vì `model_final.pth` (và
`weights/last.pth` chép ra từ nó) chỉ có khi train chạy hết.

Bốn cờ **không có tác dụng** với model này: `--fliplr`, `--flipud`, `--rot90`
vì model giữ mapper LSJ của recipe gốc (co giãn 0.1–2.0 rồi cắt ô vuông
`imgsz`, tự lật ngang), và `--mask_resolution` vì Mask2Former không có ROI
head, nó sinh mask ở stride 4 của toàn ảnh.

Ba giá trị trong lệnh là **hiệu dụng**, còn config để trống cho trainer tự
tính: `--lr 1e-4` là `1e-4 x batch/16`, `--weight_decay 0.05` là mặc định của
AdamW trong recipe, `--val_batch 8` là "theo batch train". `--momentum` có
trong danh sách nhưng AdamW không dùng. `--workers 2` là giá trị cho Linux,
thấp hơn ba model kia (8); trên Windows trainer tự đặt 0 vì paging file.

`--workers` là 2 chứ không phải 8 để tránh bị OOM killer dừng tiến trình.
detectron2 dùng `DATALOADER.NUM_WORKERS` cho cả train loader lẫn test loader,
và `EvalHook` dựng test loader mới mỗi lần chấm, nên trong lúc val có 16 tiến
trình python cùng chạy. Trên máy 30 GB RAM, điều đó làm cả ba lượt đầu bị
dừng đúng lúc val; số đo nằm trong `configs/train/mask2former_r50_d2.yaml`.
Giá trị này đổi chuỗi ngẫu nhiên của augmentation, nên **mọi fold trong bảng
phải dùng cùng một giá trị**; không ghi đè bằng cờ dòng lệnh cho riêng một
lượt.

Truyền `--lr` tay là tắt phép tự tính theo batch. `--lr_steps` phải có nháy vì
giá trị đọc bằng YAML.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/mask2former/evaluate.py \
  --config benchmark/mask2former/configs/eval/mask2former_r50_d2.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/mask2former/runs --name mask2former \
  --weights benchmark/mask2former/runs/train/<...>/weights/best.pth \
  --arch mask2former --repo benchmark/mask2former/upstream \
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
python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --list-backbones
```

| `--backbone` | mask AP COCO | ghi chú |
|---|---|---|
| `r50` | 43,7 | mặc định, cùng backbone với Mask R-CNN và SOLOv2 |
| `r101` | 44,2 | chỉ hơn R50 **0,5 điểm** |
| `swin-t` | 45,0 | đổi hẳn họ backbone sang transformer |
| `swin-s` | 46,3 | |
| `swin-b` | 46,7 | nặng, chạy `--probe` trước |
| `swin-b-in21k` | 48,1 | tiền huấn luyện IN21k: **khác nguồn dữ liệu**, không chỉ khác backbone |
| `swin-l-in21k` | 50,1 | nặng nhất repo, gần như chắc chắn hết VRAM ở batch 8 |

Config và trọng số lấy từ `MODEL_ZOO.md` của repo Mask2Former; đường dẫn
config là đường dẫn **bên trong** `upstream/`, trainer tự ghép.

```bash
python benchmark/mask2former/train.py \
  --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/mask2former/runs \
  --backbone r101 --workers 2
```

Không cần `--name`: thiếu nó thì tên thư mục run lấy theo backbone
(`mask2former-r101`), nên hai backbone không rơi vào cùng một thư mục.

Hai dòng `in21k` đổi cả nguồn tiền huấn luyện, nên chênh lệch của chúng không
quy về backbone được; nếu dùng thì phải ghi rõ trong phần phương pháp.

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
| `upstream/` | repo gốc facebookresearch/Mask2Former, submodule ghim commit |
| `cofseg/` | bản sao lõi của riêng thư mục: đọc dữ liệu, chỉ số, vòng chấm, trainer + model của model này |
| `configs/train/`, `configs/eval/` | cấu hình huấn luyện và chấm |
| `train.py`, `evaluate.py` | bản riêng, nạp `cofseg/` của thư mục này |
| `tests/` | test của thư mục |
| `runs/` | kết quả train/eval (không vào git) |
| `results/` | file kết quả nhỏ, được commit |
| `notes/experiments.md` | nhật ký thí nghiệm |

Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`)
phải giống hệt ba thư mục model kia; sửa ở đây mà không sửa ở đó thì số không
còn so được.

## upstream/ — repo gốc của Mask2Former (submodule)

| | |
|---|---|
| Nguồn | https://github.com/facebookresearch/Mask2Former |
| Commit ghim | `9b0651c` (20/05/2022, bản cuối của repo) |
| License | MIT |
| Dùng làm gì | `cofseg/models/detectron2.py` nạp `train_net.py` và gói `mask2former` làm module; op `MSDeformAttn` vá hai dòng rồi biên dịch tại chỗ (bước 9 ở trên) |

Repo con đi theo git: `git clone --recurse-submodules`, hoặc
`git submodule update --init benchmark/mask2former/upstream` nếu đã clone rồi.
Thư mục đặt tên `upstream/` chứ không phải `Mask2Former/` vì tên đó trùng tên
thư mục cha, chỉ khác hoa thường; trên Windows và macOS hai tên là một.

**Không sửa mã bên trong repo con.** Cần vá thì vá bằng code trong `cofseg/`
(như hàm `load_mask2former`).

Ngoại lệ duy nhất là `sed` ở bước 9, vá hai dòng trong
`ops/src/cuda/ms_deform_attn_cuda.cu`. Không vá từ `cofseg/` được vì đó là mã
CUDA, và op phải biên dịch xong thì mới có gì để nạp. Bản vá nằm trong repo
con nên clone lại là mất; vì thế nó là một bước ghi trong README. Hoàn tác:
`git -C benchmark/mask2former/upstream checkout .`.

Đổi commit: `git -C benchmark/mask2former/upstream checkout <commit>` rồi
commit ở repo ngoài; git ghi lại commit mới của repo con.

## Ghi chú

- **Batch 8, khác ba model kia (16).** Batch 16 tràn VRAM vì model giữ 100
  query cộng attention toàn ảnh; recipe gốc là batch 16 trên **8** GPU, tức 2
  ảnh mỗi GPU. `lr` để trống nên trainer nhân `batch/16`, lịch học vẫn đúng
  tỉ lệ recipe. Đổi backbone hay `imgsz` thì chạy `--probe` lại trước.

- **Không có kernel CUDA của `MSDeformAttn` thì hết VRAM.** Đường lùi
  `ms_deform_attn_core_pytorch` hiện vật hoá một tensor mà kernel CUDA không
  bao giờ dựng:

  ```
  torch.stack(sampling_value_list, dim=-2)   # (N*M, D, Lq, L, P)
  16*8 x 32 x 21504 x 3*4 x 4 byte = 3.94 GiB   (batch 16)
  ```

  Encoder có **6 lớp** và autograd giữ lại tensor này cho backward, nên riêng
  khoản này ~23.6 GiB ở batch 16. (`Lq = 128² + 64² + 32² = 21504` ở ô LSJ
  1024x1024; MSDeformAttn chạy fp32 vì `@autocast(enabled=False)`.) Bước 9 là
  bắt buộc.

- **Mapper LSJ của repo gốc lỗi trên ảnh nền**, đã vá bằng
  `empty_safe_mapper` trong `cofseg/models/detectron2.py` (không sửa repo
  con). Dòng lỗi:

  ```python
  instances = utils.annotations_to_instances(annos, image_shape)
  instances.gt_boxes = instances.gt_masks.get_bounding_boxes()   # <- lỗi
  ...
  if hasattr(instances, 'gt_masks'):                             # <- có kiểm
  ```

  `annotations_to_instances` chỉ gắn `gt_masks` khi `len(annos)` > 0. Dòng
  kiểm ở dưới cho thấy repo gốc biết trường đó có thể vắng, chỉ sót một chỗ.
  Recipe COCO của repo gốc không gặp lỗi này vì để
  `FILTER_EMPTY_ANNOTATIONS: True`; ở đây để `False` để giữ ảnh nền, và
  block/f1 có **2 ảnh nền trên 470 ảnh train**, đủ làm lượt chạy dừng ở
  iteration 43/235. Vá mapper thay vì bật `FILTER_EMPTY_ANNOTATIONS` cho riêng
  model này, vì bốn model phải dùng cùng một tập dữ liệu.

- **Tốn giờ máy nhất trong bốn model** dù chỉ chạy **50 epoch** (ba model kia
  100), và phần đắt là val chứ không phải train. Đo trên RTX 5090:

      train   3300 vòng x 0.508 s            = 28 phút / fold
      val     50/val_every x n_val x 1.194 s = 24 phút (f5, 48 ảnh val)
                                               ... 4h39 (f6, 280 ảnh val)

  `val_every` là tham số quyết định giờ máy lớn nhất của cả bảng: 6 fold tốn
  17.1 giờ ở mức 1 và 10.1 giờ ở mức 2. Config chọn **2**; số đo nằm trong
  `configs/train/mask2former_r50_d2.yaml`.

- **50 epoch là kết quả đo.** Lượt chạy thử 100 epoch cho thấy epoch tốt nhất
  luôn rơi trước mốc 50, phần sau chỉ tốn giờ máy. Recipe gốc dùng 50 epoch
  trên 118k ảnh COCO; bộ này có ~450 ảnh train nên số vòng thực tế ít hơn hai
  bậc, và query vẫn kịp hội tụ.

- **Tăng cường giữ LSJ của recipe gốc** (mapper trong `train_net.py` của repo
  con), không áp lật dọc hay xoay 90° như hai model R-CNN.
