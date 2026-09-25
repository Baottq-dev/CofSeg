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
#    (thêm --limit 16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f1 --runs benchmark/maskrcnn/runs --name maskrcnn --workers 8

# 2. chấm test bằng trọng số tốt nhất
RUN=$(ls -td benchmark/maskrcnn/runs/train/*_maskrcnn_block-f1_* | head -1)
python benchmark/maskrcnn/evaluate.py --config benchmark/maskrcnn/configs/eval/maskrcnn_r50_d2.yaml --weights "$RUN/weights/best.pth" --data data/export/block/f1 --split test --runs benchmark/maskrcnn/runs --name maskrcnn

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
  --weights benchmark/maskrcnn/runs/train/<...>/weights/best.pth \
  --arch maskrcnn --conf 0.05 --max_det 100 \
  --iou-thr 0.5 --band-ratio 0.02 \
  --dilation-ratio 0.02 --nsd-tau 2.0
```

Mọi tham số ở đây là **cờ thật**, không qua `--set`: `--weights` trọng số của
lần train, `--conf`/`--max_det` (và các khoá khác của lớp model) đi vào khối
`model:`, còn `--iou-thr`/`--band-ratio`/`--dilation-ratio`/`--nsd-tau` đi vào
khối `eval:`. Bốn khoá `eval:` là định nghĩa của phép đo, đổi chúng là số
không so được với ba model kia nữa: `iou_thr` ngưỡng ghép cặp, `band_ratio`
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

Ba cột giữa là của `MODEL_ZOO.md` (detectron2, COCO val2017). Cột cuối suy từ
tỉ lệ với R50 nhân số đo thật của lượt R50 — **ước lượng**, phải `--probe`
trước khi đặt lịch.

#### ViTDet — backbone transformer

| `--backbone` | mask AP COCO | tham số | ghi chú |
|---|---|---|---|
| `vit-b` | 45,9 | 86 M | bản duy nhất trong nhóm còn hy vọng vừa card |
| `vit-l` | 49,2 | 304 M | cần batch rất nhỏ |
| `vit-h` | 50,2 | 632 M | ghi cho đủ bảng, khó chạy |

Đây là **Mask R-CNN với backbone ViT** (Li et al. 2022), không phải kiến trúc
khác, nên nó nằm cùng bảng với `r50`. Chênh lệch với `r50` là +8,7 điểm — lớn
hơn hẳn mọi backbone CNN ở bảng trên (`x101` chỉ +2,3).

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --backbone vit-b --batch 4 --keep_ckpts 1 --workers 8
```

**Ba thứ lượt ViT tự đổi, và bạn phải nói ra khi báo cáo:**

| | `r50` | `vit-b` | vì sao |
|---|---|---|---|
| optimizer | SGD 0,02 | **AdamW 1e-4** | ViT tiền huấn luyện MAE, SGD 0,02 phá đặc trưng ngay vài trăm vòng đầu |
| `weight_decay` | 1e-4 | **0,1** | recipe AdamW của detectron2 |
| lr theo độ sâu | không | **layer-wise decay 0,7** | recipe ViTDet |
| kênh ảnh | BGR | **RGB** | trọng số MAE dùng mean/std ImageNet RGB |

Lịch học thì **không** đổi: `max_iter`, `lr_steps`, `warmup_iters` và số epoch
giữ nguyên, nên lượt ViT vẫn cùng ngân sách với ba model kia. Có test khoá
điều này (`tests/test_r50_khong_doi.py`).

`--batch 4` trong lệnh trên là **phỏng đoán, chưa đo**. Recipe gốc của ViTDet
là batch 64 trên 64 GPU, tức 1 ảnh/GPU; `--batch 16` gần như chắc chắn OOM.
Chạy `--probe` trước. Và vì batch khác 16 nên lượt này **không vào bảng
chính** — nó là thí nghiệm phụ, báo cáo riêng.

Model của bản ViT dựng bằng `instantiate` từ chính LazyConfig của detectron2
chứ không dựng tay: tên tham số trong checkpoint COCO do cách dựng quyết định,
dựng lại bằng tay là đoán. Sau khi nạp, `summary.json` ghi mục
`weights_loaded` nói rõ khoá nào không vào được model — đầu phân loại lệch
hình là cố ý (80 lớp → 1), backbone lệch thì không, và lúc đó có một dòng cảnh
báo in ra màn hình.

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --backbone r101 --keep_ckpts 1 --workers 8
```

Không cần `--name`: thiếu nó thì tên thư mục run lấy theo backbone
(`maskrcnn-r101`), nên hai backbone không rơi vào cùng một tên thư mục.

Trọng số COCO đi theo config đã chọn: `base_cfg` lấy
`get_checkpoint_url` của **chính** file đó.

Cascade Mask R-CNN là *kiến trúc* khác chứ không phải backbone khác — nó có
config riêng và bảng backbone riêng (`r50`, `x152`):

```bash
python benchmark/maskrcnn/train.py \
  --config benchmark/maskrcnn/configs/train/cascade_r50_d2.yaml \
  --data data/export/field/f1 --runs benchmark/maskrcnn/runs \
  --keep_ckpts 1 --workers 8
```

`new_baselines` LSJ (mask AP 40,3–42,5) vẫn chưa có trong bảng. Chúng cũng là
LazyConfig như ViTDet nên đường dựng model đã thông, nhưng phần còn lại của
recipe (SyncBN trong backbone và FPN, box head 4conv1fc) nằm trong cùng file
config đó — thêm chúng là thêm một hàng vào `BACKBONES`, chưa làm vì chưa có
ai cần. Đáng nhớ một điều từ bảng đó — R50 + LSJ 100 epoch (40,3) hơn X101 3x
(39,5): recipe, không phải backbone.

Cascade + ViTDet thì chưa: config của nó nằm trong `projects/ViTDet/configs/`,
không được đóng gói theo package detectron2 nên cần đường dẫn tới checkout.

**Đọc cái này trước khi đổi.** Sáu lượt R50 trên `field/f1..f6` cho AP50
trung bình **92,5** nhưng AP chỉ **64,6**, và epoch tốt nhất rơi vào **16–23**
trên ngân sách 50. Việc *tìm* tán gần như xong; chỗ mất điểm nằm ở IoU cao,
tức chất lượng đường biên, và model đã bão hoà trước khi hết lịch. Backbone
sâu hơn mua chủ yếu AP50 — thứ đang còn ít dư địa nhất — và thêm tham số vào
một bộ 453 ảnh train thì bão hoà còn sớm hơn.

Đổi backbone ở **một** model cũng làm mất tính chất "bốn model cùng R50 nên
chênh lệch quy về cơ chế". Mà đổi đồng bộ cả bốn thì không làm được: YOLO
không có backbone để đổi, còn SOLOv2 không có trọng số COCO cho R101 thường.
Nên để phần này **ngoài bảng chính**, báo cáo riêng như thí nghiệm phụ.

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
