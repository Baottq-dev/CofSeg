# benchmark/ — bốn model, mỗi model một thư mục độc lập

Mỗi model của bảng so sánh nằm trọn trong một thư mục, với code, config, cách
chạy và kết quả riêng. Không thư mục nào import thư mục nào, nên sửa một thư
mục không ảnh hưởng ba thư mục kia.

| Thư mục | Model | Framework | Vai trò trong bảng |
|---|---|---|---|
| `maskrcnn/` | Mask R-CNN R50-FPN | detectron2 | **mốc số 0**: mọi model khác báo Δ% mAP so với nó |
| `solov2/` | SOLOv2 R50-FPN | mmdetection | box-free: lưới + kernel động, không box |
| `yolo/` | YOLO-Seg (v8 / 11 / 26) | ultralytics | một giai đoạn, thời gian thực |
| `mask2former/` | Mask2Former R50 | detectron2 + repo gốc | query / transformer, mask toàn ảnh |

Mask R-CNN, SOLOv2 và Mask2Former dùng chung backbone ResNet-50, nên chênh
lệch giữa ba model này đến từ cơ chế chứ không từ backbone.

```
benchmark/<model>/
├─ README.md              model gì, cách cài, cách chạy, kết quả
├─ requirements.txt       gói Python riêng của model (không ghim torch)
├─ train.py, evaluate.py  huấn luyện và chấm, nạp cofseg/ của thư mục này
├─ cofseg/                bản sao lõi riêng: datasets, metrics, evaluation, models, training
├─ configs/train|eval/    cấu hình huấn luyện và chấm
├─ tests/                 test của thư mục
├─ runs/                  (không vào git) kết quả train/eval
├─ results/               file kết quả nhỏ, được commit
└─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
```

## Dựng môi trường

Dựng trên **molab** (notebook marimo, một GPU NVIDIA): Linux, **torch
`2.11.0+cu130`**, venv `uv` ở `/venv/cofseg`. CUDA toolkit cài bằng pip vào
thẳng venv, không dùng toolkit của máy và không dùng conda. Bước 0–3 dùng
chung; phần còn lại của từng model nằm trong README của model đó. Cả bốn model
dùng chung được một env: cài lần lượt phần của từng model.

Hai điều cần biết về molab:

- **Đĩa bị xoá sạch khi notebook tắt** (giới hạn 12 giờ, hoặc nhàn rỗi 90
  phút): mất code, dữ liệu, venv, mmcv đã build và mọi kết quả chưa tải về
  máy. Mỗi phiên mới làm lại từ bước 0.
- **Lệnh cài chạy trong terminal của molab**, không chạy trong ô notebook. Ô
  notebook không có lệnh `!`; muốn chạy lệnh shell trong ô thì dùng
  `subprocess.run("...", shell=True)`. Lệnh train để trong một ô riêng, không
  dùng biến của ô khác, vì ô dùng biến của ô khác sẽ tự chạy lại.

### 0. Code và kiểm máy

```bash
git clone https://github.com/Baottq-dev/CofSeg.git
cd CofSeg
nvidia-smi --query-gpu=name,driver_version --format=csv
g++ --version | head -1
```

Cần driver **≥ 580.65.06** (mốc của CUDA 13.0) và `g++` ≤ **15.x**. CUDA 13
bỏ `sm_50`–`sm_70`, nên GTX 10xx và V100 không chạy được. `nvcc` của máy không
cần kiểm: bước 3 cài toolkit riêng vào venv. Dữ liệu không nằm trong repo,
xem mục *Chạy*.

### 1. Env

```bash
uv python install 3.12
uv venv /venv/cofseg --python 3.12
source /venv/cofseg/bin/activate
uv pip install --python /venv/cofseg/bin/python pip
which python          # phải ra /venv/cofseg/bin/python
```

Molab có sẵn một Python của notebook; tạo venv riêng, không cài đè lên Python
đó.

### 2. torch

```bash
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
python -c "import torch, torchvision; from torchvision.ops import nms; b=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]).cuda(); s=torch.tensor([0.9,0.8]).cuda(); print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, '| nms', nms(b,s,0.5).tolist()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Phải in `2.11.0+cu130 13.0`, tên card và compute capability của nó. Kiểm bằng
cách **gọi op trên GPU**, không chỉ `import`: torchvision lệch bản dựng vẫn
import được rồi mới hỏng lúc gọi.

### 3. CUDA toolkit — cho Mask R-CNN, SOLOv2, Mask2Former (YOLO bỏ qua)

Toolkit phải cùng major với torch (13.x); đừng dựa vào `which nvcc` của máy,
vì `nvcc` 12.x lệch major và torch dừng lúc build. Cài vào venv, rồi bù hai
thứ wheel còn thiếu: symlink `libcudart.so` và header `nv/target` (CCCL nằm ở
gói riêng).

```bash
pip install "cuda-toolkit[nvcc,nvvm,crt,cccl]==13.0.2"

export CUDA_HOME=$(python -c "import sysconfig,os;print(os.path.join(sysconfig.get_paths()['purelib'],'nvidia','cu13'))")
ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
test -f "$CUDA_HOME/include/nv/target" || cp -rn "$(dirname $CUDA_HOME)/cuda_cccl/include/"* "$CUDA_HOME/include/"

ls $CUDA_HOME/bin/nvcc $CUDA_HOME/nvvm/bin/cicc \
   $CUDA_HOME/include/cusparse.h $CUDA_HOME/include/cublas_v2.h \
   $CUDA_HOME/include/nv/target $CUDA_HOME/lib/libcudart.so
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"   # phải ra đúng $CUDA_HOME
```

Terminal mới không mang theo biến của terminal cũ, mà thiếu `CUDA_HOME` hay
`CPATH` là build hỏng giữa chừng (lỗi `nv/target: No such file` cũng từ đây).
Ghi biến vào file activate của venv rồi cho `~/.bashrc` nạp venv:

```bash
cat >> /venv/cofseg/bin/activate <<EOF
export CUDA_HOME="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include\${CPATH:+:\$CPATH}"
EOF
echo 'source /venv/cofseg/bin/activate' >> ~/.bashrc
```

### 4. detectron2 — cho Mask R-CNN và Mask2Former

Wheel torch đặt tên `libnvrtc-builtins-<băm>.so.13.0` trong khi
`libnvrtc.so.13` tìm tên không băm; thiếu liên kết này thì detectron2 dừng ở
phép `prod` trên GPU. Lệnh dưới tự bỏ qua khi wheel đã đặt tên đúng:

```bash
P=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
S=$(find "$P/nvidia" -name "libnvrtc-builtins-*.so.13.*" | head -1)
if [ -z "$S" ]; then echo "khong tim thay ban co bam — bo qua"; else
  ln -sfn "$(basename "$S")" "$(dirname "$S")/libnvrtc-builtins.so.${S##*.so.}"
fi
```

Rồi cài gói của model (`maskrcnn` hoặc `mask2former`) và detectron2:

```bash
pip install -r benchmark/maskrcnn/requirements.txt
pip install --no-build-isolation --no-cache-dir \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

python -c "import torch, detectron2; from detectron2 import _C, model_zoo; wh=torch.rand(8,16,2).cuda(); print(detectron2.__version__, 'ext OK', float(wh.prod(dim=2).sum()))"
```

- `--no-cache-dir`: pip dùng lại wheel đã dựng cho torch cũ nếu thiếu cờ này, và
  vẫn báo "Successfully installed". Build thật mất 5–10 phút; xong trong vài
  giây nghĩa là pip lấy bản cũ.
- Dòng kiểm phải gọi cả `_C` (phần biên dịch) lẫn `model_zoo` (chỗ còn
  `import pkg_resources`, nên `requirements.txt` ghim `setuptools<82`).

### Phần cài riêng của từng model

| Model | Làm sau bước 0–2 | Xem |
|---|---|---|
| YOLO | `pip install -r benchmark/yolo/requirements.txt`; không cần toolkit | `yolo/README.md` |
| Mask R-CNN | bước 3, 4 | `maskrcnn/README.md` |
| SOLOv2 | bước 3, rồi build mmcv, cài mmengine và mmdet | `solov2/README.md` |
| Mask2Former | bước 3, 4, rồi build MSDeformAttn | `mask2former/README.md` |

Dùng lại env cũ mà torch đã đổi bản thì mọi phần biên dịch phải build lại
(lỗi `undefined symbol: ...parseSchemaOrNameERKSsb`):
`pip uninstall -y detectron2 mmcv MultiScaleDeformableAttention`,
`pip cache remove detectron2`, rồi làm lại các bước tương ứng.

## Chạy

### Dữ liệu

Dữ liệu chưa công bố nên repo không kèm theo (`data/` không nằm trong git). Mỗi
lượt chạy đọc một thư mục fold, ví dụ `data/export/field/f1`:

```
f1/
├─ images/{train,val,test}/                      ảnh
├─ labels/{train,val,test}/                      nhãn YOLO-seg, một file .txt mỗi ảnh
├─ annotations/instances_{train,val,test}.json   nhãn COCO, một lớp `canopy`
├─ data.yaml                                     đường dẫn ba split, kiểu ultralytics
└─ fold.json                                     ghi lại cách chia của fold
```

Muốn chạy trên dữ liệu của mình thì dựng đúng bố cục này (một lớp, mặt nạ dạng
đa giác). Có bản xuất gốc `data/export/dataset_v1` thì cắt bằng
`scripts/make_fold.py`. Sáu lượt luôn giống nhau (mỗi ruộng làm test một lần);
ba cách chia khác nhau ở chỗ val lấy từ đâu:

```bash
# bộ 1: khối ảnh cuối mỗi đường bay, trượt theo lượt, đệm theo đồ thị chồng lấn
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block

# bộ 2: trọn hai đường bay cuối (field_1/10/4 + field_2/10/2), cố định
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# bộ 3: trọn một ruộng làm val, xoay vòng — val là ruộng kế tiếp ruộng test
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_field.yaml --all --out-root data/export/field
```

| bộ | val nằm ở đâu | train |
|---|---|---|
| `block` | cùng đường bay với train | 5 ruộng |
| `flight` | khác đường bay, cùng ruộng | 5 ruộng |
| `field` | **khác ruộng**, đúng điều kiện của test | 4 ruộng |

Chỉ bộ `field` chọn checkpoint trên ruộng khác train; hai bộ kia chọn bằng ảnh
cùng mảnh đất với train nên lạc quan hơn test.

### Train, chấm, gộp kết quả

Mỗi model chạy 18 lượt: 6 fold × 3 bộ. Lệnh train và chấm nằm trong README của
từng model, viết sẵn cho một bộ và một fold; đổi tên bộ (`block`, `flight`,
`field`) hoặc fold (`f1`..`f6`) là ra các lượt còn lại. Số của ba bộ là ba thí
nghiệm khác nhau: so trong cùng một bộ, không so chéo.

Ba quy ước phải giữ, vì bước gộp bảng dựa vào chúng:

| | |
|---|---|
| `--runs benchmark/<model>/runs` | kết quả nằm trong thư mục của model, không lẫn với model khác |
| `--name <model>` | chỉ tên model, KHÔNG kèm fold: bộ và fold đọc từ đường dẫn dữ liệu |
| `preds/<model>_<bộ>_<fold>.json` | file dự đoán test; file trong `preds/` và `results/` đặt tên tay nên phải có bộ fold, vì ba bộ cùng đánh số f1..f6 |

Gộp kết quả bốn model thành bảng model × ruộng (hai lần chấm trùng model, bộ,
fold thì bảng giữ lần mới nhất và báo ra):

```bash
python scripts/summarize_folds.py --eval benchmark/*/runs/eval                    # cả ba bộ
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block    # chỉ một bộ
```

### Chạy thử nhanh

Mỗi model có `--probe` (dò cấu hình và VRAM rồi dừng, vài giây). `--probe` báo
`suggested_batch` lớn hơn 16 thì **giữ 16**, để batch giống nhau giữa các model
(trừ Mask2Former, 8). Lượt khói thật, vài phút mỗi lượt (YOLO không có `--limit`,
dùng `--fraction` của ultralytics):

```bash
python benchmark/maskrcnn/train.py     --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml     --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/solov2/train.py       --config benchmark/solov2/configs/train/solov2_r50_mm.yaml         --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/mask2former/train.py  --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --data data/export/field/f1 --limit 20 --epochs 1
python benchmark/yolo/train.py         --config benchmark/yolo/configs/train/yolo11s.yaml                 --data data/export/field/f1 --fraction 0.05 --epochs 1
```

Molab chỉ cấp một GPU (số 0), nên không cần `CUDA_VISIBLE_DEVICES`.

## Kết quả (runs)

`runs/` không vào git, và molab xoá đĩa khi notebook tắt, nên kết quả train và
chấm của cả bốn model được lưu trên Google Drive:

**[Drive tổng của bốn model](https://drive.google.com/drive/folders/1yn9aguCCKgqOLXVj8ae6i6nqfVoYcdTw?usp=sharing)**

Link riêng của từng model nằm trong README của model đó.

Mỗi lượt train hoặc chấm là một thư mục tên
`<ngày>_<giờ>_<model>_<bộ>-<fold>_...` trong `train/` hoặc `eval/`:

| | Thường gồm |
|---|---|
| `train/<lượt>/` | `weights/best.pt` hoặc `best.pth` (chấm bằng file này), `weights/last.*`, `config.yaml`, `summary.json`, `run.log` |
| `eval/<lượt>/` | `metrics.json` (các chỉ số), `per_region.csv` (từng tán), `per_image.csv`, `coco_metrics.json`, `cocoeval.txt`, `predictions.json` |

Tải về rồi gộp bảng bằng `scripts/summarize_folds.py --eval <thư mục eval/>`.
File nhỏ cần commit thì chép vào `benchmark/<model>/results/` (xem README ở
đó); không chép `predictions.json` và trọng số.


## Siêu tham số

Mặc định đặt theo chính bộ dữ liệu này (850 ảnh 2560×1440, tán trung vị 324 px,
trung vị 13 tán mỗi ảnh, tối đa 48), không theo recipe COCO. Nguyên tắc: giữ
recipe của tác giả, chỉ đổi những gì bộ dữ liệu bắt phải đổi, và đã đổi thì đổi
cho cả bốn.

Ba số giống nhau ở cả bốn model, vì chúng ảnh hưởng kết quả mà không thuộc về
kiến trúc:

| | giá trị | ghi chú |
|---|---|---|
| `imgsz` | **1024** | tán trung vị còn 129 px; chạy 1536 thì chênh lệch bị ghi vào cột kiến trúc |
| `batch` | **16** | Mask2Former **8**: 16 tràn VRAM vì giữ 100 query cộng attention toàn ảnh. `lr` để trống nên trainer nhân `batch/16` |
| `epochs` | **100** | Mask2Former **50**: epoch tốt nhất luôn rơi trước mốc 50 |

### Lịch học

| | Mask R-CNN | SOLOv2 | Mask2Former | nguồn |
|---|---|---|---|---|
| optimizer | SGD | SGD | AdamW | recipe |
| `lr` @ batch 16 | 0.02 | 0.01 | 1e-4 | recipe |
| `weight_decay` | 1e-4 | 1e-4 | 0.05 | recipe |
| `momentum` | 0.9 | 0.9 | (AdamW không dùng) | recipe |
| clip gradient | — | norm 35 | full_model 0.01 | recipe |
| backbone lr x | — | — | 0.1 | recipe |
| `lr_steps` | **0.7 / 0.9** | **0.7 / 0.9** | **0.7 / 0.9** | **đặt cho bộ này** (gốc 0.778/0.926, 0.75/0.917, 0.889/0.963) |
| `warmup_iters` | **0.03** | **0.03** | **0.03** | **đặt cho bộ này** (gốc 1000, 500, 10 vòng) |

Lịch ở đây ngắn hơn COCO khoảng 170 lần (1600 vòng so với 270000), nên
`lr_steps` và `warmup_iters` tính theo **phần của lịch** (dưới 1 là phần, từ 1
trở lên là số vòng tuyệt đối). Truyền `--lr` tay là tắt phép nhân `batch/16`.

### Suy luận và chấm

| | giá trị | ghi chú |
|---|---|---|
| `val_conf` | **0.05** cả bốn | gốc 0.05 / 0.1 / không có / 0.001 |
| `max_det` | 100 | ảnh dày nhất 48 tán |
| `num_queries` | 100 | chỉ Mask2Former |
| NMS IoU | 0.5 (d2) / matrix NMS (SOLOv2) / 0.7 (YOLO) | cơ chế khác nhau, không đưa về một giá trị được |
| `mask_thr` | 0.5 | SOLOv2 |

`precision`, `false_positive` và sai số đếm trong `per_region.csv` tính trên mọi
dự đoán qua ngưỡng, nên cả bốn đưa về 0.05 (Mask2Former gốc không có ngưỡng,
ra 100 dự đoán trên ảnh 14 tán). AP vì vậy thấp hơn so với ngưỡng 0.001 mà
nhiều báo cáo YOLO dùng: so nội bộ đúng, so với số trong bài báo thì không.

### Tăng cường

| | Mask R-CNN | SOLOv2 | Mask2Former | YOLO |
|---|:---:|:---:|:---:|:---:|
| lật ngang / dọc | 0.5 / 0.5 | 0.5 / 0.5 | LSJ tự lật ngang | 0.5 / 0.5 |
| xoay 90° | ✓ | — | — | — |
| co giãn | — | — | **LSJ 0.1–2.0** | 0.5 |
| dịch | — | — | crop ô vuông | 0.1 |
| màu (HSV) | ✓ | ✓ | ✓ | ✓ |
| mosaic | — | — | — | **tắt** |

- **Màu có ở cả bốn**, cùng `hsv_h 0.015 / hsv_s 0.7 / hsv_v 0.4`: bộ dữ liệu
  bay 3 ngày, 07h–16h, độ sáng chênh nhau thật. Ba framework không có
  transform giống hệt: detectron2 không có đổi sắc nên `hsv_h` khai mà không
  áp; mmdet cộng delta độ sáng thay vì nhân (`hsv_v 0.4` thành ±51) và khoá
  tương phản ở 1.0.
- **LSJ** là recipe gốc của Mask2Former nên được giữ. **Mosaic tắt** vì nó thu
  nhỏ tán, đi ngược mục tiêu giữ độ phân giải đường biên. `degrees` và
  `copy_paste` của YOLO để 0 vì chỉ YOLO có hai phép này.

### Dữ liệu và vận hành

| | giá trị | vì sao |
|---|---|---|
| `min_area` | 50 px² | chỉ loại 1 vùng trên 12138 |
| ảnh nền | giữ trong train | 2/850 ảnh; bốn model phải dùng cùng một tập |
| `amp` | bật cả bốn | |
| `val_batch` | theo batch train | |
| `workers` | 0 trên Windows, 8 trên Linux (Mask2Former 2) | mỗi worker Windows nạp lại torch, paging file không đủ |
| `seed` | 0 | cả bốn |

97.4% số vùng thuộc nhóm **large** theo thang COCO, nên `AP_small / AP_medium /
AP_large` trong `cocoeval.txt` không chia bộ này thành ba nhóm có nghĩa. Muốn
phân theo cỡ tán thì dùng `gt_area` và `gt_side` trong `per_region.csv`.

## Màn hình khi train và log

Cả bốn model in cùng một dạng: mỗi epoch một dòng (loss, lr, VRAM, mAP50-95,
mAP50, thời gian), rồi khối tổng kết cuối lượt. Tên chỉ số theo YOLO:
`mAP50-95` là AP trung bình trên IoU 0,50 → 0,95, `mAP50` là AP tại IoU 0,50;
màn hình luôn in thang 0–100, còn `results.csv` và `summary.json` giữ thang gốc
của framework. Màn hình chỉ nâng ngưỡng lên WARNING, trên đĩa không mất dòng
nào:

| File | Nội dung |
|---|---|
| `<run>/run.log` | bản sao đúng màn hình |
| `<run>/d2/log.txt` | detectron2: config, kiến trúc, từng iteration, bảng COCO |
| `<run>/mmdet/<timestamp>/<timestamp>.log` | mmengine: tương tự |
| `<run>/results.csv` | một dòng mỗi lần ghi số liệu, có cột epoch |
| `<run>/warnings.log` | cảnh báo của thư viện, mỗi nội dung một lần |

Thêm `--verbose true` để xem cảnh báo theo kiểu mặc định của framework.

## Phần chấm điểm phải giống nhau ở cả bốn thư mục

Bốn bản sao `cofseg/` xuất phát giống hệt nhau. Sửa phần **chấm điểm**
(`metrics/`, `evaluation/`, `datasets/`) ở một thư mục thì model đó được đo
bằng thước khác, và cột "Δ% mAP so với Mask R-CNN" mất nghĩa: phải sửa giống
nhau ở cả bốn, hoặc ghi rõ khác biệt khi trình bày. Sửa trainer hay wrapper
model thì không ảnh hưởng phép đo. Dùng chung của cả bốn: `data/`, `weights/`
(trọng số COCO), `scripts/make_fold.py`, `scripts/summarize_folds.py`.

## Quy ước

- Gói của model nào khai trong `benchmark/<model>/requirements.txt`; `requirements.txt`
  ở gốc repo chỉ lo `app/` và `canopyseg/`. **torch không khai ở đâu cả**: bản
  torch phụ thuộc máy, nên cài riêng ở bước 2.
- Commit message tiếng Anh, mô tả thay đổi chứ không mô tả file.
- Không commit `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`; kết
  quả cần chia sẻ thì chép file nhỏ vào `benchmark/<model>/results/`.
- Chạy test của thư mục trước khi commit: `cd benchmark/<model> && python -m pytest tests`.
