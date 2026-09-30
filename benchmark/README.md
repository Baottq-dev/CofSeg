# benchmark/ — mỗi model một thư mục độc lập, mỗi thư mục một người

Bốn model của bảng so sánh, mỗi model nằm trọn trong một thư mục: **code
riêng, config riêng, cách chạy riêng, kết quả riêng**. Không thư mục nào
import thư mục nào, không ai phải chờ ai để sửa phần của mình.

## Phân công

| Thư mục | Người phụ trách | GitHub | Model | Vai trò trong bảng |
|---|---|---|---|---|
| `maskrcnn/` | VanNguyen | @vnguyen123 | Mask R-CNN R50-FPN | **mốc số 0** — mọi model khác báo Δ% mAP so với nó |
| `solov2/` | PhuongQuynh | @Phquynh2312 | SOLOv2 R50-FPN | box-free: lưới + kernel động, không box |
| `yolo11/` | QuangBao | @Baottq-dev | YOLOv11-Seg | một giai đoạn, thời gian thực |
| `mask2former/` | AnhVu | @tranphuocanhvu2103 | Mask2Former R50 | query / transformer, mask toàn ảnh |

Ba model dùng chung backbone ResNet-50 (Mask R-CNN, SOLOv2, Mask2Former) nên
chênh lệch giữa chúng là do cơ chế, không do backbone.

## Một thư mục có gì

```
benchmark/<model>/
├─ README.md              model gì, vì sao chọn, cách chạy, trạng thái
├─ train.py               huấn luyện, nạp cofseg/ của thư mục này
├─ evaluate.py            chấm, nạp cofseg/ của thư mục này
├─ cofseg/                BẢN SAO LÕI của riêng thư mục này
│   ├─ datasets/          đọc fold COCO
│   ├─ metrics/           mask AP, Boundary AP/IoU, sai số diện tích
│   ├─ evaluation/        vòng chấm
│   ├─ models/            wrapper suy luận của model này
│   └─ training/          trainer của model này
├─ configs/train/*.yaml   cấu hình huấn luyện
├─ configs/eval/*.yaml    cấu hình chấm
├─ tests/                 test cho phần của mình
├─ runs/                  (không vào git) kết quả train/eval
├─ results/               file kết quả nhỏ, ĐƯỢC commit
└─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
```

Sửa gì trong thư mục mình cũng được — kể cả `cofseg/`. Không ai bị ảnh hưởng.

## Dựng môi trường

Linux, GPU NVIDIA, một env conda cho cả bốn model. Tám bước, chạy từ trên
xuống, mỗi bước kiểm xong mới đi tiếp. Mất ~15 phút, cộng 20–120 phút cho
bước 5 vì mmcv không có wheel cho CUDA 12.8 nên phải build từ nguồn.

Chỉ chạy một model thì theo mục *Dựng môi trường riêng* trong README của thư
mục đó, ngắn hơn.

### 1. Env và CUDA toolkit

```bash
conda create -y -n cofseg python=3.12
conda activate cofseg
conda install -y -c nvidia cuda-toolkit=12.8.1

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh <<'EOF'
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
EOF
conda activate cofseg

nvcc --version | tail -2
```

CUDA 12.8 là bản đầu tiên có `sm_120` (RTX 5090) mà vẫn phủ các đời card cũ.
Cài vào env chứ không vào máy, nên mọi máy thuê ra cùng một env; `nvcc
--version` phải in ra 12.8 trước khi đi tiếp.

### 2. torch

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch, torchvision
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print(torch.cuda.get_device_name(0))"
```

Cài torch và torchvision trong cùng một lệnh, và kiểm bằng cách **gọi op**:
torchvision lệch bản dựng vẫn `import` trót lọt rồi mới gãy lúc gọi.

### 3. Gói Python của bốn model

```bash
pip install -r benchmark/requirements.txt
```

Không cần `pip install -e .`: mỗi thư mục model tự chứa bản `cofseg/` riêng.

### 4. detectron2 — Mask R-CNN và Mask2Former

```bash
pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

python -c "import detectron2; from detectron2 import model_zoo; print(detectron2.__version__)"
```

`--no-build-isolation` vì `setup.py` của nó `import torch`. Kiểm phải gọi cả
`model_zoo` — đó là chỗ duy nhất còn `import pkg_resources`.

### 5. mmcv — SOLOv2

```bash
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 \
  pip install --no-build-isolation -e .
cd -

python -c "from mmcv.ops import nms; print('op CUDA có thật')"
```

Clone **ra ngoài repo**: thư mục `mmcv/` nằm trong gốc repo sẽ che gói đã cài,
và mọi lệnh chạy từ gốc repo báo `module 'mmcv' has no attribute '__version__'`.

Build thật chạy hàng trăm dòng `nvcc` trong hàng chục phút; xong trong vài
giây là torch chưa có trong env lúc build. `TORCH_CUDA_ARCH_LIST` đặt theo
card: `12.0` cho RTX 5090, `9.0` cho H100, `8.9` cho L40S / RTX 4090, `8.0`
cho A100.

### 6. mmengine và mmdet — SOLOv2

```bash
pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
python -c "import mmdet, mmcv, mmengine; print(mmdet.__version__, mmcv.__version__)"
```

Bản mmengine trên PyPI hỏng trên torch ≥ 2.6 và số phiên bản không phân biệt
được hai bản, nên phải lấy từ git và kiểm bằng mã nguồn — dòng kiểm phải in
`True`. Bước này phải **sau bước 5**, vì cài mmcv kéo bản PyPI về.

`sed` nới chốt của mmdet: nó khai `mmcv < 2.2.0` trong khi 2.2.0 là bản mới
nhất tồn tại.

### 7. MSDeformAttn — Mask2Former

```bash
git submodule update --init benchmark/mask2former/upstream
pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Build thẳng thư mục `ops` chứ đừng chạy `make.sh` của repo gốc: nó gọi
`setup.py install`, thứ setuptools mới đã bỏ. Bỏ bước này thì Mask2Former vẫn
chạy nhưng rơi xuống đường Python tốn 3.94 GiB VRAM mỗi lớp encoder.

### 8. Kiểm cả env

```bash
python -c "
import inspect, torch, torchvision, ultralytics, detectron2, mmcv, mmdet
import mmengine.runner.checkpoint as ckpt
from torchvision.ops import nms as tv_nms
from mmcv.ops import nms as mmcv_nms
from detectron2 import model_zoo
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, tv_nms(b, s, 0.5).tolist())
print('detectron2 ', detectron2.__version__)
print('mmcv       ', mmcv.__version__, mmcv_nms(b, s, 0.5)[1].tolist())
print('mmdet      ', mmdet.__version__, '| mmengine vá:', 'weights_only' in inspect.getsource(ckpt))
print('ultralytics', ultralytics.__version__)"

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

Đủ bảy dòng, `mmengine vá: True` và `MSDeformAttn OK` là env dùng được cho cả
bốn model.

## Siêu tham số, và số liệu chọn ra chúng

Mặc định của bốn model đặt theo chính bộ dữ liệu này, không theo recipe COCO
cũng không theo card 8 GB ở nhà. Đo trên 850 ảnh / 12138 vùng của
`data/export/dataset_v1`:

| | |
|---|---|
| ảnh | 2560 x 1440, đồng nhất cả bộ |
| tán, cạnh tương đương | trung vị **324 px**, p5 127 px, p95 620 px |
| vùng mỗi ảnh | trung vị 13, trung bình 14.3, tối đa **48** |
| ảnh nền | 2 / 850 |

Ba số phải giống nhau ở cả bốn model, vì chúng ảnh hưởng kết quả mà không
phải kiến trúc:

| | giá trị | vì sao |
|---|---|---|
| `imgsz` | **1024** | tán trung vị còn 129 px. Chạy 1536 là nhìn tán to hơn 1.5 lần, và chênh lệch đó sẽ bị ghi vào cột kiến trúc |
| `batch` | **16** | đo thật trên RTX 5090: 12.7 GB ở imgsz 1024. Mặc định cũ (4, và 2 cho YOLO) là con số của card 8 GB ở nhà |
| `epochs` | **50** | Mask2Former 100, vì query hội tụ chậm hơn — chênh lệch có chủ đích, ghi trong config của nó |

`warmup_iters` chuyển sang **phần của lịch** (`0.03`), cùng quy ước với
`lr_steps`: dưới 1 là phần, từ 1 trở lên là số vòng tuyệt đối. Lý do đo được:
500 ảnh ở batch 16 là 32 vòng/epoch, 50 epoch là 1600 vòng, nên 200 vòng cứng
thành **12.5% lịch** trong khi recipe COCO warmup chưa tới 1%. Lượt khói
28/09 cho thấy đúng điều đó: hết 3 epoch lr vẫn chưa lên tới giá trị đã đặt.

`max_det` 100 và `num_queries` 100 giữ nguyên: ảnh dày nhất có 48 tán, nên cả
hai đều dư gấp đôi.

### Hai trần độ phân giải không nằm ở imgsz

Đáng biết trước khi đọc bảng kết quả, vì dự án lấy đường biên làm trọng tâm:

| | trần | ở imgsz 1024 |
|---|---|---|
| Mask R-CNN | đầu mask **28x28** rồi phóng lên bbox | tán trung vị 129 px -> **4.6 px mỗi ô** |
| YOLO | lưới prototype `imgsz/4` = 256 | **10 px ảnh gốc** mỗi ô |

Cả hai đều nâng được: `--mask_resolution 28` cho R-CNN (thành 56x56, còn 2.3
px/ô), `--mask_ratio 2` cho YOLO (proto 512, còn 5 px). Mặc định giữ nguyên
recipe gốc để mốc số 0 không bị đổi ngầm; nâng chúng là một thí nghiệm riêng,
đáng chạy sau khi có mốc nền. SOLOv2 và Mask2Former không có trần kiểu này:
cả hai sinh mask ở stride 4 của toàn ảnh.

### Ngưỡng diện tích của COCO gần như vô nghĩa ở đây

97.4% số vùng rơi vào ô **large** theo thang COCO (small `<32²`, medium
`<96²`, tính trên ảnh gốc): chỉ 91 vùng small và 230 vùng medium trên 12138.

Nên trong bảng `cocoeval.txt`, ba dòng `AP_small` / `AP_medium` / `AP_large`
không chia được bộ này thành ba nhóm có ý nghĩa — đọc `AP_large` là gần như
đọc `AP`. Muốn cắt theo cỡ tán thì dùng `per_region.csv`, nơi mỗi vùng có
`gt_area` và `gt_side` thật.

## Chạy

Cắt fold một lần cho cả nhóm, từ gốc repo. Sáu lượt luôn giống nhau (mỗi ruộng
làm test một lần, các ruộng còn lại train); khác nhau ở chỗ cắt val ra sao.
Nhóm chạy **cả ba cách**, nên cắt cả ba bộ fold:

```bash
# bộ 1: khối ảnh cuối mỗi đường bay, trượt theo lượt, đệm theo đồ thị chồng lấn
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_block.yaml --all --out-root data/export/block

# bộ 2: trọn hai đường bay cuối (field_1/10/4 + field_2/10/2), cố định
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_flight.yaml --all --out-root data/export/flight

# bộ 3: trọn một ruộng làm val, xoay vòng — val là ruộng kế tiếp ruộng test
python scripts/make_fold.py --export data/export/dataset_v1 --val configs/dataset/val_field.yaml --all --out-root data/export/field

# bảng so ba cách trên cả sáu lượt (để viết báo cáo, không phải để chọn một)
python scripts/compare_val_splits.py
```

Ra 18 thư mục fold. Ảnh được hardlink nên gần như không tốn thêm đĩa.

Ba bộ khác nhau ở **khoảng cách giữa val và train**, và đó là điều đáng viết
trong báo cáo:

| bộ | val nằm ở đâu | train | val TB | đệm TB |
|---|---|---|---|---|
| `block` | cùng đường bay với train | 5 ruộng | 105 ảnh | 34 ảnh |
| `flight` | khác đường bay, cùng ruộng | 5 ruộng | 110 ảnh | 33 ảnh |
| `field` | **khác ruộng** — đúng điều kiện của test | 4 ruộng | 142 ảnh | 0 ảnh |

Chỉ bộ `field` chọn checkpoint bằng thước đo giống thước sẽ chấm nó. Hai bộ
kia chọn bằng ảnh cùng mảnh đất với train, tức lạc quan hơn test; nếu `field`
cho số thấp hơn thì chênh lệch đó chính là phần lạc quan, đo được.

Bộ `field` mất một ruộng trong train nhưng **gần như không mất ảnh** (567 so
với 570 của `block`): nó không phải trả khoảng đệm nào, vì hai ruộng là hai
mảnh đất không dính nhau. Chỗ nó yếu là val nhảy 48..280 ảnh và 6.9..25.1
vùng/ảnh tuỳ lượt, trong khi `block` giữ val quanh 12.7..15.9.

**Mỗi model chạy 18 lượt**: 6 fold x 3 bộ. Cả nhóm là 72 lượt. Lệnh trong
README của từng model viết sẵn cho `data/export/block/f1`; đổi `block` thành
`flight` hoặc `field`, hoặc `f1` thành `f2`..`f6`, là ra các lượt còn lại. Số
của ba bộ là ba thí nghiệm khác nhau — so trong cùng một bộ, đừng so chéo.

Thiếu giờ máy thì cắt đủ 18 thư mục nhưng chạy bộ thứ ba trên hai lượt sàng
`f4` và `f2` (khai sẵn ở `screening` trong `folds.yaml`) trước, rồi quyết.

Sau đó **mỗi người chạy model của mình** — lệnh cụ thể nằm trong README của
từng thư mục. Không có script chạy cả bốn: mỗi model một framework, một lịch,
một người chịu trách nhiệm.

Ba quy ước phải giữ, vì bước gộp bảng cuối dựa vào chúng:

| | |
|---|---|
| `--runs benchmark/<model>/runs` | kết quả rơi vào thư mục của mình, không lẫn của người khác |
| `--name <model>` | chỉ tên model, KHÔNG kèm fold |
| `preds/<model>_<bộ>_<fold>.json` | file dự đoán test, đặt ở `preds/` gốc để cả nhóm đối chiếu |

**Vì sao `--name` không kèm fold.** Fold và bộ fold đọc được từ đường dẫn dữ
liệu đã chạy, nên không cần gõ lại — và không nên gõ lại: gõ
`--name maskrcnn_f4` mà trỏ vào `data/export/block/f5` thì tên nói dối, còn
đường dẫn thì không. Bảng tổng hợp lấy theo đường dẫn và báo ra nếu hai chỗ
mâu thuẫn.

Thư mục lần chạy tự ghép đủ, không lặp:

```
runs/train/2026-09-25_091500_maskrcnn_block-f4_i1024b4e50/
                             └ model ┘ └bộ-fold┘ └tham số┘
```

**Tên file trong `preds/` và `results/` thì PHẢI có bộ fold**, vì đó là file
thường do người gõ đặt tên: ba bộ cùng đánh số f1..f6 nên
`preds/maskrcnn_f4.json` của bộ này sẽ đè của bộ kia.

Gộp kết quả bốn người thành bảng model × ruộng:

```bash
# cả ba bộ trong một bảng, có cột "bộ fold"
python scripts/summarize_folds.py --eval benchmark/*/runs/eval

# chỉ một bộ
python scripts/summarize_folds.py --eval benchmark/*/runs/eval --dataset block
```

Nếu hai lần chấm trùng (cùng model, cùng bộ, cùng fold) thì bảng giữ lần mới
nhất và **báo ra** — không im lặng đè.

## Màn hình khi train, và log đầy đủ nằm đâu

Cả bốn model in ra cùng một dạng, để bốn lượt train đọc được cạnh nhau:

```
Dữ liệu: train 470 ảnh / 7101 vùng | val 85 ảnh / 1078 vùng | test 280 ảnh / 3720 vùng (field_1)
maskrcnn: 3 epoch x 30 iteration, batch 16, imgsz 1024, lr 0.02, max_iter 90
epoch 1/3   iter 30/90   loss 2.627   lr 2.00e-02   9.3G   mAP50-95   3.76   mAP50  13.52   0:44   * tốt nhất
epoch 2/3   iter 60/90   loss 1.987   lr 2.00e-02   9.5G   mAP50-95  12.81   mAP50  33.70   0:43   * tốt nhất
epoch 3/3   iter 90/90   loss 1.800   lr 2.00e-02   9.5G   mAP50-95  27.61   mAP50  55.79   0:43   * tốt nhất

Chấm test bằng best.pth trên 280 ảnh ...

----------------------------------------------------------------
maskrcnn · 2026-09-24_144830_maskrcnn-r50-d2_block-f1_i1024b16e3
----------------------------------------------------------------
  epoch tốt nhất   3/3
  val              mAP50-95 27.61   mAP50 55.79
  test (field_1)   mAP50-95 24.10   mAP50 51.30
  thời gian        train 2:12
  trọng số         weights/best.pth
----------------------------------------------------------------
```

Trong lúc một epoch đang chạy có thanh tiến trình `tqdm` kèm loss. Thanh tự
tắt khi stdout không phải terminal (`nohup`, `> log.txt`), lúc đó chỉ còn các
dòng epoch.

### Tên chỉ số

Lấy theo cách gọi của YOLO cho **cả bốn** model. Cùng một đại lượng, mỗi khung
một tên:

| In ra | detectron2 | mmdet | Nghĩa |
|---|---|---|---|
| `mAP50-95` | `AP` | `segm_mAP` | AP trung bình trên IoU 0,50 → 0,95 |
| `mAP50` | `AP50` | `segm_mAP_50` | AP tại IoU 0,50 |

mmengine trả thang 0–1, ba khung kia trả 0–100; màn hình luôn là **0–100**.
`results.csv` và `summary.json` giữ nguyên thang gốc của từng khung.

### Log đầy đủ vẫn còn nguyên

Màn hình chỉ bị **nâng ngưỡng lên WARNING**, không mất dòng nào trên đĩa:

| File | Nội dung |
|---|---|
| `<run>/run.log` | bản sao đúng màn hình, kể cả thanh tiến trình |
| `<run>/d2/log.txt` | detectron2: config đầy đủ, kiến trúc model, từng iteration, bảng COCO |
| `<run>/mmdet/<timestamp>/<timestamp>.log` | mmengine: tương tự |
| `<run>/results.csv` | một dòng mỗi lần ghi số liệu, có cột epoch |
| `<run>/warnings.log` | cảnh báo của thư viện, mỗi nội dung một lần |

Cảnh báo từ `logging` (`WARNING`) vẫn hiện ra màn hình — ví dụ dòng
`Skip loading parameter 'cls_score' ... (81, 1024) vs (2, 1024)`, là dấu hiệu
đầu phân loại COCO 81 lớp được khởi tạo lại cho 1 lớp, đúng như mong muốn.

Cảnh báo từ module `warnings` thì đi vào `<run>/warnings.log`, mỗi nội dung
đúng một lần, và khối tổng kết cho biết có bao nhiêu loại. Chúng vốn bắn ra
**mỗi iteration** — detectron2 gọi `torch.cuda.amp.autocast` đã bị khai tử —
và vì chúng ra stderr còn thanh tiến trình vẽ ra stdout nên mỗi lần bắn là
một lần thanh bị cắt đôi.

Cơ chế gộp sẵn của Python không cứu được: nó đánh dấu "cảnh báo này hiện rồi"
trong một registry mà CPython **xoá sạch** mỗi khi danh sách bộ lọc đổi phiên
bản, và `warnings.catch_warnings()` lúc thoát luôn làm việc đó. Trong vòng lặp
train có thư viện gọi nó ở mỗi mẫu:

| | dòng cảnh báo / 30 iteration |
|---|---|
| không làm gì | 60 |
| bộ lọc `"once"`, không ai gọi `catch_warnings` | 2 |
| bộ lọc `"once"`, có `catch_warnings` mỗi iteration | **60** — thua |
| chuyển vào file (cách đang dùng) | **0** trên màn hình, 2 trong file |

Muốn xem đúng kiểu mặc định của khung thì thêm `--verbose true`. Đo trên một
lượt Mask R-CNN 3 epoch: **1474 dòng** khi bật, **khoảng 30 dòng** khi tắt —
1050 dòng trong số đó là dump config và kiến trúc model, riêng kiến trúc bị in
hai lần.

## Cái giá của việc tách rời

Bốn bản sao `cofseg/` xuất phát giống hệt nhau. Nếu một người sửa phần **chấm
điểm** (`metrics/`, `evaluation/`, `datasets/`) thì model của người đó đo bằng
một cái thước khác, và cột "Δ% mAP so với Mask R-CNN" không còn nghĩa.

Sửa trainer hay model wrapper thì thoải mái — đó là phần của bạn. Sửa phần
chấm điểm thì **phải báo nhóm** để ba người kia sửa theo, hoặc ghi rõ khác
biệt khi trình bày bảng.

Dùng chung thật sự chỉ còn: `data/` (ảnh + nhãn), `weights/` (trọng số COCO),
`scripts/make_fold.py` (cắt fold — mọi người phải dùng CÙNG bộ fold),
`scripts/summarize_folds.py` (gộp bảng).

## Quy ước làm chung

- **Gói của model nào khai trong thư mục model đó** — `benchmark/<model>/requirements.txt`.
  Cần thư viện mới thì thêm vào file của mình rồi báo nhóm, không `pip install`
  riêng rồi quên ghi. `requirements.txt` ở gốc chỉ lo `app/` và `canopyseg/`.
  **torch thì KHÔNG khai ở đó** — nó là lựa chọn của máy, không phải của
  model, nên cài riêng ở bước 1; xem mục *Dựng môi trường* ở trên.
- **Mỗi người một nhánh**, gộp vào `main` bằng merge hoặc rebase.
  **Không dùng "Squash and merge"**: squash gộp nhiều commit thành một và làm
  mất author của từng commit, tức mất dấu vết phân công.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
  Commit trong thư mục nào thì mang tên người phụ trách thư mục đó.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`.
  Kết quả muốn chia sẻ thì chép file nhỏ vào `benchmark/<model>/results/`.
- Chạy test của thư mục mình trước khi commit:
  `cd benchmark/<model> && python -m pytest tests`

## Commit đúng tên khi dùng chung một máy

Tạo `.authors.local` ở gốc repo — file này **không vào git** (tên và email
thật của bốn người không cần nằm trên GitHub), nên mỗi máy tự tạo một lần. Nội
dung hỏi trong nhóm; mẫu:

```bash
_as() { local n="$1" e="$2"; shift 2
  GIT_AUTHOR_NAME="$n" GIT_AUTHOR_EMAIL="$e" \
  GIT_COMMITTER_NAME="$n" GIT_COMMITTER_EMAIL="$e" "$@"; }
as_quynh() { _as "PhuongQuynh" "<email của Quỳnh>" "$@"; }
```

Dùng:

```bash
source .authors.local
as_quynh git commit -m "solov2: raise the score threshold for the f5 run"
git log -1 --format="A: %an <%ae>%nC: %cn <%ce>"    # kiểm lại
```

Đếm theo người: `git shortlog -sne`. Ai muốn gộp hai tên của cùng một người
(`Baottq-dev` và `QuangBao` chung một email) thì tự tạo `.mailmap` ở gốc repo
— git đọc nó từ thư mục làm việc, không cần được track.

Chỉ commit dưới tên một người khi người đó **thật sự làm phần đó**.
