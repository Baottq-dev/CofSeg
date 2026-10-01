# YOLOv11-Seg — họ một giai đoạn, thời gian thực

**Người phụ trách:** QuangBao (@Baottq-dev)
**Framework:** ultralytics (`trainer: yolo`)

## Vai trò

Đại diện nhóm real-time. Model này trả lời câu hỏi: **đưa model lên drone bay
trực tiếp thì độ chính xác giảm bao nhiêu** — so ms/ảnh và mAP với mốc 0.

## Vì sao chọn

- Model duy nhất trong bảng chạy được thời gian thực trên phần cứng nhúng.
  Cỡ **s** để so ngang với các model kia; cần số cho drone thật thì chạy thêm
  cỡ **n**.
- Mask từ 32 prototype ở stride 4 (~10 px một ô ở ảnh gốc) rồi cắt theo box:
  mịn hơn ROI 28×28 nhưng tán chạm nhau dễ dính mép cây bên — đo trên field_5.
- v11 là bản hiện hành, v8 là bản các bài trước hay dùng, cùng cơ chế mask;
  chạy cả hai nếu còn giờ (`yolo26s` cũng có sẵn config).

## Dựng môi trường riêng

Env riêng cho một mình model này, không dùng `benchmark/requirements.txt` của
cả nhóm. Chép chạy từ trên xuống; phần chung nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Nhẹ nhất trong bốn: không gói nào phải biên dịch, nên không cần CUDA toolkit,
và đây là model duy nhất chạy được trên Windows.

```bash
conda create -y -n cofseg-yolo11 python=3.12 && conda activate cofseg-yolo11
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/yolo11/requirements.txt

python -c "
import torch, torchvision, ultralytics
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print('ultralytics', ultralytics.__version__, '| GPU:', torch.cuda.get_device_name(0))"
```

Kiểm bằng cách gọi op chứ không chỉ `import`: torchvision lệch bản dựng vẫn
`import` trót lọt rồi mới gãy lúc gọi.

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
# 1. huấn luyện (thêm --epochs 1 --fraction 0.05 để khói)
python benchmark/yolo11/train.py --config benchmark/yolo11/configs/train/yolo11s.yaml --data data/export/block/f1 --runs benchmark/yolo11/runs --name yolo11s --workers 8

# 2. chấm bằng trọng số tốt nhất (weights/best.pt: chọn theo mask AP thuần,
#    không phải best.pt của ultralytics vốn cộng cả chỉ số hộp)
RUN=$(ls -td benchmark/yolo11/runs/train/*_yolo11s_block-f1_* | head -1)
python benchmark/yolo11/evaluate.py --config benchmark/yolo11/configs/eval/yolo11s.yaml --set model.weights="$RUN/weights/best.pt" --data data/export/block/f1 --split test --runs benchmark/yolo11/runs --name yolo11s

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/yolo11/runs/eval/*_yolo11s_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/yolo11s_block_f1.json
cp "$EV/metrics.json"   benchmark/yolo11/results/yolo11s_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/yolo11/results/yolo11s_block_f1_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/yolo11 && python -m pytest tests
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
python benchmark/yolo11/train.py \
  --config benchmark/yolo11/configs/train/yolo11s.yaml \
  --data data/export/field/f1 --runs benchmark/yolo11/runs --name yolo11s \
  --imgsz 1024 --batch 16 --epochs 50 --patience 0 \
  --optimizer auto --cos_lr true --amp true \
  --mask_ratio 4 --overlap_mask true \
  --fliplr 0.5 --flipud 0.5 --degrees 0 --copy_paste 0 \
  --mosaic 0 --close_mosaic 0 --scale 0.5 \
  --workers 8 --seed 0 --deterministic true --val true --plots true
```

`--workers 8` là giá trị cho máy Linux; trên Windows trainer tự đặt 0 vì mỗi
worker spawn một tiến trình mới và nạp lại torch, mà paging file mặc định
không đủ cho việc đó.

Trainer nhận **toàn bộ 114 tham số** của ultralytics, không chỉ những cái ở
trên — `--list-params` in hết. Bốn khoá bị khoá (`data`, `project`, `name`,
`exist_ok`) vì trainer tự đặt để kết quả rơi đúng thư mục run.

`optimizer: auto` nghĩa là ultralytics tự chọn AdamW và tự đặt `lr0` theo số
lớp và số epoch, nên đặt `--lr0` tay sẽ **không** có tác dụng chừng nào
optimizer còn là auto.

`--degrees`, `--copy_paste` và `--patience` để 0 là cố ý: ba model kia không
có hai tăng cường đó, và chúng chạy đủ số epoch đã khai. Bật lên là cho model
này lợi thế không đến từ kiến trúc.

Lệnh chấm, đầy đủ tham số:

```bash
python benchmark/yolo11/evaluate.py \
  --config benchmark/yolo11/configs/eval/yolo11s.yaml \
  --data data/export/field/f1 --split test \
  --runs benchmark/yolo11/runs --name yolo11s \
  --set model.weights=benchmark/yolo11/runs/train/<...>/weights/best.pt \
  --set model.imgsz=1024 --set model.conf=0.05 --set model.iou=0.7 \
  --set model.max_det=100 --set model.retina_masks=true \
  --set eval.iou_thr=0.5 --set eval.band_ratio=0.02 \
  --set eval.dilation_ratio=0.02 --set eval.nsd_tau=2.0
```

`--set model.*` đi vào khối `model:` của config chấm, `--set eval.*` vào khối
`eval:`. Bốn khoá `eval:` là định nghĩa của phép đo, đổi chúng là số không so
được với ba model kia nữa: `iou_thr` ngưỡng ghép cặp, `band_ratio` bề rộng
vành biên theo cỡ tán, `dilation_ratio` bề rộng vành của Boundary AP (2%
đường chéo ảnh, đúng bài báo), `nsd_tau` dung sai của NSD.

`model.imgsz` lúc chấm phải bằng `--imgsz` lúc train, không thì model học ở
một độ phân giải rồi bị chấm ở độ phân giải khác.

`weights/best.pt` là bản chọn theo mask AP thuần. `ultralytics/weights/best.pt`
vẫn còn, nhưng nó chọn theo fitness cộng cả chỉ số hộp nên không so được với
ba model kia.

Còn một đường chấm thứ hai, `--native`, gọi thẳng `model.val()` của ultralytics
— đúng thứ `yolo val` chạy, giữ để đối chiếu với các báo cáo YOLO khác:

```bash
python benchmark/yolo11/evaluate.py --native \
  --weights benchmark/yolo11/runs/train/<...>/weights/best.pt \
  --data data/export/field/f1 --split test --imgsz 1024 --batch 4 --max-det 100
```

### Đổi backbone

Model này **không đổi backbone được**, khác ba model kia: backbone của YOLO
gắn liền với kiến trúc, không có chỗ cắm ResNet vào. Thứ đổi được là **cỡ
model** — n / s / m / l / x, cùng kiến trúc, khác độ rộng và độ sâu:

```bash
python benchmark/yolo11/train.py --config benchmark/yolo11/configs/train/yolo11s.yaml \
  --data data/export/field/f1 --runs benchmark/yolo11/runs --name yolo11m \
  --set model=weights/yolo11m-seg.pt
```

Thiếu file thì ultralytics tự tải về đúng đường dẫn đó. Chọn cỡ `s` cho bảng
vì đo trên chính 6521 vùng của bộ này thì nút thắt là **độ phân giải** chứ
không phải dung lượng model: lưới mặt nạ của YOLO là `imgsz/4`, nên đổi `l`
lấy imgsz thấp là đánh đổi lỗ.

Hai họ YOLO khác chạy được bằng đúng đường này, config có sẵn:
`configs/train/yolov8s.yaml` và `configs/train/yolo26s.yaml`.

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

- Đường chạy **đã thử ở nhà**: probe (autobatch 13 → khuyến nghị 5 ở 1024),
  train 1 epoch trên 5 % fold f4, chấm 6 ảnh test ra `predictions.json`.
  Là model duy nhất trong bốn model đã đi hết đường train → eval.
- Chạy được trên Windows, không cần chờ máy Linux.
