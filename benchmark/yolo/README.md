# YOLO-Seg — họ một giai đoạn, thời gian thực

**Framework:** ultralytics 8.4.143 (`trainer: yolo`) · chạy được trên Linux và
Windows

Đại diện nhóm real-time: dùng một model chạy được trên phần cứng nhúng thì độ
chính xác giảm bao nhiêu so với mốc số 0 (Mask R-CNN). Cỡ **s** để so ngang với
các model kia. Mask dựng từ 32 prototype ở stride 4 (~10 px ảnh gốc mỗi ô) rồi
cắt theo box: mịn hơn ROI 28×28, nhưng ở chỗ tán chạm nhau mask dễ dính mép cây
bên. Thư mục giữ cả **họ** YOLO (v8, 11, 26; config riêng vì khác nhau ở phần
phát hiện), còn **cỡ** model thì đổi bằng `--model`.

## Cài đặt

Làm bước 0–2 của `benchmark/README.md`; model này không biên dịch gì nên không
cần CUDA toolkit. Rồi:

```bash
pip install -r benchmark/yolo/requirements.txt
python -c "
import torch, torchvision, ultralytics
from torchvision.ops import nms
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]).cuda(); s = torch.tensor([0.9,0.8]).cuda()
print('torch      ', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('torchvision', torchvision.__version__, '| nms CUDA', nms(b,s,0.5).tolist())
print('ultralytics', ultralytics.__version__)"
```

Phải in đủ ba dòng, dòng đầu có `2.11.0+cu130 13.0`, dòng cuối `ultralytics
8.4.143`. Mỗi terminal mới chỉ cần `source /venv/cofseg/bin/activate`.
`requirements.txt` cố ý **không** cài `albumentations`: ultralytics tự bật thêm
một loạt phép tăng cường khi thấy gói này, và nó làm hai pipeline huấn luyện
khác nhau mà không dòng config nào ghi lại.

## Chạy

Từ **gốc repo**. `--data` nhận thư mục fold, `--name` chỉ là tên model (quy ước
ở `benchmark/README.md`). Lệnh dưới chạy bộ **block**, fold **f1**; đổi `block`
hoặc `f1` ở cả ba chỗ là ra lượt khác.

```bash
# 1. huấn luyện (khói: thêm --epochs 1 --fraction 0.05)
python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml --data data/export/block/f1 --runs benchmark/yolo/runs --name yolo11s --workers 8

# 2. chấm bằng trọng số tốt nhất
RUN=$(ls -td benchmark/yolo/runs/train/*_yolo11s_block-f1_* | head -1)
python benchmark/yolo/evaluate.py --config benchmark/yolo/configs/eval/yolo11s.yaml --weights "$RUN/weights/best.pt" --data data/export/block/f1 --split test --runs benchmark/yolo/runs --name yolo11s

# 3. dự đoán + kết quả nhỏ
EV=$(ls -td benchmark/yolo/runs/eval/*_yolo11s_block-f1_* | head -1)
mkdir -p preds && cp "$EV/predictions.json" preds/yolo11s_block_f1.json
cp "$EV/metrics.json"   benchmark/yolo/results/yolo11s_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/yolo/results/yolo11s_block_f1_per_region.csv

# test của thư mục này
cd benchmark/yolo && python -m pytest tests
```

- Mọi tham số đều là cờ, mặc định nằm trong config (`imgsz 1024`, `batch 16`,
  `epochs 100`, `seed 0`). `--list-params` in danh sách (114 tham số của
  ultralytics), `--print-config` in config đã gộp mà không chạy, `--probe` chỉ
  dò VRAM. Bốn khoá bị khoá (`data`, `project`, `name`, `exist_ok`).
- `weights/best.pt` chọn theo mask AP thuần; `ultralytics/weights/best.pt` chọn
  theo fitness cộng cả chỉ số hộp nên không so được với ba model kia. Phép chấm
  `--native` (gọi thẳng `model.val()`) giữ để đối chiếu với báo cáo YOLO khác.
- `--degrees`, `--copy_paste`, `--patience` để 0 có chủ đích: ba model kia
  không có hai phép tăng cường đó và chạy đủ số epoch.
- `optimizer: auto` tự chọn AdamW và tự đặt `lr0`, nên `--lr0` tay không có tác
  dụng. `--workers` trên Windows tự về 0.
- Lúc chấm, `model.imgsz` phải bằng `--imgsz` lúc train. Bốn khoá `eval:`
  (`iou_thr`, `band_ratio`, `dilation_ratio`, `nsd_tau`) là định nghĩa của phép
  đo, đổi chúng thì số không còn so được với các model kia.

## Phiên bản và cỡ model (`--model`)

Model này **không đổi backbone được**; thứ đổi được là phiên bản và cỡ. `--list-models`
in danh sách. Tên trần mở rộng thành `weights/<tên>.pt` và ultralytics tự tải
nếu thiếu.

| họ | cỡ có trọng số COCO |
|---|---|
| `yolov8`, `yolo11`, `yolo26` | n s m l x |
| `yolov9` | c e |

```bash
python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml --data data/export/field/f1 --runs benchmark/yolo/runs --model yolo11m-seg --workers 8
```

`yolo12-seg` không có trọng số COCO nên không so được. Cỡ `s` được chọn vì giới hạn
của bộ này là độ phân giải (lưới mặt nạ `imgsz/4`), không phải dung lượng
model; từ cỡ `m` trở lên chạy `--probe` lại vì `batch 16 @ imgsz 1024` chỉ đo
cho cỡ `s`.

## Biến thể hàm loss (`--loss`)

Không có `--loss` thì train bằng loss gốc của ultralytics; bảng benchmark luôn
chạy như vậy. Biến thể chỉ bật bằng cờ (khối `loss:` trong config hay
`--set loss.use=...` bị chặn), mỗi lượt một biến thể.

```bash
python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s.yaml --list-losses
python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s.yaml --data data/export/field/f1 --loss dice --set loss.dice.weight=2
```

| biến thể | làm gì | tham số (mặc định) |
|---|---|---|
| `mask_iou` | điểm phân loại học **IoU mặt nạ** (cắt bằng hộp dự đoán) thay cho CIoU của hộp | `mix` 1.0, `warmup_epochs` 5, `heads` both \| o2o |
| `dice` | cộng **Dice** vào BCE của mặt nạ từng tán | `weight` 1.0 (0 = y hệt gốc) |

Tên thư mục run mang hậu tố (`..._yolo26s-seg-dice_field-f1_...`, tham số khác
mặc định cũng vào tên); `summary.json` ghi khối `loss`, `loss_<tên>.csv` cạnh
nó ghi từng epoch. Checkpoint vẫn là YOLO-Seg chuẩn, chấm bằng đúng lệnh chấm
cũ. Viết cho ultralytics 8.4.143; phiên bản khác thì `--loss` dừng ngay.

## Đầu mặt nạ A1 (`configs/train/yolo26s-dyn.yaml`)

Đầu gốc dựng mặt nạ mỗi tán bằng tổ hợp tuyến tính 32 prototype rồi cắt sát bằng
box. A1 giữ 32 prototype đó (cả trọng số COCO) làm đặc trưng, nhưng mỗi anchor
sinh 361 tham số của một mạng 1×1 nhỏ kiểu CondInst chạy trên [prototype, toạ
độ tương đối so với anchor], và mặt nạ được giữ trong cửa sổ box × 1.5 thay vì
cắt sát box. Thêm khoảng 0,07 M tham số. Chỉ dùng với YOLO26-seg; bật bằng file
config riêng, không dùng chung với `--loss`:

```bash
python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s-dyn.yaml --data data/export/field/f1
python benchmark/yolo/evaluate.py --config benchmark/yolo/configs/eval/yolo26s.yaml --weights runs/train/<...>_yolo26s-seg-dyn_field-f1_i1024b16e100/weights/best.pt --data data/export/field/f1 --split test
```

Khoá `mask_head`: `dims` (`[8, 8]`; `[]` = một lớp tuyến tính như gốc), `coords`
(`true`), `window` (`1.5`; `1.0` = cắt sát như gốc), `max_pos` (`64`). Lượt chấm
tự nhận ra checkpoint A1 và mang hậu tố `-dyn`. Checkpoint chứa lớp của
`cofseg/training/dyn_head.py` nên phải nạp từ thư mục này; chưa hỗ trợ export
ONNX/TensorRT.

## Kết quả (runs)

Kết quả train và chấm của YOLO (`runs/`) lưu trên Drive:
**[YOLO](https://drive.google.com/drive/folders/19e6Cv8VQQ0PWtqSl616A7x_e-RhaCgA0?usp=sharing)**.
Kết quả các hướng cải tiến YOLO (biến thể loss, đầu mặt nạ A1…) ở một thư mục
riêng: **[cải tiến YOLO](https://drive.google.com/drive/folders/19Gz3wBCKW1spDIwD_WQC2H04JbMcaQan?usp=drive_link)**.
Bố cục thư mục run và cách gộp bảng xem mục *Kết quả (runs)* ở
`benchmark/README.md`; file nhỏ cần commit chép vào `results/`.

## Ghi chú

- Chạy được trên Windows mà không cần thêm bước nào; trainer tự đặt `workers 0`.
- `mask_ratio 4` (mặc định) cho lưới mặt nạ `imgsz/4` = 256 ô ở imgsz 1024, tức
  ~10 px ảnh gốc mỗi ô.
- Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`)
  phải giống hệt ba thư mục model kia.
