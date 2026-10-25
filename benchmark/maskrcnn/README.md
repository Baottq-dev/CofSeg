# Mask R-CNN R50-FPN — mốc số 0

**Framework:** detectron2 (`trainer: detectron2`, `model.arch: maskrcnn`) · chỉ
chạy trên Linux

Cột mốc số 0 của bảng benchmark: mọi model khác báo cáo **Δ% mAP so với model
này** trên cùng một ruộng, nên đổi cấu hình của model này (imgsz, số epoch,
augmentation) là đổi mốc chung của cả bảng. Đây là kiến trúc gốc mà Cascade,
PointRend, Mask Transfiner đều sửa từ đó, nên chênh lệch trong bảng đọc được là
"đổi khối nào được bao nhiêu". Mask 28×28 cho mỗi ROI rồi phóng lên box: tán
trung vị 324 px ảnh gốc thì mỗi ô ≈ 12 px, biên thô, là cận dưới của trục chất
lượng biên. Xử lý từng ROI riêng nên ở chỗ tán chạm nhau mask dễ tràn sang cây
bên.

## Cài đặt

Làm bước 0–4 của `benchmark/README.md` (bước 4 cài `benchmark/maskrcnn/requirements.txt`
và detectron2). Model này không cần mmcv, và không có op CUDA riêng: ROIAlign và
NMS lấy của torchvision. `requirements.txt` ghim `setuptools<82` vì
`model_zoo` của detectron2 còn `import pkg_resources`; thiếu thì train dừng
sau khi đã nạp xong dữ liệu.

## Chạy

Từ **gốc repo**. `--data` nhận thư mục fold, `--name` chỉ là tên model (quy ước
ở `benchmark/README.md`). Lệnh dưới chạy bộ **block**, fold **f1**; đổi `block`
hoặc `f1` ở cả ba chỗ là ra lượt khác.

```bash
# 1. huấn luyện — chỉ train + val, không đụng tới split test
#    (khói: --limit 64 --epochs 3 --warmup_iters 30. Lượt ngắn phải truyền số vòng
#     tuyệt đối: warmup_iters dưới 1 là PHẦN của lịch, co về 1 vòng thì lr 0.02 làm
#     RPN nổ ngay — FloatingPointError ở vòng 13.)
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

- Mọi tham số đều là cờ, mặc định nằm trong config (`imgsz 1024`, `batch 16`,
  `epochs 100`, `lr 0.02`, `rot90 true`, `mask_resolution 14`, `seed 0`).
  `--list-params`, `--print-config`, `--probe` như các model khác.
- `lr` và `val_batch` để trống, trainer tự tính (`lr = 0.02 × batch/16`); truyền
  `--lr` tay là tắt phép đó. `--lr_steps` phải có nháy vì đọc bằng YAML.
- `--keep_ckpts 1` giữ đúng một checkpoint định kỳ trong `d2/` (mặc định của
  detectron2 giữ hết, ~350 MB mỗi epoch). `model_final.pth` và `weights/last.pth`
  chỉ có khi train chạy hết.
- Lúc chấm thêm `--arch maskrcnn`; bốn khoá `eval:` (`iou_thr`, `band_ratio`,
  `dilation_ratio`, `nsd_tau`) là định nghĩa của phép đo, đổi chúng thì số không
  còn so được với các model kia.

## Đổi backbone (`--backbone`, ngoài bảng chính)

`--list-backbones` in bảng lựa chọn. Đổi backbone ở **một** model làm mất tính
chất "ba model cùng R50 nên chênh lệch quy về cơ chế", và trên bộ này phần mất
điểm nằm ở IoU cao chứ không ở AP50 nên backbone sâu hơn ít giúp; vì vậy phần
này báo cáo riêng như thí nghiệm phụ. Không cần `--name`: tên run lấy theo
backbone (`maskrcnn-r101`).

| `--backbone` | mask AP COCO | ước lượng ở đây (VRAM · phút/fold) |
|---|---|---|
| `r50` | 37,2 | 12,7 GB · 26 phút (đo thật) |
| `r50-dconv` | 38,5 | ~13 GB · ~35 phút |
| `r50-gn` | 38,6 | ~21 GB · ~31 phút |
| `r101` | 38,6 | ~17 GB · ~34 phút |
| `x101` | 39,5 | ~27 GB · ~70 phút |

Cột ước lượng suy từ tỉ lệ trong `MODEL_ZOO.md` của detectron2 nhân số đo thật
của R50; chạy `--probe` trước khi đặt lịch.

**ViTDet** (`vit-b` 45,9 · `vit-l` 49,2 · `vit-h` 50,2 mask AP COCO) là Mask
R-CNN với backbone ViT, nên cùng bảng với `r50`. Lượt ViT tự đổi bốn thứ, phải
ghi rõ khi báo cáo:

| | `r50` | `vit-b` |
|---|---|---|
| optimizer | SGD 0,02 | **AdamW 1e-4** (ViT tiền huấn luyện MAE) |
| `weight_decay` | 1e-4 | **0,1** |
| lr theo độ sâu | không | **layer-wise decay 0,7** (`vit-l` 0,8, `vit-h` 0,9) |
| kênh ảnh | BGR | **RGB** |

Lịch học (`max_iter`, `lr_steps`, `warmup_iters`, số epoch) không đổi. Ví dụ
`--backbone vit-b --batch 4 --keep_ckpts 1`: `--batch 4` **chưa được đo** (recipe
ViTDet gốc là 1 ảnh mỗi GPU, `--batch 16` gần như chắc chắn hết VRAM), nên chạy
`--probe` trước; vì batch khác 16, lượt này không vào bảng chính. Sau khi nạp,
`summary.json` có mục `weights_loaded` cho biết khoá nào không vào được model.

Cascade Mask R-CNN là *kiến trúc* khác, có config riêng (backbone `r50`, `x152`):

```bash
python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/cascade_r50_d2.yaml --data data/export/field/f1 --runs benchmark/maskrcnn/runs --keep_ckpts 1 --workers 8
```

## Kết quả (runs)

Kết quả train và chấm của Mask R-CNN (`runs/`) lưu trên Drive:
**[Mask R-CNN](https://drive.google.com/drive/folders/1MTQ66hoZ8A5mdMd8UAbmqe8hrzwuOfId?usp=drive_link)**.
Bố cục thư mục run và cách gộp bảng xem mục *Kết quả (runs)* ở
`benchmark/README.md`; file nhỏ cần commit chép vào `results/`.

## Ghi chú

- Chỉ chạy trên Linux: repo chỉ còn đường detectron2.
- Cấu hình có trong `configs/train/` và `configs/eval/` (`maskrcnn_r50_d2`,
  `cascade_r50_d2`). Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`,
  `cofseg/datasets/`) phải giống hệt ba thư mục model kia.
