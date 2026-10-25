# Mask2Former R50 — họ query / transformer

**Framework:** detectron2 + repo Mask2Former (`trainer: detectron2`,
`model.arch: mask2former`, `model.repo: benchmark/mask2former/upstream`) · chỉ
chạy trên Linux

Đại diện hướng **query-based**: N query học được, mỗi query nhận một vật thể
qua masked attention; mask sinh ở stride 4 trên toàn ảnh, không box, không NMS.
Mask không bị box hay ROI cắt nên có thể bám đúng đường chia giữa hai tán chạm
nhau; bảng cho biết bỏ box thì biên thay đổi bao nhiêu. Điểm yếu đã biết với
vật thể dày, cùng lớp, chạm nhau: hai query cùng nhận một tán, hoặc sót tán. Số
query phải lớn hơn số tán tối đa trong một ảnh (48); đặt 100. Cùng backbone R50
với Mask R-CNN và SOLOv2.

## Cài đặt

Làm bước 0–4 của `benchmark/README.md` (bước 4 cài `benchmark/mask2former/requirements.txt`
và detectron2), rồi build op CUDA `MSDeformAttn` từ repo con `upstream/`:

```bash
git submodule update --init benchmark/mask2former/upstream
git -C benchmark/mask2former/upstream checkout .

sed -i "s/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g" \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu

pip install --no-build-isolation --no-deps \
  benchmark/mask2former/upstream/mask2former/modeling/pixel_decoder/ops

PYTHONPATH=benchmark/mask2former/upstream python -c \
  "from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn; print('MSDeformAttn OK')"
```

- `git checkout .` xoá mọi bản vá cũ trong repo con trước khi vá lại.
- `sed` vá hai dòng của repo gốc: torch 2.11 đã gỡ phép chuyển ngầm từ
  `value.type()` sang `c10::ScalarType`, nên nvcc báo `no suitable conversion
  function`; `value.scalar_type()` cùng giá trị, không đổi hành vi.
- Build thẳng thư mục `ops`, không chạy `make.sh` của repo gốc vì nó gọi
  `setup.py install`, mà setuptools mới đã bỏ lệnh này.
- **Bước này bắt buộc.** Không có kernel thì import dừng với
  `ModuleNotFoundError`; có đường lùi PyTorch cũng tốn 3,94 GiB VRAM mỗi lớp
  encoder ở batch 16 (6 lớp → ~23,6 GiB), vì nó dựng tensor
  `(N*M, D, Lq, L, P)` mà kernel CUDA không bao giờ dựng.

Dòng kiểm cuối đi qua `mask2former/__init__` → `fast_eval_api` →
`from detectron2 import _C`, nên nó đồng thời xác nhận detectron2 còn khớp torch.

## Chạy

Từ **gốc repo**. `--data` nhận thư mục fold, `--name` chỉ là tên model (quy ước
ở `benchmark/README.md`). Lệnh dưới chạy bộ **block**, fold **f1**; đổi `block`
hoặc `f1` ở cả ba chỗ là ra lượt khác.

```bash
# 1. huấn luyện — chỉ train + val, không đụng tới split test
#    (khói: --limit 64 --epochs 3 --warmup_iters 30; lượt ngắn phải truyền số vòng
#     tuyệt đối vì warmup_iters dưới 1 là PHẦN của lịch)
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

- Mọi tham số đều là cờ, mặc định nằm trong config (`imgsz 1024`, `batch 8`,
  `epochs 50`, `lr 1e-4`, `num_queries 100`, `val_every 2`, `seed 0`).
  `--list-params`, `--print-config`, `--probe` như các model khác.
- `lr` và `val_batch` để trống, trainer tự tính (`lr = 1e-4 × batch/16`); truyền
  `--lr` tay là tắt phép đó. `--lr_steps` phải có nháy vì đọc bằng YAML.
- Bốn cờ **không có tác dụng**: `--fliplr`, `--flipud`, `--rot90` (model giữ
  mapper LSJ của recipe gốc: co giãn 0.1–2.0 rồi cắt ô vuông, tự lật ngang) và
  `--mask_resolution` (không có ROI head).
- `--keep_ckpts 1` giữ đúng một checkpoint định kỳ trong `d2/` (mặc định giữ hết,
  ~530 MB mỗi epoch vì AdamW lưu hai moment).
- `--workers 2`, không phải 8: trong lúc val có 16 tiến trình python (train
  loader và test loader mỗi bên 8), làm bộ nhớ của máy 30 GB bị OOM killer dừng
  ba lượt đầu. Giá trị này đổi chuỗi ngẫu nhiên của augmentation, nên **mọi
  fold trong bảng phải dùng cùng một giá trị**.
- Lúc chấm thêm `--arch mask2former --repo benchmark/mask2former/upstream`; bốn
  khoá `eval:` (`iou_thr`, `band_ratio`, `dilation_ratio`, `nsd_tau`) là định
  nghĩa của phép đo, đổi chúng thì số không còn so được với các model kia.

## Đổi backbone (`--backbone`, ngoài bảng chính)

`--list-backbones` in bảng lựa chọn. Đổi backbone ở **một** model làm mất tính
chất "ba model cùng R50 nên chênh lệch quy về cơ chế"; vì vậy phần này báo cáo
riêng như thí nghiệm phụ. Không cần `--name`: tên run lấy theo backbone
(`mask2former-r101`).

| `--backbone` | mask AP COCO | ghi chú |
|---|---|---|
| `r50` | 43,7 | mặc định, cùng backbone với Mask R-CNN và SOLOv2 |
| `r101` | 44,2 | chỉ hơn R50 **0,5 điểm** |
| `swin-t` · `swin-s` · `swin-b` | 45,0 · 46,3 · 46,7 | đổi hẳn họ backbone sang transformer; `swin-b` nặng, chạy `--probe` trước |
| `swin-b-in21k` · `swin-l-in21k` | 48,1 · 50,1 | tiền huấn luyện IN21k: **khác nguồn dữ liệu**, phải ghi rõ trong phần phương pháp; `swin-l` gần như chắc chắn hết VRAM ở batch 8 |

Config và trọng số lấy từ `MODEL_ZOO.md` của repo Mask2Former.

## Kết quả (runs)

Kết quả train và chấm của Mask2Former (`runs/`) lưu trên Drive:
**[Mask2Former](https://drive.google.com/drive/folders/1WkDfZu0dgWV37_ukNCGogixVUG6u8WI-?usp=drive_link)**.
Bố cục thư mục run và cách gộp bảng xem mục *Kết quả (runs)* ở
`benchmark/README.md`; file nhỏ cần commit chép vào `results/`.

## `upstream/` — repo gốc (submodule)

Nguồn https://github.com/facebookresearch/Mask2Former, commit ghim `9b0651c`
(20/05/2022, bản cuối), license MIT. Repo con đi theo git: `git clone
--recurse-submodules`, hoặc lệnh `git submodule update --init` ở trên nếu đã
clone rồi. Thư mục tên `upstream/` vì `Mask2Former/` trùng tên thư mục cha khác
hoa thường (trên Windows và macOS hai tên là một).

**Không sửa mã trong repo con**; cần vá thì vá bằng code trong `cofseg/` (như
hàm `load_mask2former`). Ngoại lệ duy nhất là `sed` ở phần *Cài đặt*, vì đó là
mã CUDA phải biên dịch xong mới có gì để nạp; hoàn tác bằng
`git -C benchmark/mask2former/upstream checkout .`.

## Ghi chú

- **Batch 8, khác ba model kia (16)**: batch 16 tràn VRAM vì model giữ 100 query
  cộng attention toàn ảnh. Đổi backbone hay `imgsz` thì chạy `--probe` lại.
- **Mapper LSJ của repo gốc lỗi trên ảnh nền** (`annotations_to_instances` chỉ
  gắn `gt_masks` khi có nhãn), đã vá bằng `empty_safe_mapper` trong
  `cofseg/models/detectron2.py`. Vá mapper thay vì bật
  `FILTER_EMPTY_ANNOTATIONS`, vì bốn model phải dùng cùng một tập dữ liệu.
- **Tốn giờ máy nhất trong bốn model** dù chỉ 50 epoch, và phần đắt là val chứ
  không phải train. `val_every` quyết định giờ máy lớn nhất của cả bảng: 6 fold
  tốn 17,1 giờ ở mức 1 và 10,1 giờ ở mức 2; config chọn **2**.
- **50 epoch là kết quả đo**: lượt chạy thử 100 epoch cho thấy epoch tốt nhất
  luôn rơi trước mốc 50.
- Tăng cường giữ LSJ của recipe gốc, không áp lật dọc hay xoay 90° như hai model
  R-CNN. Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`,
  `cofseg/datasets/`) phải giống hệt ba thư mục model kia.
