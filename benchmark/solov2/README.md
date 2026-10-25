# SOLOv2 R50-FPN — họ box-free

**Framework:** mmdetection (`trainer: mmdet`, `model.arch: solov2`) · chỉ chạy
trên Linux

Đại diện cho hướng **bỏ bounding box**: chia ảnh thành lưới, ô nào chứa tâm vật
thể thì sinh kernel động, tích chập lên đặc trưng mask stride 4 của toàn ảnh.
Không RPN, không ROI, không cắt mask theo box, nên ở chỗ hai tán chạm nhau mask
không bị cắt thẳng theo cạnh box như YOLO. Điểm yếu đã biết: mỗi ô lưới chỉ đại
diện một vật thể, hai tán cùng cỡ chạm nhau mà tâm rơi vào cùng ô thì mất một
cây. Cùng backbone R50 với Mask R-CNN và Mask2Former.

## Cài đặt

Làm bước 0–3 của `benchmark/README.md` (không cần detectron2 và bước vá
`libnvrtc-builtins`). Model này cần **mmcv build từ nguồn** (không có wheel cho
CUDA 13, mất 20–120 phút), mmengine bản git và mmdet đã nới chốt phiên bản.
Thứ tự quan trọng: mmengine phải cài **sau** mmcv, vì cài mmcv kéo bản mmengine
trên PyPI về đè.

```bash
pip install -r benchmark/solov2/requirements.txt

# mmcv 2.2.0 — clone ra ngoài repo, vì thư mục mmcv/ trong gốc repo che gói đã cài
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch;print('%d.%d' % torch.cuda.get_device_capability(0))")
echo "arch = $TORCH_CUDA_ARCH_LIST"          # đọc từ card, vd 12.0 với card Blackwell
pip install ninja                          # biên dịch song song, build nhanh hơn
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
rm -rf build mmcv/_ext*.so
FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS=4 pip install --no-build-isolation -e .
cd -
ls -l ~/mmcv/mmcv/_ext*.so

# mmengine bản git (bản PyPI ném UnpicklingError khi nạp trọng số COCO trên torch >= 2.6) và nới chốt của mmdet
pip install "git+https://github.com/open-mmlab/mmengine"
MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
grep -n "mmcv_maximum_version" "$MMDET"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
```

- `TORCH_CUDA_ARCH_LIST` lấy từ card; build sai kiến trúc thì lỗi `no kernel
  image is available` chỉ hiện ở lời gọi kernel đầu tiên, sau khi build xong.
  `rm -rf build mmcv/_ext*.so` vì setuptools dùng lại file `.o` cũ.
- pip ẩn output build khi thành công. `_ext.so` cỡ **~26 MB** là bản có nhân
  CUDA; vài MB là bản chỉ có CPU, phải build lại.
- `sed` nới chốt vì mmdet khai `mmcv < 2.2.0`, trong khi 2.2.0 là bản mới nhất;
  `grep` trước để không vá hai lần.

Kiểm, mọi phép đều chạy trên GPU:

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

## Chạy

Từ **gốc repo**. `--data` nhận thư mục fold, `--name` chỉ là tên model (quy ước
ở `benchmark/README.md`). Lệnh dưới chạy bộ **block**, fold **f1**; đổi `block`
hoặc `f1` ở cả ba chỗ là ra lượt khác.

```bash
# 1. huấn luyện — chỉ train + val, không đụng tới split test
#    (khói: --limit 64 --epochs 3 --warmup_iters 30; lượt ngắn phải truyền số vòng
#     tuyệt đối vì warmup_iters dưới 1 là PHẦN của lịch)
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

- Mọi tham số đều là cờ, mặc định nằm trong config (`imgsz 1024`, `batch 16`,
  `epochs 100`, `lr 0.01`, `seed 0`). `--list-params`, `--print-config`,
  `--probe` như các model khác.
- `lr` và `val_batch` để trống, trainer tự tính (`lr = 0.01 × batch/16`); truyền
  `--lr` tay là tắt phép đó. `--lr_steps` phải có nháy vì đọc bằng YAML.
- **Không có `--rot90`**: mmdet không có transform xoay sẵn cho cả mask lẫn box,
  nên model này chỉ lật; khác hai model detectron2 có chủ đích, cần ghi chú khi
  đọc bảng. Lật ngang và dọc vẫn độc lập: `RandomFlip` nhận ba hướng loại trừ
  nhau, nên trainer ghép thành ngang `p(1-q)`, dọc `(1-p)q`, chéo `pq`.
- Tăng cường màu đi qua `PhotoMetricDistortion` của mmdet, cộng delta độ sáng
  thay vì nhân (xem *Tăng cường* ở `benchmark/README.md`).
- Bốn khoá `eval:` (`iou_thr`, `band_ratio`, `dilation_ratio`, `nsd_tau`) là
  định nghĩa của phép đo, đổi chúng thì số không còn so được với các model kia.

## Đổi backbone (`--backbone`, ngoài bảng chính)

`--list-backbones` in bảng lựa chọn. Đổi backbone ở **một** model làm mất tính
chất "ba model cùng R50 nên chênh lệch quy về cơ chế", và trên bộ này phần mất
điểm nằm ở IoU cao chứ không ở AP50 nên backbone sâu hơn ít giúp; vì vậy phần
này báo cáo riêng như thí nghiệm phụ. Không cần `--name`: tên run lấy theo
backbone (`solov2-r101-dcn`).

| `--backbone` | mask AP COCO | ghi chú |
|---|---|---|
| `r50` | 37,5 | mặc định, cùng backbone với Mask R-CNN và Mask2Former |
| `r101-dcn` | 41,2 | R101 **cộng** deformable conv: hai thay đổi cùng lúc |
| `x101-dcn` | 42,4 | nặng nhất, chạy `--probe` trước |
| `light-r18` | 29,7 | bản nhẹ, mask stride thô hơn |
| `light-r50` | 33,7 | bản nhẹ, mask stride thô hơn |

SOLOv2 **không có R101 thường**: có config `solov2_r101_fpn_ms-3x_coco.py` nhưng
metafile không kèm trọng số COCO, nên chọn bản đó là khởi đầu từ ImageNet và
không so được với các model kia vốn bắt đầu từ COCO. Số mask AP lấy từ
`configs/solov2/metafile.yml` của mmdet 3.3.0.

## Kết quả (runs)

Kết quả train và chấm của SOLOv2 (`runs/`) lưu trên Drive:
**[SOLOv2](https://drive.google.com/drive/folders/18K4VMu5QfDbYLlfFKQJdnYf7nSUG_Muy?usp=drive_link)**.
Bố cục thư mục run và cách gộp bảng xem mục *Kết quả (runs)* ở
`benchmark/README.md`; file nhỏ cần commit chép vào `results/`.

## Ghi chú

- Chỉ chạy trên Linux: mmcv không cài được trên Windows.
- `configs/mmdet/solov2_r50_coffee.py` là file ghi đè config mmdet (`num_classes`
  và ngưỡng test), dùng chung cho mọi backbone.
- Phần chấm điểm (`cofseg/metrics/`, `cofseg/evaluation/`, `cofseg/datasets/`)
  phải giống hệt ba thư mục model kia.
