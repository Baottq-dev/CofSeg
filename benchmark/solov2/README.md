# SOLOv2 R50-FPN — họ box-free

**Người phụ trách:** PhuongQuynh (@Phquynh2312)
**Framework:** mmdetection (`trainer: mmdet`, `model.arch: solov2`)

## Vai trò

Đại diện cho hướng **bỏ bounding box**: chia ảnh thành lưới, ô nào chứa tâm
vật thể thì ô đó sinh kernel động, tích chập lên đặc trưng mask stride 4 của
toàn ảnh. Không RPN, không ROI, không cắt mask theo box.

## Vì sao chọn

- Cách tìm vật thể khác hẳn Mask R-CNN (ROI) và YOLO (một giai đoạn cắt theo
  box) — bảng cần có mặt hướng này để đủ các nhóm.
- Mask không bị box cắt, nên chỗ hai tán chạm nhau không dính mép cây bên như
  YOLO. Đây là điểm cần soi trên field_5.
- Điểm yếu cần soi: mỗi ô lưới chỉ đại diện được một vật thể — hai tán cùng cỡ
  chạm nhau mà tâm rơi vào cùng ô thì mất một cây.
- Cùng backbone R50 với Mask R-CNN và Mask2Former.

## Dựng môi trường riêng

Phần này dành cho trường hợp chạy **một mình model này**, env riêng, không dùng
env chung của `benchmark/requirements.txt`. Lệnh ở đây đủ để chép chạy từ đầu
đến cuối; phần *vì sao* (vì sao CUDA 12.8, vì sao mmcv không có wheel, vì sao
mmcv dựng bản rỗng mà không báo lỗi) nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Model **lâu cài nhất**: `mmcv` không có wheel cho CUDA 12.8 nên phải build từ
nguồn, mất 20–120 phút.

```bash
conda create -y -n cofseg-solov2 python=3.12 && conda activate cofseg-solov2
nvcc --version | tail -2
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/solov2/requirements.txt

# mmcv từ nguồn. torch PHẢI có mặt trước bước này.
python -c "import torch; print(torch.__version__, torch.version.cuda)"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv && FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 pip install --no-build-isolation -e . && cd ..
python -c "from mmcv.ops import nms; print('op CUDA co that')"

# mmengine bản PyPI hỏng trên torch >= 2.6; đè bằng bản git, SAU khi cài mmcv
pip install "git+https://github.com/open-mmlab/mmengine"

# mmdet 3.3.0 khai mmcv < 2.2.0, mà 2.2.0 là bản mới nhất tồn tại
MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"
```

`nvcc --version` phải ra **12.x**. Lệch major với `torch.version.cuda` là gói
biên dịch từ nguồn gãy ở `build_ext`, bằng một câu không hề nhắc tới torch.
Máy có nvcc 13.x thì cài `cuda-toolkit=12.8.1` vào chính env — xem
`benchmark/README.md` mục *Cài mmcv*.

`TORCH_CUDA_ARCH_LIST` đặt theo card: `12.0` cho RTX 5090, `8.9` cho RTX 4090 /
L40S, `8.0` cho A100, `9.0` cho H100. Nhiều card thì ngăn bằng dấu chấm phẩy.

**Ba cái bẫy của model này, cả ba đều báo xanh:**

| | dấu hiệu | hậu quả |
|---|---|---|
| mmcv build lúc chưa có torch | xong trong vài giây, không dòng `nvcc` nào | gói mmcv không có `_ext`, SOLOv2 chết ở forward đầu tiên |
| mmengine lấy từ PyPI | dòng `Successfully installed ... mmengine-0.10.7` lúc cài mmcv | `UnpicklingError` lúc nạp trọng số COCO |
| quên nới chốt mmdet | — | `AssertionError: MMCV==2.2.0 is used but incompatible` |

`setup.py` của mmcv bắt luôn `ModuleNotFoundError` của torch rồi đi tiếp, nên
`MMCV_WITH_OPS=1` và `FORCE_CUDA=1` thành vô hiệu mà pip vẫn in `Successfully
installed mmcv-2.2.0`. Build thật phải chạy hàng chục phút.

Số phiên bản **không** phân biệt được hai bản mmengine — nhánh `main` chưa tăng
số sau lần phát hành cuối. Kiểm bằng bản vá:

```bash
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"
python -c "import mmdet, mmcv, mmengine; print(mmdet.__version__, mmcv.__version__)"
python -c "
import torch, torchvision
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())"
```

`True` là đúng bản git; `False` là bản PyPI hỏng vẫn còn đó.

Torchvision có phần mở rộng C++ link vào `libtorch`: bản dựng cho CUDA khác sẽ
`import` trót lọt rồi gãy lúc gọi op. Vì vậy cài torch và torchvision **trong
cùng một lệnh**, đừng cài rời.

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
# 1. huấn luyện (thêm --set data.limit=16 --epochs 1 để khói, kiểm đường chạy trước)
python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --data data/export/block/f1 --runs benchmark/solov2/runs --name solov2 --workers 8

# 2. dự đoán test do trainer ghi ra -> preds/ để cả nhóm dùng chung
RUN=$(ls -td benchmark/solov2/runs/train/*_solov2_block-f1_* | head -1)
mkdir -p preds && cp "$RUN/predictions.json" preds/solov2_block_f1.json

# 3. chấm: Boundary AP, Boundary IoU, sai số diện tích, bảng từng vùng
python benchmark/solov2/evaluate.py --config benchmark/solov2/configs/eval/_coco.yaml --file preds/solov2_block_f1.json --data data/export/block/f1 --split test --runs benchmark/solov2/runs --name solov2

# 4. chép file kết quả nhỏ vào results/ (runs/ không vào git)
EV=$(ls -td benchmark/solov2/runs/eval/*_solov2_block-f1_* | head -1)
cp "$EV/metrics.json"   benchmark/solov2/results/solov2_block_f1_metrics.json
cp "$EV/per_region.csv" benchmark/solov2/results/solov2_block_f1_per_region.csv

# test của thư mục này, chạy trước khi commit
cd benchmark/solov2 && python -m pytest tests
```

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

- **Chưa chạy thật lần nào.** mmcv không cài được trên Windows nên ở nhà chỉ
  kiểm được phần dựng config. Việc đầu tiên trên máy Linux, một lượt khói:

  ```bash
  python benchmark/solov2/train.py --config benchmark/solov2/configs/train/solov2_r50_mm.yaml --data data/export/block/f1 --set data.limit=16 --epochs 1
  ```
- Khác biệt có chủ đích so với hai model detectron2: **không có xoay 90°**
  (mmdet không có transform sẵn cho mask + box), chỉ lật ngang/dọc/chéo. Nhớ
  ghi chú khi đọc bảng.
- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  Nới dòng kiểm đó bằng `sed`; xem `benchmark/README.md` mục *Cài mmcv*.
