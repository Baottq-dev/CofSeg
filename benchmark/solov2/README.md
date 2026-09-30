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

Env riêng cho một mình model này, không dùng `benchmark/requirements.txt` của
cả nhóm. Chép chạy từ trên xuống; phần chung nằm ở `benchmark/README.md` mục
*Dựng môi trường*.

Lâu cài nhất: `mmcv` không có wheel cho CUDA 12.8 nên phải build từ nguồn,
mất 20–120 phút.

```bash
conda create -y -n cofseg-solov2 python=3.12 && conda activate cofseg-solov2
conda install -y -c nvidia cuda-toolkit=12.8.1
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r benchmark/solov2/requirements.txt

git clone --branch v2.2.0 --depth 1 https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 pip install --no-build-isolation -e .
cd -
python -c "from mmcv.ops import nms; print('op CUDA co that')"

pip install "git+https://github.com/open-mmlab/mmengine"
python -c "import inspect, mmengine.runner.checkpoint as c; print('weights_only' in inspect.getsource(c))"

MMDET=$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$MMDET"

python -c "
import torch, torchvision, mmdet, mmcv
from torchvision.ops import nms
b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]); s = torch.tensor([0.9, 0.8])
print(torch.__version__, torch.version.cuda, '| tv', torchvision.__version__, nms(b, s, 0.5).tolist())
print('mmdet', mmdet.__version__, '| mmcv', mmcv.__version__)"
```

Clone mmcv **ra ngoài repo**: thư mục `mmcv/` nằm trong gốc repo sẽ che gói đã
cài, và mọi lệnh chạy từ gốc repo báo `module 'mmcv' has no attribute
'__version__'`.

Ba dòng kiểm ở trên không bỏ được, vì cả ba kiểu hỏng đều báo xanh lúc cài:
mmcv dựng bản rỗng nếu chưa có torch, mmengine bản PyPI ném `UnpicklingError`
lúc nạp trọng số COCO, và mmdet chốt `mmcv < 2.2.0` trong khi 2.2.0 là bản mới
nhất tồn tại. `TORCH_CUDA_ARCH_LIST` đặt theo card: `12.0` cho RTX 5090, `8.9`
cho RTX 4090 / L40S, `8.0` cho A100, `9.0` cho H100.

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
  Nới dòng kiểm đó bằng `sed`; xem `benchmark/README.md` mục *Dựng môi
  trường*, bước 6.
