Hướng dẫn chạy 

## 0. Yêu cầu

- GPU NVIDIA có CUDA (khuyến nghị ≥ 12GB VRAM cho SAM2 hiera-large). Không có GPU vẫn chạy nhưng rất chậm (đặt `sam.device: cpu`).
- Python 3.10, ~20GB đĩa trống cho weights + tiles.
- Khuyến nghị Linux/WSL2 (rasterio & pycocotools dễ cài hơn Windows thuần).

## 1. Cài môi trường

```
conda create -n coffee python=3.12 -y
conda activate coffee
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install -e .
```

## 2. Tải weights SAM2

```
mkdir -p weights
# tải 4 model sam 2.1 từ https://github.com/facebookresearch/sam2 -> đặt vào weights/
```
## 3. tải data

Giải nén thư mục iachim_dataset_export/ vào data/ <br>
Cấu trúc thư mục data sẽ là
```
data/
├─iachim_dataset_export/
    ├─data_compressed/
    ├─predictions/
└─masks/
```

## 3. chạy

```
python -m app.server -r      
```

