# Chạy benchmark trên máy lab / máy thuê

Máy Linux (Ubuntu, GPU ≥ 24 GB, CUDA toolkit có `nvcc`, có internet). Mọi lệnh
chạy từ gốc repo.

Mã đi theo repo: `git clone --recurse-submodules <repo>` (Mask2Former là
submodule trong `third_party/`); thư viện Linux-only (detectron2, mmcv) nằm
trong `requirements.txt` với marker `sys_platform`; trọng số theo
`configs/weights.yaml`. Không phải clone hay tải gì bằng tay.

Mỗi model nằm trọn trong một thư mục `benchmark/<model>/`: code riêng
(`cofseg/`), config riêng, `train.py` và `evaluate.py` riêng.
`benchmark/run.py` chỉ gọi lần lượt bốn thư mục đó. Ai sửa model của mình thì
sửa trong thư mục mình — xem `benchmark/README.md` để biết ai phụ trách gì.

## 1. Ở máy nhà: chuẩn bị dữ liệu

1. Gán nốt nhãn, khởi động lại app để có code mới.
2. Xuất từ app: scope rỗng, **`split_by: none`**, format **`coco` + `yolo`**,
   tên `all_v2`. Bản xuất đã chia sẵn train/val/test **không dùng được**: fold
   phải cắt theo ruộng.
3. Đóng gói (PowerShell có sẵn `tar`):

       tar -cf all_v2.tar -C data/export all_v2

4. Đưa lên máy Linux: repo (`git clone` hoặc `git bundle`) + `all_v2.tar` (~1 GB).

## 2. Ở máy lab / máy thuê

```bash
conda create -y -n cofseg python=3.12 && conda activate cofseg

python scripts/setup_env.py                  # torch, detectron2, mmcv/mmdet, Mask2Former, trọng số (~20 phút)
python scripts/prepare_data.py all_v2.tar    # giải nén + cắt 6 fold (hardlink, không tốn đĩa)
python benchmark/run.py f4 --smoke           # vài iteration mỗi model: kiểm đường chạy TRƯỚC
python benchmark/run.py f4 f2 f1 f6 f3 f5    # sáu fold, f4 f2 trước để có số sớm
python scripts/pack_results.py               # -> results_<ngày>.tar mang về
```

`setup_env.py` chạy sáu bước và dừng ngay khi một bước lỗi; sửa xong chạy tiếp
bằng `--from <tên bước>` (xem `--list`). Không tự tạo conda env: kích hoạt env
từ trong một tiến trình Python không có tác dụng ra ngoài, nên bước đầu chỉ
kiểm và bảo bạn cần gõ gì.

`benchmark/run.py` gọi `train.py` rồi `evaluate.py` của từng thư mục, một model
lỗi thì ghi nhận và chạy tiếp model sau. Mỗi model để lại
`preds/<model>_<fold>.json`, lần chạy đầy đủ trong `benchmark/<model>/runs/`,
file kết quả nhỏ trong `benchmark/<model>/results/`.

Chạy một phần:

```bash
python benchmark/run.py f4 --only solov2
python benchmark/run.py f4 --only maskrcnn,mask2former --epochs 30 --batch 2
python benchmark/run.py --list
bash -c 'nohup python benchmark/run.py f4 f2 f1 f6 f3 f5 > runs/remote_all.log 2>&1 &'
```

Thời gian ước trên 4090 @1024, batch 4, ~600 ảnh train: Mask R-CNN ~40 phút,
SOLOv2 ~45 phút, YOLO11s ~1 h, Mask2Former (100 epoch) ~3 h → một fold ~5,5 h,
sáu fold ~33 h.

## 3. Về máy nhà: dựng bảng

```bash
tar -xf results_<ngày>.tar
python benchmark/check_copies.py                          # bốn bản chấm còn giống nhau không
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
```

Nếu muốn chấm lại mọi file dự đoán bằng **một** vòng chấm duy nhất (bản gốc
`canopyseg/`) thay vì bản của từng người:

```bash
python scripts/score_remote.py --preds preds --export data/export
python scripts/summarize_folds.py --eval runs/eval
```

## Chưa kiểm ở nhà

`setup_env.py`, Mask R-CNN, Mask2Former (detectron2) và SOLOv2 (mmdet) chưa
chạy thật trên Linux (máy phát triển là Windows, không dựng detectron2/mmcv).
Lần đầu: `python scripts/setup_env.py --only check`, rồi từng bước một, rồi
`python benchmark/run.py f4 --smoke`. Sửa chỗ lệch **trong thư mục của model
đó**. Nhánh YOLO11 đã chạy thử ở nhà (1 epoch, 5 % ảnh, chấm 6 ảnh test).

Hai chỗ dễ vấp đã biết:

- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  bước `mmdet` của `setup_env.py` nới dòng kiểm đó. Nếu `from mmcv.ops import
  nms` vẫn lỗi thì torch/mmcv lệch ABI: kiểm `pip show torch mmcv`.
- `--no-build-isolation` khi cài `requirements.txt`: detectron2 và SAM 2 import
  torch lúc build, nên torch phải cài trước (bước `torch` đã làm).
