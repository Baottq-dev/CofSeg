# Chạy benchmark trên máy lab / máy thuê

Máy Linux (Ubuntu, GPU 24 GB, CUDA toolkit có `nvcc`, có internet). Mọi lệnh
chạy từ gốc repo. Máy đó **chỉ huấn luyện và xuất dự đoán**; điểm số chấm lại
ở máy nhà bằng `scripts/score_remote.py` để mọi model đi qua cùng một vòng chấm.

Mã đi theo repo: `git clone --recurse-submodules <repo>` (Mask2Former là
submodule trong `third_party/`); thư viện Linux-only (detectron2, mmcv) nằm
trong `requirements.txt` với marker `sys_platform`; trọng số theo
`configs/weights.yaml`. Không phải clone hay tải gì bằng tay.

Mỗi model nằm trọn trong một thư mục `benchmark/<model>_<người>/`: code riêng
(`cofseg/`), config riêng, `run.sh` riêng. `run_fold.sh` chỉ gọi lần lượt bốn
`run.sh` đó. Ai sửa model của mình thì sửa trong thư mục mình.

## 1. Ở máy nhà: chuẩn bị dữ liệu

1. Gán nốt nhãn (fold f4 test = field_4), khởi động lại app để có code mới.
2. Xuất từ app: scope rỗng, `split_by: none`, format `coco` + `yolo`, tên `all_v2`.
3. Đóng gói (PowerShell có sẵn `tar`):

       tar -cf all_v2.tar -C data/export all_v2

4. Đưa lên máy thuê: repo (`git clone` hoặc `git bundle`) + `all_v2.tar` (~1 GB).

## 2. Ở máy lab / máy thuê

    bash scripts/remote/setup.sh                 # env cofseg, detectron2, mmcv/mmdet, op Mask2Former, trọng số (~20 phút)
    bash scripts/remote/prepare_data.sh all_v2.tar   # giải nén + cắt 6 fold (hardlink, không tốn đĩa)
    bash scripts/remote/run_fold.sh f4 --smoke   # vài iteration mỗi model: kiểm đường chạy TRƯỚC
    nohup bash scripts/remote/run_all.sh > runs/remote_all.log 2>&1 &   # 6 fold, f4 f2 trước
    bash scripts/remote/pack_results.sh          # -> results_<ngày>.tar mang về

`run_fold.sh` gọi `benchmark/<...>/run.sh` của bốn thư mục theo thứ tự
Mask R-CNN → SOLOv2 → YOLO11s → Mask2Former, dừng ngay khi một cái lỗi. Mỗi
model để lại `preds/<model>_<fold>.json` (COCO results của split test), kết
quả chấm trong `benchmark/<...>/results/`, log `runs/remote_<fold>.log`.
Trước khi chạy nó gọi `benchmark/check_copies.py` để biết bốn bản chấm điểm
còn giống nhau không.

Chạy một model thôi thì gọi thẳng thư mục đó:

    bash benchmark/solov2_phuongquynh/run.sh f4 --smoke

Chạy lại một phần / đổi tham số:

    bash scripts/remote/run_fold.sh f2 --only solov2,mask2former
    bash scripts/remote/run_fold.sh f4 --epochs 30 --batch 2
    FOLDS="f3 f5" bash scripts/remote/run_all.sh --only maskrcnn

Thời gian ước trên 4090 @1024, batch 4, ~600 ảnh train: YOLO11s ~1 h,
Mask R-CNN ~40 phút, SOLOv2 ~45 phút, Mask2Former (100 epoch) ~3 h → một fold
~5,5 h, sáu fold ~33 h (Cascade thêm ~55 phút/fold nếu chạy).

## 3. Về máy nhà: chấm

    tar -xf results_<ngày>.tar
    python benchmark/check_copies.py
    python scripts/summarize_folds.py --eval benchmark/*/runs/eval

Nếu muốn chấm lại mọi file dự đoán bằng MỘT vòng chấm duy nhất (bản gốc
canopyseg/) thay vì bản của từng người:

    python scripts/score_remote.py --preds preds --export data/export
    python scripts/summarize_folds.py --eval runs/eval

## Chưa kiểm ở nhà

`setup.sh`, Mask R-CNN, Mask2Former (detectron2) và SOLOv2 (mmdet) chưa chạy
thật trên Linux (máy phát triển là Windows, không dựng detectron2/mmcv). Lần
đầu: chạy `setup.sh` từng khối, rồi `run_fold.sh f4 --smoke`, sửa tại chỗ nếu
API lệch — sửa trong thư mục của model đó. Nhánh YOLO11 đã chạy thử ở nhà
(1 epoch, 5 % ảnh, chấm 6 ảnh test). Hai chỗ dễ vấp đã biết:

- mmdet 3.3.0 khai `mmcv < 2.2` nhưng wheel dựng sẵn cho torch 2.4 là 2.2.0;
  `setup.sh` nới dòng kiểm đó. Nếu `from mmcv.ops import nms` vẫn lỗi thì
  torch/mmcv lệch ABI: kiểm `pip show torch mmcv`.
- `--no-build-isolation` khi `pip install -r requirements.txt`: detectron2 và
  SAM 2 import torch lúc build, torch phải cài trước (setup.sh đã làm).
