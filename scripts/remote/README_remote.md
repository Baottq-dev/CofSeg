# Chạy benchmark trên máy thuê

Máy thuê Linux (Ubuntu, GPU 24 GB, CUDA toolkit có `nvcc`, có internet).
Mọi lệnh chạy từ gốc repo. Máy thuê **chỉ huấn luyện và xuất dự đoán**; điểm
số chấm lại ở máy nhà bằng `scripts/score_remote.py` để mọi model đi qua cùng
một vòng chấm.

## 1. Ở máy nhà: chuẩn bị dữ liệu

1. Gán nốt nhãn (fold f4 test = field_4), khởi động lại app để có code mới.
2. Xuất từ app: scope rỗng, `split_by: none`, format `coco` + `yolo`, tên `all_v2`.
3. Đóng gói (PowerShell có sẵn `tar`):

       tar -cf all_v2.tar -C data/export all_v2

4. Đưa lên máy thuê: repo (`git clone` hoặc `git bundle`) + `all_v2.tar` (~1 GB).

## 2. Ở máy thuê

    bash scripts/remote/setup.sh                 # env cofseg, detectron2, Mask2Former, trọng số  (~15 phút)
    bash scripts/remote/prepare_data.sh all_v2.tar   # giải nén + cắt 6 fold (hardlink, không tốn đĩa)
    bash scripts/remote/run_fold.sh f4 --smoke   # vài iteration mỗi model: kiểm đường chạy TRƯỚC
    nohup bash scripts/remote/run_all.sh > runs/remote_all.log 2>&1 &   # 6 fold, f4 f2 trước
    bash scripts/remote/pack_results.sh          # -> results_<ngày>.tar mang về

`run_fold.sh` chạy tuần tự 4 lệnh `scripts/train.py` (YOLO11s → Mask R-CNN →
Cascade → Mask2Former, mỗi cái `--workers 8`), dừng ngay khi một lệnh lỗi,
và để lại `preds/<model>_<fold>.json` (COCO results của split test) + log
`runs/remote_<fold>.log`.

Chạy lại một phần / đổi tham số:

    bash scripts/remote/run_fold.sh f2 --only cascade,mask2former
    bash scripts/remote/run_fold.sh f4 --imgsz 2048 --batch 2 --epochs 30
    FOLDS="f3 f5" bash scripts/remote/run_all.sh --only maskrcnn

Thời gian ước trên 4090 @1024, batch 4, ~600 ảnh train: YOLO11s ~1 h,
Mask R-CNN ~40 phút, Cascade ~55 phút, Mask2Former (100 epoch) ~3 h → một fold
~6 h, sáu fold ~35 h.

## 3. Về máy nhà: chấm

    tar -xf results_<ngày>.tar
    python scripts/score_remote.py --preds preds --export data/export
    python scripts/summarize_folds.py --eval runs/eval

## Chưa kiểm ở nhà

`setup.sh` và ba model detectron2 chưa chạy thật trên Linux (máy phát triển
là Windows, không dựng detectron2). Lần đầu: chạy `setup.sh` từng khối, rồi
`run_fold.sh f4 --smoke`, sửa tại chỗ nếu API detectron2 lệch, rồi ghim
`D2_REF` / `M2F_REF` trong `setup.sh` vào commit đã chạy được. Nhánh YOLO11 của
`run_fold.sh` đã chạy thử ở nhà (1 epoch, 5 % ảnh, chấm 6 ảnh test).
