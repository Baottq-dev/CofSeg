# Kết quả — Mask2Former R50

`runs/` không vào git, nên sau mỗi lần chấm chép các file nhỏ về đây. Tên file
phải có cả bộ fold lẫn fold (`<model>_<bộ>_<fold>`), vì ba bộ fold cùng đánh
số f1..f6 và file của bộ này sẽ đè file của bộ kia. Ví dụ bộ `field`, fold
`f4`, chạy từ gốc repo:

```bash
EV=$(ls -td benchmark/mask2former/runs/eval/*_mask2former_field-f4_* | head -1)
RUN=$(ls -td benchmark/mask2former/runs/train/*_mask2former_field-f4_* | head -1)
cp "$EV/metrics.json"    benchmark/mask2former/results/mask2former_field_f4_metrics.json
cp "$EV/per_region.csv"  benchmark/mask2former/results/mask2former_field_f4_per_region.csv
cp "$RUN/results.csv"    benchmark/mask2former/results/mask2former_field_f4_train.csv
```

**Không** chép `predictions.json` (vài chục MB) hay file trọng số vào đây:
chúng ở lại `runs/`, `weights/` và được đóng gói bằng
`scripts/pack_results.py`.

Bảng tổng hợp cả bốn model:

```bash
python scripts/summarize_folds.py --eval benchmark/*/runs/eval
```
