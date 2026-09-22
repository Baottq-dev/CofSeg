# Kết quả — SOLOv2 R50-FPN

`runs/` không vào git nên chép file nhỏ về đây sau mỗi lần chấm, đặt tên theo
`<model>_<fold>`:

    cp runs/eval/<...>_solov2_f4_test/metrics.json     results/solov2_f4_metrics.json
    cp runs/eval/<...>_solov2_f4_test/per_region.csv   results/solov2_f4_per_region.csv
    cp runs/train/<...>/results.csv                    results/solov2_f4_train.csv

**Không** chép `predictions.json` (vài chục MB) hay file trọng số vào đây —
chúng đi theo `scripts/remote/pack_results.sh` và ở lại `runs/`, `weights/`.

Bảng tổng hợp cả bốn model dựng bằng:

    python scripts/summarize_folds.py --eval runs/eval
