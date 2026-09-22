#!/usr/bin/env bash
# Mask R-CNN R50-FPN — huấn luyện một fold rồi chấm split test. Của VanNguyen.
#
#     bash benchmark/maskrcnn_vannguyen/run.sh f4
#     bash benchmark/maskrcnn_vannguyen/run.sh f4 --smoke            # vài iteration, kiểm đường chạy
#     bash benchmark/maskrcnn_vannguyen/run.sh f4 --epochs 30 --batch 2
#
# Chạy từ GỐC REPO (data/ và weights/ dùng chung). Mọi thứ khác nằm trong thư
# mục này: code (cofseg/), config, kết quả (runs/). Không đụng thư mục người khác.
set -euo pipefail
cd "$(dirname "$0")/../.."
HERE="benchmark/maskrcnn_vannguyen"
FOLD="${1:?tên fold, vd f4}"; shift || true
SMOKE=0; EXTRA=()
for a in "$@"; do case "$a" in --smoke) SMOKE=1 ;; *) EXTRA+=("$a") ;; esac; done

ROOT="data/export/$FOLD"
[ -f "$ROOT/annotations/instances_test.json" ] || { echo "Không thấy $ROOT: chạy scripts/make_fold.py trước"; exit 1; }
mkdir -p "$HERE/runs" "$HERE/results" preds
SM=(); [ $SMOKE = 1 ] && SM=(--set data.limit=16 --epochs 1)

python "$HERE/scripts/train.py" --config "$HERE/configs/train/maskrcnn_r50_d2.yaml"   --set data.root="$ROOT" --runs "$HERE/runs" --name "maskrcnn-$FOLD" "${EXTRA[@]}" "${SM[@]}"

RUN=$(ls -td "$HERE/runs/train/"*_maskrcnn-"$FOLD"_* | head -1)
[ -f "$RUN/predictions.json" ] || { echo "$RUN không có predictions.json"; exit 1; }
cp "$RUN/predictions.json" "preds/maskrcnn_$FOLD.json"
cp "$RUN/test_metrics.json" "$HERE/results/maskrcnn_${FOLD}_test_metrics.json"
[ -f "$RUN/results.csv" ] && cp "$RUN/results.csv" "$HERE/results/maskrcnn_${FOLD}_train.csv"

# Chấm lại bằng vòng chấm trong thư mục này -> Boundary AP/IoU, sai số diện tích.
python "$HERE/scripts/evaluate.py" --config "$HERE/configs/eval/_coco.yaml"   --file "preds/maskrcnn_$FOLD.json" --set data.root="$ROOT" --split test   --runs "$HERE/runs" --name "maskrcnn_$FOLD"
EV=$(ls -td "$HERE/runs/eval/"*_maskrcnn_"$FOLD"_* | head -1)
cp "$EV/metrics.json"    "$HERE/results/maskrcnn_${FOLD}_metrics.json"
cp "$EV/per_region.csv"  "$HERE/results/maskrcnn_${FOLD}_per_region.csv"
echo "== xong. Kết quả: $HERE/results/maskrcnn_${FOLD}_metrics.json"
