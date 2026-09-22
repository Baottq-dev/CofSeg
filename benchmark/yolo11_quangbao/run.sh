#!/usr/bin/env bash
# YOLOv11-Seg — huấn luyện một fold rồi chấm split test. Của QuangBao.
#
#     bash benchmark/yolo11_quangbao/run.sh f4
#     bash benchmark/yolo11_quangbao/run.sh f4 --smoke            # 1 epoch, 5 % ảnh
#     bash benchmark/yolo11_quangbao/run.sh f4 --imgsz 1024 --batch 4
#
# Chạy từ GỐC REPO (data/ và weights/ dùng chung). Mọi thứ khác nằm trong thư
# mục này: code (cofseg/), config, kết quả (runs/). Không đụng thư mục người khác.
set -euo pipefail
cd "$(dirname "$0")/../.."
HERE="benchmark/yolo11_quangbao"
FOLD="${1:?tên fold, vd f4}"; shift || true
SMOKE=0; EXTRA=()
for a in "$@"; do case "$a" in --smoke) SMOKE=1 ;; *) EXTRA+=("$a") ;; esac; done

ROOT="data/export/$FOLD"
[ -f "$ROOT/data.yaml" ] || { echo "Không thấy $ROOT/data.yaml: chạy scripts/make_fold.py trước"; exit 1; }
mkdir -p "$HERE/runs" "$HERE/results" preds
SM=(); [ $SMOKE = 1 ] && SM=(--epochs 1 --fraction 0.05)

python "$HERE/scripts/train.py" --config "$HERE/configs/train/yolo11s.yaml"   --set data.yaml="$ROOT/data.yaml" --runs "$HERE/runs" --name "yolo11s-$FOLD" "${EXTRA[@]}" "${SM[@]}"

RUN=$(ls -td "$HERE/runs/train/"*_yolo11s-"$FOLD"_* | head -1)
BEST="$RUN/ultralytics/weights/best.pt"; [ -f "$BEST" ] || BEST="$RUN/ultralytics/weights/last.pt"
[ -f "$RUN/results.csv" ] && cp "$RUN/results.csv" "$HERE/results/yolo11s_${FOLD}_train.csv"

python "$HERE/scripts/evaluate.py" --config "$HERE/configs/eval/yolo11s.yaml"   --set model.weights="$BEST" --set data.root="$ROOT" --split test   --runs "$HERE/runs" --name "yolo11s_$FOLD"
EV=$(ls -td "$HERE/runs/eval/"*_yolo11s_"$FOLD"_* | head -1)
cp "$EV/predictions.json" "preds/yolo11s_$FOLD.json"
cp "$EV/metrics.json"     "$HERE/results/yolo11s_${FOLD}_metrics.json"
cp "$EV/per_region.csv"   "$HERE/results/yolo11s_${FOLD}_per_region.csv"
echo "== xong. Kết quả: $HERE/results/yolo11s_${FOLD}_metrics.json"
