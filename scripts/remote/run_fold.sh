#!/usr/bin/env bash
# Huấn luyện 4 model chính trên một fold rồi gom dự đoán test về preds/.
#
#     bash scripts/remote/run_fold.sh f4
#     bash scripts/remote/run_fold.sh f4 --smoke                 # vài iteration, kiểm đường chạy
#     bash scripts/remote/run_fold.sh f4 --only maskrcnn,solov2  # chạy lại một phần
#     bash scripts/remote/run_fold.sh f4 --only cascade          # model ngoài bộ chính
#     bash scripts/remote/run_fold.sh f4 --imgsz 2048 --batch 2 --epochs 30
#
# Mỗi model một lệnh scripts/train.py, tuần tự, dừng ngay khi một lệnh lỗi.
# Model xong thì preds/<model>_<fold>.json là file COCO results của split
# TEST — thứ mang về máy nhà chấm bằng scripts/score_remote.py. Log: runs/remote_<fold>.log
#
# Bộ chính: yolo11s, maskrcnn (mốc 0), solov2, mask2former. cascade vẫn có
# code/config, chạy khi gọi tên qua --only.
set -euo pipefail
FOLD="${1:?tên fold, vd f4}"; shift
WORKERS=8; ONLY="yolo11s,maskrcnn,solov2,mask2former"; SMOKE=0; IMGSZ=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --workers) WORKERS="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    --imgsz) IMGSZ="$2"; EXTRA+=(--imgsz "$2"); shift ;;
    *) EXTRA+=("$1") ;;          # --epochs 30, --batch 2, ... đi thẳng vào train.py
  esac
  shift
done

cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook)"; conda activate "${ENV_NAME:-cofseg}"
ROOT="data/export/$FOLD"
[ -f "$ROOT/annotations/instances_test.json" ] || { echo "Không thấy $ROOT: chạy prepare_data.sh trước"; exit 1; }
mkdir -p preds runs
LOG="runs/remote_${FOLD}.log"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date '+%F %T') fold $FOLD  only=$ONLY  smoke=$SMOKE  workers=$WORKERS  extra=${EXTRA[*]:-}"

want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
latest() { ls -td "$1"* 2>/dev/null | head -1; }

# ---- YOLO11s (ultralytics) ---------------------------------------------------
if want yolo11s; then
  NAME="yolo11s-seg-$FOLD"
  SM=(); [ $SMOKE = 1 ] && SM=(--epochs 1 --fraction 0.05)
  python scripts/train.py --config configs/train/yolo11s.yaml \
    --set data.yaml="$ROOT/data.yaml" --workers "$WORKERS" --name "$NAME" "${EXTRA[@]}" "${SM[@]}"
  RUN=$(latest "runs/train/*_${NAME}_")
  BEST="$RUN/ultralytics/weights/best.pt"
  [ -f "$BEST" ] || BEST="$RUN/ultralytics/weights/last.pt"
  EV=(); [ -n "$IMGSZ" ] && EV=(--imgsz "$IMGSZ")
  python scripts/evaluate.py --config configs/eval/yolo11s.yaml \
    --set model.weights="$BEST" --set data.root="$ROOT" --split test --name "yolo11s-$FOLD" "${EV[@]}"
  cp "$(latest "runs/eval/*_yolo11s-${FOLD}_")/predictions.json" "preds/yolo11s_${FOLD}.json"
  echo "== yolo11s xong: preds/yolo11s_${FOLD}.json"
fi

# ---- detectron2 (Mask R-CNN mốc 0, Mask2Former, Cascade) và mmdet (SOLOv2) ----
# Cả hai trainer để lại cùng bố cục: predictions.json + test_metrics.json{segm}.
run_cfg() {   # $1 = tên model trong preds, $2 = config train
  NAME="$2-$FOLD"
  SM=(); [ $SMOKE = 1 ] && SM=(--set data.limit=16 --epochs 1)
  python scripts/train.py --config "configs/train/$2.yaml" \
    --set data.root="$ROOT" --workers "$WORKERS" --name "$NAME" "${EXTRA[@]}" "${SM[@]}"
  RUN=$(latest "runs/train/*_${NAME}_")
  [ -f "$RUN/predictions.json" ] || { echo "$RUN không có predictions.json"; exit 1; }
  cp "$RUN/predictions.json" "preds/$1_${FOLD}.json"
  echo "== $1 xong: preds/$1_${FOLD}.json  ($(python -c "import json;d=json.load(open('$RUN/test_metrics.json'));print({k:round(v,1) for k,v in d.get('segm',{}).items() if k in ('AP','AP50','AP75')})"))"
}
want maskrcnn    && run_cfg maskrcnn    maskrcnn_r50_d2
want solov2      && run_cfg solov2      solov2_r50_mm
want mask2former && run_cfg mask2former mask2former_r50_d2
want cascade     && run_cfg cascade     cascade_r50_d2

echo "== $(date '+%F %T') fold $FOLD xong. preds/:"; ls -1 preds/*_"$FOLD".json
