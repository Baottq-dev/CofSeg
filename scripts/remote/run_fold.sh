#!/usr/bin/env bash
# Chạy cả bốn model của bảng benchmark trên một fold, tuần tự.
#
#     bash scripts/remote/run_fold.sh f4
#     bash scripts/remote/run_fold.sh f4 --smoke                 # vài iteration, kiểm đường chạy
#     bash scripts/remote/run_fold.sh f4 --only maskrcnn,solov2  # chạy lại một phần
#     bash scripts/remote/run_fold.sh f4 --epochs 30 --batch 2
#
# Script này KHÔNG biết model nào chạy thế nào: nó gọi run.sh trong thư mục
# của từng người (benchmark/<model>_<người>/run.sh), mỗi thư mục có bản code
# riêng. Ai sửa model của mình thì sửa trong thư mục mình, file này không đổi.
#
# Mỗi model xong để lại preds/<model>_<fold>.json (COCO results của split TEST)
# và kết quả chấm trong benchmark/<...>/results/. Log: runs/remote_<fold>.log
set -euo pipefail
FOLD="${1:?tên fold, vd f4}"; shift || true
ONLY="maskrcnn,solov2,yolo11s,mask2former"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift ;;
    *) EXTRA+=("$1") ;;          # --smoke, --epochs 30, --batch 2, ... đi thẳng vào run.sh
  esac
  shift
done

cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook)"; conda activate "${ENV_NAME:-cofseg}"
[ -f "data/export/$FOLD/annotations/instances_test.json" ] || {
  echo "Không thấy data/export/$FOLD: chạy prepare_data.sh trước"; exit 1; }
mkdir -p preds runs
LOG="runs/remote_${FOLD}.log"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date '+%F %T') fold $FOLD  only=$ONLY  extra=${EXTRA[*]:-}"

# Bốn bản chấm điểm phải giống nhau thì Δ% mAP giữa các model mới có nghĩa.
python benchmark/check_copies.py || echo "!! bản chấm điểm đã lệch — xem cảnh báo ở trên"

want() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
run() {   # $1 = tên ngắn, $2 = thư mục thành viên
  echo "== $(date '+%F %T') $1 -> benchmark/$2"
  bash "benchmark/$2/run.sh" "$FOLD" "${EXTRA[@]:-}"
}
want maskrcnn    && run maskrcnn    maskrcnn_vannguyen
want solov2      && run solov2      solov2_phuongquynh
want yolo11s     && run yolo11s     yolo11_quangbao
want mask2former && run mask2former mask2former_anhvu

echo "== $(date '+%F %T') fold $FOLD xong. preds/:"; ls -1 preds/*_"$FOLD".json
