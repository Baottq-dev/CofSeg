#!/usr/bin/env bash
# Chạy trọn bộ fold `field`: 6 biến thể model x 6 fold, train rồi eval.
#
# Chạy lại bao nhiêu lần cũng được. Script KHÔNG giữ file trạng thái riêng —
# nó đọc thẳng thư mục kết quả mỗi lần khởi động và làm tiếp phần còn thiếu.
# Đó là điều kiện để sống sót qua `SIGKILL`: máy thuê bị thu hồi, OOM killer
# của hệ điều hành, hay mất điện đều không cho tiến trình kịp ghi gì cả, nên
# một file trạng thái sẽ lệch với thực tế đúng vào lúc cần nó nhất.
#
#     bash scripts/run_all_field.sh              # chạy / chạy tiếp
#     bash scripts/run_all_field.sh --dry-run    # chỉ in việc sẽ làm
#     CUDA_VISIBLE_DEVICES=1 bash scripts/run_all_field.sh
#
# Chạy lại cả bảng trên một môi trường khác thì ĐỔI GỐC KẾT QUẢ, đừng ghi đè:
#
#     RUNS=runs_cu130 bash scripts/run_all_field.sh
#
# Vì "đã xong" được suy ra từ chính thư mục kết quả, để chung gốc là script
# thấy lượt cũ rồi bỏ qua sạch — đúng thứ ta không muốn khi mục đích là chấm
# lại trên env mới. Tách gốc cũng giữ được lượt cũ để so hai môi trường.
#
# Mất SSH là mất SIGHUP, cái đó tránh được — chạy trong tmux:
#
#     tmux new -s bench 'bash scripts/run_all_field.sh 2>&1 | tee -a runs/bench.log'
#
# KHÔNG truyền siêu tham số nào ở đây. epochs, batch, val_every, màu... đều
# nằm trong config và đã có test khoá. Sáu lượt YOLO đầu tiên chạy 100 epoch
# bằng cờ dòng lệnh trong khi config ghi 50, và không ai phát hiện ra cho tới
# lúc dựng bảng — một con số chỉ tồn tại trong lịch sử shell là con số không
# kiểm toán được.

set -uo pipefail          # KHÔNG -e: một lượt hỏng thì ghi nhận rồi đi tiếp
cd "$(dirname "$0")/.."   # luôn chạy từ gốc repo

PY=${PY:-python}
BO=${BO:-field}                       # bộ fold: field | block | flight
FOLDS=${FOLDS:-"f1 f2 f3 f4 f5 f6"}
RUNS=${RUNS:-runs}                    # gốc chứa runs/train và runs/eval
LOGS="$RUNS/joblog"
TIENDO="$RUNS/progress.tsv"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# model | thư mục | tên config (dùng chung cho train và eval) | tên lần chạy | đuôi checkpoint
# Xếp theo GIỜ MÁY TĂNG DẦN, và chạy trọn từng model trước khi sang model sau:
# bảng tổng hợp cần đủ 6 fold mới tính được trung bình một model, nên xong sớm
# một HÀNG có giá trị hơn xong dở nhiều hàng. Máy chết giữa chừng thì phần đã
# xong vẫn dùng được ngay.
CONG_VIEC=(
  "yolo|yolov8s|yolov8s-seg|pt"
  "yolo|yolo11s|yolo11s-seg|pt"
  "yolo|yolo26s|yolo26s-seg|pt"
  "maskrcnn|maskrcnn_r50_d2|maskrcnn-r50-d2|pth"
  "solov2|solov2_r50_mm|solov2-r50-mm|pth"
  "mask2former|mask2former_r50_d2|mask2former-r50-d2|pth"
)

# Ba model này nối tiếp được lượt bị ngắt (detectron2 `resume_or_load`, mmdet
# có nhánh riêng). YOLO thì KHÔNG: `benchmark/yolo/cofseg/training/yolo.py`
# không hề đụng tới cờ `resume`, mà `base.py` nói rõ trainer không nối tiếp
# được thì phải nêu lý do chứ đừng im lặng train lại từ đầu. Truyền --resume
# cho nó sẽ rơi đúng vào cái bẫy đó, nên lượt YOLO dở thì xoá đi chạy lại —
# mất nhiều nhất 11 phút, rẻ hơn mọi cách khác.
noi_tiep_duoc() { [ "$1" != "yolo" ]; }

moi_nhat() { ls -td $1 2>/dev/null | head -1; }   # cố ý không quote: cần glob

ghi() { printf '%s\t%s\n' "$(date +%H:%M:%S)" "$*"; }

ghi_tien_do() {   # model fold giai_doan trang_thai giay
  [ "$DRY" = 1 ] && return 0
  mkdir -p "$(dirname "$TIENDO")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" "$5" >> "$TIENDO"
}

chay() {  # nhan  file_log  lenh...
  local nhan="$1" log="$2"; shift 2
  if [ "$DRY" = 1 ]; then ghi "     [dry] $*"; return 0; fi
  mkdir -p "$(dirname "$log")"
  local t0=$SECONDS
  if "$@" >>"$log" 2>&1; then
    ghi "     xong $nhan  ($((SECONDS - t0))s)"; return 0
  fi
  ghi "     HỎNG $nhan  ($((SECONDS - t0))s) — xem $log"; return 1
}

tong=0; xong=0; bo_qua=0; hong=0
[ "$DRY" = 1 ] || mkdir -p "$LOGS"
ghi "bộ $BO | folds: $FOLDS | runs: $RUNS | dry-run: $DRY"
ghi "GPU: ${CUDA_VISIBLE_DEVICES:-mặc định}"

for viec in "${CONG_VIEC[@]}"; do
  IFS='|' read -r thumuc cfg ten duoi <<< "$viec"
  for fold in $FOLDS; do
    tong=$((tong + 1))
    nhan="$ten/$fold"
    data="data/export/$BO/$fold"
    tag="${BO}-${fold}"

    if [ ! -d "$data" ]; then
      ghi "$nhan: THIẾU $data — cắt fold trước (scripts/make_fold.py)"
      ghi_tien_do "$ten" "$fold" data thieu 0; hong=$((hong + 1)); continue
    fi

    # --- đã eval xong chưa? Đó là mốc duy nhất nói cả việc đã trọn vẹn.
    ev=$(moi_nhat "$RUNS/eval/*_${ten}_${tag}_test_*")
    if [ -n "$ev" ] && [ -f "$ev/metrics.json" ]; then
      ghi "$nhan: bỏ qua (đã có $ev)"
      bo_qua=$((bo_qua + 1)); continue
    fi

    # --- train: xong / dở / chưa có
    tr=$(moi_nhat "$RUNS/train/*_${ten}_${tag}_*")
    if [ -n "$tr" ] && [ -f "$tr/summary.json" ]; then
      ghi "$nhan: train đã xong ($tr)"
    else
      lenh=("$PY" "benchmark/$thumuc/train.py"
            --config "benchmark/$thumuc/configs/train/$cfg.yaml"
            --data "$data" --runs "$RUNS")
      if [ -n "$tr" ]; then
        if noi_tiep_duoc "$thumuc"; then
          ghi "$nhan: train dở, nối tiếp $tr"
          lenh+=(--resume "$tr")
        else
          ghi "$nhan: train dở và $thumuc không nối tiếp được — xoá $tr rồi chạy lại"
          [ "$DRY" = 1 ] || rm -rf "$tr"
        fi
      else
        ghi "$nhan: train mới"
      fi
      t0=$SECONDS
      if ! chay "train $nhan" "$LOGS/${ten}_${fold}_train.log" "${lenh[@]}"; then
        ghi_tien_do "$ten" "$fold" train hong $((SECONDS - t0))
        hong=$((hong + 1)); continue
      fi
      ghi_tien_do "$ten" "$fold" train xong $((SECONDS - t0))
      tr=$(moi_nhat "$RUNS/train/*_${ten}_${tag}_*")
    fi

    [ "$DRY" = 1 ] && { ghi "$nhan: [dry] eval"; continue; }

    # `weights/best.pt(h)` là bản chọn theo mask AP THUẦN. Với YOLO đừng lấy
    # `ultralytics/weights/best.pt`: ultralytics chọn theo fitness trộn cả chỉ
    # số hộp, tức một tiêu chí khác ba model kia — xem FITNESS_KEY trong
    # benchmark/yolo/cofseg/training/yolo.py.
    w="$tr/weights/best.$duoi"
    if [ ! -f "$w" ]; then
      ghi "$nhan: train báo xong mà không thấy $w"
      ghi_tien_do "$ten" "$fold" eval thieu_weight 0; hong=$((hong + 1)); continue
    fi

    t0=$SECONDS
    if chay "eval $nhan" "$LOGS/${ten}_${fold}_eval.log" \
        "$PY" "benchmark/$thumuc/evaluate.py" \
        --config "benchmark/$thumuc/configs/eval/$cfg.yaml" \
        --weights "$w" --data "$data" --split test --runs "$RUNS"; then
      ghi_tien_do "$ten" "$fold" eval xong $((SECONDS - t0)); xong=$((xong + 1))
    else
      ghi_tien_do "$ten" "$fold" eval hong $((SECONDS - t0)); hong=$((hong + 1))
    fi
  done
done

echo
ghi "tổng $tong việc | vừa xong $xong | đã có sẵn $bo_qua | hỏng $hong"
[ "$hong" -gt 0 ] && ghi "chạy lại script này: nó bỏ qua phần đã xong và làm tiếp phần hỏng"

# Bảng model x ruộng khi đã đủ. Thiếu lượt nào thì ô đó in "—", không phải lỗi.
if [ "$DRY" = 0 ]; then
  echo
  ghi "bảng tổng hợp:"
  "$PY" scripts/summarize_folds.py --eval "$RUNS/eval" --dataset "$BO" || true
fi
