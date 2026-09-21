#!/usr/bin/env bash
# Chạy đủ 6 fold, hai fold sàng trước để có số sớm. Mọi tuỳ chọn đi thẳng
# vào run_fold.sh (--only, --smoke, --epochs ...).
#     nohup bash scripts/remote/run_all.sh > runs/remote_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
FOLDS="${FOLDS:-f4 f2 f1 f6 f3 f5}"
for f in $FOLDS; do
  bash scripts/remote/run_fold.sh "$f" "$@"
done
echo "== xong $FOLDS. Gói kết quả: bash scripts/remote/pack_results.sh"
