#!/usr/bin/env bash
# Gói thứ cần mang về máy nhà: dự đoán test (chấm lại bằng score_remote.py),
# trọng số tốt nhất, results.csv, config và env của mỗi lần chạy, log.
#     bash scripts/remote/pack_results.sh            # -> results_<ngày>.tar
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-results_$(date +%F_%H%M).tar}"
LIST=$(mktemp)
{
  ls preds/*.json 2>/dev/null
  ls runs/remote_*.log 2>/dev/null
  for r in runs/train/*/; do
    for f in weights/best.pt weights/best.pth weights/d2_config.yaml results.csv config.yaml env.json \
             dataset_check.json summary.json test_metrics.json predictions.json run.log; do
      [ -f "$r$f" ] && echo "$r$f"
    done
  done
  for r in runs/eval/*/; do
    for f in metrics.json cocoeval.txt per_region.csv per_image.csv config.yaml; do
      [ -f "$r$f" ] && echo "$r$f"
    done
  done
} > "$LIST"
tar -cf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "== $OUT  ($(du -h "$OUT" | cut -f1)); $(tar -tf "$OUT" | wc -l) file"
