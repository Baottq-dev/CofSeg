#!/usr/bin/env bash
# Giải nén bản xuất chưa chia rồi cắt đủ fold.
#     bash scripts/remote/prepare_data.sh all_v2.tar
#
# all_v2.tar tạo ở máy nhà, từ gốc repo (PowerShell/Windows 10+ có sẵn tar):
#     tar -cf all_v2.tar -C data/export all_v2
# Bên trong là một thư mục all_v2/ với images/, annotations/instances.json,
# labels/, data.yaml — đúng thứ app/ xuất với split_by=none.
set -euo pipefail
TAR="${1:?đường dẫn all_v2.tar}"
cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook)"; conda activate "${ENV_NAME:-cofseg}"

mkdir -p data/export
tar -xf "$TAR" -C data/export
[ -f data/export/all_v2/annotations/instances.json ] || { echo "Không thấy all_v2/annotations/instances.json sau khi giải nén"; exit 1; }
# data.yaml của bản xuất ghi path tuyệt đối của máy nhà; make_fold ghi lại
# theo máy này nên không cần sửa.
python scripts/make_fold.py --export data/export/all_v2 --all
ls -d data/export/f*
