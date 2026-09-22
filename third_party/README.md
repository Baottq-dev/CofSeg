# third_party/

Mã nguồn của người khác mà repo cần nhưng không có trên PyPI. Mỗi thư mục là
một **git submodule** ghim đúng một commit, nên `git clone --recurse-submodules`
(hoặc `git submodule update --init` sau khi `pull`) là có đủ, không phụ thuộc
mạng lúc chạy `setup.sh`.

| Thư mục | Nguồn | Commit | License | Dùng cho |
|---|---|---|---|---|
| `Mask2Former/` | https://github.com/facebookresearch/Mask2Former | `9b0651c` (20/05/2022, bản cuối) | MIT | `canopyseg/models/detectron2.py` nạp `train_net.py` + gói `mask2former` làm module; op `MSDeformAttn` biên dịch bằng `pip install` trong `mask2former/modeling/pixel_decoder/ops/` (`scripts/remote/setup.sh`) |

Thư viện có pip package (detectron2, mmdet, sam2, ultralytics) **không** để ở
đây — chúng ghim trong `requirements.txt`.

Cập nhật một submodule: `git -C third_party/<tên> checkout <commit>` rồi commit
ở repo ngoài (git ghi lại commit mới của repo con). Không sửa mã bên trong
repo con; cần vá thì vá bằng code ở `canopyseg/` (như `load_mask2former`).
