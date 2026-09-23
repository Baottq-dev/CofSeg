# benchmark/ — mỗi model một thư mục độc lập, mỗi thư mục một người

Bốn model của bảng so sánh, mỗi model nằm trọn trong một thư mục: **code
riêng, config riêng, cách chạy riêng, kết quả riêng**. Không thư mục nào
import thư mục nào, không ai phải chờ ai để sửa phần của mình.

## Phân công

| Thư mục | Người phụ trách | GitHub | Model | Vai trò trong bảng |
|---|---|---|---|---|
| `maskrcnn/` | VanNguyen | @vnguyen123 | Mask R-CNN R50-FPN | **mốc số 0** — mọi model khác báo Δ% mAP so với nó |
| `solov2/` | PhuongQuynh | @Phquynh2312 | SOLOv2 R50-FPN | box-free: lưới + kernel động, không box |
| `yolo11/` | QuangBao | @Baottq-dev | YOLOv11-Seg | một giai đoạn, thời gian thực |
| `mask2former/` | AnhVu | @tranphuocanhvu2103 | Mask2Former R50 | query / transformer, mask toàn ảnh |

Ba model dùng chung backbone ResNet-50 (Mask R-CNN, SOLOv2, Mask2Former) nên
chênh lệch giữa chúng là do cơ chế, không do backbone.

## Một thư mục có gì

```
benchmark/<model>/
├─ README.md              model gì, vì sao chọn, cách chạy, trạng thái
├─ train.py               huấn luyện, nạp cofseg/ của thư mục này
├─ evaluate.py            chấm, nạp cofseg/ của thư mục này
├─ cofseg/                BẢN SAO LÕI của riêng thư mục này
│   ├─ datasets/          đọc fold COCO
│   ├─ metrics/           mask AP, Boundary AP/IoU, sai số diện tích
│   ├─ evaluation/        vòng chấm
│   ├─ models/            wrapper suy luận của model này
│   └─ training/          trainer của model này
├─ configs/train/*.yaml   cấu hình huấn luyện
├─ configs/eval/*.yaml    cấu hình chấm
├─ tests/                 test cho phần của mình
├─ runs/                  (không vào git) kết quả train/eval
├─ results/               file kết quả nhỏ, ĐƯỢC commit
└─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
```

Sửa gì trong thư mục mình cũng được — kể cả `cofseg/`. Không ai bị ảnh hưởng.

## Chạy

Từ **gốc repo** (để `data/` và `weights/` dùng chung):

```bash
# cắt fold một lần cho cả nhóm
python scripts/make_fold.py --export data/export/all_v2 --all

# cả bốn model trên một fold
python benchmark/run.py f4 --smoke      # vài iteration, kiểm đường chạy trước
python benchmark/run.py f4

# một người chạy model của mình
python benchmark/run.py f4 --only solov2
python benchmark/run.py f4 f2 --only yolo11 --imgsz 1024 --batch 4

# hoặc gọi thẳng train/evaluate của thư mục mình
python benchmark/yolo11/train.py --config benchmark/yolo11/configs/train/yolo11s.yaml     --set data.yaml=data/export/f4/data.yaml --runs benchmark/yolo11/runs
```

`benchmark/run.py` gọi `train.py` rồi `evaluate.py` của từng thư mục, để lại
`preds/<model>_<fold>.json` ở gốc và file kết quả trong
`benchmark/<model>/results/`. Một model lỗi thì ghi nhận và chạy tiếp model sau.

## Cái giá của việc tách rời

Bốn bản sao `cofseg/` xuất phát giống hệt nhau. Nếu một người sửa phần **chấm
điểm** (`metrics/`, `evaluation/`, `datasets/`) thì model của người đó đo bằng
một cái thước khác, và cột "Δ% mAP so với Mask R-CNN" không còn nghĩa.

Sửa trainer hay model wrapper thì thoải mái — đó là phần của bạn. Sửa phần
chấm điểm thì **phải báo nhóm** để ba người kia sửa theo, hoặc ghi rõ khác
biệt khi trình bày bảng.

Dùng chung thật sự chỉ còn: `data/` (ảnh + nhãn), `weights/` (trọng số COCO),
`scripts/make_fold.py` (cắt fold — mọi người phải dùng CÙNG bộ fold),
`scripts/summarize_folds.py` (gộp bảng), `requirements.txt` (một env).

## Quy ước làm chung

- **Một env Python duy nhất.** Cần thư viện mới thì thêm vào `requirements.txt`
  rồi báo nhóm, không `pip install` riêng rồi quên ghi.
- **Mỗi người một nhánh**, gộp vào `main` bằng merge hoặc rebase.
  **Không dùng "Squash and merge"**: squash gộp nhiều commit thành một và làm
  mất author của từng commit, tức mất dấu vết phân công.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
  Commit trong thư mục nào thì mang tên người phụ trách thư mục đó.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`.
  Kết quả muốn chia sẻ thì chép file nhỏ vào `benchmark/<model>/results/`.
- Chạy test của thư mục mình trước khi commit:
  `cd benchmark/<model> && python -m pytest tests`

## Commit đúng tên khi dùng chung một máy

Tạo `.authors.local` ở gốc repo (không vào git), nội dung theo
`docs/danh_tinh_thanh_vien.md`:

```bash
source .authors.local
as_quynh git commit -m "solov2: raise the score threshold for the f5 run"
git log -1 --format="A: %an <%ae>%nC: %cn <%ce>"    # kiểm lại
```

Đếm theo người: `git shortlog -sne`.

Chỉ commit dưới tên một người khi người đó **thật sự làm phần đó**.
