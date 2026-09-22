# benchmark/ — mỗi model một thư mục độc lập, mỗi thư mục một người

Bốn model của bảng so sánh, mỗi model nằm trọn trong một thư mục: **code
riêng, config riêng, cách chạy riêng, kết quả riêng**. Không thư mục nào
import thư mục nào, không ai phải chờ ai để sửa phần của mình.

## Phân công

| Thư mục | Người phụ trách | GitHub | Model | Vai trò trong bảng |
|---|---|---|---|---|
| `maskrcnn_vannguyen/` | VanNguyen | @vnguyen123 | Mask R-CNN R50-FPN | **mốc số 0** — mọi model khác báo Δ% mAP so với nó |
| `solov2_phuongquynh/` | PhuongQuynh | @Phquynh2312 | SOLOv2 R50-FPN | box-free: lưới + kernel động, không box |
| `yolo11_quangbao/` | QuangBao | @Baottq-dev | YOLOv11-Seg | một giai đoạn, thời gian thực |
| `mask2former_anhvu/` | AnhVu | @tranphuocanhvu2103 | Mask2Former R50 | query / transformer, mask toàn ảnh |

Ba model dùng chung backbone ResNet-50 (Mask R-CNN, SOLOv2, Mask2Former) nên
chênh lệch giữa chúng là do cơ chế, không do backbone.

## Một thư mục có gì

```
benchmark/<model>_<người>/
├─ README.md              model gì, vì sao chọn, cách chạy, trạng thái
├─ cofseg/                BẢN SAO LÕI của riêng thư mục này
│   ├─ datasets/          đọc fold COCO
│   ├─ metrics/           mask AP, Boundary AP/IoU, sai số diện tích
│   ├─ evaluation/        vòng chấm
│   ├─ models/            wrapper suy luận của model này
│   └─ training/          trainer của model này
├─ configs/train/*.yaml   cấu hình huấn luyện
├─ configs/eval/*.yaml    cấu hình chấm
├─ scripts/train.py       bản riêng, nạp cofseg/ của thư mục này
├─ scripts/evaluate.py    bản riêng
├─ run.sh                 train một fold -> chấm -> chép kết quả
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

# một người chạy model của mình
bash benchmark/yolo11_quangbao/run.sh f4 --smoke      # kiểm đường chạy trước
bash benchmark/yolo11_quangbao/run.sh f4

# cả bốn model trên một fold (gọi run.sh của từng thư mục)
bash scripts/remote/run_fold.sh f4
```

`run.sh` để lại `preds/<model>_<fold>.json` ở gốc và các file kết quả trong
`benchmark/<...>/results/`.

## Cái giá của việc tách rời, và cách kiểm soát

Bốn bản sao `cofseg/` ban đầu giống hệt nhau. Nếu một người sửa phần **chấm
điểm** (`metrics/`, `evaluation/`, `datasets/`) thì model của người đó đo bằng
một cái thước khác, và cột "Δ% mAP so với Mask R-CNN" không còn nghĩa.

```bash
python benchmark/check_copies.py          # 9 file chấm điểm x 4 bản, so với canopyseg/
python benchmark/check_copies.py --diff   # lệch ở dòng nào
python benchmark/check_copies.py --sync   # chép bản gốc đè lên chỗ lệch
```

`run_fold.sh` gọi script này trước khi chạy. Sửa trainer hay model wrapper thì
thoải mái — đó là phần của bạn; nhưng sửa phần chấm điểm thì **phải báo nhóm**,
hoặc ghi rõ khác biệt khi trình bày bảng.

Dùng chung thật sự chỉ còn: `data/` (ảnh + nhãn), `weights/` (trọng số COCO),
`scripts/make_fold.py` (cắt fold — mọi người phải dùng CÙNG bộ fold),
`requirements.txt` (một env).

## Quy ước làm chung

- **Một env Python duy nhất.** Cần thư viện mới thì thêm vào `requirements.txt`
  rồi báo nhóm, không `pip install` riêng rồi quên ghi.
- **Mỗi người một nhánh**, gộp vào `main` bằng merge hoặc rebase.
  **Không dùng "Squash and merge"**: squash gộp nhiều commit thành một và làm
  mất author của từng commit, tức mất dấu vết phân công.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
  Commit trong thư mục nào thì mang tên người phụ trách thư mục đó.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/`, `benchmark/*/runs/`.
  Kết quả muốn chia sẻ thì chép file nhỏ vào `benchmark/<...>/results/`.
- Chạy test của thư mục mình trước khi commit:
  `cd benchmark/<...> && python -m pytest tests`

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
