# members/ — mỗi model một thư mục, mỗi thư mục một người phụ trách

Khung benchmark (dữ liệu, chỉ số, vòng chấm, trainer) nằm ở `canopyseg/` và
dùng chung cho cả bốn model — đó là điều kiện để các con số so được với nhau.
Thư mục ở đây là phần của từng người: config của model mình, code riêng nếu
cần, nhật ký thí nghiệm và kết quả.

## Phân công

| Thư mục | Người phụ trách | GitHub | Model | Vai trò trong bảng |
|---|---|---|---|---|
| `maskrcnn/` | VanNguyen | @vnguyen123 | Mask R-CNN R50-FPN | **mốc số 0** — mọi model khác báo Δ% mAP so với nó |
| `solov2/` | PhuongQuynh | @Phquynh2312 | SOLOv2 R50-FPN | họ box-free: lưới + kernel động, không box |
| `yolo11/` | QuangBao | @Baottq-dev | YOLOv11-Seg | họ một giai đoạn, thời gian thực |
| `mask2former/` | AnhVu | @tranphuocanhvu2103 | Mask2Former R50 | họ query/transformer, mask toàn ảnh |

Ba model dùng chung backbone ResNet-50 (Mask R-CNN, SOLOv2, Mask2Former) nên
chênh lệch giữa chúng là do cơ chế, không do backbone.

## Ai sửa được phần nào

| Phần | Ai |
|---|---|
| `members/<model>/` | chỉ người phụ trách model đó |
| `canopyseg/` — lõi dùng chung | cả nhóm, báo trước khi sửa |
| `scripts/`, `configs/` (fold, `_base.yaml`, `_base_d2.yaml`, `weights.yaml`), `requirements.txt` | cả nhóm, báo trước khi sửa |
| `app/` — annotator | QuangBao |

Lý do lõi không chia theo người: đổi một dòng trong cách tính Boundary AP hay
cách cắt fold là đổi số của **cả bốn** model. Sửa xong phải báo để mọi người
chạy lại, không tự sửa rồi im lặng.

`.github/CODEOWNERS` ghi đúng bảng trên; GitHub sẽ tự gán người review theo
thư mục bị đụng (có hiệu lực khi bật branch protection cho `main`).

## Bố cục một thư mục thành viên

```
members/<model>/
├─ README.md              model gì, vì sao chọn, cách chạy, trạng thái
├─ configs/train/*.yaml   cấu hình huấn luyện
├─ configs/eval/*.yaml    cấu hình chấm
├─ notes/experiments.md   nhật ký: chạy gì, ra số gì, vì sao đổi tham số
├─ results/               file kết quả nhỏ (runs/ không vào git)
└─ plugin.py              (tuỳ chọn) code riêng: trainer, hook, tăng cường
```

`plugin.py` được `scripts/train.py` và `scripts/evaluate.py` nạp trước khi tra
registry, nên `@register("trainer", "ten_cua_ban")` trong đó dùng được ngay
từ `trainer:` của config. Đây là chỗ thử ý riêng mà không đụng `canopyseg/`.

Ba model chạy qua detectron2 (Mask R-CNN, Cascade, Mask2Former) dùng chung
recipe ở `configs/train/_base_d2.yaml` — sửa file đó là đổi số của cả ba.

## Quy ước làm chung

- **Một env Python duy nhất.** Cần thư viện mới thì thêm vào `requirements.txt`
  rồi báo nhóm, không `pip install` riêng rồi quên ghi — máy người khác sẽ hỏng.
- **Mỗi người một nhánh**, gộp vào `main` bằng merge hoặc rebase.
  **Không dùng "Squash and merge"**: squash gộp nhiều commit thành một và làm
  mất author của từng commit, tức mất dấu vết phân công.
- **Commit message tiếng Anh**, mô tả thay đổi chứ không mô tả file.
  Commit mang tên người thật sự làm phần đó.
- **Không commit** `data/`, `runs/`, `weights/`, `docs/` (đã trong `.gitignore`).
  Kết quả muốn chia sẻ thì chép file nhỏ vào `members/<model>/results/`.

## Commit đúng tên khi dùng chung một máy

Cả nhóm đang code trên một máy nên phải đặt author thủ công cho từng commit.
Tạo `.authors.local` ở gốc repo (file này **không** vào git):

```bash
# source .authors.local   ->   as_quynh git commit -m "solov2: ..."
_as() {  # $1 = tên, $2 = email, còn lại là lệnh git
  local n="$1" e="$2"; shift 2
  GIT_AUTHOR_NAME="$n"  GIT_AUTHOR_EMAIL="$e" \
  GIT_COMMITTER_NAME="$n" GIT_COMMITTER_EMAIL="$e" "$@"
}
as_nguyen() { _as "VanNguyen"   "doanvantuanst357@gmail.com"   "$@"; }
as_quynh()  { _as "PhuongQuynh" "ttpquynh2312@gmail.com"       "$@"; }
as_bao()    { _as "QuangBao"    "baottqdeveloper@gmail.com"    "$@"; }
as_vu()     { _as "AnhVu"       "tranphuocanhvu2103@gmail.com" "$@"; }
```

Kiểm sau mỗi commit: `git log -1 --format="A: %an <%ae>%nC: %cn <%ce>"`
Đếm theo người: `git shortlog -sne`

Chỉ commit dưới tên một người khi người đó **thật sự làm phần đó**. Log ghi
tên người không tham gia là khai sai công với người chấm.
