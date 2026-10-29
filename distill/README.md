# distill/ — cải tiến YOLO26-seg cho CoffeeSeg

Thư mục này chứa mọi phần cải tiến của đợt nghiên cứu Q1. Hiện có:
- **trò nhẹ** (YOLO26-seg độ rộng 0,375 hoặc 0,25 với ba thay đổi ở đầu; độ rộng chọn sau đợt 1);
- **giám sát mặt nạ ngoài hộp** (K-A).

Sau này thêm chắt lọc tri thức, SOLOv2 + 8 phép và đo Jetson.

`benchmark/` chỉ giữ các model đem so, chạy công thức chuẩn. Kế hoạch các đợt chạy, cổng của từng đợt và lệnh molab nằm trong tài liệu nghiên cứu của nhóm, ngoài git.

## Quan hệ với benchmark/

- **Dùng lại, không sửa, không chép:**
  - `distill/` dùng phần train, bộ đọc dữ liệu và bộ chấm của `benchmark/yolo/cofseg`.
  - `distill/train.py` và `distill/evaluate.py` nạp gói `distill`, rồi chạy nguyên `benchmark/yolo/train.py` và `benchmark/yolo/evaluate.py`.
  - Mọi cờ của hai script đó giữ nguyên nghĩa.
- **Một chiều:** `benchmark/` không bao giờ import `distill/`.
- **Cùng bộ chấm:** số của `distill/` so thẳng được với bảng benchmark.
- **Trainer riêng `distill_yolo`** (`trainer.py`) kế thừa `YoloTrainer` của benchmark. Nó thêm hai khối config:
  - `student:` dựng trò thay cho model của config;
  - `outside_box:` thay loss mặt nạ.

  Mọi thứ khác giữ nguyên: mặc định cho ảnh nadir, chọn `best.pt` theo mask AP, `summary.json`, `--aug`.

## File

| File | Nội dung |
|---|---|
| `__init__.py` | Đặt `benchmark/yolo` lên đầu `sys.path` cho `import cofseg`; tắt tự cài gói của ultralytics (xem mục Lưu ý) |
| `blocks.py` | Kiểm tham số, đặt nhãn tên run, mô tả cho run.log của hai khối `student`, `outside_box`; không import torch |
| `student.py` | Trò: `Proto26S8`, `Segment26Student`, dựng từ yaml YOLO26-seg, lấy trọng số từ checkpoint COCO (chép nguyên hoặc chọn đều) |
| `outside_box.py` | Loss mặt nạ trong hộp nhãn giãn, nhật ký epoch `loss_outside_box.csv` |
| `trainer.py` | Trainer `distill_yolo`; lớp SegmentationTrainer dựng trò và gắn loss, ghép được với lớp của `--aug` |
| `train.py`, `evaluate.py` | Lệnh train và chấm (chạy từ gốc repo) |
| `configs/train/tro.yaml`, `configs/eval/tro.yaml` | Trò 0,375, lấy trọng số từ YOLO26s-seg COCO, tên run `tro-seg` |
| `configs/train/tro-025.yaml`, `configs/eval/tro-025.yaml` | Trò 0,25, lấy trọng số từ YOLO26n-seg COCO, tên run `tro-seg-w0.25` |
| `configs/train/yolo26s-ka.yaml`, `configs/eval/yolo26s-ka.yaml` | YOLO26s-seg + giám sát ngoài hộp, tên run `yolo26s-seg-ka`; lúc chấm cắt ×1,1 |
| `tests/` | 54 test, dựng model từ yaml với trọng số ngẫu nhiên, không cần `data/` hay `weights/` |

## Trò (`student:`)

Trò là YOLO26-seg với ba thay đổi ở đầu. Đợt 1 chạy hai độ rộng (quyết định 29/10) rồi chọn một:

| Độ rộng | Tham số | GFLOPs (1024×576, sau fuse) | So với YOLO26s-seg | Nguồn trọng số |
|---|---:|---:|---:|---|
| 0,375 (`tro.yaml`) | 5,76 triệu | 22,04 | −55,5% | YOLO26s-seg COCO, chọn đều |
| 0,25 (`tro-025.yaml`) | 2,57 triệu | 9,95 | −79,9% | YOLO26n-seg COCO, gần như chép nguyên |

Mỗi thay đổi ở đầu tắt riêng được:

| Khoá | Mặc định | Nghĩa |
|---|---|---|
| `width` | 0.375 | Hệ số độ rộng; không được rộng hơn checkpoint nguồn |
| `nm` | 16 | Số prototype (gốc 32) |
| `proto_s8` | true | Lớp 3×3 thứ hai của Proto chạy ở stride 8, trước bước phóng to (gốc: sau, ở stride 4). Chỉ đổi thứ tự, prototype ra vẫn ở stride 4 |
| `drop_p3` | true | Bỏ nhánh hộp/điểm/hệ số ở P3: anchor chỉ ở stride 16 và 32. Proto vẫn nhận P3–P5 |
| `init` | select | `select`: lấy trọng số từ checkpoint COCO ở khoá `model:` của config. `scratch`: khởi tạo ngẫu nhiên |

**Khởi tạo** (mọi model benchmark đều khởi đầu từ COCO):
- **Quy tắc chung:** trọng số lấy từ checkpoint COCO ở khoá `model:` của config, gọi là nguồn.
  - Tensor cùng cỡ thì chép nguyên.
  - Tensor hẹp hơn thì chọn đều theo Weight Selection (Xu và cộng sự, ICLR 2024): mỗi chiều kênh S của nguồn → T của trò lấy các chỉ số ⌊k·S/T⌋.
  - Nguồn hẹp hơn trò thì lượt train dừng.
- **Trò 0,25 ← YOLO26n-seg:** cùng độ rộng.
  - 663 tensor chép nguyên, 167 tensor chọn: các lớp theo số prototype (32 → 16) và nhánh điểm.
  - Nhánh điểm của YOLO26n COCO rộng 80 kênh vì COCO có 80 lớp; train 1 lớp thì chỉ 64 kênh. Trò lấy 64/80 kênh. YOLO26n-seg của benchmark thì để phần này ngẫu nhiên, vì ultralytics bỏ qua tensor lệch cỡ.
- **Trò 0,375 ← YOLO26s-seg:** không có checkpoint COCO cùng độ rộng.
  - Mọi khối kênh của trò bằng đúng 0,75 khối của nguồn (nhánh tách đôi của C3k2, các đầu vào của Concat, bốn bản sao của SPPF), nên phép chọn đều tự khớp giữa lớp xuất và lớp nhận.
  - Attention đổi số head (4 → 3) mà giữ cỡ head, nên ở đó chọn nguyên head.
  - Kiểm (test): lớp đầu của trò trùng tuyệt đối các kênh đã chọn; ở lớp sâu, đặc trưng còn tương quan với đúng các kênh đã chọn, còn chọn tiền tố hay khởi tạo ngẫu nhiên thì không.
  - Phạm vi: đây là cách khởi tạo, không phải nén. Với trọng số COCO, tương quan giảm còn khoảng 0,1 sau vài khối. Đó là lý do đợt 1 chạy cả hai độ rộng.
- **Lớp cuối của nhánh lớp và semseg** (80 lớp COCO → 1) giữ khởi tạo, như mọi lượt fine-tune của benchmark.

**Ablation (đợt 5):** đổi từng khoá bằng `--set`. Tên run tự mang hậu tố của mọi khoá khác mặc định. Ví dụ đầu gốc ở cùng độ rộng là `--set student.nm=32 --set student.proto_s8=false --set student.drop_p3=false`, ra tên `tro-seg-nm32-ps4-p3`.

## Giám sát mặt nạ ngoài hộp (`outside_box:`)

**Lúc train:**
- Loss mặt nạ tính trong hộp nhãn giãn `train_expand` (1,25) lần quanh tâm, kẹp trong lưới.
- Mẫu số vẫn là diện tích hộp nhãn gốc. Phần trong hộp vì thế giữ y như mô hình gốc; vành mới cộng thêm, với tán bên cạnh và đất là nhãn âm.
- Hộp, điểm, prototype, hệ số và bộ gán nhãn không đổi. Hệ số 1 trùng từng số với loss gốc (test).

**Lúc chấm:**
- Bộ chấm của benchmark đã có `crop_expand`, và vốn lấy hộp từ mặt nạ.
- Config chấm `yolo26s-ka.yaml` đặt `crop_expand: 1.1` (`infer_expand`).

**Nhật ký:** `loss_outside_box.csv` ghi mỗi epoch phần loss trong hộp, phần vành, và tỉ lệ ô trong vành còn logit dương (`_leak`). `_leak` giảm dần là dấu hiệu giám sát mới có tác dụng.

## Chạy

Từ gốc repo, env `cofseg`:

```bash
python distill/train.py --config distill/configs/train/tro.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/tro.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

python distill/train.py --config distill/configs/train/tro-025.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/tro-025.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

python distill/train.py --config distill/configs/train/yolo26s-ka.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/yolo26s-ka.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0
```

**Kiểm trước khi chạy:** `--print-config` in tham số cuối cùng mà không train.

**Nơi lưu kết quả:** lượt nào của `distill/` cũng ghi vào thư mục kết quả riêng (`distill_seed<seed>`), không ghi lẫn vào thư mục của benchmark. Tên run lấy theo config (`tro-seg`, `tro-seg-w0.25`, `yolo26s-seg-ka`). Các khối `student`, `outside_box` cùng tên gốc được ghi vào `summary.json` (khoá `distill`).

**`evaluate.py` đối chiếu trước khi chấm:**
- Tên config chấm phải trùng tên gốc của lượt train. Gõ `--name` thì bỏ qua bước này.
- Lượt train có giám sát ngoài hộp thì `crop_expand` phải đúng `infer_expand`.

## Test

```bash
python -m pytest distill/tests
```

Mất khoảng nửa phút trên CPU. Phép đo GFLOPs cần `thop`, thiếu thì test đó tự bỏ qua.

## Lưu ý

- **Chấm trò bằng `distill/evaluate.py`, không bằng `benchmark/yolo/evaluate.py`.**
  - Checkpoint của trò chứa lớp của gói `distill`.
  - Ở nơi không import được `distill`, ultralytics coi đó là gói thiếu và tự `pip install distill` từ PyPI — một gói lạ. `distill/__init__.py` đặt `YOLO_AUTOINSTALL=false` để chặn việc này.
  - Checkpoint của giám sát ngoài hộp là SegmentationModel chuẩn, nạp được ở mọi nơi.
- **Đoạn chép:** `Proto26S8.forward` và `Segment26Student.forward` chép từ ultralytics 8.4.143. Trainer dừng nếu phiên bản khác.
- **Loss của lượt val lúc train vẫn là loss gốc.** Model EMA tự dựng hàm loss của nó. Val chỉ dùng để chọn `best.pt` theo mask AP, nên không ảnh hưởng.
