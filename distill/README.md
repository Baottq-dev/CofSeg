# distill/ — cải tiến YOLO26-seg cho CoffeeSeg

Thư mục này chứa mọi phần cải tiến của đợt nghiên cứu Q1. Hiện có:
- **trò nhẹ** (YOLO26-seg độ rộng 0,375 hoặc 0,25 với ba thay đổi ở đầu; đợt 1 chọn 0,25);
- **giám sát mặt nạ ngoài hộp** (K-A);
- **chắt lọc tri thức** từ ba thầy seed 0 cùng fold: mặt nạ từ mô hình gốc YOLO26s, SOLOv2 và Mask2Former, đặc trưng từ YOLO26s.

Sau này thêm SOLOv2 + 8 phép và đo Jetson.

`benchmark/` chỉ giữ các model đem so, chạy công thức chuẩn. Kế hoạch các đợt chạy, cổng của từng đợt và lệnh molab nằm trong tài liệu nghiên cứu của nhóm, ngoài git.

## Quan hệ với benchmark/

- **Dùng lại, không sửa, không chép:**
  - `distill/` dùng phần train, bộ đọc dữ liệu và bộ chấm của `benchmark/yolo/cofseg`.
  - `distill/train.py` và `distill/evaluate.py` nạp gói `distill`, rồi chạy nguyên `benchmark/yolo/train.py` và `benchmark/yolo/evaluate.py`.
  - Mọi cờ của hai script đó giữ nguyên nghĩa.
- **Một chiều:** `benchmark/` không bao giờ import `distill/`.
- **Cùng bộ chấm:** số của `distill/` so thẳng được với bảng benchmark.
- **Trainer riêng `distill_yolo`** (`trainer.py`) kế thừa `YoloTrainer` của benchmark. Nó thêm ba khối config:
  - `student:` dựng trò thay cho model của config;
  - `outside_box:` thay loss mặt nạ;
  - `kd:` chắt lọc: dataset train mang đích mặt nạ của thầy, loss mặt nạ học đích trộn, loss PKD trên đặc trưng.

  Mọi thứ khác giữ nguyên: mặc định cho ảnh nadir, chọn `best.pt` theo mask AP, `summary.json`, `--aug`.

## File

| File | Nội dung |
|---|---|
| `__init__.py` | Đặt `benchmark/yolo` lên đầu `sys.path` cho `import cofseg`; tắt tự cài gói của ultralytics (xem mục Lưu ý) |
| `blocks.py` | Kiểm tham số, đặt nhãn tên run, mô tả cho run.log của ba khối `student`, `outside_box`, `kd`; không import torch |
| `student.py` | Trò: `Proto26S8`, `Segment26Student`, dựng từ yaml YOLO26-seg, lấy trọng số từ checkpoint COCO (chép nguyên hoặc chọn đều) |
| `outside_box.py` | Loss mặt nạ trong hộp nhãn giãn, nhật ký epoch `loss_outside_box.csv` |
| `teachers.py` | Bảng ba thầy (thư mục benchmark, config chấm, tên run, file trọng số); tìm trọng số và lượt chấm của thầy theo fold |
| `teacher_infer.py` | Pha 0: ba thầy suy luận trên ảnh train của một fold bằng `evaluate.py --split train` của benchmark |
| `teacher_masks.py` | Ghép dự đoán của thầy với tán nhãn, ghi đích mặt nạ (ảnh 4 kênh) và `meta.json` |
| `kd_data.py` | Dataset train mang đích thầy qua các phép biến đổi của ultralytics; với `--aug ag1`, ghép với `AugDataset` của benchmark và mang đích theo phép chuyển vị |
| `kd_loss.py` | Loss mặt nạ với đích trộn, PKD trên đặc trưng, lớp 1×1 khớp kênh, nhật ký epoch `loss_kd.csv` |
| `trainer.py` | Trainer `distill_yolo`; lớp SegmentationTrainer dựng trò, gắn loss, dataset và thầy của chắt lọc; ghép được với lớp của `--aug`; chắt lọc mặt nạ chỉ ghép với `ag1` |
| `train.py`, `evaluate.py` | Lệnh train và chấm (chạy từ gốc repo) |
| `configs/train/tro.yaml`, `configs/eval/tro.yaml` | Trò 0,375, lấy trọng số từ YOLO26s-seg COCO, tên run `tro-seg` |
| `configs/train/tro-025.yaml`, `configs/eval/tro-025.yaml` | Trò 0,25, lấy trọng số từ YOLO26n-seg COCO, tên run `tro-seg-w0.25` |
| `configs/train/yolo26s-ka.yaml`, `configs/eval/yolo26s-ka.yaml` | YOLO26s-seg + giám sát ngoài hộp, tên run `yolo26s-seg-ka`; lúc chấm cắt ×1,1 |
| `configs/train/tro-025-kd.yaml`, `configs/eval/tro-025-kd.yaml` | Trò 0,25 + chắt lọc (đợt 2), tên run `tro-seg-kd-w0.25` |
| `configs/eval/tro-025-kd-ag1.yaml` | Chấm trò 0,25 + chắt lọc train với `--aug ag1`, tên run `tro-seg-kd-w0.25-ag1`; train dùng `configs/train/tro-025-kd.yaml` kèm `--aug ag1` |
| `tests/` | 100 test, dựng model, ảnh, nhãn và dự đoán của thầy tại chỗ, không cần `data/` hay `weights/` |

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

## Chắt lọc (`kd:`)

Ba bước, theo thứ tự, cho từng fold:

1. **Thầy suy luận trên ảnh train** (`teacher_infer.py`; cần ultralytics, mmdetection và detectron2, tức molab).
   - Thầy là lượt train seed 0 của benchmark cùng fold: mô hình gốc YOLO26s, SOLOv2, Mask2Former.
   - Mỗi thầy là một lệnh `benchmark/<model>/evaluate.py --split train` với config chấm của benchmark, nên mặt nạ dùng để dạy đúng là mặt nạ đã chấm.
   - Thầy đã có lượt chấm train thì bỏ qua.
2. **Dựng đích mặt nạ** (`teacher_masks.py`; chỉ cần numpy, opencv, pycocotools):
   - Mỗi thầy ghép dự đoán với tán nhãn trong `labels/train/*.txt` theo IoU mặt nạ, một-một, IoU ≥ 0,5. Dự đoán không ghép được bị bỏ, vì dạy theo chúng tương đương thêm nhãn.
   - Đích của tán là trung bình mặt nạ của các thầy ghép được, thu về cỡ ảnh train bằng INTER_AREA. Mặt nạ thầy là nhị phân (bộ chấm), nên trong lòng tán đích là tỉ lệ thầy đồng ý, ở mép là tỉ lệ phủ. Tán không có thầy nào ghép được thì chỉ học nhãn.
   - `--mode best` lấy thầy khớp nhãn nhất; `--teachers yolo26s` cho bản một thầy. Đây là các dòng ablation thầy.
   - Ghi `<out>/<fold>/maps/<ảnh>.png` và `meta.json`. Ảnh đích có 4 kênh: mã tán 1, giá trị 1, mã tán 2, giá trị 2. Mã tán là số dòng trong file nhãn; 255 là không có tán.
3. **Train** với khối `kd:`:
   - `kd_data.py` đặt ảnh đích vào `labels["semantic_mask"]`. Ultralytics 8.4.143 tự đưa khoá này qua mọi phép hình học (co giãn, dịch, lật, letterbox) với nội suy gần nhất. Mã tán đi theo tán qua cột lớp. `KDFormat` giải mã và theo đúng thứ tự Format sắp tán theo diện tích.
   - Loss mặt nạ: đích của tán có thầy là (1 − α)·nhãn + α·thầy, vùng tính là hộp nhãn như gốc. BCE tuyến tính theo đích, nên tổng trọng số loss mặt nạ không đổi; α = 0 trùng từng số với loss gốc.
   - Đặc trưng: YOLO26s thầy cùng fold, đóng băng, chạy trên đúng batch đã biến đổi.
     - Ba cặp đặc trưng: P4, P5 (đầu vào của đầu) và đầu ra `feat_fuse` của Proto.
     - Lớp 1×1 (`kd_adapter`) đưa trò về số kênh của thầy.
     - Loss PKD: chuẩn hoá từng kênh rồi MSE / 2, tức 1 − tương quan Pearson. Trong hộp nhãn và ngoài hộp lấy trung bình riêng, ngoài hộp nhân `feat_bg`.
   - Sau khi train, `kd_adapter` được gỡ khỏi các file trong `<run>/weights`, nên `best.pt` chỉ còn đúng kiến trúc của trò.
   - Với `--aug ag1`, dataset train là lớp ghép của `KDDataset` và `AugDataset` của benchmark. Phép chuyển vị của ag1 không nằm trong ultralytics, nên `KDTranspose` bọc nó: ảnh bị chuyển vị thì ảnh đích cũng vậy. Phép bọc không rút thêm số ngẫu nhiên. Tên run có thêm hậu tố `-ag1`; lúc chấm dùng `configs/eval/tro-025-kd-ag1.yaml`.

| Khoá | Mặc định | Nghĩa |
|---|---|---|
| `targets` | — | Thư mục đích (`teacher_masks.py --out`); null = không chắt lọc mặt nạ |
| `mask_alpha` | 0.5 | α của đích trộn |
| `mask_expand` | 1.0 | Vùng tính loss mặt nạ: hộp nhãn giãn ngần này lần |
| `teacher_runs` | — | Thư mục runs có YOLO26s seed 0 của fold; null = không chắt lọc đặc trưng |
| `feat_weight` | 6.0 | Hệ số của loss PKD (giá trị của bài PKD) |
| `feat_bg` | 0.5 | Trọng số phần ngoài hộp nhãn |

**Nhật ký:**
- `loss_kd.csv` ghi mỗi epoch:
  - số neo dương của loss mặt nạ và số neo có đích thầy, cho từng đầu one2one, one2many;
  - loss PKD của từng cặp đặc trưng.
- Nhật ký train của ultralytics có thêm cột `kd_loss`. Ở val cột này là 0, vì model EMA giữ loss gốc.

**Chưa ghép:** trainer dừng nếu gặp một trong hai tổ hợp sau:
- `kd` với `outside_box`: cả hai cùng thay loss mặt nạ;
- chắt lọc mặt nạ với phép dán `--aug pa1` hay `ag4`: tán dán đè lên tán có sẵn mà ảnh đích không biết. Chỉ chắt lọc đặc trưng thì phép nào cũng được.

## Chạy

Từ gốc repo, env `cofseg`:

```bash
python distill/train.py --config distill/configs/train/tro.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/tro.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

python distill/train.py --config distill/configs/train/tro-025.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/tro-025.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

python distill/train.py --config distill/configs/train/yolo26s-ka.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/yolo26s-ka.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

# chắt lọc, cho từng fold: thầy suy luận, dựng đích, rồi train
python distill/teacher_infer.py --data data/export/field/f1 --weights-from runs/train_2 --runs runs/distill_teachers
python distill/teacher_masks.py --data data/export/field/f1 --evals runs/distill_teachers/eval --out runs/distill_teachers/targets/mean3
python distill/train.py --config distill/configs/train/tro-025-kd.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0
python distill/evaluate.py --config distill/configs/eval/tro-025-kd.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

# chắt lọc + 8 phép xoay/lật (--aug ag1), tên run tro-seg-kd-w0.25-ag1
python distill/train.py --config distill/configs/train/tro-025-kd.yaml --data data/export/field/f1 --runs distill_seed0 --seed 0 --aug ag1
python distill/evaluate.py --config distill/configs/eval/tro-025-kd-ag1.yaml --weights distill_seed0/train/<run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0
```

**Kiểm trước khi chạy:** `--print-config` in tham số cuối cùng mà không train.

**Nơi lưu kết quả:** lượt nào của `distill/` cũng ghi vào thư mục kết quả riêng (`distill_seed<seed>`), không ghi lẫn vào thư mục của benchmark. Tên run lấy theo config (`tro-seg`, `tro-seg-w0.25`, `yolo26s-seg-ka`, `tro-seg-kd-w0.25`). Các khối `student`, `outside_box`, `kd` cùng tên gốc được ghi vào `summary.json` (khoá `distill`).

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
- **Đoạn chép:** trainer dừng nếu ultralytics khác phiên bản 8.4.143, vì các đoạn sau chép từ bản đó:
  - `Proto26S8.forward` và `Segment26Student.forward`;
  - vòng lặp `calculate_segmentation_loss` (`kd_loss.py`);
  - `_format_segments` (`kd_data.py`).
- **Loss của lượt val lúc train vẫn là loss gốc.** Model EMA tự dựng hàm loss của nó. Val chỉ dùng để chọn `best.pt` theo mask AP, nên không ảnh hưởng.
