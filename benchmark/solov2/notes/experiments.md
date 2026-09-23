# Nhật ký thí nghiệm — SOLOv2 R50-FPN

Người ghi: PhuongQuynh. Mỗi lần chạy thật một dòng, ghi ngay sau khi chạy xong —
ba tuần sau không ai nhớ vì sao lần đó đặt batch 2.

| Ngày | Fold | Cấu hình khác mặc định | mAP | Boundary AP | Boundary IoU | Nhận xét |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

Cột "cấu hình khác mặc định" ghi đúng phần ghi đè, ví dụ `--imgsz 2048 --batch 2`
hoặc `--set train.lr=0.005`; để trống nghĩa là chạy y như config trong
`configs/`. Số lấy từ `runs/eval/<...>/metrics.json` sau khi chấm bằng
`benchmark/solov2/evaluate.py`, không lấy số detectron2/mmdet in ra lúc train (khác
cách tính Boundary).

## Ghi chú dài

Chỗ viết những thứ không nhét vừa một ô: vì sao đổi tham số, lỗi gặp phải và
cách sửa, ảnh chụp trường hợp model sai. Một mục một lần, mới nhất lên đầu.

### 2026-09-22 — khởi tạo
Chưa chạy lần nào.
