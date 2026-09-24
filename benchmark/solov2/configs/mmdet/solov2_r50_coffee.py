# SOLOv2 R50-FPN cho tán cà phê — phần GHI ĐÈ lên config zoo của mmdet
# (solov2/solov2_r50_fpn_ms-3x_coco.py, đóng gói sẵn trong gói mmdet).
#
# Đây không phải config mmdet hoàn chỉnh: cofseg/training/mmdet.py nạp
# config zoo rồi gộp file này lên, sau đó tự đặt dữ liệu (fold), độ phân giải,
# lịch học, checkpoint theo khối `train:` của configs/train/solov2_r50_mm.yaml.
# Không dùng `_base_ = ['mmdet::...']` vì cú pháp đó cần pkg_resources, thứ
# setuptools 82 đã bỏ (81.0.0 còn, 82.0.1 hết). requirements.txt ghim
# setuptools < 82 cho detectron2, nhưng đường nạp config ở đây vẫn không dựa
# vào cái ghim đó — SOLOv2 cài được mà không cần detectron2.
#
# Chỉ để ở đây những gì thuộc về MODEL mà recipe COCO không hợp với task.
model = dict(
    # Một lớp: tán cà phê.
    mask_head=dict(num_classes=1),
    # Ngưỡng điểm và số vật thể tối đa lúc val/test; trainer ghi đè theo
    # train.val_conf / train.max_det. mask_thr 0.5 giữ như recipe.
    test_cfg=dict(score_thr=0.05, max_per_img=100),
)
