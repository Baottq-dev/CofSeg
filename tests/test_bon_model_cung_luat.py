"""Bốn model phải chạy cùng một luật thì bảng so sánh mới đo được cơ chế.

Tăng cường, ngân sách epoch và cách chọn checkpoint đều là thứ ảnh hưởng tới
kết quả mà KHÔNG phải kiến trúc. Lệch một trong ba là bảng đo nhầm thứ khác.

Hai chênh lệch còn lại là có chủ đích và không bỏ được, ghi ở đây để không ai
"sửa" nhầm:

- Mask2Former giữ mapper LSJ của recipe gốc (co giãn 0.1-2.0 rồi cắt ô vuông).
  Bỏ nó là bỏ recipe của tác giả.
- Hai model detectron2 có xoay bội 90 độ, SOLOv2 và YOLO không: mmdet không có
  transform xoay cho mask + box, còn ultralytics chỉ xoay góc bất kỳ (có nội
  suy), không phải cùng một phép biến đổi.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo11")


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _train_block(model: str, name: str) -> dict:
    return _yaml(BENCH / model / "configs" / "train" / name).get("train") or {}


# ------------------------------------------------------- tăng cường giống nhau
def test_yolo_khong_co_tang_cuong_rieng_no_moi_co():
    """`degrees` và `copy_paste` chỉ YOLO có.

    Bật chúng là cho một model thêm dữ liệu huấn luyện mà ba model kia không
    được hưởng, rồi ghi chênh lệch đó vào cột "kiến trúc".
    """
    t = _train_block("yolo11", "yolo26s.yaml")
    assert float(t.get("degrees", 0)) == 0.0, "degrees phải 0: ba model kia không xoay góc bất kỳ"
    assert float(t.get("copy_paste", 0)) == 0.0, "copy_paste phải 0: chỉ YOLO có"


@pytest.mark.parametrize("model,cfg", [
    ("maskrcnn", "_base_d2.yaml"), ("mask2former", "_base_d2.yaml"),
    ("solov2", "solov2_r50_mm.yaml"), ("yolo11", "yolo26s.yaml")])
def test_lat_ngang_doc_giong_nhau_o_ca_bon(model, cfg):
    """Lật ngang và lật dọc là mẫu số chung: bốn model đều phải có, cùng 0.5."""
    t = _train_block(model, cfg)
    assert float(t["fliplr"]) == 0.5, f"{model}: fliplr lệch"
    assert float(t["flipud"]) == 0.5, f"{model}: flipud lệch"


# --------------------------------------------------- cùng ngân sách, cùng luật
def test_yolo_khong_dung_som():
    """Dừng ở epoch 40 rồi so với model chạy đủ 100 là so hai ngân sách."""
    t = _train_block("yolo11", "yolo26s.yaml")
    assert int(t.get("patience", 0)) == 0, "patience phải 0 (tắt dừng sớm)"


def test_yolo_chon_best_theo_mask_ap():
    """Ultralytics chọn best.pt theo `seg.fitness() + box.fitness()`, tức trộn
    cả chỉ số HỘP vào. Ba model kia chọn thuần mask AP (`segm/AP`,
    `coco/segm_mAP`), nên trainer phải giữ thêm một bản theo đúng cột đó."""
    src = (BENCH / "yolo11" / "cofseg" / "training" / "yolo.py").read_text(encoding="utf-8")
    assert 'FITNESS_KEY = "metrics/mAP50-95(M)"' in src, "thiếu cột chọn best"
    assert "add_callback" in src and "on_fit_epoch_end" in src, \
        "thiếu callback giữ best theo mask AP"
    assert "(M)" in src and "mAP50-95(B)" not in src, "không được chọn best theo chỉ số hộp"


@pytest.mark.parametrize("model", MODELS)
def test_best_nam_cung_mot_cho_o_ca_bon(model):
    """`run_dir/weights/best.*` — để lệnh chấm viết giống nhau cho cả bốn."""
    src_dir = BENCH / model / "cofseg" / "training"
    src = "\n".join(p.read_text(encoding="utf-8") for p in src_dir.glob("*.py"))
    assert 'run_dir / "weights"' in src, f"{model}: best không về run_dir/weights"


# ------------------------------------------------ val không chạy ở batch 1 nữa
@pytest.mark.parametrize("model,mod", [
    ("maskrcnn", "detectron2"), ("mask2former", "detectron2"), ("solov2", "mmdet")])
def test_val_theo_batch_cua_train(model, mod):
    """Mặc định của cả detectron2 lẫn mmdet là batch 1 khi chấm.

    Đo trên máy thuê: train batch 16 dùng 12.7 GB, val batch 1 dùng 2.5 GB và
    tốn GẤP ĐÔI thời gian train dù ít hơn bốn lần số ảnh. Suy luận không giữ
    đồ thị cho backward nên batch của train luôn vừa.
    """
    src = (BENCH / model / "cofseg" / "training" / f"{mod}.py").read_text(encoding="utf-8")
    assert '"val_batch": None' in src, f"{model}: thiếu tham số val_batch"
    assert 'a.get("val_batch") or a["batch"]' in src or \
           'args.get("val_batch") or batch' in src, \
        f"{model}: val_batch không rơi về batch của train"


def test_val_cua_mmdet_chay_amp_nhu_train():
    """AmpOptimWrapper chỉ bọc bước tối ưu hoá; vòng val vẫn FP32 nếu không
    khai fp16 cho ValLoop."""
    src = (BENCH / "solov2" / "cofseg" / "training" / "mmdet.py").read_text(encoding="utf-8")
    assert 'val_cfg=dict(type="ValLoop", fp16=bool(args["amp"]))' in src, \
        "ValLoop không theo cờ amp của train"


def test_d2_khong_con_de_batch_cham_mac_dinh():
    """detectron2 phải override build_test_loader; mặc định của nó là 1."""
    for model in ("maskrcnn", "mask2former"):
        src = (BENCH / model / "cofseg" / "training" / "detectron2.py").read_text(encoding="utf-8")
        assert "def build_test_loader" in src, f"{model}: không override loader chấm"
        assert "batch_size=val_batch" in src, f"{model}: loader chấm không nhận val_batch"
