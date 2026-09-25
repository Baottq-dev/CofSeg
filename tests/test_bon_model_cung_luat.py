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
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo")


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
    t = _train_block("yolo", "yolo26s.yaml")
    assert float(t.get("degrees", 0)) == 0.0, "degrees phải 0: ba model kia không xoay góc bất kỳ"
    assert float(t.get("copy_paste", 0)) == 0.0, "copy_paste phải 0: chỉ YOLO có"


@pytest.mark.parametrize("model,cfg", [
    ("maskrcnn", "_base_d2.yaml"), ("mask2former", "_base_d2.yaml"),
    ("solov2", "solov2_r50_mm.yaml"), ("yolo", "yolo26s.yaml")])
def test_lat_ngang_doc_giong_nhau_o_ca_bon(model, cfg):
    """Lật ngang và lật dọc là mẫu số chung: bốn model đều phải có, cùng 0.5."""
    t = _train_block(model, cfg)
    assert float(t["fliplr"]) == 0.5, f"{model}: fliplr lệch"
    assert float(t["flipud"]) == 0.5, f"{model}: flipud lệch"


# --------------------------------------------------- cùng ngân sách, cùng luật
def test_yolo_khong_dung_som():
    """Dừng ở epoch 40 rồi so với model chạy đủ 100 là so hai ngân sách."""
    t = _train_block("yolo", "yolo26s.yaml")
    assert int(t.get("patience", 0)) == 0, "patience phải 0 (tắt dừng sớm)"


def test_yolo_chon_best_theo_mask_ap():
    """Ultralytics chọn best.pt theo `seg.fitness() + box.fitness()`, tức trộn
    cả chỉ số HỘP vào. Ba model kia chọn thuần mask AP (`segm/AP`,
    `coco/segm_mAP`), nên trainer phải giữ thêm một bản theo đúng cột đó."""
    src = (BENCH / "yolo" / "cofseg" / "training" / "yolo.py").read_text(encoding="utf-8")
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


# -------------------------------- bộ mặc định, đo từ chính bộ dữ liệu này
#: Ba số này phải giống nhau ở cả bốn model, nếu không bảng đo nhầm thứ khác.
CHUNG = {"imgsz": 1024, "batch": 16}


@pytest.mark.parametrize("model,cfg", [
    ("maskrcnn", "_base_d2.yaml"), ("mask2former", "_base_d2.yaml"),
    ("solov2", "solov2_r50_mm.yaml"), ("yolo", "yolo26s.yaml")])
def test_bon_model_cung_imgsz_va_batch(model, cfg):
    """imgsz và batch không phải lựa chọn của từng người.

    imgsz quyết định cỡ tán mà model nhìn thấy: tán trung vị 324 px ở ảnh gốc
    2560x1440 còn 129 px ở imgsz 1024. Một model chạy 1536 là nó nhìn tán to
    hơn 1.5 lần, và chênh lệch đó sẽ bị ghi vào cột "kiến trúc".

    batch 16 đo thật trên RTX 5090: 12.7 GB ở imgsz 1024. Mặc định cũ là 4
    (hai model detectron2/mmdet) và 2 (YOLO) — con số của card 8 GB ở nhà.
    """
    t = _train_block(model, cfg)
    for k, v in CHUNG.items():
        assert int(t[k]) == v, f"{model}: {k}={t[k]}, phải {v}"


@pytest.mark.parametrize("model,cfg,epochs", [
    ("maskrcnn", "_base_d2.yaml", 50), ("solov2", "solov2_r50_mm.yaml", 50),
    ("yolo", "yolo26s.yaml", 50),
    ("mask2former", "mask2former_r50_d2.yaml", 100)])
def test_ngan_sach_epoch(model, cfg, epochs):
    """50 epoch cho ba model; Mask2Former 100 vì query hội tụ chậm hơn — đó là
    chênh lệch có chủ đích, ghi trong config của nó."""
    t = _train_block(model, cfg)
    assert int(t["epochs"]) == epochs, f"{model}: epochs={t['epochs']}, phải {epochs}"


@pytest.mark.parametrize("model,cfg", [
    ("maskrcnn", "_base_d2.yaml"), ("mask2former", "_base_d2.yaml"),
    ("solov2", "solov2_r50_mm.yaml")])
def test_warmup_theo_phan_cua_lich(model, cfg):
    """200 vòng cứng là 12.5% lịch khi batch 16 (500 ảnh -> 32 vòng/epoch,
    50 epoch -> 1600 vòng), trong khi recipe COCO warmup chưa tới 1%. Thấy rõ
    ở lượt khói 28/09: hết 3 epoch lr vẫn chưa lên tới giá trị đã đặt.

    Ghi theo PHẦN thì đổi batch không phải tính lại.
    """
    t = _train_block(model, cfg)
    w = float(t["warmup_iters"])
    assert 0 < w < 1, f"{model}: warmup_iters={w} là số vòng cứng, phải là phần của lịch"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_dau_mask_cua_rcnn_khai_ro_do_phan_giai(model):
    """Đầu mask của R-CNN dự đoán ở 28x28 rồi phóng lên bbox, nên với tán
    trung vị 129 px ở imgsz 1024 thì mỗi ô nuốt 4.6 px. Đó là trần đường biên
    của model, KHÔNG phải imgsz — một con số phải nhìn thấy được, không nằm
    ẩn trong recipe."""
    src = (BENCH / model / "cofseg" / "training" / "detectron2.py").read_text(encoding="utf-8")
    assert '"mask_resolution"' in src, f"{model}: độ phân giải đầu mask không lộ ra"
    assert "ROI_MASK_HEAD.POOLER_RESOLUTION" in src, f"{model}: không nối vào config d2"


# ------------------------------------- thư mục run không được phình vô hạn
@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_d2_khong_giu_moi_checkpoint(model):
    """DefaultTrainer dựng PeriodicCheckpointer KHÔNG truyền max_to_keep, nên
    mặc định của detectron2 là giữ hết: một file mỗi epoch.

    Mask R-CNN ~350 MB mỗi file (trọng số + buffer momentum) x 50 epoch là
    17 GB; Mask2Former ~530 MB (AdamW giữ hai moment) x 100 epoch là 53 GB.
    Nhân 18 lượt fold thì không đĩa nào chịu nổi.
    """
    src = (BENCH / model / "cofseg" / "training" / "detectron2.py").read_text(encoding="utf-8")
    assert '"keep_ckpts"' in src, f"{model}: không khai số checkpoint giữ lại"
    assert "max_to_keep=keep_ckpts" in src,         f"{model}: PeriodicCheckpointer vẫn dùng mặc định giữ hết"


def test_solov2_da_gioi_han_checkpoint():
    """mmdet có sẵn max_keep_ckpts; chỉ cần không ai gỡ nó ra."""
    src = (BENCH / "solov2" / "cofseg" / "training" / "mmdet.py").read_text(encoding="utf-8")
    assert "max_keep_ckpts=1" in src, "solov2: CheckpointHook không còn giới hạn"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_giu_du_de_resume_duoc(model):
    """Giữ ÍT NHẤT một checkpoint định kỳ, không phải không giữ cái nào.

    `model_final.pth` — và `weights/last.pth` mà trainer chép ra từ nó — chỉ
    có khi train chạy hết. Lượt chạy bị ngắt giữa chừng chỉ còn checkpoint
    định kỳ gần nhất và file `last_checkpoint` trỏ vào nó.
    """
    src = (BENCH / model / "cofseg" / "training" / "detectron2.py").read_text(encoding="utf-8")
    assert "max(1, int(a[" in src and "keep_ckpts" in src,         f"{model}: keep_ckpts phải được kẹp về tối thiểu 1"


# ------------------------------------------------------------- đổi backbone
def _bang_backbone(model: str) -> dict:
    """BACKBONES của trainer trong thư mục đó, đọc bằng AST để khỏi cần
    detectron2/mmcv — hai gói không dựng được trên máy phát triển."""
    import ast

    rel = "training/mmdet.py" if model == "solov2" else "training/detectron2.py"
    ten = "ZOO_BACKBONES" if model == "solov2" else "BACKBONES"
    src = (BENCH / model / "cofseg" / rel).read_text(encoding="utf-8")
    cay = ast.parse(src)
    # mmdet ghép URL từ hằng U ở đầu file; gom mọi hằng chuỗi cấp module lại
    # để _hang() tra được, thay vì trả về tên biến.
    hang = {}
    for node in cay.body:
        d = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(d, ast.Name) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            hang[d.id] = node.value.value
    for node in ast.walk(cay):
        dich = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(dich, ast.Name) and dich.id == ten:
            # Các giá trị là dict(...) nên literal_eval không nuốt được; dựng tay.
            ra = {}
            for arch, bang in zip(node.value.keys, node.value.values):
                o = {}
                for k, v in zip(bang.keys, bang.values):
                    o[k.value] = {kw.arg: _hang(kw.value, hang) for kw in v.keywords}
                ra[arch.value] = o
            return ra
    raise AssertionError(f"{model}: không thấy {ten}")


def _hang(node, hang: dict):
    """Chuỗi ghép nhiều dòng ("a" "b") và ghép bằng + với hằng cấp module."""
    import ast

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return hang.get(node.id, node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _hang(node.left, hang) + _hang(node.right, hang)
    return ast.unparse(node)


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former", "solov2"))
def test_bang_backbone_co_r50_va_mo_ta(model):
    """R50 phải còn là một lựa chọn: nó là mốc mà ba model chia nhau, bỏ nó ra
    khỏi bảng là không quay về mốc được nữa. Mỗi ô phải có `note` vì
    --list-backbones in cột đó để người chạy chọn bằng số."""
    bang = _bang_backbone(model)
    assert bang, f"{model}: bảng backbone rỗng"
    for arch, o in bang.items():
        assert o, f"{model}/{arch}: không có backbone nào"
        for ten, spec in o.items():
            assert spec.get("config") or spec.get("lazy"), \
                f"{model}/{arch}/{ten}: không có config yaml lẫn LazyConfig"
            assert not (spec.get("config") and spec.get("lazy")), \
                f"{model}/{arch}/{ten}: khai cả hai đường dựng model"
            assert spec.get("note"), f"{model}/{arch}/{ten}: thiếu note cho --list-backbones"
    goc = "r50" if model != "maskrcnn" else "r50"
    assert goc in bang[list(bang)[0]], f"{model}: bảng không còn {goc}"


def test_backbone_detectron2_dung_dung_duong_dung_model():
    """Hai đường dựng model, và mỗi ô phải nói rõ nó đi đường nào.

    `config` là yaml -> get_cfg() + merge_from_file, DefaultTrainer dựng model.
    `lazy` là LazyConfig .py -> trainer instantiate model rồi giao lại.
    Nhầm đường thì lỗi chỉ hiện ra trên máy thuê: merge_from_file gặp file .py
    là gãy ngay, còn instantiate một yaml thì không có gì để instantiate.
    """
    for model in ("maskrcnn", "mask2former"):
        for arch, o in _bang_backbone(model).items():
            for ten, spec in o.items():
                if spec.get("lazy"):
                    assert spec["lazy"].endswith(".py"), f"{model}/{arch}/{ten}"
                    # Recipe của bản lazy khác hẳn R-CNN; thiếu một mảnh là
                    # lặng lẽ train ViT bằng SGD 0.02 hoặc bằng kênh BGR.
                    assert {"vit", "optim", "input_format", "checkpoint"} <= set(spec), \
                        f"{model}/{arch}/{ten}: ô LazyConfig thiếu phần recipe"
                else:
                    assert spec["config"].endswith(".yaml"), \
                        f"{model}/{arch}/{ten}: {spec['config']} không phải config yaml"


def test_backbone_mmdet_deu_co_trong_so_coco():
    """mmdet có config không kèm checkpoint (R101 không-DCN); chọn phải bản đó
    là lặng lẽ khởi đầu từ ImageNet thay vì COCO."""
    for arch, o in _bang_backbone("solov2").items():
        for ten, spec in o.items():
            assert spec.get("checkpoint", "").startswith("https://"), \
                f"solov2/{arch}/{ten}: không có trọng số COCO"


def test_backbone_zoo_cua_detectron2_ton_tai():
    """Đường dẫn trong bảng phải là đường dẫn model_zoo có thật KÈM checkpoint.
    Cần detectron2 nên chỉ chạy trên máy thuê; ở nhà thì bỏ qua."""
    model_zoo = pytest.importorskip("detectron2.model_zoo")
    for arch, o in _bang_backbone("maskrcnn").items():
        for ten, spec in o.items():
            model_zoo.get_config_file(spec["config"])
            assert model_zoo.get_checkpoint_url(spec["config"]), f"maskrcnn/{arch}/{ten}"
