"""Lệnh trong README phải chạy được, không phải gần đúng.

Tài liệu chép-dán là loại dễ mục nhất: đổi tên một tham số trong trainer thì
lệnh trong README thành sai mà không có gì báo. Test ở đây đối chiếu từng cờ
trong "Lệnh đầy đủ" với danh sách tham số THẬT của trainer và của lớp model.

Đọc mặc định bằng `ast` chứ không import: hai trainer cần detectron2/mmcv,
thứ không có trên máy phát triển.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"

#: model -> (file trainer, tên hằng mặc định, file model, tên lớp model)
CAI_DAT = {
    "maskrcnn": ("training/detectron2.py", "D2_DEFAULTS", "models/detectron2.py", "Detectron2Model"),
    "mask2former": ("training/detectron2.py", "D2_DEFAULTS", "models/detectron2.py", "Detectron2Model"),
    "solov2": ("training/mmdet.py", "MM_DEFAULTS", "models/mmdet.py", "MMDetModel"),
    "yolo": ("training/yolo.py", None, "models/yolo_seg.py", "YoloSegModel"),
}
MODELS = tuple(CAI_DAT)

#: khối `eval:` của config chấm — xem EVAL_DEFAULTS trong evaluate.py
EVAL_KEYS = {"iou_thr", "band_ratio", "nsd_tau", "dilation_ratio"}

#: cờ của chính train.py, không phải của trainer
CUA_TRAIN_PY = {"config", "data", "runs", "name", "set", "probe",
                "print-config", "list-params", "log-clean"}

#: khoá khối `model:` trong config TRAIN (chọn kiến trúc và trọng số khởi đầu).
#: Khác với đối số lớp model, thứ khối `model:` của config CHẤM nhận.
KHOA_MODEL_CONFIG = {"arch", "repo", "config", "config_file", "weights"}


def _khoi_lenh(muc: str, moc: str) -> str:
    """Khối ```bash chứa `moc`, cắt đúng hàng rào mở và đóng."""
    for khoi in re.findall(r"```bash\n(.*?)```", muc, re.S):
        if moc in khoi:
            return khoi
    raise AssertionError(f"không thấy khối lệnh chứa {moc!r}")


def _readme(model: str) -> str:
    return (BENCH / model / "README.md").read_text(encoding="utf-8")


def _muc(doc: str, tieu_de: str) -> str:
    """Nội dung một mục `###`, tới tiêu đề `##`/`###` kế tiếp."""
    i = doc.index(tieu_de)
    sau = doc[i + len(tieu_de):]
    ket = re.search(r"^#{2,3} ", sau, re.M)
    return sau[: ket.start()] if ket else sau


def _mac_dinh_trainer(model: str) -> set[str] | None:
    """Khoá của hằng mặc định, lấy bằng ast (không import trainer)."""
    rel, ten, _, _ = CAI_DAT[model]
    if ten is None:
        return None
    cay = ast.parse((BENCH / model / "cofseg" / rel).read_text(encoding="utf-8"))
    for node in cay.body:
        dich = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(dich, ast.Name) and dich.id == ten:
            return {k.value for k in node.value.keys}
    raise AssertionError(f"{model}: không thấy {ten}")


def _tham_so_model(model: str) -> set[str]:
    """Tên đối số của hàm khởi tạo lớp model."""
    _, _, rel, lop = CAI_DAT[model]
    cay = ast.parse((BENCH / model / "cofseg" / rel).read_text(encoding="utf-8"))
    for node in ast.walk(cay):
        if isinstance(node, ast.ClassDef) and node.name == lop:
            for con in node.body:
                if isinstance(con, ast.FunctionDef) and con.name == "__init__":
                    return {a.arg for a in con.args.args} - {"self"}
    raise AssertionError(f"{model}: không thấy {lop}.__init__")


def _co_trong_ultralytics(ten: str) -> bool:
    from ultralytics.cfg import get_cfg

    return ten in vars(get_cfg())


# --------------------------------------------------------------- có mục chưa
@pytest.mark.parametrize("model", MODELS)
def test_co_muc_lenh_day_du_va_doi_backbone(model):
    doc = _readme(model)
    assert "### Lệnh đầy đủ" in doc, f"{model}: thiếu mục lệnh đầy đủ"
    assert "### Đổi backbone" in doc, f"{model}: thiếu mục đổi backbone"


@pytest.mark.parametrize("model", MODELS)
def test_lenh_day_du_co_ca_train_va_eval(model):
    khoi = _muc(_readme(model), "### Lệnh đầy đủ")
    assert f"benchmark/{model}/train.py" in khoi, f"{model}: thiếu lệnh train"
    assert f"benchmark/{model}/evaluate.py" in khoi, f"{model}: thiếu lệnh chấm"
    assert "--data data/export/field/f1" in khoi, f"{model}: lệnh mẫu không chạy trên fold thật"


# ------------------------------------------------- mọi cờ đều là cờ có thật
@pytest.mark.parametrize("model", ("maskrcnn", "mask2former", "solov2"))
def test_co_train_deu_la_tham_so_that(model):
    """Cờ trong lệnh train phải có trong hằng mặc định của chính trainer đó."""
    khoi = _muc(_readme(model), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "train.py")
    hop_le = _mac_dinh_trainer(model) | CUA_TRAIN_PY
    for co in re.findall(r"--([a-z0-9_-]+)", lenh):
        assert co in hop_le, f"{model}: --{co} không phải tham số của trainer"


def test_co_train_cua_yolo_deu_la_tham_so_ultralytics():
    khoi = _muc(_readme("yolo"), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "train.py")
    for co in re.findall(r"--([a-z0-9_-]+)", lenh):
        assert co in CUA_TRAIN_PY or _co_trong_ultralytics(co), \
            f"yolo: --{co} không có trong cfg của ultralytics"


#: Cờ của evaluate.py không đi vào khối model: — script tự xử lý.
CUA_EVAL_PY = {"config", "data", "split", "limit", "runs", "name", "weights",
               "native", "native-data", "device", "batch", "file", "set"}
#: Cờ đi vào khối eval:, viết bằng gạch ngang.
CO_EVAL = {"iou-thr", "band-ratio", "nsd-tau", "dilation-ratio"}


@pytest.mark.parametrize("model", MODELS)
def test_lenh_cham_khong_con_dung_set(model):
    """Lệnh trong README phải là cờ thật. `--set` còn trong mã làm lối thoát,
    nhưng lệnh mẫu mà dùng nó thì người đọc sẽ học theo."""
    khoi = _muc(_readme(model), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "evaluate.py")
    assert "--set " not in lenh, f"{model}: lệnh chấm vẫn còn --set"


@pytest.mark.parametrize("model", MODELS)
def test_co_cham_deu_la_co_that(model):
    """Mọi cờ trong lệnh chấm phải là cờ của evaluate.py, khoá của khối eval:,
    hoặc đối số khởi tạo của chính lớp model."""
    khoi = _muc(_readme(model), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "evaluate.py")
    hop_le = _tham_so_model(model) | KHOA_MODEL_CONFIG | CUA_EVAL_PY | CO_EVAL
    for co in re.findall(r"--([a-z0-9_-]+)", lenh):
        assert co in hop_le, f"{model}: --{co} không phải cờ của evaluate.py"


@pytest.mark.parametrize("model", MODELS)
def test_bon_nguong_cham_van_duoc_ghi_ra(model):
    """Bốn khoá của khối eval: là định nghĩa phép đo; lệnh mẫu phải ghi chúng
    ra để ai đọc cũng thấy con số mình đang so là con số nào."""
    khoi = _muc(_readme(model), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "evaluate.py")
    thieu = [c for c in sorted(CO_EVAL) if f"--{c} " not in lenh]
    assert not thieu, f"{model}: lệnh chấm thiếu {thieu}"


def _backbone_cua(model: str, cfg_name: str) -> set:
    """Tên backbone hợp lệ, hỏi chính trainer của thư mục đó."""
    import importlib.util
    import sys

    d = BENCH / model
    sys.path.insert(0, str(d))
    try:
        for m in [k for k in sys.modules if k == "cofseg" or k.startswith("cofseg.")]:
            del sys.modules[m]
        spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cofseg"] = mod
        spec.loader.exec_module(mod)
        import cofseg.config as cfgmod
        import cofseg.training  # noqa: F401  đăng ký trainer
        from cofseg.registry import resolve

        c = cfgmod.load(d / "configs" / "train" / cfg_name)
        return set(resolve("trainer", c["trainer"]).backbones(c))
    finally:
        sys.path.remove(str(d))
        for m in [k for k in sys.modules if k == "cofseg" or k.startswith("cofseg.")]:
            del sys.modules[m]


# ------------------------------------------------------------ đổi backbone
@pytest.mark.parametrize("model", ("maskrcnn", "mask2former", "solov2"))
def test_doi_backbone_nhac_cai_gia_phai_tra(model):
    """Ba model dùng chung R50 nên chênh lệch quy về cơ chế. Đổi backbone MỘT
    model là mất tính chất đó, và README phải nói ra."""
    khoi = _muc(_readme(model), "### Đổi backbone")
    assert "R50" in khoi and "cơ chế" in khoi, \
        f"{model}: không nói vì sao ba model phải chung backbone"


def test_yolo_noi_ro_khong_doi_backbone_duoc():
    khoi = _muc(_readme("yolo"), "### Đổi backbone")
    assert "không đổi backbone được" in khoi, "yolo: phải nói rõ nó khác ba model kia"
    assert "yolo11m-seg.pt" in khoi, "yolo: thiếu cách đổi CỠ model"


def test_checkpoint_mask2former_khop_model_zoo():
    """URL trọng số trong bảng BACKBONES phải có thật trong MODEL_ZOO.md.

    Bảng nằm ở trainer chứ không ở README: URL là thứ mã dùng, để trong tài
    liệu thì hai bên trôi ra xa nhau lúc nào không biết.
    """
    zoo = BENCH / "mask2former" / "upstream" / "MODEL_ZOO.md"
    if not zoo.exists():
        pytest.skip("chưa init submodule")
    noi_dung = zoo.read_text(encoding="utf-8")
    src = (BENCH / "mask2former" / "cofseg" / "training" / "detectron2.py").read_text(
        encoding="utf-8")
    # Chỉ URL của Mask2Former: bảng còn chứa ViTDet, thuộc model zoo khác.
    urls = re.findall(r'"(https://dl\.fbaipublicfiles\.com/maskformer/\S+?)"\s*\n?\s*"?(\S*\.pkl)"',
                      src)
    urls = [a + b for a, b in urls]
    assert len(urls) >= 7, f"mask2former: bảng backbone chỉ có {len(urls)} trọng số"
    for u in urls:
        assert u.rsplit("/", 2)[-2] + "/" + u.rsplit("/", 1)[-1] in noi_dung, \
            f"mask2former: {u} không có trong MODEL_ZOO.md"


def test_checkpoint_solov2_la_ban_co_trong_so_that():
    """mmdet có config R101 không-DCN nhưng KHÔNG phát hành trọng số COCO cho
    nó; chọn nhầm bản đó là khởi đầu từ ImageNet."""
    src = (BENCH / "solov2" / "cofseg" / "training" / "mmdet.py").read_text(encoding="utf-8")
    assert "solov2_r101_dcn_fpn_3x_coco" in src, "solov2: thiếu trọng số R101-DCN"
    assert "solov2/solov2_r101_fpn_ms-3x_coco.py" not in src, \
        "solov2: R101 không-DCN không có trọng số COCO, không được đưa vào bảng"
    khoi = _muc(_readme("solov2"), "### Đổi backbone")
    assert "r101-dcn" in khoi, "solov2: README không nói cách đi lên R101"
    assert "không có R101 thường" in khoi, \
        "solov2: không cảnh báo bản R101 thiếu trọng số COCO"


@pytest.mark.parametrize("model,cfg", (
    ("maskrcnn", "maskrcnn_r50_d2.yaml"),
    ("mask2former", "mask2former_r50_d2.yaml"),
    ("solov2", "solov2_r50_mm.yaml"),
))
def test_lenh_doi_backbone_dung_co_that(model, cfg):
    """Mục đổi backbone phải dùng --backbone, và tên backbone phải có trong
    bảng của chính trainer đó."""
    khoi = _muc(_readme(model), "### Đổi backbone")
    assert "--set " not in khoi, f"{model}: mục đổi backbone vẫn còn --set"
    ten = re.findall(r"--backbone ([a-z0-9-]+)", khoi)
    assert ten, f"{model}: mục đổi backbone không có lệnh --backbone nào"
    co = _backbone_cua(model, cfg)
    for t in ten:
        assert t in co, f"{model}: --backbone {t} không có trong bảng ({sorted(co)})"
