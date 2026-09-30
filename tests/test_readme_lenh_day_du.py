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
    "yolo11": ("training/yolo.py", None, "models/yolo_seg.py", "YoloSegModel"),
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
    khoi = _muc(_readme("yolo11"), "### Lệnh đầy đủ")
    lenh = _khoi_lenh(khoi, "train.py")
    for co in re.findall(r"--([a-z0-9_-]+)", lenh):
        assert co in CUA_TRAIN_PY or _co_trong_ultralytics(co), \
            f"yolo11: --{co} không có trong cfg của ultralytics"


@pytest.mark.parametrize("model", MODELS)
def test_set_model_va_set_eval_deu_dung_khoa(model):
    """`--set model.k=` phải là đối số của lớp model; `--set eval.k=` phải
    thuộc bốn khoá định nghĩa phép đo."""
    doc = _readme(model)
    hop_le = _tham_so_model(model) | KHOA_MODEL_CONFIG
    for k in re.findall(r"--set model\.([a-z0-9_]+)=", doc):
        assert k in hop_le, f"{model}: model.{k} không phải đối số của lớp model"
    for k in re.findall(r"--set eval\.([a-z0-9_]+)=", doc):
        assert k in EVAL_KEYS, f"{model}: eval.{k} không phải khoá của khối eval"


# ------------------------------------------------------------ đổi backbone
@pytest.mark.parametrize("model", ("maskrcnn", "mask2former", "solov2"))
def test_doi_backbone_nhac_cai_gia_phai_tra(model):
    """Ba model dùng chung R50 nên chênh lệch quy về cơ chế. Đổi backbone MỘT
    model là mất tính chất đó, và README phải nói ra."""
    khoi = _muc(_readme(model), "### Đổi backbone")
    assert "R50" in khoi and "cơ chế" in khoi, \
        f"{model}: không nói vì sao ba model phải chung backbone"


def test_yolo_noi_ro_khong_doi_backbone_duoc():
    khoi = _muc(_readme("yolo11"), "### Đổi backbone")
    assert "không đổi backbone được" in khoi, "yolo11: phải nói rõ nó khác ba model kia"
    assert "yolo11m-seg.pt" in khoi, "yolo11: thiếu cách đổi CỠ model"


def test_checkpoint_mask2former_khop_model_zoo():
    """URL trọng số phải là URL có thật trong MODEL_ZOO.md của submodule."""
    zoo = BENCH / "mask2former" / "upstream" / "MODEL_ZOO.md"
    if not zoo.exists():
        pytest.skip("chưa init submodule")
    noi_dung = zoo.read_text(encoding="utf-8")
    doc = _readme("mask2former")
    urls = re.findall(r"https://dl\.fbaipublicfiles\.com/\S+\.pkl", doc)
    assert urls, "mask2former: mục đổi backbone không có trọng số nào"
    for u in urls:
        assert u.rsplit("/", 2)[-2] + "/" + u.rsplit("/", 1)[-1] in noi_dung, \
            f"mask2former: {u} không có trong MODEL_ZOO.md"


def test_checkpoint_solov2_la_ban_co_trong_so_that():
    """mmdet có config R101 không-DCN nhưng KHÔNG phát hành trọng số COCO cho
    nó; chọn nhầm bản đó là khởi đầu từ ImageNet."""
    khoi = _muc(_readme("solov2"), "### Đổi backbone")
    assert "solov2_r101_dcn_fpn_3x_coco" in khoi, "solov2: thiếu trọng số R101-DCN"
    assert "không** DCN" in khoi or "không DCN" in khoi, \
        "solov2: không cảnh báo bản R101 thiếu trọng số COCO"
