"""requirements.txt ở gốc lo app/, mỗi model lo phần của mình.

Gộp chung một file là điều KHÔNG làm được nữa, và lý do là hai mốc phiên bản
không giao nhau:

    mmcv (SOLOv2)      chỉ có wheel tới torch 2.4 / cu121
    GPU Blackwell      đòi torch >= 2.7 (cu121 không có kernel sm_120)

Test này giữ cho việc tách đó không bị gộp lại bằng một lần "dọn dẹp" thiện
chí, và giữ đúng cái ghim mà thiếu nó thì train chết giữa chừng.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo11", "solov2", "maskrcnn", "mask2former")


def _pins(path: Path) -> dict[str, str]:
    """Tên gói -> dòng đầy đủ, bỏ chú thích và dòng --option."""
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[=<>!;\[ ]", line, 1)[0].strip().lower()
        if name:
            out[name] = line
    return out


@pytest.fixture(scope="module")
def root_pins():
    return _pins(ROOT / "requirements.txt")


@pytest.mark.parametrize("model", MODELS)
def test_moi_model_co_file_rieng(model):
    assert (ROOT / "benchmark" / model / "requirements.txt").exists()


@pytest.mark.parametrize("model", MODELS)
def test_moi_model_khai_du_goi_cofseg_dung_chung(model):
    """cofseg/ của mọi thư mục đều import numpy, cv2, pycocotools, yaml, tqdm."""
    pins = _pins(ROOT / "benchmark" / model / "requirements.txt")
    for goi in ("numpy", "opencv-python", "pycocotools", "pyyaml", "tqdm", "torch"):
        assert goi in pins, f"{model}: thiếu {goi}"


def test_goc_khong_con_goi_cua_benchmark(root_pins):
    """app/ và canopyseg/ không import gói nào trong số này — đã kiểm cả repo."""
    for goi in ("mmcv", "mmdet", "mmengine", "ultralytics", "timm", "shapely"):
        assert goi not in root_pins, f"{goi} là của benchmark, không phải của app/"


def test_goc_van_du_cho_annotator(root_pins):
    for goi in ("fastapi", "uvicorn", "pydantic", "hydra-core", "iopath",
                "pycocotools", "opencv-python", "numpy", "torch"):
        assert goi in root_pins, f"app/ cần {goi}"


def test_solov2_khoa_dung_torch_241_cu121():
    """Mốc cứng: wheel mmcv mới nhất là torch2.4.0/cu121, không có gì hơn."""
    txt = (ROOT / "benchmark" / "solov2" / "requirements.txt").read_text(encoding="utf-8")
    pins = _pins(ROOT / "benchmark" / "solov2" / "requirements.txt")
    assert pins["torch"] == "torch==2.4.1+cu121"
    assert pins["mmcv"] == "mmcv==2.2.0"
    assert "download.openmmlab.com/mmcv/dist/cu121/torch2.4" in txt, \
        "thiếu --find-links thì pip đi build mmcv từ nguồn"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_hai_model_detectron2_ghim_setuptools(model):
    """setuptools 82 bỏ pkg_resources, mà model_zoo của detectron2 0.6 cần nó.

    Không ghim thì train chết SAU khi đã nạp xong dữ liệu, và bước cài vẫn báo
    thành công vì `import detectron2` không kéo model_zoo vào.
    """
    pins = _pins(ROOT / "benchmark" / model / "requirements.txt")
    assert pins.get("setuptools") == "setuptools<82"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_detectron2_khong_nam_trong_requirements(model):
    """Nó build từ nguồn và tự quyết biên dịch CUDA hay không tuỳ máy lúc cài."""
    pins = _pins(ROOT / "benchmark" / model / "requirements.txt")
    assert "detectron2" not in pins


def test_mask2former_khai_du_goi_cua_repo_goc():
    """scipy (matcher Hungarian), shapely (train_net.py), timm (backbone Swin)."""
    pins = _pins(ROOT / "benchmark" / "mask2former" / "requirements.txt")
    for goi in ("scipy", "shapely", "timm"):
        assert goi in pins


def test_yolo11_co_y_khong_cai_albumentations():
    """ultralytics TỰ BẬT tăng cường khi thấy gói này, im lặng bỏ qua khi không.

    Có hay không có nó cho ra hai pipeline huấn luyện khác nhau mà không dòng
    config nào nói ra. Để ngoài là lựa chọn có chủ đích, và phải ghi lại.
    """
    path = ROOT / "benchmark" / "yolo11" / "requirements.txt"
    assert "albumentations" not in _pins(path)
    assert "albumentations" in path.read_text(encoding="utf-8"), \
        "bỏ gói thì phải để lại ghi chú vì sao, không thì lần sau có người thêm vào"


def test_setup_env_tro_dung_cac_file_do():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "setup_env", ROOT / "scripts" / "setup_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.BENCH_REQS) == set(MODELS)
    for rel in mod.BENCH_REQS.values():
        assert (ROOT / rel).exists(), rel
