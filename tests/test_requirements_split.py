"""Bố cục requirements: gốc lo app/, mỗi model lo phần của mình, torch một chỗ.

    requirements.txt                    app/ + canopyseg
    benchmark/<model>/requirements.txt  gói của model đó, KHÔNG có torch
    benchmark/requirements.txt          -r cả bốn, để cài một env

Bất biến quan trọng nhất: **không file nào ghim torch**. Bản torch phụ thuộc
kiến trúc GPU chứ không phụ thuộc kiến trúc mạng, nên nó nằm ở hằng `TORCH`
trong scripts/setup_env.py, chọn bằng `--cuda`. Ghim trong từng file model là
cách chắc chắn để bốn file trôi ra xa nhau rồi không cài chung một env được.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("yolo11", "solov2", "maskrcnn", "mask2former")


def _lines(path: Path) -> list[str]:
    """Dòng còn hiệu lực: bỏ chú thích và dòng trống."""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _pins(lines) -> dict[str, str]:
    pins = {}
    for line in lines:
        if line.startswith("-"):
            continue
        name = re.split(r"[=<>!;\[ ]", line, 1)[0].strip().lower()
        if name:
            pins[name] = line
    return pins


def _req(model: str) -> Path:
    return BENCH / model / "requirements.txt"


# ------------------------------------------------------------- bố cục cơ bản
def test_file_gop_dan_ve_ca_bon():
    lines = _lines(BENCH / "requirements.txt")
    for model in MODELS:
        assert f"-r {model}/requirements.txt" in lines


@pytest.mark.parametrize("model", MODELS)
def test_moi_model_co_file_rieng(model):
    assert _req(model).exists()


# ------------------------------------------ bất biến: không file nào ghim torch
@pytest.mark.parametrize("model", MODELS)
def test_khong_file_model_nao_ghim_torch(model):
    """Ghim ở đây là bước đầu tiên để bốn file trôi ra xa nhau.

    torch phụ thuộc GPU chứ không phụ thuộc model; nó ở TORCH trong
    scripts/setup_env.py, chọn bằng --cuda.
    """
    for line in _lines(_req(model)):
        assert not line.startswith(("torch=", "torch>", "torchvision",
                                    "--extra-index-url", "--index-url")),             f"{model}: torch thuộc setup_env.py --cuda, không thuộc file này"


def test_moi_file_nhac_cai_torch_truoc():
    """Cài file này mà quên torch thì pip kéo bản CPU từ PyPI, và im lặng."""
    for model in MODELS:
        assert "torch" in _req(model).read_text(encoding="utf-8"),             f"{model}: phải có ghi chú cài torch trước"


# ------------------------------------------------------------- nội dung từng file
@pytest.mark.parametrize("model", MODELS)
def test_moi_file_khai_du_goi_cofseg_dung_chung(model):
    """cofseg/ của mọi thư mục đều import numpy, cv2, pycocotools, yaml, tqdm.

    Mỗi file tự khai đủ, không `-r` sang file khác: mở ra là đọc được ngay
    thư mục này cần gì, không phải lần theo hai ba file.
    """
    pins = _pins(_lines(_req(model)))
    for goi in ("numpy", "opencv-python", "pycocotools", "pyyaml", "tqdm"):
        assert goi in pins, f"{model}: thiếu {goi}"


def test_goc_khong_con_goi_cua_benchmark():
    """app/ và canopyseg/ không import gói nào trong số này — đã kiểm cả repo."""
    pins = _pins(_lines(ROOT / "requirements.txt"))
    for goi in ("mmcv", "mmdet", "mmengine", "ultralytics", "timm", "shapely"):
        assert goi not in pins, f"{goi} là của benchmark, không phải của app/"


def test_goc_van_du_cho_annotator():
    pins = _pins(_lines(ROOT / "requirements.txt"))
    for goi in ("fastapi", "uvicorn", "pydantic", "hydra-core", "iopath",
                "pycocotools", "opencv-python", "numpy", "torch"):
        assert goi in pins, f"app/ cần {goi}"


def test_mmcv_khong_nam_trong_requirements_cua_solov2():
    """Cách cài phụ thuộc MÁY: có máy lấy wheel, có máy phải build 20-120 phút.

    Một dòng requirements không rẽ nhánh theo GPU được — cùng lý do detectron2
    không nằm trong file của Mask R-CNN. Nó thuộc setup_env.py.
    """
    assert "mmcv" not in _pins(_lines(_req("solov2")))


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_hai_model_detectron2_ghim_setuptools(model):
    """setuptools 82 bỏ pkg_resources, mà model_zoo của detectron2 0.6 cần nó.

    Không ghim thì train chết SAU khi đã nạp xong dữ liệu, và bước cài vẫn báo
    thành công vì `import detectron2` không kéo model_zoo vào.
    """
    assert _pins(_lines(_req(model))).get("setuptools") == "setuptools<82"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_detectron2_khong_nam_trong_requirements(model):
    assert "detectron2" not in _pins(_lines(_req(model)))


def test_mask2former_khai_du_goi_cua_repo_goc():
    """scipy (matcher Hungarian), shapely (train_net.py), timm (backbone Swin)."""
    pins = _pins(_lines(_req("mask2former")))
    for goi in ("scipy", "shapely", "timm"):
        assert goi in pins


def test_yolo11_co_y_khong_cai_albumentations():
    """ultralytics TỰ BẬT tăng cường khi thấy gói này, im lặng bỏ qua khi không.

    Có hay không có nó cho ra hai pipeline huấn luyện khác nhau mà không dòng
    config nào nói ra. Để ngoài là lựa chọn có chủ đích, và phải ghi lại.
    """
    path = _req("yolo11")
    assert "albumentations" not in _pins(_lines(path))
    assert "albumentations" in path.read_text(encoding="utf-8"), \
        "bỏ gói thì phải để lại ghi chú vì sao, không thì lần sau có người thêm vào"


# ------------------------------------------------------------------ setup_env
@pytest.fixture(scope="module")
def setup_env():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "setup_env", ROOT / "scripts" / "setup_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_setup_env_tro_dung_cac_file_do(setup_env):
    assert set(setup_env.BENCH_REQS) == set(MODELS)
    for rel in setup_env.BENCH_REQS.values():
        assert (ROOT / rel).exists(), rel


def test_moi_lua_chon_cuda_deu_co_nhan_conda(setup_env):
    """Thông báo lỗi của step_check đưa lệnh `conda install ... cuda-toolkit`.

    Thêm một lựa chọn --cuda mà quên nhãn là người dùng nhận lệnh sai.
    """
    for cuda in setup_env.TORCH:
        setup_env.CUDA = cuda
        tag = setup_env.torch_cuda_tag()
        assert re.fullmatch(r"\d+\.\d", tag), tag
        assert tag in setup_env.CONDA_LABEL, f"thiếu nhãn conda cho CUDA {tag}"
    setup_env.CUDA = "121"


def test_torch_va_torchvision_cung_mot_chi_muc_cuda(setup_env):
    for cuda, pkgs in setup_env.TORCH.items():
        for pkg in pkgs:
            assert pkg.endswith(f"+cu{cuda}"), f"{pkg} không khớp cu{cuda}"


def test_co_lua_chon_cho_blackwell(setup_env):
    """cu121 không có kernel sm_120; thiếu lựa chọn CUDA 13 là RTX 50xx bó tay."""
    assert any(int(c) >= 128 for c in setup_env.TORCH), \
        "cần ít nhất một lựa chọn CUDA >= 12.8 cho Blackwell"
