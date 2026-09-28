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
    goc = setup_env.CUDA
    try:
        for cuda in setup_env.TORCH:
            setup_env.CUDA = cuda
            tag = setup_env.torch_cuda_tag()
            assert re.fullmatch(r"\d+\.\d", tag), tag
            assert tag in setup_env.CONDA_LABEL, f"thiếu nhãn conda cho CUDA {tag}"
    finally:
        setup_env.CUDA = goc        # fixture dùng chung cả module, đừng để rò


def test_torch_va_torchvision_cung_mot_chi_muc_cuda(setup_env):
    for cuda, pkgs in setup_env.TORCH.items():
        for pkg in pkgs:
            assert pkg.endswith(f"+cu{cuda}"), f"{pkg} không khớp cu{cuda}"


def test_co_lua_chon_cho_blackwell(setup_env):
    """cu121 không có kernel sm_120; thiếu lựa chọn CUDA 13 là RTX 50xx bó tay."""
    assert any(int(c) >= 128 for c in setup_env.TORCH), \
        "cần ít nhất một lựa chọn CUDA >= 12.8 cho Blackwell"


# ------------------------------------------------- tài liệu phải là lệnh thật
READMES = [ROOT / "README.md", BENCH / "README.md"] + [
    BENCH / m / "README.md" for m in MODELS]


@pytest.mark.parametrize("path", READMES, ids=lambda p: str(p.relative_to(ROOT)))
def test_readme_khong_day_nguoi_doc_sang_setup_env(path):
    """README phải ghi LỆNH THẬT, không trỏ sang script bao.

    setup_env.py chỉ để chạy nhanh. Ai dựng môi trường lần đầu cần thấy đúng
    lệnh pip/conda sẽ chạy, để còn sửa được khi máy mình khác.
    """
    if not path.exists():
        pytest.skip(f"{path.name} chưa có")
    assert "setup_env" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [BENCH / m / "requirements.txt" for m in MODELS]
                                 + [BENCH / "requirements.txt"],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_requirements_cung_ghi_lenh_that(path):
    assert "setup_env" not in path.read_text(encoding="utf-8")


def test_commit_detectron2_trong_readme_khop_voi_script(setup_env):
    """Hai chỗ ghi cùng một commit; lệch là người đọc cài nhầm bản."""
    commit = setup_env.D2.split("@")[-1]
    assert len(commit) == 40, commit
    assert commit in (BENCH / "README.md").read_text(encoding="utf-8"), \
        "benchmark/README.md ghim commit detectron2 khác với scripts/setup_env.py"


def test_lenh_va_dong_ma_trong_readme_con_dung():
    """Ba chuỗi README bảo người ta sed/ghim; kiểm chúng vẫn khớp nguồn thật."""
    doc = (BENCH / "README.md").read_text(encoding="utf-8")

    # chỉ mục wheel mmcv
    assert "download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html" in doc

    # dòng raise mà lệnh sed nhắm vào, trong submodule Mask2Former
    src = (BENCH / "mask2former" / "upstream" / "mask2former" / "modeling"
           / "pixel_decoder" / "ops" / "functions" / "ms_deform_attn_func.py")
    if src.exists():
        assert "raise ModuleNotFoundError(info_string)" in src.read_text(encoding="utf-8"), \
            "upstream đã đổi dòng raise; lệnh sed trong README thành vô hiệu"
    assert "raise ModuleNotFoundError(info_string)" in doc


# ------------------------------------------- mặc định phải chạy được trên 5090
def test_mac_dinh_la_cuda_13(setup_env):
    """cu121 không có kernel sm_120, nên nó không thể là mặc định.

    Kế hoạch chạy là thuê RTX 5090. Mặc định cu121 nghĩa là ai làm theo tài
    liệu cũng dựng ra một env mà SOLOv2 chết ở lời gọi kernel đầu tiên.
    """
    assert setup_env.CUDA == "130"
    assert int(setup_env.CUDA) >= 128, "phải >= 12.8 mới có sm_120"


def test_mac_dinh_build_mmcv_tu_nguon(setup_env):
    """Trên cu130 không có wheel mmcv nào; lấy wheel là cài nhầm bản cu121."""
    assert setup_env.BUILD_MMCV is True


@pytest.mark.parametrize(
    "path",
    [BENCH / m / "requirements.txt" for m in MODELS] + [BENCH / "requirements.txt"],
    ids=lambda p: str(p.relative_to(ROOT)))
def test_lenh_mau_trong_header_dung_cu130(path):
    """Header mỗi file có lệnh cài torch làm mẫu; nó phải là lệnh chạy được."""
    doc = path.read_text(encoding="utf-8")
    assert "whl/cu130" in doc, f"{path.name}: lệnh mẫu còn trỏ CUDA cũ"


def test_readme_dat_cu130_len_truoc():
    """Đọc từ trên xuống phải gặp cu130 trước cu121."""
    for path in (ROOT / "README.md", BENCH / "README.md"):
        doc = path.read_text(encoding="utf-8")
        assert doc.index("cu130") < doc.index("cu121"), \
            f"{path.name}: cu121 xuất hiện trước cu130"


def test_solov2_khong_ghim_mmengine_tu_pypi():
    """Trên torch >= 2.6 bản PyPI ném UnpicklingError lúc nạp checkpoint COCO.

    mmengine 0.10.7 ra 04/03/2025, bản vá weights_only merge 25/10/2025, và
    không có bản phát hành nào sau đó. Phải lấy từ git.
    """
    doc = _req("solov2").read_text(encoding="utf-8")
    assert "mmengine" not in _pins(_lines(_req("solov2")))
    assert "github.com/open-mmlab/mmengine" in doc, "phải chỉ đường lấy từ git"


# ------------------------------- nvcc của máy phải khớp major với torch
def test_readme_bao_kiem_nvcc_truoc_khi_cai_torch():
    """Bước xem nvcc phải đứng TRƯỚC lệnh cài torch đầu tiên.

    Bỏ thứ tự này thì lỗi không rơi ở bước cài torch mà rơi tận lúc build
    detectron2/mmcv, với một câu báo không nhắc gì tới torch:

        RuntimeError: The detected CUDA version (12.8) mismatches the
        version that was used to compile PyTorch (13.0)

    Gặp thật trên máy Vast RTX 5090, 28/09/2026.
    """
    doc = (BENCH / "README.md").read_text(encoding="utf-8")
    assert doc.index("nvcc --version") < doc.index("pip install torch=="), \
        "lệnh cài torch đứng trước bước kiểm nvcc"
    assert "mismatches the version" in doc, "thiếu câu báo lỗi để người ta tra ra"


@pytest.mark.parametrize(
    "path",
    [BENCH / m / "requirements.txt" for m in MODELS] + [BENCH / "requirements.txt"],
    ids=lambda p: str(p.relative_to(ROOT)))
def test_header_chi_duong_cho_may_nvcc_12(path):
    """Header nêu lệnh cu130 làm mẫu, nhưng phần lớn ảnh máy thuê cho 5090 có
    nvcc 12.8 — phải nói rõ đổi sang cu128, không thì lệnh mẫu là cái bẫy."""
    doc = path.read_text(encoding="utf-8")
    assert "cu128" in doc, f"{path.name}: lệnh mẫu không nhắc nhánh nvcc 12.x"


def test_readme_neu_ca_hai_nhanh_torch_deu_cung_phien_ban():
    """Đổi nhánh CUDA không được đổi phiên bản torch, không thì hai người cùng
    nhóm chạy hai bản torch khác nhau mà tưởng là cùng."""
    doc = (BENCH / "README.md").read_text(encoding="utf-8")
    for pin in ("torch==2.11.0+cu130", "torch==2.11.0+cu128"):
        assert pin in doc, f"thiếu {pin}"
