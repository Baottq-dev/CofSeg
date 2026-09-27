"""Bố cục requirements: gốc lo app/, mỗi model lo phần của mình, torch một chỗ.

    requirements.txt                  app/ + canopyseg
    benchmark/torch.txt               torch  <- DUY NHẤT một chỗ ghim
    benchmark/base.txt                -r torch.txt + gói cofseg dùng chung
    benchmark/<model>/requirements.txt  -r ../base.txt + gói riêng
    benchmark/requirements.txt        -r cả bốn, để cài một env

Bất biến quan trọng nhất và cũng là thứ dễ vỡ nhất: **cả bốn file phải dẫn về
cùng một bản torch**. Vỡ cái đó thì `pip install -r benchmark/requirements.txt`
gãy vì xung đột, tức mất khả năng cài chung một env.

torch là lựa chọn của MÁY (kiến trúc GPU), không phải của model. Ghim nó trong
từng file model là cách chắc chắn để bốn file trôi ra xa nhau.
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


def _resolve(path: Path) -> list[str]:
    """Mở hết chuỗi `-r` lồng nhau, theo đúng cách pip làm (tương đối với file)."""
    out = []
    for line in _lines(path):
        if line.startswith("-r "):
            out += _resolve((path.parent / line[3:].strip()).resolve())
        else:
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
@pytest.mark.parametrize("model", MODELS)
def test_moi_model_co_file_rieng_va_dan_ve_base(model):
    assert "-r ../base.txt" in _lines(_req(model))


def test_base_dan_ve_torch():
    assert "-r torch.txt" in _lines(BENCH / "base.txt")


def test_file_gop_dan_ve_ca_bon():
    lines = _lines(BENCH / "requirements.txt")
    for model in MODELS:
        assert f"-r {model}/requirements.txt" in lines


# ---------------------------------------------- bất biến: một bản torch duy nhất
@pytest.mark.parametrize("model", MODELS)
def test_khong_file_model_nao_tu_ghim_torch(model):
    """Ghim ở đây là bước đầu tiên để bốn file trôi ra xa nhau."""
    for line in _lines(_req(model)):
        assert not line.startswith(("torch=", "torch>", "torchvision", "--extra-index-url")), \
            f"{model}: torch phải ghim ở benchmark/torch.txt, không phải ở đây"


def test_bon_model_cung_dan_ve_mot_ban_torch():
    """Đây chính là điều kiện để `pip install -r benchmark/requirements.txt` chạy."""
    lay = lambda m: {l for l in _resolve(_req(m))
                     if l.startswith(("torch", "--extra-index-url"))}
    chuan = lay(MODELS[0])
    assert chuan, "không thấy dòng torch nào sau khi mở chuỗi -r"
    for model in MODELS[1:]:
        assert lay(model) == chuan, f"{model} dẫn về bản torch khác"


def test_torch_txt_chi_bat_mot_khoi():
    """Hai khối cùng bật thì pip thấy hai bản torch và gãy."""
    lines = _lines(BENCH / "torch.txt")
    assert len([l for l in lines if l.startswith("torch==")]) == 1
    assert len([l for l in lines if l.startswith("--extra-index-url")]) == 1


def test_torch_va_torchvision_cung_mot_chi_muc_cuda():
    lines = _lines(BENCH / "torch.txt")
    tag = next(re.search(r"/whl/cu(\d+)", l)[1] for l in lines
               if l.startswith("--extra-index-url"))
    for l in lines:
        if l.startswith(("torch==", "torchvision==")):
            assert l.endswith(f"+cu{tag}"), f"{l} không khớp chỉ mục cu{tag}"


# ------------------------------------------------------------- nội dung từng file
@pytest.mark.parametrize("model", MODELS)
def test_goi_cofseg_dung_chung_nam_o_base(model):
    """cofseg/ của mọi thư mục đều import numpy, cv2, pycocotools, yaml, tqdm."""
    pins = _pins(_resolve(_req(model)))
    for goi in ("numpy", "opencv-python", "pycocotools", "pyyaml", "tqdm", "torch"):
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


def test_setup_env_doc_cuda_tu_torch_txt(setup_env):
    """Đổi khối trong torch.txt thì mọi thông báo lỗi phải nói đúng phiên bản."""
    tag = setup_env.torch_cuda_tag()
    assert re.fullmatch(r"\d+\.\d", tag), tag
    lines = _lines(BENCH / "torch.txt")
    url = next(l for l in lines if l.startswith("--extra-index-url"))
    assert f"cu{tag.replace('.', '')}" in url
    assert tag in setup_env.CONDA_LABEL, \
        f"thiếu nhãn kênh conda cho CUDA {tag}; thông báo lỗi sẽ đưa lệnh sai"
