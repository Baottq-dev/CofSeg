"""Bước chấm không được làm lại cùng một phép tính, và không được đổi kết quả.

Ba chỗ lãng phí đã đo được trên chính bộ này, mỗi chỗ đều nằm trong code của
repo chứ không phải của thư viện:

1. `full_frame` dựng khung bool theo thứ tự hàng rồi `encode_mask` mới đổi
   sang uint8 thứ tự cột — hai lần chép CẢ khung 2560x1440 cho mỗi dự đoán.
   5.55 ms -> 1.62 ms khi cấp khung sẵn đúng dạng.
2. `assd`, `hd95`, `nsd` và sai số biên có dấu mỗi hàm tự dựng lại đường biên
   và chạy lại `distanceTransform` — bốn lượt cho cùng một cặp mặt nạ.
   6.44 ms -> 2.59 ms mỗi vùng khi dùng chung một `Surfaces`.
3. `test_pipeline` của SOLOv2 nạp nhãn thật mà `CocoMetric` không bao giờ
   đọc: 24.3 ms và 32 MiB mỗi ảnh, vứt đi nguyên vẹn.

Điều kiện của cả ba: KẾT QUẢ KHÔNG ĐỔI. Test dưới đây so bản mới với đúng
phép tính của bản cũ, chép nguyên văn, và đòi trùng tuyệt đối — không phải
xấp xỉ, vì cả ba đều là bỏ việc thừa chứ không phải đổi công thức.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
MODELS = ("maskrcnn", "mask2former", "solov2", "yolo")

mask_utils = pytest.importorskip("pycocotools.mask")
cv2 = pytest.importorskip("cv2")

H, W = 1440, 2560      # đúng kích thước ảnh thật của bộ này
S = 340                # cửa sổ cục bộ điển hình: tán trung vị 324 px + đệm


def _cofseg(model: str):
    """Nạp gói `cofseg` của riêng một thư mục model."""
    d = BENCH / model
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    for k in [m for m in sys.modules if m == "cofseg" or m.startswith("cofseg.")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("cofseg", d / "cofseg" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cofseg"] = m
    spec.loader.exec_module(m)
    return m


def _src(model: str, *parts: str) -> str:
    return (BENCH / model / "cofseg" / Path(*parts)).read_text(encoding="utf-8")


class _Pred:
    """Đủ giống Prediction cho `full_frame`: mặt nạ cục bộ + góc trên trái."""

    def __init__(self, mask, origin):
        self.mask, self.origin = mask, origin


def _tron(size: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


#: Cặp (dự đoán, vùng thật) phủ các trường hợp mà chỉ số biên dễ sai nhất.
CAP = {
    "lệch nhẹ": (_tron(S, 173, 167, 158), _tron(S, 170, 170, 162)),
    "trùng khít": (_tron(S, 170, 170, 150), _tron(S, 170, 170, 150)),
    "co vào": (_tron(S, 170, 170, 120), _tron(S, 170, 170, 162)),
    "phình ra": (_tron(S, 170, 170, 170), _tron(S, 170, 170, 140)),
    "rời hẳn": (_tron(S, 60, 60, 40), _tron(S, 280, 280, 40)),
    "dự đoán rỗng": (np.zeros((S, S), bool), _tron(S, 170, 170, 150)),
    "vùng thật rỗng": (_tron(S, 170, 170, 150), np.zeros((S, S), bool)),
    "cả hai rỗng": (np.zeros((S, S), bool), np.zeros((S, S), bool)),
}


# --------------------------------------------------------- 1. mã hoá RLE
@pytest.mark.parametrize("model", MODELS)
def test_khung_day_du_cap_san_dung_dang_pycocotools_doc(model):
    """uint8 + thứ tự cột, để `encode_mask` khỏi phải chép lại cả khung."""
    _cofseg(model)
    from cofseg.evaluation import coco_eval as ce

    out = ce.full_frame(_Pred(_tron(200, 100, 100, 60), (300, 200)), (H, W))
    assert out.dtype == np.uint8, f"{model}: {out.dtype}, sẽ bị chép lại khi mã hoá"
    assert out.flags.f_contiguous, f"{model}: thứ tự hàng, sẽ bị hoán vị khi mã hoá"


@pytest.mark.parametrize("model", MODELS)
def test_rle_trung_tung_byte_voi_cach_cu(model):
    """Chuỗi RLE phải giống hệt — nếu lệch thì mọi con số AP đã cũ đều sai.

    Bao cả mặt nạ tràn ra ngoài khung, vì đó là chỗ phép dán dễ lệch nhất.
    """
    _cofseg(model)
    from cofseg.evaluation import coco_eval as ce

    rng = np.random.default_rng(1)
    for _ in range(30):
        r = int(rng.integers(30, 300))
        ox, oy = int(rng.integers(0, W)) - r, int(rng.integers(0, H)) - r
        win = _tron(2 * r, r, r, r)
        moi = ce.encode_mask(ce.full_frame(_Pred(win, (ox, oy)), (H, W)))

        # Nguyên văn cách cũ: khung bool thứ tự hàng -> astype -> asfortranarray.
        cu_frame = np.zeros((H, W), bool)
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + 2 * r), min(H, oy + 2 * r)
        if x1 > x0 and y1 > y0:
            cu_frame[y0:y1, x0:x1] = win[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        rle = mask_utils.encode(np.asfortranarray(cu_frame.astype(np.uint8)))
        cu = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
        assert cu == moi, f"{model}: RLE lệch ở mặt nạ r={r} tại ({ox}, {oy})"


# ------------------------------------------------ 2. chỉ số biên dùng chung
def _cu_surface_distances(B, pred, gt):
    """Nguyên văn `surface_distances` của bản cũ."""
    pe, ge = B.edge(pred), B.edge(gt)
    dp, dg = B._dist_to(pe), B._dist_to(ge)
    if dp is None or dg is None:
        return None
    return dp[ge], dg[pe]


def _cu(B, pred, gt, tau=2.0):
    """Bốn chỉ số theo đúng cách bản cũ tính — mỗi cái một lượt riêng."""
    sd = _cu_surface_distances(B, pred, gt)
    if sd is None:
        assd = hd95 = nsd = float("nan")
    else:
        a, b = sd
        n = a.size + b.size
        assd = float("nan") if n == 0 else float((a.sum() + b.sum()) / n)
        both = np.concatenate(sd)
        hd95 = float("nan") if both.size == 0 else float(np.percentile(both, 95))
        nsd = (float("nan") if n == 0
               else float(((a <= tau).sum() + (b <= tau).sum()) / n))
    pe, ge = B.edge(pred), B.edge(gt)
    dp = B._dist_to(pe)
    if dp is None or not ge.any():
        signed = np.array([], dtype=np.float64)
    else:
        signed = (np.where(pred.astype(bool)[ge], 1.0, -1.0)
                  * dp[ge].astype(np.float64))
    return assd, hd95, nsd, signed


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ten", sorted(CAP))
def test_surfaces_cho_dung_so_nhu_bon_ham_roi(model, ten):
    """Dùng chung là bỏ việc lặp, không phải đổi công thức: phải trùng tuyệt đối."""
    _cofseg(model)
    from cofseg.metrics import boundary as B

    pred, gt = CAP[ten]
    s = B.surfaces(pred, gt)
    nan = float("nan")
    moi = (s.assd() if s else nan, s.hd95() if s else nan, s.nsd(2.0) if s else nan,
           s.signed() if s else np.array([], dtype=np.float64))
    cu = _cu(B, pred, gt)
    for k, (a, b) in enumerate(zip(cu, moi)):
        if isinstance(a, float):
            assert (np.isnan(a) and np.isnan(b)) or a == b, f"{model}/{ten}/{k}"
        else:
            assert np.array_equal(a, b), f"{model}/{ten}/{k}"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ten", sorted(CAP))
def test_bon_ham_roi_van_tra_dung_so_do(model, ten):
    """Bốn hàm mức module vẫn phải dùng được một mình (README và metrics/ xuất
    chúng), chỉ là chậm hơn."""
    _cofseg(model)
    from cofseg.metrics import boundary as B

    pred, gt = CAP[ten]
    cu_assd, cu_hd95, cu_nsd, cu_signed = _cu(B, pred, gt)
    for goi, cu in ((B.assd(pred, gt), cu_assd), (B.hd95(pred, gt), cu_hd95),
                    (B.normalized_surface_dice(pred, gt, 2.0), cu_nsd)):
        assert (np.isnan(goi) and np.isnan(cu)) or goi == cu, f"{model}/{ten}"
    assert np.array_equal(B.signed_boundary_error(pred, gt), cu_signed)


@pytest.mark.parametrize("model", MODELS)
def test_region_row_chi_dung_khoang_cach_bien_mot_lan(model):
    """Vòng chấm phải đi đường dùng chung, không gọi lại bốn hàm rời.

    Đây là điều kiện để 24 s xuống 9.6 s mỗi split; gọi `B.assd(p, g)` trở
    lại là lặng lẽ trả về mức cũ mà không test nào khác bắt được.
    """
    src = _src(model, "evaluation", "runner.py")
    than = src[src.index("def _region_row"):src.index("def _spurious_row")]
    assert "B.surfaces(" in than, f"{model}: không dựng Surfaces dùng chung"
    for ten in ("B.assd(", "B.hd95(", "B.normalized_surface_dice(",
                "B.signed_boundary_error("):
        assert ten not in than, f"{model}: {ten} tính lại trường khoảng cách"


# ----------------------------------------- 3. SOLOv2 không nạp nhãn lúc chấm
def _pipeline(ten_bien: str) -> list[str]:
    """Danh sách `type=` của một pipeline trong build_cfg, đọc bằng AST."""
    src = (BENCH / "solov2" / "cofseg" / "training" / "mmdet.py").read_text(
        encoding="utf-8")
    cay = ast.parse(src)
    for node in ast.walk(cay):
        if not isinstance(node, ast.Assign):
            continue
        dich = node.targets[0]
        if not (isinstance(dich, ast.Name) and dich.id == ten_bien):
            continue
        if not isinstance(node.value, ast.List):
            continue
        ra = []
        for phan_tu in node.value.elts:
            for kw in getattr(phan_tu, "keywords", []):
                if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                    ra.append(kw.value.value)
        return ra
    raise AssertionError(f"không thấy {ten_bien} trong mmdet.py")


def test_pipeline_cham_cua_solov2_khong_nap_nhan():
    """`CocoMetric` đọc nhãn từ ann_file, nên `LoadAnnotations` ở đây là công
    thuần tuý: rã đa giác thành bitmap 2560x1440 rồi vứt."""
    assert "LoadAnnotations" not in _pipeline("test_pipeline")


def test_pipeline_train_cua_solov2_van_nap_nhan():
    """Cùng lúc đó vòng train thì BẮT BUỘC phải có — bỏ nhầm là train không lỗi
    mà chỉ đơn giản không học được gì."""
    assert "LoadAnnotations" in _pipeline("train_pipeline")


# ----------------------------------- 4. detectron2 không được đoán imgsz nữa
@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_thieu_d2_config_thi_dung_chu_khong_doan(model):
    """Sáu lượt chấm test đã chạy ở cạnh dài 1333 trong khi train ở 1024, im
    lặng, vì chỗ này rơi về mặc định của detectron2 thay vì báo lỗi."""
    src = _src(model, "models", "detectron2.py")
    khoi = src[src.index("saved = Path(config_file)"):]
    khoi = khoi[:khoi.index("cfg.MODEL.WEIGHTS")]
    assert "raise SystemExit" in khoi, f"{model}: vẫn đoán imgsz trong im lặng"
    assert "1333" in khoi, f"{model}: lỗi phải nói ra con số mặc định gây hại"


@pytest.mark.parametrize("model", ("maskrcnn", "mask2former"))
def test_model_detectron2_nhan_imgsz_tu_dong_lenh(model):
    """`--imgsz 1024` phải đi tới được model: đó là lối thoát khi không có
    d2_config.yaml, và nó cũng là thứ gắn hậu tố _i1024 vào tên thư mục run."""
    _cofseg(model)
    from cofseg.models import model_param_names

    assert "imgsz" in (model_param_names("detectron2") or set()), model
