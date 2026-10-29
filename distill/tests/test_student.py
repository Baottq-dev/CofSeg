"""Trò: chi phí đúng thiết kế, đầu hai cấp, Proto stride 8, phép chọn trọng số khớp giữa các lớp, train và
suy luận được qua đường của ultralytics và bộ chấm của benchmark."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from distill.blocks import check_student  # noqa: E402
from distill.student import (  # noqa: E402
    Proto26S8, Segment26Student, _heads, _uniform, build_student, find_student_head, student_yaml,
)


def _build(teacher, **block):
    return build_student(check_student(block), teacher.yaml, nc=1, teacher=teacher)


def _layers(m, x, keep):
    """Đầu ra của các lớp trong `keep`, chạy tuần tự như BaseModel._predict_once."""
    out, ys = {}, []
    for i, mod in enumerate(m.model[:-1]):
        if mod.f != -1:
            x = ys[mod.f] if isinstance(mod.f, int) else [x if j == -1 else ys[j] for j in mod.f]
        x = mod(x)
        ys.append(x)
        if i in keep:
            out[i] = x
    return out


def _corr(a, b):
    a, b = a.flatten(2), b.flatten(2)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return float(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)).mean())


def test_chi_phi_dung_so_da_do(teacher):
    """5,76 triệu tham số, 22,04 GFLOPs ở 1024×576 sau fuse: số của tro_nhe/chi_phi_tro.log."""
    m, _ = _build(teacher)
    f = copy.deepcopy(m).eval().fuse()
    assert sum(p.numel() for p in f.parameters()) == 5_762_950
    thop = pytest.importorskip("thop")
    g = thop.profile(f, inputs=(torch.zeros(1, 3, 576, 1024),), verbose=False)[0] * 2 / 1e9
    assert g == pytest.approx(22.04, abs=0.01)


def test_tro_025_tu_yolo26n_chep_gan_het(teacher_n):
    """Trò 0,25 lấy từ YOLO26n-seg cùng độ rộng: 2,57 triệu tham số, 9,95 GFLOPs; backbone và neck chép
    nguyên, chỉ chọn ở các lớp theo số prototype (32 -> 16) và nhánh điểm (COCO rộng 80 kênh vì 80 lớp,
    1 lớp thì 64 — benchmark YOLO26n-seg để ngẫu nhiên phần này, trò lấy 64/80)."""
    m, counts = build_student(check_student({"width": 0.25}), teacher_n.yaml, nc=1, teacher=teacher_n)
    assert counts == {"chép": 663, "chọn": 167, "bỏ qua": 14}
    for i in range(23):
        for k, v in m.model[i].state_dict().items():
            assert torch.equal(v, teacher_n.model[i].state_dict()[k].to(v.dtype)), (i, k)
    f = copy.deepcopy(m).eval().fuse()
    assert sum(p.numel() for p in f.parameters()) == 2_567_234
    thop = pytest.importorskip("thop")
    g = thop.profile(f, inputs=(torch.zeros(1, 3, 576, 1024),), verbose=False)[0] * 2 / 1e9
    assert g == pytest.approx(9.95, abs=0.01)


def test_nguon_hep_hon_tro_bi_chan(teacher_n):
    """Trò 0,375 lấy nhầm từ YOLO26n: phải dừng, không được lặng lẽ giữ khởi tạo ngẫu nhiên."""
    with pytest.raises(SystemExit, match="hẹp hơn trò"):
        build_student(check_student({}), teacher_n.yaml, nc=1, teacher=teacher_n)


def test_dau_hai_cap_va_proto_stride_8(teacher):
    m, counts = _build(teacher)
    head = m.model[-1]
    assert type(head) is Segment26Student and type(head.proto) is Proto26S8
    assert head.nl == 2 and head.f == [16, 19, 22]
    assert head.stride.tolist() == [16.0, 32.0] and m.stride.tolist() == [16.0, 32.0]
    assert all(len(getattr(head, n)) == 2 for n in ("cv2", "cv3", "cv4", "one2one_cv2", "one2one_cv3", "one2one_cv4"))
    assert head.nm == 16 and head.proto.cv3.conv.out_channels == 16
    # 12 tensor lớp cuối của nhánh lớp (3 cấp × 2 đầu × weight, bias) + 2 của semseg: 80 lớp COCO -> 1.
    assert counts["bỏ qua"] == 14
    assert find_student_head(m) is head


def test_lop_3x3_thu_hai_chay_o_stride_8(teacher):
    m, _ = _build(teacher)
    seen = {}
    proto = m.model[-1].proto
    # Hook trả về None: trả về giá trị khác là thay đầu ra của lớp.
    proto.cv2.register_forward_hook(lambda mod, i, o: seen.update(cv2=i[0].shape[-2:]))
    m.train()
    out = m(torch.zeros(1, 3, 128, 128))["one2many"]
    assert tuple(seen["cv2"]) == (16, 16)                # 128 / 8
    p, sem = out["proto"]
    assert p.shape == (1, 16, 32, 32)                    # prototype vẫn ở stride 4
    assert out["boxes"].shape[-1] == 8 * 8 + 4 * 4      # anchor chỉ ở stride 16 và 32


def test_tat_tung_thay_doi_ve_lai_dau_goc(teacher):
    from ultralytics.nn.modules.block import Proto26
    from ultralytics.nn.modules.head import Segment26

    m, _ = _build(teacher, nm=32, proto_s8=False, drop_p3=False)
    head = m.model[-1]
    assert type(head) is Segment26 and type(head.proto) is Proto26
    assert head.nl == 3 and head.stride.tolist() == [8.0, 16.0, 32.0]


def test_yaml_tro_khong_doi_yaml_thay(teacher):
    goc = copy.deepcopy(teacher.yaml)
    d = student_yaml(teacher.yaml, check_student({}))
    assert teacher.yaml == goc
    assert d["scale"] == "t" and d["scales"]["t"] == [0.5, 0.375, 1024] and d["head"][-1][-1][1] == 16


def test_chi_so_chon_deu_va_chon_head():
    assert _uniform(8, 6).tolist() == [0, 1, 2, 4, 5, 6]
    # Hai khối 4 kênh -> hai khối 3 kênh: chỉ số của khối sau vẫn nằm trong khối sau.
    assert _uniform(8, 6)[3:].tolist() == [4, 5, 6]
    assert _heads(4, 3, 2).tolist() == [0, 1, 2, 3, 4, 5]
    assert _heads(4, 2, 2).tolist() == [0, 1, 4, 5]


def test_lop_dau_trung_thay_tren_kenh_da_chon(teacher):
    """Conv đầu tiên nhận đủ 3 kênh ảnh, nên kênh ra của trò phải trùng đúng kênh đã chọn của thầy."""
    m, _ = _build(teacher)
    m.eval()
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = m.model[0](x), teacher.model[0](x)
    assert torch.allclose(a, b[:, _uniform(b.shape[1], a.shape[1])], atol=1e-5)


def test_chon_deu_khop_giua_cac_lop(teacher):
    """Phép chọn khớp giữa lớp xuất và lớp nhận (C3k2 tách đôi, Concat) thì đặc trưng của trò còn tương
    quan với đúng các kênh đã chọn của thầy; chọn tiền tố (sai khối) hay khởi tạo ngẫu nhiên thì không."""
    m, _ = _build(teacher, drop_p3=False, proto_s8=False)
    r, _ = _build(teacher, drop_p3=False, proto_s8=False, init="scratch")
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 128), torch.linspace(0, 1, 192), indexing="ij")
    x = torch.stack([torch.sin(9 * xx) * yy, torch.cos(7 * yy), (xx * yy) ** 0.5])[None]
    keep = [2, 4, 16]
    for mod in (m, r):
        mod.eval()
    with torch.no_grad():
        ft, fs, fr = _layers(teacher, x, keep), _layers(m, x, keep), _layers(r, x, keep)
    for i in keep:
        t, s = ft[i], fs[i]
        sel = _corr(s, t[:, _uniform(t.shape[1], s.shape[1])])
        pre = _corr(s, t[:, : s.shape[1]])
        rnd = _corr(fr[i], t[:, _uniform(t.shape[1], s.shape[1])])
        assert sel > pre + 0.15 and sel > rnd + 0.15, (i, sel, pre, rnd)


def test_attention_chon_nguyen_head(teacher):
    """qkv của trò = các hàng của head 0, 1, 2 ở thầy (4 head -> 3), cột chọn đều theo luồng dư."""
    m, _ = _build(teacher)
    s_att, t_att = m.model[10].m[0].attn, teacher.model[10].m[0].attn
    assert (t_att.num_heads, s_att.num_heads) == (4, 3)
    rows = _heads(4, 3, 2 * s_att.key_dim + s_att.head_dim)
    cols = _uniform(t_att.qkv.conv.in_channels, s_att.qkv.conv.in_channels)
    want = t_att.qkv.conv.weight[rows][:, cols]
    assert torch.equal(s_att.qkv.conv.weight, want)
    hd = _heads(4, 3, s_att.head_dim)
    assert torch.equal(s_att.pe.conv.weight, t_att.pe.conv.weight[hd])
    assert torch.equal(s_att.proj.conv.weight,
                       t_att.proj.conv.weight[_uniform(256, 192)][:, hd])


def test_lop_cuoi_cua_nhanh_lop_giu_khoi_tao(teacher):
    """Không chép hàng của lớp COCO nào sang lớp tán: giống fine-tune từ COCO của benchmark."""
    m, _ = _build(teacher)
    r, _ = _build(teacher, init="scratch")
    head_m, head_r = m.model[-1], r.model[-1]
    assert torch.equal(head_m.cv3[0][2].bias, head_r.cv3[0][2].bias)   # bias_init theo stride, không theo thầy


def test_loss_va_lan_lui_tren_tro(teacher, batch):
    from ultralytics.cfg import get_cfg

    m, _ = _build(teacher)
    m.args = get_cfg()
    m.train()
    preds = m(torch.rand(2, 3, 128, 128))
    loss, items = m.loss(batch, preds)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert m.model[-1].cv4[0][2].weight.grad is not None
    assert m.model[-1].proto.cv2.conv.weight.grad.abs().sum() > 0


def test_checkpoint_nap_duoc_va_cham_qua_lop_boc(teacher, tmp_path):
    """best.pt của trò đi qua YOLO() và lớp bọc chấm của benchmark như mọi checkpoint YOLO."""
    import numpy as np
    from ultralytics.cfg import get_cfg

    from cofseg.models.yolo_seg import YoloSegModel

    m, _ = _build(teacher)
    m.args = get_cfg()
    m.names = {0: "canopy"}
    p = tmp_path / "tro.pt"
    torch.save({"model": copy.deepcopy(m).half(), "train_args": {"task": "segment", "imgsz": 128}}, p)
    y = YoloSegModel(str(p), imgsz=128, conf=0.0, max_det=5)
    assert y.mask_head == "gốc"
    assert find_student_head(y.model.model) is not None
    preds = y.predict(np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    assert len(preds) <= 5 and all(pr.mask.dtype == bool for pr in preds)
    # Phép cắt nới hộp của bộ chấm cũng dùng được với trò (lượt trò + giám sát ngoài hộp ở đợt 4).
    YoloSegModel(str(p), imgsz=128, conf=0.0, max_det=5, crop_expand=1.1)
