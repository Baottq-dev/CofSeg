"""Trò nhẹ: YOLO26-seg độ rộng 0,375 hoặc 0,25 với ba thay đổi ở đầu (configs/train/tro.yaml, tro-025.yaml).

Chi phí đo bằng thop (2 × MACs, 1024×576, đồ thị đã fuse, 1 lớp); test giữ đúng các số đó. Ba thay đổi,
mỗi cái tắt riêng được bằng khối `student:` để làm ablation:

- **Proto stride 8.** Proto26 gốc: feat_fuse -> cv1 (3×3, s8) -> ConvTranspose x2 -> cv2 (3×3, s4) -> cv3.
  Ở đây cv2 chạy TRƯỚC bước phóng to, ở s8, nên lớp 3×3 thứ hai rẻ đi 4 lần. Chỉ đổi thứ tự, không đổi lớp
  nào: cv2 là c_ -> c_ nên đặt trước hay sau ConvTranspose đều khớp kênh, và prototype ra vẫn ở stride 4.
- **Bỏ nhánh phát hiện P3.** Hộp, điểm và hệ số chỉ đoán ở P4, P5 (stride 16, 32). Tán ở đây lớn (trung vị
  cạnh 325 px ảnh gốc, ~130 px ở imgsz 1024), nên anchor stride 8 không cần. Proto vẫn nhận P3–P5 như gốc:
  đầu chỉ bỏ ba nhánh ở cấp 0, giữ bề rộng các nhánh còn lại (vẫn tính theo kênh P3 như gốc), đúng cấu hình
  đã đo chi phí.
- **16 prototype** thay 32: vector hệ số cần đoán ngắn đi một nửa.

**Khởi tạo** (`init: select`, mặc định): lấy trọng số từ checkpoint COCO ở khoá `model:` của config, gọi là
nguồn. Tensor cùng cỡ thì chép nguyên; tensor hẹp hơn thì chọn đều theo Weight Selection (Xu và cộng sự,
"Initializing Models with Larger Ones", ICLR 2024): mỗi chiều kênh S của nguồn -> T của trò lấy các chỉ số
⌊k·S/T⌋, k = 0..T−1. Nguồn không được hẹp hơn trò.

- Trò 0,25 lấy từ YOLO26n-seg COCO, cùng độ rộng: gần như mọi tensor chép nguyên, chỉ các lớp theo số
  prototype (32 -> 16) phải chọn. Đây là khởi tạo COCO thật, như YOLO26n-seg trong benchmark.
- Trò 0,375 không có checkpoint COCO cùng độ rộng, nên lấy từ YOLO26s-seg COCO. Mọi khối kênh của trò (nhánh
  cv1 tách đôi của C3k2, các đầu vào của Concat, bốn bản sao của SPPF) đều bằng đúng 0,75 khối của nguồn,
  nên phép chọn đều tự khớp giữa lớp xuất và lớp nhận: chỉ số k trong khối j của trò rơi vào đúng khối j
  của nguồn. Ngoại lệ là Attention: số head đổi (4 -> 3) còn cỡ head giữ nguyên (64), nên ở đó chọn nguyên
  head. Cách này giữ trọn bộ lọc tầng nông nhưng không giữ hàm của nguồn ở lớp sâu (tương quan đặc trưng còn
  khoảng 0,1 sau vài khối): đây là khởi tạo, không phải nén.

Lớp cuối của nhánh lớp và của semseg (80 lớp COCO -> 1) giữ khởi tạo như mọi lượt fine-tune của benchmark.
`init: scratch` bỏ bước này.

Bỏ nhánh P3 làm SAU khi chọn trọng số, nên nhánh P4, P5 nhận đúng trọng số của chính chúng ở thầy.

Checkpoint của trò chứa lớp ở đây, nên muốn nạp nó phải import được `distill` (xem distill/__init__.py và
evaluate.py).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Attention, Proto26
from ultralytics.nn.modules.head import Detect, Segment26
from ultralytics.nn.tasks import SegmentationModel, yaml_model_load

#: Tên cỡ của trò trong khối `scales` của yaml. Ultralytics chỉ biết n/s/m/l/x; khoá `scale` của yaml thắng
#: phép đoán theo tên file, nên một chữ mới là đủ.
SCALE = "t"


class Proto26S8(Proto26):
    """Proto26 với lớp 3×3 thứ hai (cv2) chạy ở stride 8, trước bước phóng to.

    Chỉ tạo bằng cách đổi lớp của một Proto26 đã dựng (to_student), nên tên và cỡ mọi lớp giữ nguyên.
    """

    def forward(self, x, return_semantic: bool = True):
        # Bốn dòng gộp P3–P5 chép Proto26.forward (ultralytics 8.4.143); chỉ dòng tạo p đổi thứ tự.
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            feat = feat + F.interpolate(f(x[i + 1]), scale_factor=2 ** (i + 1), mode="nearest")
        p = self.cv3(self.upsample(self.cv2(self.cv1(self.feat_fuse(feat)))))
        if self.training and return_semantic:
            return p, self.semseg(feat)
        return p


class Segment26Student(Segment26):
    """Segment26 chỉ đoán hộp, điểm và hệ số ở P4, P5; Proto vẫn nhận P3–P5.

    `f` của lớp vẫn là ba cấp như gốc; `det_skip` cấp đầu chỉ bị bỏ ở nhánh phát hiện. stride, anchor và
    bộ gán nhãn đều đọc từ `feats` mà nhánh phát hiện trả về, nên tự đi theo hai cấp còn lại.
    """

    det_skip: int = 1

    def forward(self, x):
        # Chép Segment26.forward (ultralytics 8.4.143); chỉ khác ở x đưa vào nhánh phát hiện.
        outputs = Detect.forward(self, x[self.det_skip:])
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


# ---------------------------------------------------------------- dựng
def student_yaml(base, spec: dict) -> dict:
    """yaml của trò từ yaml của YOLO26-seg: thêm cỡ `t`, đổi số prototype. Không đổi `base`."""
    d = copy.deepcopy(yaml_model_load(base) if isinstance(base, str) else base)
    ref = d["scales"].get("s")
    if ref is None or d["head"][-1][2] != "Segment26":
        raise SystemExit("Trò dựng từ yaml YOLO26-seg (có cỡ s và đầu Segment26); yaml này không phải.")
    d["scales"] = {**d["scales"], SCALE: [ref[0], float(spec["width"]), ref[2]]}
    d["scale"] = SCALE
    # Tên trong dòng tóm tắt model của ultralytics; giữ tên yaml gốc thì log in "YOLO26s-seg summary".
    d["yaml_file"] = "tro-seg.yaml"
    head = list(d["head"][-1])
    args = list(head[-1])
    args[1] = int(spec["nm"])
    head[-1] = args
    d["head"] = [*d["head"][:-1], head]
    return d


def to_student(model: SegmentationModel, spec: dict) -> SegmentationModel:
    """Đổi TẠI CHỖ đầu của một YOLO26-seg đã dựng sang đầu của trò (Proto stride 8, bỏ nhánh P3)."""
    head = model.model[-1]
    if type(head) is not Segment26:
        raise SystemExit(f"Trò viết cho đầu Segment26; model này có {type(head).__name__}.")
    if spec["proto_s8"]:
        head.proto.__class__ = Proto26S8
    if spec["drop_p3"]:
        k = Segment26Student.det_skip
        for name in ("cv2", "cv3", "cv4", "one2one_cv2", "one2one_cv3", "one2one_cv4"):
            branch = getattr(head, name, None)
            if branch is not None:
                setattr(head, name, nn.ModuleList(list(branch)[k:]))
        head.nl -= k
        head.stride = head.stride[k:].clone()
        head.anchors, head.strides, head.shape = torch.empty(0), torch.empty(0), None
        head.__class__ = Segment26Student
        model.stride = head.stride
    return model


def _uniform(s: int, t: int) -> torch.Tensor:
    """⌊k·s/t⌋, k = 0..t−1: t chỉ số rải đều trên s."""
    return torch.arange(t) * s // t


def _heads(nh_s: int, nh_t: int, block: int) -> torch.Tensor:
    """Chỉ số kênh khi chọn nguyên head: nh_t head rải đều trong nh_s, mỗi head một khối `block` kênh liền."""
    h = _uniform(nh_s, nh_t)
    return (h[:, None] * block + torch.arange(block)[None]).reshape(-1)


def select_weights(student: nn.Module, teacher: nn.Module) -> dict:
    """Nạp vào `student` trọng số chọn đều từ `teacher` cùng kiến trúc, rộng hơn hoặc bằng.

    Hai model phải dựng từ cùng một yaml, chỉ khác độ rộng (và số prototype), để tên tham số trùng nhau.
    Trả về số đếm cho nhật ký: chép nguyên, chọn, bỏ qua (giữ khởi tạo).
    """
    tsd = {k: v.detach().float() for k, v in teacher.state_dict().items()}
    ssd = student.state_dict()
    # Attention: chọn nguyên head. qkv xếp theo head: mỗi head [q (key_dim), k (key_dim), v (head_dim)];
    # v, pe và đầu vào của proj xếp theo head, mỗi head head_dim kênh. Đầu ra của proj quay về luồng dư
    # của khối, nên chọn đều như mọi chỗ khác.
    special: dict[str, dict[int, torch.Tensor]] = {}
    for p, m in student.named_modules():
        if not isinstance(m, Attention):
            continue
        tm = teacher.get_submodule(p)
        if (m.head_dim, m.key_dim) != (tm.head_dim, tm.key_dim):
            raise SystemExit(f"{p}: cỡ head của trò ({m.head_dim}, {m.key_dim}) khác thầy ({tm.head_dim}, {tm.key_dim})")
        qkv = _heads(tm.num_heads, m.num_heads, 2 * m.key_dim + m.head_dim)
        hd = _heads(tm.num_heads, m.num_heads, m.head_dim)
        for leaf in ("conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"):
            special[f"{p}.qkv.{leaf}"] = {0: qkv}
            special[f"{p}.pe.{leaf}"] = {0: hd}
        special[f"{p}.proj.conv.weight"] = {1: hd}
    nc_t, nc_s = teacher.model[-1].nc, student.model[-1].nc
    new, n = {}, {"chép": 0, "chọn": 0, "bỏ qua": 0}
    for name, s in ssd.items():
        t = tsd.get(name)
        if t is None or t.ndim != s.ndim:
            n["bỏ qua"] += 1
            continue
        # Nguồn hẹp hơn trò (vd trò 0,375 lấy từ YOLO26n): bỏ qua lặng lẽ thì gần như mọi tensor giữ khởi tạo
        # ngẫu nhiên mà log vẫn ghi là "chọn trọng số". Dừng hẳn.
        if any(b > a for a, b in zip(t.shape, s.shape)):
            raise SystemExit(f"{name}: nguồn {tuple(t.shape)} hẹp hơn trò {tuple(s.shape)}. Khoá `model:` của "
                             "config phải là checkpoint COCO rộng bằng hoặc hơn trò (0,25: yolo26n-seg; 0,375: yolo26s-seg).")
        # Lớp cuối của nhánh lớp và của semseg: 80 lớp COCO không có nghĩa cho 1 lớp tán.
        if nc_t != nc_s and any((a, b) == (nc_t, nc_s) for a, b in zip(t.shape, s.shape)):
            n["bỏ qua"] += 1
            continue
        x = t
        for d, (a, b) in enumerate(zip(t.shape, s.shape)):
            if a != b:
                x = x.index_select(d, special.get(name, {}).get(d, _uniform(a, b)))
        new[name] = x.to(s.dtype)
        n["chép" if x.shape == t.shape else "chọn"] += 1
    student.load_state_dict(new, strict=False)
    return n


def build_student(spec: dict, base_yaml, nc: int, ch: int = 3, teacher: nn.Module | None = None,
                  verbose: bool = False) -> tuple[SegmentationModel, dict | None]:
    """Dựng trò theo `spec` từ yaml của YOLO26-seg; chọn trọng số từ `teacher` nếu init = select.

    Trả về (model, số đếm của phép chọn hoặc None).
    """
    # Dựng im lặng: lúc này đầu còn nhánh P3, dòng tóm tắt in ra sẽ sai cỡ. In lại sau khi đổi đầu.
    model = SegmentationModel(student_yaml(base_yaml, spec), nc=nc, ch=ch, verbose=False)
    counts = None
    if spec["init"] == "select":
        if teacher is None:
            raise SystemExit("student.init = select cần một checkpoint COCO ở khoá `model:` của config.")
        counts = select_weights(model, teacher)
    to_student(model, spec)
    if verbose:
        model.info()
    return model, counts


def source_name(model) -> str:
    """Tên checkpoint nguồn cho nhật ký: đường dẫn .pt nếu có, không thì tên yaml."""
    pt = getattr(model, "pt_path", None)
    if pt:
        return str(pt).replace("\\", "/").rsplit("/", 1)[-1]
    return str(getattr(model, "yaml", {}).get("yaml_file", "?"))


def find_student_head(model) -> Segment26Student | None:
    """Đầu của trò trong một nn.Module, bản EMA hay AutoBackend; None nếu không có."""
    for cand in (model, getattr(getattr(model, "backend", None), "model", None)):
        if isinstance(cand, nn.Module):
            for m in cand.modules():
                if isinstance(m, Segment26Student):
                    return m
    return None
