"""Biến thể hàm loss của YOLO-seg, bật bằng cờ `--loss` của train.py.

Không có `--loss` thì KHÔNG có gì ở đây chạy: không gắn callback, không nạp
module tính loss, ultralytics dùng đúng hàm loss gốc của nó. Bảng benchmark
phải được train bằng loss gốc, nên mặc định đó là bất biến chứ không phải
một lựa chọn trong config.

Vì vậy biến thể chỉ bật được bằng cờ. Khối `loss:` trong file config hay
`--set loss.use=...` đều bị chặn: một config mang sẵn biến thể thì lệnh train
trông y hệt lệnh baseline mà kết quả lại khác. Tham số của một biến thể đã bật
thì chỉnh bằng `--set loss.<tên>.<tham số>=<giá trị>`.

File này chỉ chứa bảng tra và phần kiểm tham số, không import torch hay
ultralytics: `--list-losses` và `--print-config` phải chạy được tức thì. Phần
tính loss của mỗi biến thể nằm ở module khai trong khoá `module` của nó
(vd mask_iou_loss.py) và chỉ được nạp trong install().

Module đó phải có `build_criterion(model, params)` và `EpochLog(path)`; có
thể có thêm `check_model(model, params)` để chặn tổ hợp model không hợp.
"""

from __future__ import annotations

import difflib
import importlib
from pathlib import Path

#: Phần lõi của mask_iou chép ~25 dòng từ `v8DetectionLoss.get_assigned_targets
#: _and_loss`. Phiên bản khác thì đoạn chép có thể lệch với thư viện mà không
#: lỗi nào hiện ra, nên install() dừng hẳn thay vì chạy.
ULTRALYTICS_VERSION = "8.4.143"


def _check_mask_iou(p: dict) -> None:
    mix = p["mix"]
    if isinstance(mix, bool) or not isinstance(mix, (int, float)) or not 0.0 <= float(mix) <= 1.0:
        raise SystemExit(f"loss.mask_iou.mix phải là số trong [0, 1], nhận được {mix!r}")
    wu = p["warmup_epochs"]
    if isinstance(wu, bool) or not isinstance(wu, int) or wu < 0:
        raise SystemExit(f"loss.mask_iou.warmup_epochs phải là số nguyên >= 0, nhận được {wu!r}")
    if p["heads"] not in ("both", "o2o"):
        raise SystemExit(f"loss.mask_iou.heads phải là 'both' hoặc 'o2o', nhận được {p['heads']!r}")


def _check_dice(p: dict) -> None:
    w = p["weight"]
    if isinstance(w, bool) or not isinstance(w, (int, float)) or float(w) < 0.0:
        raise SystemExit(f"loss.dice.weight phải là số >= 0, nhận được {w!r}")


def _number(name: str, v, lo: float = 0.0) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) < lo:
        raise SystemExit(f"loss.{name} phải là số >= {lo:g}, nhận được {v!r}")


def _check_vfl(p: dict) -> None:
    _number("vfl.alpha", p["alpha"])
    _number("vfl.gamma", p["gamma"])


def _check_ncb(p: dict) -> None:
    _number("ncb.sigma", p["sigma"])


def _integer(name: str, v, lo: int) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise SystemExit(f"loss.{name} phải là số nguyên >= {lo}, nhận được {v!r}")


def _check_boundary(p: dict) -> None:
    _number("boundary.weight", p["weight"])
    _integer("boundary.clip", p["clip"], 1)
    _integer("boundary.start_epoch", p["start_epoch"], 0)
    _integer("boundary.ramp_epochs", p["ramp_epochs"], 0)


def _check_repgt(p: dict) -> None:
    _number("repgt.weight", p["weight"])
    s = p["sigma"]
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0.0 <= float(s) < 1.0:
        raise SystemExit(f"loss.repgt.sigma phải là số trong [0, 1), nhận được {s!r}")


def _check_alpha_ciou(p: dict) -> None:
    _number("alpha_ciou.alpha", p["alpha"], 1.0)


def _unit(name: str, v) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
        raise SystemExit(f"loss.{name} phải là số trong [0, 1], nhận được {v!r}")


def _check_lm1(p: dict) -> None:
    _number("lm1.weight", p["weight"])
    _number("lm1.margin", p["margin"])
    _number("lm1.temp", p["temp"], 1e-6)
    _integer("lm1.k", p["k"], 1)
    _integer("lm1.start_epoch", p["start_epoch"], 0)
    _integer("lm1.ramp_epochs", p["ramp_epochs"], 0)
    _unit("lm1.bg_iou", p["bg_iou"])
    _integer("lm1.bg_top", p["bg_top"], 1)
    _unit("lm1.q_min", p["q_min"])
    if p["bg_top"] < p["k"]:
        raise SystemExit(f"loss.lm1.bg_top ({p['bg_top']}) phải >= loss.lm1.k ({p['k']})")


def _check_lm3(p: dict) -> None:
    _number("lm3.alpha", p["alpha"], 1.0)
    _unit("lm3.tau_s", p["tau_s"])
    _unit("lm3.tau_u", p["tau_u"])
    _number("lm3.kappa", p["kappa"], 1e-6)


#: tên -> mô tả, tham số (mặc định, nhãn ngắn trong tên thư mục, giải thích), hàm kiểm.
#: Thứ tự khai ở đây là thứ tự trong tên thư mục, nên `--loss b,a` và
#: `--loss a,b` ra cùng một tên.
LOSSES: dict = {
    "mask_iou": {
        "doc": "L1: điểm phân loại học IoU MẶT NẠ (cắt bằng hộp dự đoán) thay cho CIoU của hộp",
        "params": {
            "mix": (1.0, "mix", "tỉ trọng IoU mặt nạ trong mục tiêu phân loại: 0 = y hệt gốc, 1 = thuần IoU mặt nạ"),
            "warmup_epochs": (5, "wu", "số epoch đầu tăng tuyến tính tỉ trọng từ 0 lên mix"),
            "heads": ("both", "h", "both = cả hai đầu của YOLO26; o2o = chỉ đầu one-to-one (đầu dùng lúc suy luận)"),
        },
        "check": _check_mask_iou,
        "module": "mask_iou_loss",
    },
    "dice": {
        "doc": "L2: cộng Dice vào BCE của mặt nạ từng tán, cùng vùng giám sát (trong box GT)",
        "params": {
            "weight": (1.0, "w", "hệ số của Dice so với BCE: 0 = y hệt gốc, 1 = nặng ngang BCE"),
        },
        "check": _check_dice,
        "module": "dice_loss",
    },
    "nosem": {
        "doc": "đối chứng của TC2: tắt loss ngữ nghĩa phụ của Proto26 (chỉ YOLO26), mọi loss khác giữ nguyên",
        "params": {},
        "check": lambda p: None,
        "module": "nosem_loss",
    },
    "vfl": {
        "doc": "TH2: Varifocal thay BCE của điểm, chỉ ở đầu one-to-many; đích TAL giữ nguyên",
        "params": {
            "alpha": (0.75, "a", "trọng số âm tính alpha·p^gamma; dương tính nhân với chính đích của nó"),
            "gamma": (2.0, "g", "số mũ của p ở âm tính: âm tính càng dễ càng nhẹ"),
        },
        "check": _check_vfl,
        "module": "vfl_loss",
    },
    "ncb": {
        "doc": "TH1: BCE mặt nạ với nhãn mềm theo tỉ lệ phủ của từng ô (NCB của T1, Proto gốc, không GPR)",
        "params": {
            "sigma": (2.4, "s", "Gauss làm mềm nhãn, px ảnh đầu vào (bằng nhiễu nhãn đã đo); 0 = chỉ tỉ lệ phủ"),
        },
        "check": _check_ncb,
        "module": "ncb_loss",
        # Tỉ lệ phủ phải tính từ nhãn raster ở đủ độ phân giải: mask_ratio 4
        # chỉ còn nhãn nhị phân ở lưới 1/4, đúng thứ NCB thay đi.
        "train": {"mask_ratio": 1},
    },
    "boundary": {
        "doc": "TH3: BCE mặt nạ + λ·Boundary loss, khoảng cách có dấu tới biên nhãn cắt ở ±clip ô",
        "params": {
            "weight": (0.1, "w", "λ sau khi tăng xong; 0 = y hệt gốc"),
            "clip": (5, "c", "cắt khoảng cách ở bao nhiêu ô lưới mặt nạ (1 ô ≈ 10 px ảnh gốc)"),
            "start_epoch": (10, "st", "λ = 0 trong ngần này epoch đầu"),
            "ramp_epochs": (10, "r", "sau đó tăng tuyến tính lên weight trong ngần này epoch"),
        },
        "check": _check_boundary,
        "module": "boundary_loss",
    },
    "repgt": {
        "doc": "TH4: CIoU + λ·RepGT (phạt hộp lấn sang tán bên cạnh), chỉ ở đầu one-to-many",
        "params": {
            "weight": (0.5, "w", "λ của RepGT so với CIoU; 0 = y hệt gốc"),
            "sigma": (0.5, "s", "ngưỡng của Smooth_ln, trong [0, 1): trên ngưỡng thì phạt tuyến tính"),
        },
        "check": _check_repgt,
        "module": "repgt_loss",
    },
    "alpha_ciou": {
        "doc": "TH5: α-CIoU thay CIoU trong loss hộp, cả hai đầu; bộ gán TAL vẫn dùng CIoU",
        "params": {
            "alpha": (3.0, "a", "luỹ thừa của cả ba phần CIoU; 1 = y hệt gốc, lớn hơn = dồn vào hộp IoU cao"),
        },
        "check": _check_alpha_ciou,
        "module": "alpha_ciou_loss",
    },
    "lm1": {
        "doc": "LM1: đẩy ứng viên vẽ đúng của mỗi tán vượt mức nền thật của ảnh, một chiều, chỉ đầu one-to-many",
        "params": {
            "weight": (0.5, "w", "λ sau khi tăng xong; 0 = y hệt gốc"),
            "margin": (1.0, "m", "biên Δ (logit) phải vượt mức nền"),
            "temp": (0.25, "t", "nhiệt độ T của softplus"),
            "k": (3, "k", "mức nền = logit lớn thứ k trong các neo nền của ảnh"),
            "start_epoch": (5, "st", "λ = 0 trong ngần này epoch đầu"),
            "ramp_epochs": (5, "r", "sau đó tăng tuyến tính lên weight trong ngần này epoch"),
            "bg_iou": (0.1, "bi", "neo âm chỉ tính là nền khi mặt nạ có IoU < ngưỡng này với mọi tán"),
            "bg_top": (300, "bt", "số neo âm điểm cao nhất được xét khi tìm mức nền"),
            "q_min": (0.5, "q", "tán chỉ được đẩy khi ứng viên tốt nhất có IoU mặt nạ >= ngưỡng này"),
        },
        "check": _check_lm1,
        "module": "lm1_loss",
    },
    "lm3": {
        "doc": "LM3: α-CIoU chỉ cho tán đã chắc (điểm và CIoU lớn nhất của tán vượt ngưỡng), tán yếu giữ CIoU gốc",
        "params": {
            "alpha": (3.0, "a", "luỹ thừa của α-CIoU ở tán bật cổng; 1 = y hệt gốc"),
            "tau_s": (0.6, "ts", "ngưỡng điểm lớn nhất của tán để bật cổng"),
            "tau_u": (0.7, "tu", "ngưỡng CIoU hộp lớn nhất của tán để bật cổng"),
            "kappa": (0.05, "kp", "độ mềm của cổng sigmoid"),
        },
        "check": _check_lm3,
        "module": "lm3_loss",
    },
}


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def apply(cfg: dict, spec: str | None) -> str:
    """Ghi biến thể loss đã kiểm vào `cfg["loss"]`, trả về hậu tố cho tên run.

    Không có `spec` thì trả về "" và cfg không có khoá `loss`. Hậu tố gồm tên
    biến thể và mọi tham số khác mặc định, để hai cấu hình khác nhau không bao
    giờ ra cùng một tên thư mục.
    """
    block = cfg.get("loss")
    if block is not None and not isinstance(block, dict):
        raise SystemExit(f"khoá `loss` phải là khối lồng nhau, nhận được {block!r}")
    if block and "use" in block:
        raise SystemExit(
            "Biến thể loss chỉ bật bằng cờ --loss, không ghi `loss.use` trong config "
            "hay --set: lệnh train phải tự nói nó có dùng loss khác gốc hay không."
        )
    if not spec:
        if block:
            raise SystemExit(
                f"Có `loss.{next(iter(block))}...` (từ config hoặc --set) nhưng không có --loss. "
                "Tham số loss chỉ có tác dụng kèm cờ --loss <tên>."
            )
        cfg.pop("loss", None)
        return ""

    asked = [s.strip() for s in str(spec).split(",") if s.strip()]
    for name in asked:
        if name not in LOSSES:
            near = difflib.get_close_matches(name, sorted(LOSSES), n=3, cutoff=0.5)
            hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
            raise SystemExit(f"--loss {name!r} không có.{hint} Xem --list-losses")
    use = [n for n in LOSSES if n in asked]
    if len(use) > 1:
        # Không phải luật cấm, mà vì ghép thẳng sẽ chạy sai trong im lặng: mỗi
        # biến thể thay một chỗ khác của hàm loss, và mask_iou tự tính BCE
        # mặt nạ trong bản chép riêng, không gọi single_mask_loss mà dice cài
        # đè. Lượt chạy sẽ mang tên `mask-iou-dice` trong khi Dice không hề
        # được tính. Muốn ghép thì phải cho mask_iou đi qua cùng một hàm mặt
        # nạ, kèm test chứng minh cả hai tác dụng cùng có mặt.
        raise SystemExit(
            f"--loss {','.join(use)}: mỗi lượt chỉ bật được một biến thể loss. "
            "Ghép thẳng thì biến thể sau không chạy mà tên run vẫn ghi là có."
        )
    for extra in set(block or {}) - set(use):
        raise SystemExit(f"Có tham số cho `loss.{extra}` nhưng --loss không bật {extra!r}.")

    out: dict = {"use": use}
    parts = []
    for name in use:
        spec_ = LOSSES[name]
        given = (block or {}).get(name) or {}
        if not isinstance(given, dict):
            raise SystemExit(f"`loss.{name}` phải là khối tham số, nhận được {given!r}")
        for k in given:
            if k not in spec_["params"]:
                near = difflib.get_close_matches(k, sorted(spec_["params"]), n=3, cutoff=0.5)
                hint = f" Ý bạn là: {', '.join(near)}?" if near else ""
                raise SystemExit(f"loss {name!r} không có tham số {k!r}.{hint} Xem --list-losses")
        defaults = {k: v[0] for k, v in spec_["params"].items()}
        merged = {**defaults, **given}
        spec_["check"](merged)
        # Tham số train mà biến thể bắt buộc: không tự đặt hộ, để lệnh train
        # tự nói ra nó khác baseline ở đâu.
        for k, need in spec_.get("train", {}).items():
            got = (cfg.get("train") or {}).get(k)
            if got != need:
                raise SystemExit(
                    f"--loss {name} cần train.{k}={need} (đang là {got!r}). "
                    f"Thêm --set train.{k}={need} vào lệnh train."
                )
        out[name] = merged
        diff = [f"{spec_['params'][k][1]}{_fmt(v)}" for k, v in merged.items() if v != defaults[k]]
        parts.append("-".join([name.replace("_", "-"), *diff]))
    out["tag"] = "-".join(parts)
    cfg["loss"] = out
    return out["tag"]


def describe() -> str:
    """Bảng cho --list-losses."""
    lines = [f"{len(LOSSES)} biến thể loss (bỏ trống --loss = loss gốc của ultralytics):", ""]
    for name, spec_ in LOSSES.items():
        lines.append(f"  {name}: {spec_['doc']}")
        for k, (default, _tag, doc) in spec_["params"].items():
            lines.append(f"      {name + '.' + k:<24} mặc định {default!r:<7} {doc}")
        for k, need in spec_.get("train", {}).items():
            lines.append(f"      cần --set train.{k}={need}")
    lines += ["", "Bật:       --loss dice           (mỗi lượt một biến thể)",
              "Chỉnh:     --loss dice --set loss.dice.weight=2"]
    return "\n".join(lines)


def describe_active(block: dict | None) -> str:
    """Một dòng cho run.log, in cả khi không bật gì: phép kiểm im lặng là phép
    kiểm không ai biết đã chạy hay chưa."""
    if not block:
        return "Loss: mặc định của ultralytics"
    return "Loss: " + "; ".join(
        f"{n} ({', '.join(f'{k} {_fmt(v)}' for k, v in block[n].items())})" if block[n] else n
        for n in block["use"]
    )


def install(yolo_model, block: dict, run_dir: str | Path) -> None:
    """Gắn biến thể loss vào lượt train sắp chạy của `yolo_model`.

    Ultralytics chỉ tạo hàm loss khi lần đầu cần tới (`BaseModel.loss`), và
    `on_train_start` bắn sau khi trainer dựng xong model nhưng trước batch đầu
    tiên, nên gắn ở đó là đủ — không sửa dòng nào của thư viện. Hàm loss chỉ
    sống trên model đang train: checkpoint lưu bản EMA và gỡ `criterion` ra
    trước khi lưu (`trainer.py`, save_model), nên best.pt vẫn là
    SegmentationModel chuẩn, `YOLO(best.pt)` nạp được ở mọi nơi.

    Nhật ký từng epoch ghi vào `run_dir/loss_<tên>.csv`.
    """
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise SystemExit(
            f"--loss viết cho ultralytics {ULTRALYTICS_VERSION}, môi trường đang có "
            f"{ultralytics.__version__}. Biến thể loss dựa vào nội bộ của thư viện "
            "(đoạn chép, chữ ký hàm) nên phải đối chiếu lại trước khi chạy."
        )
    (name,) = block["use"]  # apply() chỉ cho một biến thể mỗi lượt
    mod = importlib.import_module(f".{LOSSES[name]['module']}", __package__)
    p = block[name]
    check = getattr(mod, "check_model", None)
    if check is not None:
        check(yolo_model.model, p)
    log = mod.EpochLog(Path(run_dir) / f"loss_{name}.csv")

    def on_train_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        m = unwrap_model(trainer.model)
        m.criterion = mod.build_criterion(m, p)

    def on_train_epoch_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        log.start(unwrap_model(trainer.model).criterion, trainer.epoch)

    def on_train_epoch_end(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        log.end(unwrap_model(trainer.model).criterion, trainer.epoch)

    yolo_model.add_callback("on_train_start", on_train_start)
    yolo_model.add_callback("on_train_epoch_start", on_train_epoch_start)
    yolo_model.add_callback("on_train_epoch_end", on_train_epoch_end)
