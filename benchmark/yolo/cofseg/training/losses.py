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
tính loss nằm ở mask_iou_loss.py và chỉ được nạp trong install().
"""

from __future__ import annotations

import difflib
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
            lines.append(f"      {name}.{k:<14} mặc định {default!r:<7} {doc}")
    lines += ["", "Bật:       --loss mask_iou",
              "Chỉnh:     --loss mask_iou --set loss.mask_iou.mix=0.5"]
    return "\n".join(lines)


def describe_active(block: dict | None) -> str:
    """Một dòng cho run.log, in cả khi không bật gì: phép kiểm im lặng là phép
    kiểm không ai biết đã chạy hay chưa."""
    if not block:
        return "Loss: mặc định của ultralytics"
    return "Loss: " + "; ".join(
        f"{n} ({', '.join(f'{k} {_fmt(v)}' for k, v in block[n].items())})" for n in block["use"]
    )


def install(yolo_model, block: dict, log_path: str | Path) -> None:
    """Gắn biến thể loss vào lượt train sắp chạy của `yolo_model`.

    Ultralytics chỉ tạo hàm loss khi lần đầu cần tới (`BaseModel.loss`), và
    `on_train_start` bắn sau khi trainer dựng xong model nhưng trước batch đầu
    tiên, nên gắn ở đó là đủ — không sửa dòng nào của thư viện. Hàm loss chỉ
    sống trên model đang train: checkpoint lưu bản EMA và gỡ `criterion` ra
    trước khi lưu (`trainer.py`, save_model), nên best.pt vẫn là
    SegmentationModel chuẩn, `YOLO(best.pt)` nạp được ở mọi nơi.
    """
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise SystemExit(
            f"--loss viết cho ultralytics {ULTRALYTICS_VERSION}, môi trường đang có "
            f"{ultralytics.__version__}. Phần gán nhãn được chép từ thư viện nên phải "
            "đối chiếu lại trước khi chạy."
        )
    from .mask_iou_loss import build_criterion, is_end2end, EpochLog

    p = block["mask_iou"]
    if p["heads"] == "o2o" and not is_end2end(yolo_model.model):
        raise SystemExit(
            "loss.mask_iou.heads=o2o chỉ có nghĩa với model đầu-cuối (YOLO26). "
            "YOLOv8/11 chỉ có một đầu, dùng heads=both."
        )
    log = EpochLog(log_path)

    def on_train_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        m = unwrap_model(trainer.model)
        m.criterion = build_criterion(m, p)

    def on_train_epoch_start(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        log.start(unwrap_model(trainer.model).criterion, trainer.epoch)

    def on_train_epoch_end(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        log.end(unwrap_model(trainer.model).criterion, trainer.epoch)

    yolo_model.add_callback("on_train_start", on_train_start)
    yolo_model.add_callback("on_train_epoch_start", on_train_epoch_start)
    yolo_model.add_callback("on_train_epoch_end", on_train_epoch_end)
