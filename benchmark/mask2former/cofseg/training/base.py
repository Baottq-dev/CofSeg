"""Hợp đồng cho trainer.

benchmark/mask2former/train.py chỉ làm việc với lớp này. Thêm Mask R-CNN, detectron2, hay
model tự viết = thêm một file trong cofseg/training/ có @register, không
sửa dòng nào ở script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Trainer(ABC):
    """Nhận config + thư mục run, trả về tóm tắt kèm đường dẫn trọng số."""

    def __init__(self, cfg: dict, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)

    # --------------------------------------------------------------- thời gian
    def _time_row(self, seconds) -> str:
        """Dòng "thời gian" của khối cuối, tách train khỏi val khi biết được.

        Gộp hai số làm một rồi dán nhãn "train" là nói dối: val chạy batch 1
        nên trên bộ này nó tốn gấp đôi train, và ai đọc cũng sẽ tưởng train
        mới là chỗ hết giờ máy. Trainer nào giữ một reporter có `train_total`
        và `val_total` thì hai số đó được in riêng; không có thì in tổng, và
        nhãn nói đúng rằng đó là tổng.
        """
        from .. import progress

        rep = getattr(self, "_reporter", None)
        train_s = getattr(rep, "train_total", 0.0) or 0.0
        val_s = getattr(rep, "val_total", 0.0) or 0.0
        if train_s and val_s:
            return (f"train {progress.fmt_time(train_s)}"
                    f" + val {progress.fmt_time(val_s)}")
        return f"train + val {progress.fmt_time(seconds)}"

    # -------------------------------------------------------------------- đặt tên
    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        """Hậu tố ngắn mô tả cấu hình THẬT, để tên thư mục không nói dối.

        Là classmethod vì tên thư mục phải có trước khi dựng trainer — trainer
        nhận run_dir trong hàm khởi tạo. Mỗi họ model tự quyết tham số nào đáng
        đưa vào tên; mặc định không thêm gì.
        """
        return ""

    #: Khoá trong `data:` nhận thư mục fold. detectron2/mmdet đọc thẳng thư
    #: mục (`data.root`); ultralytics đọc file mô tả bên trong nó
    #: (`data.yaml`). Nhờ khai ở đây mà `--data <thư-mục-fold>` viết giống
    #: nhau cho mọi model, thay vì người chạy phải nhớ model nào cần cái gì.
    DATA_KEY = "root"

    @classmethod
    def data_arg(cls, path: str) -> tuple[str, str]:
        """Thư mục fold -> (khoá config, giá trị) cho --data."""
        p = str(path).replace("\\", "/").rstrip("/")
        return (f"data.{cls.DATA_KEY}",
                f"{p}/data.yaml" if cls.DATA_KEY == "yaml" else p)

    # ----------------------------------------------------------------- backbone
    #: arch -> {tên ngắn: phần config quyết định backbone}. Rỗng nghĩa là họ
    #: model này không đổi backbone được; YOLO là trường hợp đó, backbone của
    #: nó gắn liền với kiến trúc nên chỗ đổi là CỠ model, không phải backbone.
    BACKBONES: dict = {}

    @classmethod
    def arch_of(cls, cfg: dict) -> str:
        """Kiến trúc config đang chọn — nó quyết định bảng backbone nào áp dụng."""
        m = cfg.get("model")
        return str(m.get("arch", "")) if isinstance(m, dict) else ""

    @classmethod
    def backbones(cls, cfg: dict) -> dict:
        return cls.BACKBONES.get(cls.arch_of(cfg), {})

    @classmethod
    def apply_backbone(cls, cfg: dict, name: str) -> str:
        """Đặt backbone `name` vào `cfg`, trả về tên gợi ý cho thư mục run.

        Mặc định này hợp với detectron2: `model.config_file` chọn kiến trúc và
        `model.weights` chọn checkpoint COCO đi kèm. Trainer nào khai config
        kiểu khác thì cài đè.
        """
        bang = cls.backbones(cfg)
        arch = cls.arch_of(cfg) or "model này"
        if not bang:
            raise SystemExit(f"{arch} không đổi backbone được.")
        if name not in bang:
            raise SystemExit(
                f"--backbone {name!r} không có với {arch}. Có: {', '.join(sorted(bang))}"
            )
        spec = bang[name]
        m = cfg.setdefault("model", {})
        m["config_file"] = spec["config"]
        # checkpoint None = suy ra từ chính config đó (model_zoo của detectron2).
        if spec.get("checkpoint"):
            m["weights"] = spec["checkpoint"]
        else:
            m.pop("weights", None)
        return f"{cls.arch_of(cfg)}-{name}"

    @classmethod
    def apply_model(cls, cfg: dict, name: str) -> str:
        """Đổi model/cỡ model. Chỉ họ YOLO dùng; ba họ kia đổi bằng --backbone."""
        raise SystemExit(
            "--model chỉ dùng cho họ YOLO. Model này đổi kiến trúc bằng --config "
            "và đổi backbone bằng --backbone."
        )

    @classmethod
    def describe_backbones(cls, cfg: dict) -> str:
        bang = cls.backbones(cfg)
        arch = cls.arch_of(cfg) or "model này"
        if not bang:
            return f"{arch} không đổi backbone được."
        rong = max(len(k) for k in bang)
        dong = [f"{len(bang)} backbone dùng được với arch {arch!r}:", ""]
        for k in sorted(bang):
            spec = bang[k]
            dong.append(f"  {k:<{rong}}  {spec.get('note', spec['config'])}")
        return "\n".join(dong)

    # ------------------------------------------------------------------ tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        """Tên -> mặc định của mọi tham số mà khối `train:` nhận.

        benchmark/mask2former/train.py dùng nó để bắt lỗi gõ sai tên và để in --list-params.
        None nghĩa là trainer không liệt kê được (mọi tên đều được nhận) — đó là
        một lựa chọn phải có chủ đích, vì gõ sai sẽ không có gì báo.
        """
        return None

    @classmethod
    def param_names(cls) -> set[str] | None:
        d = cls.param_defaults()
        return None if d is None else set(d)

    @classmethod
    def locked_params(cls) -> frozenset[str]:
        """Khoá trainer tự đặt để kết quả rơi đúng thư mục run; ghi đè từ dòng
        lệnh sẽ làm hỏng chính chỗ ghi kết quả nên bị chặn ngay."""
        return frozenset()

    # ------------------------------------------------------------------ vòng đời
    def prepare(self) -> dict:
        """Kiểm dữ liệu trước khi đụng tới GPU.

        Mặc định không làm gì. Nên cài đè: phát hiện nhãn hỏng ở đây rẻ hơn
        nhiều so với phát hiện sau ba tiếng huấn luyện.
        """
        return {}

    def probe(self) -> dict:
        """Ước lượng VRAM cần cho cấu hình hiện tại, không huấn luyện thật."""
        return {"supported": False}

    @abstractmethod
    def fit(self) -> dict:
        """Huấn luyện. Trả về dict có tối thiểu khoá 'weights'."""
