"""Hợp đồng cho trainer.

benchmark/yolo11/train.py chỉ làm việc với lớp này. Thêm Mask R-CNN, detectron2, hay
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

    # ------------------------------------------------------------------ tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        """Tên -> mặc định của mọi tham số mà khối `train:` nhận.

        benchmark/yolo11/train.py dùng nó để bắt lỗi gõ sai tên và để in --list-params.
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
