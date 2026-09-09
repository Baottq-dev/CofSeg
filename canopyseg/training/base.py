"""Hợp đồng cho trainer.

scripts/train.py chỉ làm việc với lớp này. Thêm Mask R-CNN, detectron2, hay
model tự viết = thêm một file trong canopyseg/training/ có @register, không
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
