"""Huấn luyện YOLO-seg qua ultralytics.

KHÔNG viết lại trainer — ultralytics làm tốt rồi. File này chỉ lo bốn việc mà
ultralytics không biết về dữ liệu này:

1. Mặc định của ultralytics sai cho ảnh chụp thẳng đứng từ UAV. flipud=0.0 và
   degrees=0.0 giả định ảnh có chiều "trên" cố định; ảnh nadir thì không.
   Lật dọc và xoay bất kỳ đều là biến đổi hợp lệ, bỏ không dùng là phí dữ liệu.

2. imgsz=640 phá hỏng bài toán này. Ảnh gốc 2560 px, lưới mặt nạ của YOLO là
   imgsz/4, nên ở 640 mỗi ô mặt nạ nuốt 16 px ảnh gốc. Tán ở đây có cạnh
   tương đương trung vị 325 px và toàn bộ giá trị nằm ở đường biên.

3. mask_ratio=4 hạ tiếp độ phân giải MỤC TIÊU huấn luyện thêm 4 lần nữa. Với
   dự án lấy đường biên làm trọng tâm thì đây là tham số đáng chú ý, không
   phải chi tiết phụ.

4. 8 GB VRAM và ảnh 2560x1440: đoán batch rồi OOM sau ba tiếng là kịch bản
   phải tránh, nên có bước dò trước.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..registry import register
from .base import Trainer

# Mặc định riêng cho ảnh UAV nadir, đè lên mặc định của ultralytics.
# Chỉ liệt kê thứ CỐ Ý khác đi; còn lại để ultralytics quyết.
AERIAL_DEFAULTS: dict = {
    "flipud": 0.5,  # ảnh nadir không có chiều "trên"
    "fliplr": 0.5,
    "degrees": 180.0,  # xoay bất kỳ đều hợp lệ
    "deterministic": True,
    "seed": 0,
}


@register("trainer", "yolo")
class YoloTrainer(Trainer):
    """Trainer cho mọi biến thể YOLO-seg của ultralytics (v8/v9/11/26)."""

    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        self.model_name: str = cfg["model"]
        self.data_yaml = Path(cfg["data"]["yaml"])
        self.train_args: dict = {**AERIAL_DEFAULTS, **(cfg.get("train") or {})}

    # ------------------------------------------------------------------ chuẩn bị
    def prepare(self) -> dict:
        """Để ultralytics tự xác nhận bộ dữ liệu trước khi đụng GPU.

        Dùng chính bộ đọc của thư viện sẽ tiêu thụ nhãn, chứ không tự viết
        phép kiểm riêng: cái ta cần biết là NÓ có đọc được không.
        """
        from ultralytics.data.utils import check_det_dataset

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Không thấy {self.data_yaml}. Chạy scripts/prepare_yolo_dataset.py trước."
            )
        info = check_det_dataset(str(self.data_yaml), autodownload=False)
        out = {
            "data_yaml": str(self.data_yaml),
            "nc": info.get("nc"),
            "names": info.get("names"),
            "splits": {
                k: str(info[k]) for k in ("train", "val", "test") if info.get(k)
            },
        }
        (self.run_dir / "dataset_check.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    # ---------------------------------------------------------------------- dò
    def probe(self) -> dict:
        """Ước lượng batch lớn nhất chạy được ở imgsz đang đặt.

        Dùng autobatch của ultralytics: nó nạp thật, chạy thật, đo thật. Kết
        quả là ƯỚC LƯỢNG — huấn luyện thật còn thêm bộ nhớ của dataloader và
        phân mảnh, nên nên lùi một bậc so với con số nó đưa ra.
        """
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.autobatch import check_train_batch_size

        if not torch.cuda.is_available():
            return {"supported": False, "reason": "không có CUDA"}

        imgsz = int(self.train_args.get("imgsz", 640))
        # Số vật thể mỗi ảnh ảnh hưởng lớn tới bộ nhớ của nhánh mặt nạ.
        max_obj = int(self.cfg.get("data", {}).get("max_objects_per_image", 64))
        model = YOLO(self.model_name)
        free_before = torch.cuda.mem_get_info()[0] / 2**20
        try:
            batch = check_train_batch_size(
                model.model.to("cuda"),
                imgsz=imgsz,
                amp=bool(self.train_args.get("amp", True)),
                max_num_obj=max_obj,
            )
        finally:
            del model
            torch.cuda.empty_cache()
        return {
            "supported": True,
            "imgsz": imgsz,
            "suggested_batch": int(batch),
            "recommended_batch": max(1, int(batch) - 1),
            "free_mb_before": round(free_before),
            "note": "lùi một bậc so với suggested; dataloader và phân mảnh chưa tính vào",
        }

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        from ultralytics import YOLO

        model = YOLO(self.model_name)
        args = dict(self.train_args)
        # Ultralytics tự quản lý cây thư mục riêng của nó; neo vào run_dir để
        # mọi thứ của một lần chạy nằm chung một chỗ.
        args.update(
            data=str(self.data_yaml),
            project=str(self.run_dir),
            name="ultralytics",
            exist_ok=True,
        )
        results = model.train(**args)

        save_dir = Path(getattr(results, "save_dir", self.run_dir / "ultralytics"))
        weights = save_dir / "weights"
        out = {
            "weights": {
                "best": str(weights / "best.pt") if (weights / "best.pt").exists() else None,
                # last.pt cũng phải chấm: val chỉ có 20 ảnh nên best.pt được
                # chọn theo một tín hiệu rất nhiễu.
                "last": str(weights / "last.pt") if (weights / "last.pt").exists() else None,
            },
            "save_dir": str(save_dir),
            "args": {k: v for k, v in args.items() if not k.startswith("_")},
        }
        metrics = getattr(results, "results_dict", None)
        if metrics:
            out["ultralytics_metrics"] = {k: float(v) for k, v in metrics.items()}
        return out
