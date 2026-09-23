"""Chấm bằng chính bộ đánh giá của thư viện đã huấn luyện model.

Với ultralytics, đó là `model.val()`. Không viết lại: đó đúng là thứ `yolo val`
chạy, nên con số trùng với mọi báo cáo YOLO khác và không thể trôi lệch khi
ultralytics đổi cách tính.

Hàm này chỉ lo hai việc mà bản thân `val()` không lo: neo kết quả vào thư mục
run của dự án, và trả về số liệu ở dạng máy đọc được.
"""

from __future__ import annotations

from pathlib import Path


def validate(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 1024,
    batch: int = 8,
    conf: float | None = None,
    iou: float = 0.7,
    max_det: int = 300,
    device=None,
    run_dir: str | Path | None = None,
    save_json: bool = True,
    plots: bool = True,
    **extra,
) -> dict:
    """Chạy `model.val()` và trả về số liệu + đường dẫn kết quả."""
    from ultralytics import YOLO

    kw = dict(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch,
        iou=iou,
        max_det=max_det,
        # save_json xuất predictions.json theo định dạng COCO results — đầu vào
        # cho Boundary AP về sau, và cũng để chấm lại bằng công cụ khác.
        save_json=save_json,
        plots=plots,
        verbose=True,
        **extra,
    )
    if conf is not None:
        kw["conf"] = conf
    if device is not None:
        kw["device"] = device
    if run_dir is not None:
        # Đường dẫn TUYỆT ĐỐI: đường dẫn tương đối bị ultralytics nối vào thư
        # mục runs của chính nó.
        kw.update(project=str(Path(run_dir).resolve()), name="ultralytics",
                  exist_ok=True)

    results = YOLO(str(weights)).val(**kw)

    save_dir = Path(getattr(results, "save_dir", run_dir or "."))
    out: dict = {"save_dir": str(save_dir), "args": {k: str(v) for k, v in kw.items()}}
    rd = getattr(results, "results_dict", None)
    if rd:
        out["metrics"] = {k: float(v) for k, v in rd.items()}
    speed = getattr(results, "speed", None)
    if speed:
        out["speed_ms"] = {k: float(v) for k, v in speed.items()}
    pj = save_dir / "predictions.json"
    out["predictions_json"] = str(pj) if pj.exists() else None
    return out
