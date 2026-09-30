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
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from .. import progress
from ..registry import register
from . import memory
from .base import Trainer

# Mặc định riêng cho ảnh UAV nadir, đè lên mặc định của ultralytics.
# Chỉ liệt kê thứ CỐ Ý khác đi; còn lại để ultralytics quyết.
AERIAL_DEFAULTS: dict = {
    "flipud": 0.5,  # ảnh nadir không có chiều "trên"
    "fliplr": 0.5,
    "deterministic": True,
    "seed": 0,
}

#: Cột chỉ số mà ta chọn checkpoint theo. Ultralytics chọn best.pt theo
#: `SegmentMetrics.fitness = seg.fitness() + box.fitness()`, tức trộn cả chỉ
#: số HỘP vào; ba model kia chọn thuần theo mask AP. Không đổi được bằng tham
#: số, nên trainer tự giữ thêm một bản best theo đúng cột này.
FITNESS_KEY = "metrics/mAP50-95(M)"


@register("trainer", "yolo")
class YoloTrainer(Trainer):
    """Trainer cho mọi biến thể YOLO-seg của ultralytics (v8/v9/11/26)."""

    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        self.model_name: str = cfg["model"]
        self.data_yaml = Path(cfg["data"]["yaml"])
        self.train_args: dict = {**AERIAL_DEFAULTS, **(cfg.get("train") or {})}
        self._local_yaml: Path | None = None

    # -------------------------------------------------------------------- tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        """Danh sách sống của ultralytics, không chép tay để khỏi lệch phiên bản."""
        try:
            from ultralytics.cfg import get_cfg
        except ImportError:
            return None
        return dict(vars(get_cfg()))

    @classmethod
    def locked_params(cls) -> frozenset[str]:
        return frozenset({"data", "project", "name", "exist_ok"})

    # -------------------------------------------------------------------- đặt tên
    #: Tham số đưa vào tên thư mục, kèm tiền tố ngắn. Ba cái này quyết định cả
    #: chất lượng lẫn chi phí của lần chạy, và cũng chính là ba cái hay bị ghi
    #: đè từ dòng lệnh — tức ba cái dễ khiến nhãn tĩnh trong config thành sai.
    TAG_KEYS = (("imgsz", "i"), ("batch", "b"), ("epochs", "e"))
    DATA_KEY = "yaml"      # ultralytics đọc data/export/<bộ>/<fold>/data.yaml

    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        args = {**AERIAL_DEFAULTS, **(cfg.get("train") or {})}
        return "".join(
            f"{prefix}{args[key]}" for key, prefix in cls.TAG_KEYS if key in args
        )

    # ------------------------------------------------------------------ chuẩn bị
    def _portable_yaml(self) -> Path:
        """data.yaml của fold mang `path:` TUYỆT ĐỐI của máy đã cắt fold.

        Cắt fold ở Windows rồi train ở Linux thì dòng đó thành rác, mà hỏng
        theo kiểu khó đọc nhất: `F:/CoffeeSeg/...` không phải đường dẫn tuyệt
        đối trên POSIX, nên ultralytics nối nó vào thư mục dataset của chính
        nó (`check_det_dataset`: `if not path.exists() and not
        path.is_absolute(): path = (DATASETS_DIR / path).resolve()`) và báo
        thiếu một đường dẫn ghép chẳng ai viết bao giờ:

            /home/student/.../LightCoral/F:/CoffeeSeg/data/export/block/f1/images/val

        Thông tin đó vốn thừa: data.yaml LUÔN nằm ở gốc fold, cạnh images/ và
        labels/ — đó là bố cục mà scripts/make_fold.py cắt ra. Nên ghi đè
        `path:` bằng chính thư mục chứa nó, tuyệt đối theo máy đang chạy.

        Bản sửa ghi vào run_dir chứ không đụng vào data/ — data/ là dữ liệu
        dùng chung của cả nhóm, và bản trong run_dir còn nói lại được sau này
        lượt chạy ấy đã nạp đúng cái gì.
        """
        if self._local_yaml is None:
            doc = yaml.safe_load(self.data_yaml.read_text(encoding="utf-8")) or {}
            doc["path"] = str(self.data_yaml.resolve().parent).replace("\\", "/")
            out = self.run_dir / "data.yaml"
            out.write_text(
                yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self._local_yaml = out
        return self._local_yaml

    def prepare(self) -> dict:
        """Để ultralytics tự xác nhận bộ dữ liệu trước khi đụng GPU.

        Dùng chính bộ đọc của thư viện sẽ tiêu thụ nhãn, chứ không tự viết
        phép kiểm riêng: cái ta cần biết là NÓ có đọc được không.
        """
        from ultralytics.data.utils import check_det_dataset

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Không thấy {self.data_yaml}. Cắt fold trước: scripts/make_fold.py --export <bản xuất> --all"
            )
        info = check_det_dataset(str(self._portable_yaml()), autodownload=False)
        out = {
            "data_yaml": str(self.data_yaml),
            "data_yaml_used": str(self._portable_yaml()),
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

        CẢNH BÁO, đo được trên chính máy này: autobatch báo cột backward là
        `nan`, tức nó chỉ tính bộ nhớ CHIỀU THUẬN. Nó đề xuất batch 5 ở
        imgsz=1536 (ước 4.7/8 GB), nhưng chạy thật batch 4 đã chạm 7.2 GB.

        Vượt ngưỡng đó thì driver NVIDIA không báo OOM mà âm thầm tràn sang
        RAM hệ thống: 1536/batch 2 chạy 1.25 it/s, còn 1280/batch 4 tụt xuống
        179 s/it — chậm 220 lần mà không có lỗi nào. Nên coi con số ở đây là
        cận trên lạc quan và LUÔN xác nhận bằng một lần chạy ngắn có nhìn cột
        GPU_mem; ngưỡng an toàn là 0.88 x VRAM (7.0 GB trên card 8 GB này),
        xem training/memory.py.
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
            # Hệ số 0.4 rút từ đo thật: autobatch nói 5, thực tế chạy được 2.
            "recommended_batch": max(1, int(int(batch) * 0.4)),
            "free_mb_before": round(free_before),
            "wall_gb": memory.wall_gb(),
            "note": (
                "suggested chỉ tính chiều thuận nên lạc quan ~2.5 lần; "
                "xác nhận bằng một lần chạy ngắn, giữ GPU_mem dưới wall_gb "
                "để không tràn sang RAM hệ thống"
            ),
        }

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        from ultralytics import YOLO

        model = YOLO(self.model_name)
        args = dict(self.train_args)
        # Ultralytics tự quản lý cây thư mục riêng của nó; neo vào run_dir để
        # mọi thứ của một lần chạy nằm chung một chỗ.
        args.update(
            data=str(self._portable_yaml().resolve()),
            # Đường dẫn TUYỆT ĐỐI: với đường dẫn tương đối, ultralytics nối nó
            # vào thư mục runs của chính nó và kết quả rơi vào
            # runs/segment/<đường dẫn của ta>/ thay vì vào run_dir.
            project=str(self.run_dir.resolve()),
            name="ultralytics",
            exist_ok=True,
        )
        self.warned = progress.warnings_to_file(self.run_dir / "warnings.log")
        picked = self._best_by_mask_ap(model)
        t0 = time.time()
        results = model.train(**args)
        train_seconds = round(time.time() - t0, 1)

        # train() trả về SegmentMetrics, và thực thể đó KHÔNG có save_dir (đã
        # kiểm trên ultralytics 8.4.143: hasattr -> False). Nên nơi ghi được
        # suy thẳng từ project/name ta vừa truyền vào, chứ không dò thuộc tính
        # rồi im lặng rơi vào giá trị mặc định.
        save_dir = Path(getattr(results, "save_dir", None) or args["project"])
        if not (save_dir / "weights").is_dir():
            save_dir = self.run_dir / "ultralytics"
        weights = save_dir / "weights"
        best, last = weights / "best.pt", weights / "last.pt"
        # Bản chọn theo mask AP thuần, về đúng bố cục của ba model kia
        # (run_dir/weights/best.pt) để lệnh chấm viết giống nhau cho cả bốn.
        mine = self.run_dir / "weights"
        mine.mkdir(exist_ok=True)
        if picked.path is not None and picked.path.exists():
            shutil.copy2(picked.path, mine / "best.pt")
        elif best.exists():
            shutil.copy2(best, mine / "best.pt")
        if last.exists():
            shutil.copy2(last, mine / "last.pt")
        if not best.exists() and not last.exists():
            # Không có trọng số nghĩa là lần chạy hỏng, dù ultralytics không
            # ném lỗi. Báo ra ngay thay vì trả về summary rỗng trông như thành công.
            raise RuntimeError(
                f"Huấn luyện kết thúc nhưng không có trọng số nào trong {weights}"
            )
        out = {
            "weights": {
                "best": str(mine / "best.pt") if (mine / "best.pt").exists() else None,
                # last.pt cũng phải chấm: val chỉ có 20 ảnh nên best.pt được
                # chọn theo một tín hiệu rất nhiễu.
                "last": str(mine / "last.pt") if (mine / "last.pt").exists() else None,
                "ultralytics_best": str(best) if best.exists() else None,
            },
            "best_epoch": picked.epoch,
            "best_val_AP": round(picked.value * 100, 4) if picked.value is not None else -1.0,
            "save_dir": str(save_dir),
            "args": {k: v for k, v in args.items() if not k.startswith("_")},
        }
        metrics = getattr(results, "results_dict", None)
        if metrics:
            out["ultralytics_metrics"] = {k: float(v) for k, v in metrics.items()}
        out["train_seconds"] = train_seconds
        self._print_summary(metrics or {}, train_seconds, mine / "best.pt")
        return out

    def _best_by_mask_ap(self, model):
        """Giữ thêm một bản best chọn thuần theo mask AP trên val.

        Ultralytics chọn best.pt theo `seg.fitness() + box.fitness()`, tức
        cộng cả chỉ số hộp vào, còn ba model kia chọn theo mask AP. Không có
        tham số nào đổi được, và callback `on_fit_epoch_end` chạy SAU
        `save_model()`, nên cách sạch nhất là chép lại `last.pt` mà
        ultralytics vừa ghi cho epoch này — không đụng một dòng nào của thư viện.
        """
        picked = SimpleNamespace(value=None, epoch=-1, path=None)
        store = self.run_dir / "weights"

        def on_fit_epoch_end(trainer):
            got = (trainer.metrics or {}).get(FITNESS_KEY)
            if got is None or not Path(trainer.last).exists():
                return
            got = float(got)
            if picked.value is not None and got <= picked.value:
                return
            store.mkdir(exist_ok=True)
            out = store / "best_mask_ap.pt"
            shutil.copy2(trainer.last, out)
            picked.value, picked.epoch, picked.path = got, int(trainer.epoch) + 1, out

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        return picked

    def _print_summary(self, metrics: dict, seconds, weights) -> None:
        """Khối cuối lượt chạy, cùng dạng với ba model kia.

        ultralytics đã có thanh tiến trình và bảng mỗi epoch, nên ở đây không
        cần hook nào — chỉ cần kết thúc giống nhau để bốn lượt train đọc được
        cạnh nhau. Lấy chỉ số của MẶT NẠ, hậu tố (M), chứ không phải (B) là
        của hộp: bài toán là phân vùng thực thể.
        """
        def pct(key):
            got = metrics.get(key)
            return None if got is None else float(got) * 100

        rows = [("epoch", str(self.train_args.get("epochs", "—"))),
                ("val (mặt nạ)",
                 f"mAP50-95 {progress.fmt_num(pct('metrics/mAP50-95(M)'))}   "
                 f"mAP50 {progress.fmt_num(pct('metrics/mAP50(M)'))}"),
                ("thời gian", self._time_row(seconds))]
        seen = getattr(getattr(self, "warned", None), "seen", ())
        if seen:
            rows.append(("cảnh báo", f"{len(seen)} loại  ->  warnings.log"))
        try:
            shown = Path(weights).relative_to(self.run_dir)
        except ValueError:
            shown = weights
        rows.append(("trọng số", str(shown)))
        print(progress.summary(f"{self.model_name} · {self.run_dir.name}", rows), flush=True)
