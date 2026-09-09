"""Huấn luyện một model bất kỳ. Model do config chỉ định, không phải script.

    python scripts/train.py --config configs/train/yolo26s.yaml
    python scripts/train.py --config configs/train/yolo26s.yaml --probe
    python scripts/train.py --config configs/train/yolo26s.yaml --print-config

Truyền siêu tham số thẳng trên dòng lệnh — mọi tham số của trainer đều nhận:

    python scripts/train.py --config configs/train/yolo26s.yaml \
        --epochs 50 --imgsz 1280 --batch 3 --lr0 0.003 --optimizer AdamW

Tên viết gạch nối cũng được (--cos-lr = --cos_lr). Cờ không kèm giá trị nghĩa
là bật: --amp tương đương --amp true. Gõ sai tên thì script BÁO LỖI kèm gợi ý,
thay vì im lặng bỏ qua rồi để bạn chờ ba tiếng mới biết tham số không vào.

Vẫn giữ --set cho các khoá lồng nhau ngoài nhóm train, vd --set data.yaml=...

Script này KHÔNG biết YOLO tồn tại. Nó đọc khoá `trainer` trong config, tra sổ
đăng ký, rồi gọi ba phương thức của hợp đồng. Thêm Mask R-CNN hay model tự viết
= thêm một file trong canopyseg/training/, script giữ nguyên.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canopyseg import artifacts  # noqa: E402
from canopyseg import config as cfgmod  # noqa: E402
from canopyseg import console  # noqa: E402
from canopyseg import training  # noqa: F401,E402 - nạp để đăng ký trainer
from canopyseg.registry import available, resolve  # noqa: E402

console.setup()

# Trainer tự đặt bốn khoá này để kết quả rơi đúng thư mục run; đè lên chúng sẽ
# làm hỏng chính chỗ ghi kết quả, nên chặn ngay ở dòng lệnh cho rõ ràng.
LOCKED = {"data", "project", "name", "exist_ok"}


def trainer_param_names(trainer_kind: str) -> set[str] | None:
    """Tập tham số hợp lệ của trainer, để bắt lỗi gõ sai. None = không kiểm được."""
    if trainer_kind == "yolo":
        try:
            from ultralytics.cfg import get_cfg

            return set(vars(get_cfg()))
        except ImportError:
            return None
    return None


def parse_hparams(tokens: list[str], valid: set[str] | None) -> dict:
    """Biến các đối số lạ thành ghi đè cho khối `train:` của config."""
    out: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(
                f"Không hiểu đối số {tok!r}. Siêu tham số phải có dạng --ten giatri."
            )
        body = tok[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            key, raw = body, tokens[i + 1]
            i += 1
        else:
            # Cờ trần: --amp nghĩa là --amp true.
            key, raw = body, "true"
        key = key.replace("-", "_")

        if key in LOCKED:
            raise SystemExit(
                f"--{key} bị khoá: trainer tự đặt nó để kết quả rơi đúng thư mục run. "
                f"Đổi tên lần chạy bằng --name, đổi bộ dữ liệu bằng --set data.yaml=..."
            )
        if valid is not None and key not in valid:
            near = difflib.get_close_matches(key, sorted(valid), n=3, cutoff=0.6)
            hint = f" Ý bạn là: {', '.join('--' + n for n in near)}?" if near else ""
            raise SystemExit(
                f"Trainer không có tham số {key!r}.{hint}\n"
                f"Xem toàn bộ tham số: --list-params"
            )
        out[key] = yaml.safe_load(raw)
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="ghi đè khoá lồng nhau bất kỳ, vd --set data.yaml=duong/dan.yaml",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="chỉ ước lượng VRAM rồi dừng, không huấn luyện",
    )
    ap.add_argument(
        "--print-config",
        action="store_true",
        help="in tham số cuối cùng sau khi gộp mọi nguồn rồi dừng",
    )
    ap.add_argument(
        "--list-params",
        action="store_true",
        help="liệt kê mọi siêu tham số trainer nhận, kèm giá trị mặc định",
    )
    ap.add_argument("--runs", default="runs", help="thư mục gốc chứa kết quả")
    ap.add_argument("--name", default=None, help="tên lần chạy (mặc định lấy từ config)")
    a, extra = ap.parse_known_args()

    cfg = cfgmod.load(a.config, a.overrides)
    if "trainer" not in cfg:
        raise SystemExit(f"config thiếu khoá 'trainer'. Hiện có: {available('trainer')}")

    valid = trainer_param_names(cfg["trainer"])
    if a.list_params:
        if valid is None:
            raise SystemExit(f"Không liệt kê được tham số cho trainer {cfg['trainer']!r}.")
        from ultralytics.cfg import get_cfg

        d = vars(get_cfg())
        print(f"{len(d)} tham số trainer {cfg['trainer']!r} nhận (khoá cứng: {sorted(LOCKED)}):\n")
        for k in sorted(d):
            mark = "  [khoá]" if k in LOCKED else ""
            print(f"  --{k:<20} mặc định {d[k]!r}{mark}")
        return 0

    # Siêu tham số dòng lệnh thắng config, vì chúng cụ thể hơn.
    hp = parse_hparams(extra, valid)
    if hp:
        cfg.setdefault("train", {}).update(hp)

    if a.print_config:
        merged = resolve("trainer", cfg["trainer"])(cfg, Path(".")).train_args
        print("Tham số cuối cùng đưa vào trainer:\n")
        for k in sorted(merged):
            src = "  <-- dòng lệnh" if k in hp else ""
            print(f"  {k:<18} {merged[k]!r}{src}")
        return 0

    name = a.name or cfg.get("name") or Path(a.config).stem
    run_dir = artifacts.create_run_dir(a.runs, name)
    artifacts.write_env(run_dir)
    artifacts.snapshot_config(run_dir, cfg)
    print(f"Lần chạy: {run_dir}")
    if hp:
        print("Ghi đè từ dòng lệnh:", json.dumps(hp, ensure_ascii=False))

    trainer = resolve("trainer", cfg["trainer"])(cfg, run_dir)

    info = trainer.prepare()
    if info:
        print("Dữ liệu:", json.dumps(info, ensure_ascii=False)[:400])

    if a.probe:
        p = trainer.probe()
        print("\nDò VRAM:", json.dumps(p, indent=2, ensure_ascii=False))
        (run_dir / "probe.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    summary = trainer.fit()
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\nXong. Trọng số:", json.dumps(summary.get("weights"), ensure_ascii=False))
    print("Kết quả:", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
