# Bản của benchmark/mask2former: chạy hoàn toàn trong thư mục này (cofseg/ là bản sao
# lõi của riêng thư mục). Chạy từ GỐC REPO để data/ và weights/ dùng chung:
#
#     python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml
"""Huấn luyện một model bất kỳ. Model do config chỉ định, không phải script.

    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml
    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --probe
    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --print-config

Truyền siêu tham số thẳng trên dòng lệnh — mọi tham số của trainer đều nhận:

    python benchmark/mask2former/train.py --config benchmark/mask2former/configs/train/mask2former_r50_d2.yaml --epochs 100 --imgsz 640 --batch 16 
    
Tên viết gạch nối cũng được (--cos-lr = --cos_lr). Cờ không kèm giá trị nghĩa
là bật: --amp tương đương --amp true. Gõ sai tên thì script BÁO LỖI kèm gợi ý,
thay vì im lặng bỏ qua rồi để bạn chờ ba tiếng mới biết tham số không vào.

Vẫn giữ --set cho các khoá lồng nhau ngoài nhóm train, vd --set data.yaml=...

Script này KHÔNG biết YOLO tồn tại. Nó đọc khoá `trainer` trong config, tra sổ
đăng ký, rồi gọi ba phương thức của hợp đồng. Danh sách tham số hợp lệ và các
khoá bị khoá cũng do trainer khai (Trainer.param_defaults / locked_params), nên
thêm Mask R-CNN hay model tự viết = sửa cofseg/training/ của thư mục này,
script giữ nguyên.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # thư mục của model này

from cofseg import artifacts  # noqa: E402
from cofseg import cli  # noqa: E402
from cofseg import config as cfgmod  # noqa: E402
from cofseg import console  # noqa: E402
from cofseg import runlog  # noqa: E402
from cofseg import training  # noqa: F401,E402 - nạp để đăng ký trainer
from cofseg.registry import available, resolve  # noqa: E402


console.setup()


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
    ap.add_argument(
        "--log-clean",
        action="store_true",
        help="run.log gộp mỗi dòng một lần và bỏ mã màu (mặc định: chép nguyên văn)",
    )
    ap.add_argument("--data", default=None, metavar="THƯ_MỤC_FOLD",
                    help="thư mục fold, vd data/export/block/f1 — viết thẳng ra để "
                         "nhìn lệnh là biết đang train fold nào")
    ap.add_argument("--runs", default="runs", help="thư mục gốc chứa kết quả")
    ap.add_argument("--name", default=None, help="tên lần chạy (mặc định lấy từ config)")
    a, extra = ap.parse_known_args()

    # --data đứng TRƯỚC --set trong danh sách ghi đè, nên --set thắng nếu ai
    # đó dùng cả hai — cụ thể hơn thì thắng, như mọi chỗ khác.
    overrides = list(a.overrides)
    if a.data:
        cfg0 = cfgmod.load(a.config)
        cls0 = resolve("trainer", cfg0["trainer"]) if "trainer" in cfg0 else None
        if cls0 is None:
            raise SystemExit(f"config thiếu khoá 'trainer'. Hiện có: {available('trainer')}")
        key, val = cls0.data_arg(a.data)
        overrides.insert(0, f"{key}={val}")
    cfg = cfgmod.load(a.config, overrides)
    if "trainer" not in cfg:
        raise SystemExit(f"config thiếu khoá 'trainer'. Hiện có: {available('trainer')}")
    trainer_cls = resolve("trainer", cfg["trainer"])
    locked = trainer_cls.locked_params()

    if a.list_params:
        d = trainer_cls.param_defaults()
        if d is None:
            raise SystemExit(f"Trainer {cfg['trainer']!r} không liệt kê tham số của nó.")
        print(f"{len(d)} tham số trainer {cfg['trainer']!r} nhận (khoá cứng: {sorted(locked)}):\n")
        print(cli.describe_params(d, locked))
        return 0

    # Siêu tham số dòng lệnh thắng config, vì chúng cụ thể hơn.
    hp = cli.parse_overrides(extra, trainer_cls.param_names(), locked, what="trainer")
    if hp:
        cfg.setdefault("train", {}).update(hp)

    if a.print_config:
        merged = trainer_cls(cfg, Path(".")).train_args
        print("Tham số cuối cùng đưa vào trainer:\n")
        for k in sorted(merged):
            src = "  <-- dòng lệnh" if k in hp else ""
            print(f"  {k:<18} {merged[k]!r}{src}")
        return 0

    name = a.name or cfg.get("name") or Path(a.config).stem
    # Nhãn sinh từ tham số ĐÃ GỘP (config + dòng lệnh), nên tên thư mục luôn
    # mô tả đúng thứ vừa chạy kể cả khi bạn ghi đè imgsz hay batch.
    # Bộ fold đứng trước nhãn tham số: hai bộ fold cùng đặt tên f1..f6, nên
    # thiếu nó thì hai lần chạy khác hẳn nhau ra tên thư mục giống hệt.
    ds = artifacts.dataset_tag(cfg)
    run_dir = artifacts.create_run_dir(
        a.runs, "probe" if a.probe else "train", name,
        "_".join(p for p in (ds, trainer_cls.run_tag(cfg)) if p)
    )
    artifacts.write_env(run_dir, cfg)
    artifacts.snapshot_config(run_dir, cfg)
    print(f"Lần chạy: {run_dir}")

    # Từ đây trở đi mọi thứ hiện trên màn hình được chép nguyên văn vào
    # run.log, kể cả output của ultralytics. Xuất xứ lần chạy nằm ở env.json
    # và config.yaml cùng thư mục, nên log không cần header.
    with runlog.capture(run_dir / "run.log", raw=not a.log_clean) as log_path:
      # Bắt lỗi BÊN TRONG khối with: nếu để ngoại lệ thoát ra, luồng đã được
      # trả về nguyên trạng trước khi Python in traceback, và traceback sẽ
      # không có trong log — đúng lúc cần nó nhất.
      try:
        if hp:
            print("Ghi đè từ dòng lệnh:", json.dumps(hp, ensure_ascii=False))

        trainer = trainer_cls(cfg, run_dir)

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
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print("\nXong. Trọng số:", json.dumps(summary.get("weights"), ensure_ascii=False))
        print("Kết quả:", run_dir)
        print("Nhật ký:", log_path)
      except SystemExit:
        raise
      except BaseException:                       # gồm cả KeyboardInterrupt
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
