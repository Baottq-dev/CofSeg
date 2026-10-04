# Bản của benchmark/yolo: chạy hoàn toàn trong thư mục này (cofseg/ là bản sao
# lõi của riêng thư mục). Chạy từ GỐC REPO để data/ và weights/ dùng chung:
#
#     python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml
"""Huấn luyện một model bất kỳ. Model do config chỉ định, không phải script.

    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml
    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml --probe
    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml --print-config

Truyền siêu tham số thẳng trên dòng lệnh — mọi tham số của trainer đều nhận:

    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo11s.yaml --epochs 100 --imgsz 640 --batch 16

Chạy khói dùng --fraction của ultralytics (tỉ lệ tập train), không có --limit:

    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s.yaml --data data/export/field/f1 --epochs 2 --fraction 0.05
    
Tên viết gạch nối cũng được (--cos-lr = --cos_lr). Cờ không kèm giá trị nghĩa
là bật: --amp tương đương --amp true. Gõ sai tên thì script BÁO LỖI kèm gợi ý,
thay vì im lặng bỏ qua rồi để bạn chờ ba tiếng mới biết tham số không vào.

Vẫn giữ --set làm lối thoát cho khoá lồng nhau chưa có cờ riêng.

Hàm loss mặc định là loss gốc của framework. Biến thể chỉ bật bằng cờ, và tên
thư mục run mang hậu tố của nó (yolo26s-seg-mask-iou_...):

    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s.yaml --data data/export/field/f1 --loss mask_iou
    python benchmark/yolo/train.py --config benchmark/yolo/configs/train/yolo26s.yaml --list-losses

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
from cofseg import progress  # noqa: E402
from cofseg import runlog  # noqa: E402
from cofseg import training  # noqa: F401,E402 - nạp để đăng ký trainer
from cofseg.registry import available, resolve  # noqa: E402


console.setup()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        # Không cho viết tắt. Mặc định của argparse nhận mọi tiền tố không
        # nhập nhằng, nên `--conf 0.05` bị nuốt thành `--config 0.05` và lỗi
        # hiện ra ở tận chỗ mở file. Siêu tham số của model phải rơi xuống
        # parse_known_args để đi đúng đường kiểm tên.
        allow_abbrev=False,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="lối thoát cho khoá lồng nhau chưa có cờ riêng, vd --set model.repo=...",
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
    ap.add_argument("--model", default=None, metavar="TÊN",
                    help="đổi phiên bản/cỡ model, vd yolo11m-seg — xem --list-models")
    ap.add_argument("--list-models", dest="list_models", action="store_true",
                    help="liệt kê phiên bản và cỡ có trọng số COCO rồi dừng")
    ap.add_argument("--loss", default=None, metavar="TÊN[,TÊN]",
                    help="biến thể hàm loss, vd --loss mask_iou. Bỏ trống = loss gốc "
                         "của framework — bảng benchmark luôn train như vậy. Xem --list-losses")
    ap.add_argument("--list-losses", dest="list_losses", action="store_true",
                    help="liệt kê biến thể loss và tham số của chúng rồi dừng")
    ap.add_argument("--data", default=None, metavar="THƯ_MỤC_FOLD",
                    help="thư mục fold, vd data/export/block/f1 — viết thẳng ra để "
                         "nhìn lệnh là biết đang train fold nào")
    # Không có --limit: trainer YOLO không đọc `data.limit`, nên cờ đó từng nhận
    # giá trị rồi lặng lẽ train trên cả fold. Chạy khói thì dùng tham số của
    # ultralytics: --fraction 0.05 (tỉ lệ tập train).
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

    if a.list_models:
        print(trainer_cls.describe_models(cfg))
        return 0
    if a.list_losses:
        print(trainer_cls.describe_losses())
        return 0

    # Đổi kiến trúc phải xong TRƯỚC khi đọc khối train: và trước khi đặt tên
    # thư mục run — tên thư mục phải nói đúng thứ vừa chạy. Hai cờ này ghi vào
    # khối `model:`, không phải khối `train:`, nên chúng không đi qua
    # parse_overrides.
    goi_y =trainer_cls.apply_model(cfg, a.model) if a.model else None
    # Biến thể loss cũng phải xong trước khi đặt tên, vì nó vào tên thư mục.
    # Gọi cả khi không có cờ: trainer chặn `loss.*` lọt vào từ config/--set mà
    # thiếu --loss, thay vì để nó nằm im trong config.yaml như thể đã dùng.
    hau_to_loss = trainer_cls.apply_loss(cfg, a.loss)

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
        print("\nloss:", json.dumps(cfg["loss"], ensure_ascii=False) if cfg.get("loss")
              else "gốc của framework (không có --loss)")
        return 0

    name = a.name or goi_y or cfg.get("name") or Path(a.config).stem
    # --name gõ tay được giữ nguyên văn như mọi khi; tên tự sinh thì mang hậu
    # tố loss, để lượt L1 không bao giờ trùng tên với lượt baseline cùng fold.
    if hau_to_loss and not a.name:
        name = f"{name}-{hau_to_loss}"
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
            # Đổ nguyên JSON rồi cắt ở ký tự thứ 400 là cách cũ; chỗ cắt đó rơi
            # đúng vào khoá "val", nên số ảnh val và test không bao giờ hiện ra.
            print("Dữ liệu:", progress.data_line(info)
                  or json.dumps(info, ensure_ascii=False)[:400])

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
