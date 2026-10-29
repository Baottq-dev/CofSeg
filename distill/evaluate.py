"""Chấm một lượt train của distill/ bằng đúng bộ chấm của benchmark/yolo.

Chạy từ GỐC REPO:

    python distill/evaluate.py --config distill/configs/eval/tro.yaml --weights <run>/weights/best.pt --data data/export/field/f1 --split test --runs distill_seed0

Phải chấm qua script này, không qua benchmark/yolo/evaluate.py: checkpoint của trò chứa lớp của gói distill,
và chỉ script này đặt gốc repo vào sys.path trước khi nạp trọng số.

Trước khi chấm, script đối chiếu lượt train (khối `distill` trong summary.json cạnh trọng số) với config chấm:

- tên config chấm phải trùng tên gốc của lượt train (vd tro-seg, yolo26s-seg-ka), để không chấm trò bằng
  config của YOLO26s hay ngược lại; gõ --name thì bỏ qua bước này;
- lượt train có `outside_box` thì phép cắt lúc chấm (crop_expand) phải đúng hệ số đã chốt (infer_expand).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Python đặt thư mục của script (distill/) lên đầu sys.path; bỏ nó đi để student.py, trainer.py... không
# thành module cấp cao nhất, rồi đặt gốc repo vào để `import distill` được.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import distill  # noqa: E402
import distill.student  # noqa: E402,F401 - lớp của trò phải nạp được trước khi đọc checkpoint
from distill.trainer import train_distill  # noqa: E402


def _flag(argv: list[str], *names: str) -> str | None:
    """Giá trị của cờ dạng `--k v` hoặc `--k=v`; None nếu không có."""
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def check(argv: list[str]) -> str:
    """Đối chiếu lượt train với lệnh chấm; trả về một dòng để in, dừng nếu lệch."""
    from cofseg import config as cfgmod

    weights, conf = _flag(argv, "--weights"), _flag(argv, "--config")
    if not weights or not conf:
        return "distill: thiếu --weights hoặc --config, để bộ chấm của benchmark tự báo."
    block = train_distill(weights)
    if block is None:
        return f"distill: không thấy khối `distill` trong summary.json cạnh {weights}; không đối chiếu."
    cfg = cfgmod.load(conf)
    name = _flag(argv, "--name")
    if name is None and cfg.get("name") != block["name"]:
        raise SystemExit(
            f"Config chấm {conf} mang tên {cfg.get('name')!r}, còn lượt train là {block['name']!r}. "
            "Dùng config chấm cùng tên, hoặc gõ --name nếu đó đúng là ý bạn."
        )
    ob = block.get("outside_box")
    raw = _flag(argv, "--crop_expand", "--crop-expand")
    used = float(raw) if raw is not None else float((cfg.get("model") or {}).get("crop_expand", 1.0))
    if ob and abs(used - ob["infer_expand"]) > 1e-9:
        raise SystemExit(
            f"Lượt train có giám sát ngoài hộp, chốt cắt mặt nạ ×{ob['infer_expand']:g} lúc chấm; lệnh này "
            f"cắt ×{used:g}. Dùng configs/eval của distill, hoặc gõ --crop_expand {ob['infer_expand']:g}."
        )
    return f"distill: lượt train {block['name']!r}, crop_expand {used:g}, khớp config chấm."


if __name__ == "__main__":
    print(check(sys.argv[1:]), flush=True)
    runpy.run_path(str(distill.BENCH_YOLO / "evaluate.py"), run_name="__main__")
