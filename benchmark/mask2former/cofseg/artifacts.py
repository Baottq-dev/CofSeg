"""Thư mục kết quả cho mỗi lần chạy, kèm dấu vết để tái lập.

Bố cục:

    runs/<viec>/<thoi-diem>_<ten>_<bo-fold>_<nhan-tham-so>/

    runs/train/2026-09-25_155401_maskrcnn-r50-d2_block-f4_i1024b4e50/
    runs/train/2026-09-25_181233_maskrcnn-r50-d2_flight-f4_i1024b4e50/
    runs/probe/2026-09-25_130145_maskrcnn-r50-d2_block-f4_i1024b4e50/

Bốn quyết định, mỗi cái sửa một khuyết điểm đã gặp thật:

1. THỜI ĐIỂM ĐỨNG TRƯỚC. Sắp xếp theo tên cũng là sắp theo thời gian, trong
   mọi trình duyệt file và mọi lệnh ls. Đặt tên trước thì các cấu hình khác
   nhau xen kẽ nhau và không lần ra được thứ tự đã chạy.

2. NHÃN THAM SỐ SINH TỪ CẤU HÌNH THẬT, không phải từ nhãn tĩnh trong config.
   Đây là lỗi đã xảy ra: thư mục tên "yolo26s_seg_1536" trong khi lần chạy
   thật dùng imgsz=640. Tên thư mục không được phép nói dối.

3. TÁCH THEO LOẠI VIỆC. Một lần dò VRAM 30 giây không nên nằm lẫn với một lần
   train 4 tiếng.

4. BỘ FOLD NẰM TRONG TÊN. Hai cách chia val cho hai bộ fold cùng đặt tên
   f1..f6, nên hai lần chạy khác hẳn nhau lại ra tên thư mục giống hệt. Nhìn
   `..._maskrcnn-r50-d2_i1024b4e50` thì không biết nó train trên bộ nào; nhìn
   `..._maskrcnn-r50-d2_block-f4_i1024b4e50` thì biết.

Trong mỗi thư mục luôn có config đã dùng và env.json (phiên bản thư viện, GPU,
commit git, repo sạch hay bẩn). Thiếu những thứ đó thì ba tháng sau không ai
nói được con số sinh ra từ đâu.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import config as cfgmod

# Gộp mọi ký tự không an toàn cho tên file thành một dấu gạch ngang. Windows
# cấm \ / : * ? " < > | và cả tên có dấu; giữ danh sách cho phép thay vì danh
# sách cấm để không sót.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str) -> str:
    return _UNSAFE.sub("-", str(text)).strip("-_.") or "unnamed"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_dir(
    root: str | Path, kind: str, name: str, tag: str = "", when=None
) -> Path:
    """runs/<kind>/<thời-điểm>_<name>[_<tag>]/ — luôn là thư mục mới.

    `tag` mô tả cấu hình THẬT của lần chạy (vd "i640b4e100"); trainer sinh ra
    nó qua Trainer.run_tag() nên mỗi họ model tự quyết tham số nào đáng ghi.
    """
    ts = (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    stem = "_".join(p for p in (ts, slugify(name), slugify(tag) if tag else "") if p)
    parent = Path(root) / slugify(kind)
    d, n = parent / stem, 2
    # Hai lần chạy trong cùng một giây vẫn phải ra hai thư mục khác nhau, thay
    # vì ném lỗi và làm mất cả lần chạy.
    while d.exists():
        d = parent / f"{stem}~{n}"
        n += 1
    d.mkdir(parents=True)
    return d


def write_env(run_dir: str | Path, cfg: dict | None = None,
              name: str = "env.json") -> dict:
    """Ghi lại môi trường. Gọi TRƯỚC khi chạy để có dấu vết cả khi chạy hỏng.

    Có `cfg` thì ghi kèm bộ fold đã dùng (đọc từ fold.json của thư mục dữ
    liệu): ba tháng sau mở một thư mục kết quả là biết ngay nó train trên bộ
    nào, cắt val kiểu gì, bỏ bao nhiêu ảnh làm đệm.

    `name` khác mặc định khi nối tiếp một lượt chạy: bản env của lượt đầu
    phải còn nguyên, vì một lượt bị ngắt rồi chạy tiếp có thể đã đổi máy, đổi
    phiên bản thư viện, hoặc đổi commit — và đúng những thứ đó mới giải thích
    được vì sao nửa sau khác nửa đầu.
    """
    env: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        # Repo bẩn nghĩa là commit ở trên KHÔNG mô tả đủ code đã chạy.
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": {},
    }
    # detectron2/mmdet/mmcv/mmengine là framework của chính ba model đang so;
    # thiếu chúng ở đây thì env.json không mô tả được thứ đã chạy, mà trên máy
    # thuê đó là bản ghi duy nhất còn lại về môi trường.
    for mod in ("torch", "torchvision", "ultralytics", "detectron2",
                "mmdet", "mmcv", "mmengine", "numpy", "cv2", "pycocotools"):
        try:
            m = __import__(mod)
            env["packages"][mod] = getattr(m, "__version__", "?")
        except ImportError:
            env["packages"][mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**20
                ),
                "cuda": torch.version.cuda,
            }
    except Exception:  # noqa: BLE001 - môi trường hỏng không được chặn việc chạy
        pass

    if cfg is not None:
        env["dataset"] = fold_facts(cfg)

    Path(run_dir, name).write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return env


def snapshot_config(run_dir: str | Path, cfg: dict) -> None:
    cfgmod.dump(cfg, Path(run_dir) / "config.yaml")


def check_resume(run_dir: str | Path, cfg: dict) -> str:
    """Xác nhận `run_dir` nối tiếp được, trả về tên file env cho lượt này.

    Nối tiếp một lượt chạy bằng THAM SỐ KHÁC thì summary.json cuối cùng mô tả
    một cấu hình chỉ đúng với nửa sau của lượt đó, còn results.csv thì trộn
    hai lịch học vào một đường cong — không dòng log nào nói ra. Đó đúng loại
    sai lặng lẽ mà repo này biến thành lỗi dừng ở mọi chỗ khác, nên ở đây
    cũng vậy: so nguyên văn config đã gộp với bản lượt trước chụp lại.

    env.json của lượt đầu được giữ nguyên; lượt nối tiếp ghi sang
    env.resume1.json, resume2.json... vì máy, phiên bản thư viện hay commit
    có thể đã khác, và chính những thứ đó giải thích nửa sau khác nửa đầu.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"--resume trỏ vào {run_dir}, không phải một thư mục lần chạy.")
    snap = run_dir / "config.yaml"
    if not snap.is_file():
        raise SystemExit(f"Không thấy {snap}: thư mục này không phải một lần chạy "
                         "do train.py tạo, hoặc nó hỏng ngay trước khi kịp ghi config.")
    cu = snap.read_text(encoding="utf-8")
    moi = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    if cu != moi:
        import difflib

        diff = "".join(difflib.unified_diff(cu.splitlines(True), moi.splitlines(True),
                                            fromfile="config.yaml (lượt trước)",
                                            tofile="lệnh lần này", n=1))
        raise SystemExit(
            "Config của lệnh này khác config lượt đang nối tiếp:\n"
            + diff.rstrip()
            + "\n\nNối tiếp bằng tham số khác cho ra một lượt chạy mà summary.json "
              "mô tả sai. Gõ lại đúng lệnh cũ, hoặc bỏ --resume để chạy lượt mới.")
    n = 1
    while (run_dir / f"env.resume{n}.json").exists():
        n += 1
    return f"env.resume{n}.json"


# --------------------------------------------------------- bộ fold của lần chạy
def data_root(cfg: dict) -> Path | None:
    """Thư mục fold mà config trỏ vào, bất kể trainer nào.

    detectron2/mmdet đọc `data.root`; ultralytics đọc `data.yaml` nằm trong
    chính thư mục fold đó.
    """
    d = cfg.get("data") if isinstance(cfg, dict) else None
    if not isinstance(d, dict):
        return None
    if d.get("root"):
        return Path(str(d["root"]))
    if d.get("yaml"):
        return Path(str(d["yaml"])).parent
    return None


def dataset_tag(cfg: dict) -> str:
    """Nhãn ngắn nhận ra BỘ FOLD: 'block-f4', 'flight-f4'.

    Đi vào tên thư mục lần chạy, vào tên file dự đoán, và vào khoá của bảng
    tổng hợp. Thiếu nó thì `maskrcnn_f4` của hai bộ trông y hệt nhau và cái
    chạy sau đè cái trước.
    """
    p = data_root(cfg)
    if p is None:
        return ""
    parts = [x for x in p.parts if x not in ("", ".", "..")]
    if not parts:
        return ""
    fold = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else ""
    return f"{parent}-{fold}" if parent and parent != "export" else fold


def train_run_dir(weights) -> Path | None:
    """Thư mục lượt train đã sinh ra `weights` — nơi có summary.json.

    Hai bố cục đang dùng: `<run>/weights/best.pth` (detectron2, mmdet, và bản
    mirror của yolo) và `<run>/ultralytics/weights/best.pt` (bản của chính
    ultralytics). Đi ngược lên tối đa bốn bậc là đủ cho cả hai, và dừng ở thư
    mục ĐẦU TIÊN có summary.json nên không phải đoán theo tên thư mục.
    """
    if not weights:
        return None
    for d in list(Path(str(weights)).parents)[:4]:
        if (d / "summary.json").is_file():
            return d
    return None


def train_args(weights) -> dict | None:
    """Khối `args` trong summary.json của lượt train đã sinh ra `weights`."""
    d = train_run_dir(weights)
    if d is None:
        return None
    try:
        doc = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    args = doc.get("args")
    return args if isinstance(args, dict) else None


def train_imgsz(weights) -> int | None:
    """imgsz của lượt train đã sinh ra `weights`; None nếu không tra được."""
    try:
        return int((train_args(weights) or {})["imgsz"])
    except (KeyError, TypeError, ValueError):
        return None


def check_imgsz(weights, used, explicit: bool = False) -> str:
    """Dừng nếu chấm ở độ phân giải KHÁC lượt train, trừ khi cố ý.

    Đây là lỗi đã xảy ra và đã ăn mất sáu lượt chấm: cả sáu lần train Mask
    R-CNN chạy ở cạnh dài 1024, nhưng lúc chấm thiếu d2_config.yaml nên
    detectron2 rơi về mặc định của nó (800/1333) và chấm ở 1333 mà không một
    dòng log nào khác đi. Cùng lúc YOLO chấm ở đúng 1024, nên bảng so sánh
    giữa hai model là so ở hai độ phân giải. Chỉ đọc lại log mới phát hiện ra.

    Hàng rào ở `Detectron2Model` chặn trường hợp THIẾU file config. Hàm này
    chặn trường hợp còn lại, nguy hiểm hơn vì không thiếu gì cả: file có mặt
    nhưng con số trong đó khác con số đang dùng.

    `explicit` là khi người chạy gõ thẳng `--imgsz`. Lúc đó lệch là chủ ý
    (khảo sát độ phân giải) nên chỉ kêu to chứ không dừng. imgsz lấy từ file
    config thì KHÔNG tính là chủ ý: config eval mặc định 1024, mà lượt train
    640 vẫn nạp đúng config đó — đúng cái bẫy cần chặn.

    Trả về một dòng để ghi vào run.log, kể cả khi mọi thứ khớp: một phép kiểm
    im lặng là một phép kiểm không ai biết đã chạy hay chưa.
    """
    want = train_imgsz(weights)
    if used is None:
        return "imgsz: model không có khái niệm này, bỏ qua bước đối chiếu."
    if want is None:
        return (f"imgsz: đang chấm ở {used}; không tra được lượt train "
                f"(không thấy summary.json cạnh {weights!r}) nên KHÔNG đối chiếu được.")
    if int(want) == int(used):
        return f"imgsz: {used}, khớp lượt train ({train_run_dir(weights)})."
    msg = (f"imgsz lệch: lượt train chạy ở {want}, lần chấm này ở {used}.\n"
           f"  Nguồn: {train_run_dir(weights)}/summary.json -> args.imgsz = {want}\n"
           "  Chấm ở độ phân giải model chưa từng thấy thì con số không so được "
           "với bất kỳ lượt nào khác.")
    if explicit:
        return "CẢNH BÁO: " + msg + "\n  Bỏ qua vì --imgsz được gõ thẳng (chủ ý khảo sát)."
    raise SystemExit(
        msg + "\n  Sửa config/trọng số cho khớp, hoặc gõ thẳng --imgsz "
        f"{want} nếu đó đúng là ý bạn."
    )


def fold_facts(cfg: dict) -> dict | None:
    """Những gì fold.json của bộ dữ liệu nói về lần chạy này."""
    p = data_root(cfg)
    if p is None or not (p / "fold.json").is_file():
        return None
    try:
        doc = json.loads((p / "fold.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    vs = doc.get("val_split") or {}
    return {
        "root": p.as_posix(),
        "tag": dataset_tag(cfg),
        "fold": doc.get("fold"),
        "source_sha1": doc.get("source_sha1"),
        "val_method": vs.get("method"),
        "dropped": (doc.get("dropped") or {}).get("count"),
        "splits": {k: v.get("images") for k, v in (doc.get("splits") or {}).items()},
        "leak": (vs.get("audit") or {}).get("leak"),
    }
