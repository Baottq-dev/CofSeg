"""Thanh tiến trình và dòng tổng kết mỗi epoch, dùng chung cho cả bốn trainer.

Vì sao cần: ba trong bốn khung huấn luyện báo tiến độ theo ITERATION và đổ ra
màn hình vài trăm dòng cấu hình + kiến trúc model trước khi chạy. Đo trên một
lượt Mask R-CNN 3 epoch thật: 1050 / 1474 dòng (71%) là dump config và model,
trong đó model bị in HAI LẦN. Chữ "epoch" không xuất hiện lần nào suốt lúc
train. Nhìn vào đó không đánh giá được lượt train đang tốt hay hỏng.

Module này KHÔNG thay hệ thống log của thư viện, nó đứng cạnh:

- `hush()` gỡ handler ra CONSOLE và giữ nguyên handler ra FILE, nên bản log
  đầy đủ vẫn nằm trong run dir (detectron2 ghi `d2/log.txt`, mmengine ghi
  `<timestamp>.log`). Không mất gì, chỉ thôi in ra màn hình.
- `Bar` bọc tqdm, và tự tắt khi stdout không phải terminal — chạy `nohup` hay
  `> log.txt` thì chỉ còn các dòng epoch, không có ký tự \r rác.
- `epoch_line()`, `summary()`, `data_line()` là hàm THUẦN: không đụng tqdm,
  không đụng khung nào, nên test được ở máy không cài detectron2 lẫn mmdet.

TÊN CỘT thống nhất theo cách gọi của YOLO cho cả bốn model:

    mAP50-95   detectron2 gọi `AP`,   mmdet gọi `segm_mAP`
    mAP50      detectron2 gọi `AP50`, mmdet gọi `segm_mAP_50`

Cùng một đại lượng, ba cách gọi. Bảng so sánh cuối cùng chỉ đọc được bằng mắt
nếu bốn model in ra giống nhau.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: Thứ tự cột trong dòng epoch. Khoá là tên ta dùng, không phải tên của khung.
METRICS = ("mAP50-95", "mAP50")


# --------------------------------------------------------------------- logger
def hush(*loggers, level: int = logging.WARNING):
    """Nâng ngưỡng của handler ra MÀN HÌNH lên `level`; handler ra file giữ nguyên.

    Nhận TÊN hoặc thẳng đối tượng Logger. Phải nhận được đối tượng vì
    `MMLogger` của mmengine gọi `Logger.__init__(self, "mmengine")` mà KHÔNG
    đăng ký vào `logging.Logger.manager.loggerDict`. Nên
    `logging.getLogger("mmengine")` trả về một logger khác hẳn, rỗng handler,
    và bịt nó thì không tác động gì tới log thật.

    Nâng ngưỡng chứ không gỡ hẳn, vì hai loại dòng nằm ở hai mức khác nhau:
    dump config và kiến trúc model là INFO, còn những dòng đáng đọc như
    "Skip loading parameter ... incompatible shapes" là WARNING. Gỡ handler
    thì mất cả hai; nâng ngưỡng thì chỉ mất loại thứ nhất.

    `logging.FileHandler` là lớp con của `StreamHandler`, nên phải loại trừ nó
    tường minh — nếu không thì bịt luôn đường ghi ra đĩa, tức là xoá mất bản
    log đầy đủ thay vì chỉ giấu nó khỏi màn hình.

    Trả về [(handler, ngưỡng cũ)] để `unhush()` trả lại được.
    """
    changed: list[tuple[logging.Handler, int]] = []
    for item in loggers:
        lg = item if isinstance(item, logging.Logger) else logging.getLogger(item)
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                changed.append((h, h.level))
                h.setLevel(level)
    return changed


def unhush(changed) -> None:
    for h, old in changed:
        h.setLevel(old)


class _Drop(logging.Filter):
    def __init__(self, name: str, startswith: str):
        super().__init__()
        self.name_ = name
        self.head = startswith

    def filter(self, record) -> bool:
        return not (record.name == self.name_
                    and record.getMessage().startswith(self.head))


def drop_from_console(*loggers, name: str, startswith: str):
    """Bỏ MỘT loại bản ghi khỏi màn hình; file vẫn giữ nguyên.

    Chỉ dùng cho bản ghi TRÙNG LẶP, không dùng để giấu lỗi. Trường hợp cụ
    thể: `detectron2.engine.train_loop` bắt ngoại lệ, gọi `logger.exception`
    rồi `raise` lại. Nên cùng một traceback ra màn hình hai lần — bản của
    detectron2 trước, rồi bản Python in ở tầng ngoài. Bản thứ hai chứa đủ
    khung của bản thứ nhất CỘNG THÊM khung của train.py, nên bỏ bản đầu khỏi
    màn hình không mất thông tin nào.

    Lọc đặt trên HANDLER ra màn hình chứ không trên logger, nên d2/log.txt
    vẫn nhận cả hai.
    """
    f = _Drop(name, startswith)
    for item in loggers:
        lg = item if isinstance(item, logging.Logger) else logging.getLogger(item)
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.addFilter(f)
    return f


class _Dedupe(logging.Filter):
    """Mỗi nội dung cảnh báo đúng một lần, đếm bằng set của chính ta."""

    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def filter(self, record) -> bool:
        msg = record.getMessage()
        if msg in self.seen:
            return False
        self.seen.add(msg)
        return True


def warnings_to_file(path):
    """Cảnh báo đi vào FILE thay vì stderr, mỗi nội dung đúng một lần.

    Vì sao không dùng bộ lọc "once" của chính module warnings — đã thử và nó
    KHÔNG ăn ở đây. CPython đánh dấu "cảnh báo này hiện rồi" trong một
    registry, và `already_warned()` XOÁ SẠCH registry đó mỗi khi danh sách
    filter đổi phiên bản. Mà `warnings.catch_warnings()` lúc thoát luôn gọi
    `_filters_mutated()`. Trong vòng lặp train có thư viện dùng
    catch_warnings ở mỗi mẫu, nên dấu vừa ghi xong đã bị xoá:

        catch_warnings() mỗi iteration = False ->  1 dòng
        catch_warnings() mỗi iteration = True  -> 30 dòng

    Đó cũng là lý do cảnh báo lặp lại NGAY TỪ ĐẦU, trước khi ai đụng vào bộ
    lọc: cơ chế gộp mặc định của Python bị vô hiệu bởi đúng chuyện đó.

    Nên ở đây giành quyền ở ĐƯỜNG RA chứ không phải ở bộ lọc:
    `captureWarnings` đẩy mọi cảnh báo qua logging, rồi một set của ta gộp
    lại. Không registry nào xoá được set đó.

    KHÔNG đụng vào `warnings.filters`. Đụng vào là bật lên cả những cảnh báo
    Python đang ẩn mặc định — `DeprecationWarning` của pycocotools chẳng hạn
    — tức là làm màn hình bẩn thêm chứ không sạch đi.

    Trả về bộ lọc, để cuối lượt chạy đếm được có bao nhiêu loại.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.captureWarnings(True)
    lg = logging.getLogger("py.warnings")
    lg.propagate = False          # không cho rơi lên root rồi ra stderr
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    dedupe = _Dedupe()
    handler.addFilter(dedupe)
    lg.addHandler(handler)
    return dedupe


# ------------------------------------------------------------------ định dạng
def fmt_time(seconds: float | None) -> str:
    """132.4 -> '2:12'; 4521 -> '1:15:21'."""
    if seconds is None:
        return "—"
    s = int(round(float(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_num(x, nd: int = 2) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def epoch_line(epoch: int, epochs: int, it: int, iters: int, *, loss=None,
               lr=None, mem=None, metrics: dict | None = None,
               best: bool = False, seconds=None, val_seconds=None) -> str:
    """Một dòng cho một epoch đã xong.

    Bề rộng cột lấy từ tổng số epoch/iteration nên các dòng thẳng hàng nhau mà
    không cần biết trước sẽ chạy bao lâu.
    """
    we, wi = len(str(epochs)), len(str(iters))
    parts = [f"epoch {epoch:>{we}}/{epochs}", f"iter {it:>{wi}}/{iters}"]
    if loss is not None:
        parts.append(f"loss {float(loss):.3f}")
    if lr is not None:
        parts.append(f"lr {float(lr):.2e}")
    if mem is not None:
        parts.append(f"{float(mem):.1f}G")
    for k in METRICS:
        v = (metrics or {}).get(k)
        if v is not None:
            parts.append(f"{k} {float(v):6.2f}")
    # Tách train và val: một epoch 0:53 mà 0:33 trong đó là chấm val thì
    # con số gộp nói sai về tốc độ huấn luyện.
    if seconds is not None:
        parts.append(fmt_time(seconds) if val_seconds is None
                     else f"{fmt_time(seconds)} + val {fmt_time(val_seconds)}")
    line = "   ".join(parts)
    return line + "   * tốt nhất" if best else line


def summary(title: str, rows) -> str:
    """Khối tổng kết cuối lượt chạy. `rows` là các cặp (nhãn, giá trị)."""
    rows = [(str(k), str(v)) for k, v in rows if v not in (None, "")]
    w = max((len(k) for k, _ in rows), default=0)
    body = "\n".join(f"  {k:<{w}}   {v}" for k, v in rows)
    # Gạch ngang lấy theo dòng DÀI NHẤT thật sự in ra, kể cả tiêu đề, chứ
    # không ước theo bề rộng nhãn — nhãn ngắn mà giá trị dài thì khung hở.
    rule = "-" * max([len(title)] + [len(x) for x in body.splitlines()])
    return f"\n{rule}\n{title}\n{rule}\n{body}\n{rule}"


def data_line(info: dict) -> str | None:
    """`prepare()` kiểu COCO -> một dòng. None nếu không phải dạng đó.

    Thay cho việc đổ JSON rồi cắt ở ký tự thứ 400 — chỗ cắt đó rơi đúng vào
    khoá "val", nên số ảnh val và test không bao giờ hiện ra.
    """
    if not isinstance(info, dict):
        return None
    parts = []
    for sp in ("train", "val", "test"):
        d = info.get(sp)
        if not isinstance(d, dict) or "images" not in d:
            return None
        chunk = f"{sp} {d['images']} ảnh / {d.get('regions', '?')} vùng"
        fields = d.get("fields")
        if sp == "test" and isinstance(fields, dict) and len(fields) == 1:
            chunk += f" ({next(iter(fields))})"
        parts.append(chunk)
    return " | ".join(parts)


# ------------------------------------------------------------------- thanh
class Bar:
    """tqdm nếu dùng được, im lặng nếu không. Không bao giờ là lý do gãy train.

    Tự tắt khi stdout không phải terminal: `nohup python train.py > log.txt`
    thì thanh tiến trình chỉ để lại hàng nghìn ký tự \r trong file chứ không
    ai nhìn, trong khi các dòng epoch vẫn ra đủ.

    `leave=True` (mặc định) GIỮ thanh lại sau khi xong. Thanh đã hoàn tất là
    thứ duy nhất nói được tốc độ thật (`1.53it/s`) và thời gian riêng của
    phần đó, tách khỏi thời gian chấm val. Xoá đi là mất số đó.
    """

    def __init__(self, total: int, desc: str = "", enabled: bool = True,
                 leave: bool = True):
        self.total = int(total)
        self.n = 0
        self.bar = None
        if not enabled or not self._tty():
            return
        try:
            from tqdm import tqdm
        except ImportError:
            return
        self.bar = tqdm(total=self.total, desc=desc, unit="it", leave=leave,
                        dynamic_ncols=True, file=sys.stdout)

    @staticmethod
    def _tty() -> bool:
        try:
            return bool(sys.stdout.isatty())
        except (AttributeError, ValueError):
            return False

    def advance(self, n: int = 1, note: str = "") -> None:
        self.n += n
        if self.bar is None:
            return
        if note:
            self.bar.set_postfix_str(note, refresh=False)
        self.bar.update(n)

    def write(self, msg: str) -> None:
        """In một dòng mà không làm vỡ thanh đang vẽ."""
        if self.bar is None:
            print(msg, flush=True)
        else:
            self.bar.write(msg)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
