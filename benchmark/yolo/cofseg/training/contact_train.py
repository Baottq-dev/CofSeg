"""TC2: nhãn phụ có biên tiếp giáp cho đầu `semseg` của Proto26.

Đầu `semseg` của YOLO26 chỉ chạy lúc train và bị xoá khi fuse
(`block.py:2014-2021` của 8.4.143). Nó đọc `feat` = P3 + P4↑ + P5↑ trước
`feat_fuse`, nên gradient của nó đi vào đúng phần đặc trưng mà Proto và các
đầu phát hiện cùng dùng. Nhãn gốc của nó là hợp mọi tán; TC2 thêm hai kênh
(xem contact_targets.py):

    kênh 0..nc-1  tiền cảnh, y như gốc (BCE + Dice)
    kênh nc       dải tiếp giáp (BCE + Dice)
    kênh nc+1     khoảng cách tới biên (L1 của sigmoid, chỉ trên tiền cảnh)

    sem_loss = gốc + box_gain · (w_band · dải + w_dist · khoảng cách)

Mọi thứ nằm trong ô `sem_loss`, nên vẫn đi qua `w_o2m` của ProgLoss như gốc:
chỉ nhãn đổi, lịch trọng số giữ nguyên. Với w_band = w_dist = 0, loss trùng
từng số với gốc; test giữ điều đó.

Ba quyết định cài đặt:

- Lớp cuối của `semseg` (1x1, ra nc kênh) đổi sang nc + 2 kênh TRƯỚC khi nạp
  trọng số. Lớp này không nhận trọng số COCO (COCO có 80 lớp) nên thứ tự không
  đổi gì so với gốc, nhưng --resume cần nạp checkpoint 3 kênh vào lớp 3 kênh.
  Lớp mới khởi tạo trong `fork_rng` với seed riêng: trạng thái ngẫu nhiên
  chung giữ y như lượt gốc cùng seed.
- Lớp model giữ nguyên là SegmentationModel. Hàm loss gắn vào model đang
  train ở `on_train_start`, như cơ chế --loss, nên checkpoint là checkpoint
  chuẩn, nạp được ở mọi nơi mà không cần import gói này. Bản EMA mà validator
  chấm dùng loss gốc; ở chế độ eval `semseg` không chạy nên không sao.
- Hàm loss GỌI LẠI v8SegmentationLoss.loss với kênh tiền cảnh, không chép
  đoạn nào của thư viện.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.modules.block import Proto26
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

from .contact_targets import boundary_distance, contact_band, to_semseg_grid
from .mask_iou_loss import is_end2end

#: Bias khởi đầu của kênh dải: sigmoid(-4) ≈ 0,018, cỡ tỉ lệ pixel dải trong
#: ảnh, để vài trăm bước đầu không phải trả một loss lớn chỉ vì khởi tạo.
BAND_BIAS = -4.0

#: Số bước giãn tối đa khi tính khoảng cách. Ở lưới nhãn 1/4, tán trung vị
#: rộng 31-52 ô; scale 1,5 của augmentation đưa bán kính lên khoảng 40 ô.
MAX_ITER = 64


def find_proto26(model) -> Proto26:
    """Proto26 của một model YOLO26-seg (nn.Module trần hoặc bản EMA)."""
    for m in model.modules():
        if isinstance(m, Proto26):
            return m
    raise SystemExit(
        "TC2 cần đầu `semseg` của Proto26, tức một model yolo26*-seg. "
        "YOLOv8/YOLO11 dùng Proto cũ, không có đầu này."
    )


def expand_semseg(model, seed: int = 0) -> nn.Conv2d:
    """Đổi lớp cuối của `semseg` từ nc sang nc + 2 kênh ra, tại chỗ.

    Kênh tiền cảnh khởi tạo mặc định như gốc; kênh dải: trọng số 0, bias
    BAND_BIAS; kênh khoảng cách: trọng số 0, bias 0. Gọi lần hai thì không
    làm gì: lượt --resume dựng lại model rồi mới nạp checkpoint đã 3 kênh.
    """
    proto = find_proto26(model)
    last = proto.semseg[-1]
    nc = int(model.yaml["nc"]) if hasattr(model, "yaml") else last.out_channels
    if last.out_channels == nc + 2:
        return last
    if last.out_channels != nc:
        raise RuntimeError(f"Lớp cuối của semseg có {last.out_channels} kênh ra, mong đợi {nc}.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new = nn.Conv2d(last.in_channels, nc + 2, 1, bias=True)
    with torch.no_grad():
        new.weight[nc:].zero_()
        new.bias[nc] = BAND_BIAS
        new.bias[nc + 1] = 0.0
    proto.semseg[-1] = new.to(last.weight.device, last.weight.dtype)
    return proto.semseg[-1]


class ContactSemLoss(v8SegmentationLoss):
    """v8SegmentationLoss với hai kênh nhãn phụ của TC2 cộng vào `sem_loss`."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 band: int = 2, w_band: float = 0.25, w_dist: float = 0.25):
        super().__init__(model, tal_topk, tal_topk2)
        self.band = int(band)
        self.w_band = float(w_band)
        self.w_dist = float(w_dist)
        # số batch, Σ loss tiền cảnh, Σ loss dải, Σ loss khoảng cách,
        # Σ xác suất dải trên pixel dải, số pixel dải,
        # Σ xác suất dải trên tiền cảnh ngoài dải, số pixel đó.
        self.stats = torch.zeros(8, dtype=torch.float64, device=self.device)

    def loss(self, preds, batch):
        proto = preds["proto"]
        if not (isinstance(proto, tuple) and len(proto) == 2):
            return super().loss(preds, batch)  # chế độ eval: không có semseg
        proto_t, sem = proto
        nc = self.nc
        if sem.shape[1] != nc + 2:
            raise RuntimeError(
                f"Đầu semseg ra {sem.shape[1]} kênh, TC2 cần {nc + 2}: model chưa qua expand_semseg()."
            )
        masks = batch["masks"].to(self.device)
        if not self.overlap or masks.dim() != 3:
            raise RuntimeError("TC2 dựng nhãn từ bản đồ chỉ số tán: cần train.overlap_mask: true.")
        total, items = super().loss({**preds, "proto": (proto_t, sem[:, :nc])}, batch)

        band_t, dist_t, fg_t = to_semseg_grid(
            contact_band(masks, self.band), boundary_distance(masks, MAX_ITER), masks > 0, sem.shape[-2:]
        )
        logit_band = sem[:, nc:nc + 1].float()
        p_dist = sem[:, nc + 1:nc + 2].float().sigmoid()
        l_band = self.bcedice_loss(logit_band, band_t)
        l_dist = ((p_dist - dist_t).abs() * fg_t).sum() / fg_t.sum().clamp(min=1.0)
        extra = (self.w_band * l_band + self.w_dist * l_dist) * self.hyp.box

        bs = proto_t.shape[0]
        add = torch.zeros_like(total)
        add[4] = extra * bs
        total = total + add
        items = dict(items)
        sem_a = items["sem_loss"] / self.hyp.box
        items["sem_loss"] = items["sem_loss"] + extra.detach()

        with torch.no_grad():
            p_band = logit_band.sigmoid()
            in_band = band_t > 0.5
            out_band = (fg_t > 0.5) & ~in_band
            self.stats += torch.stack([
                torch.ones((), dtype=torch.float64, device=self.device),
                sem_a.double(), l_band.double(), l_dist.double(),
                p_band[in_band].double().sum(), in_band.sum().double(),
                p_band[out_band].double().sum(), out_band.sum().double(),
            ])
        return total, items


def build_criterion(model, block: dict):
    """Hàm loss TC2 thay cho `model.init_criterion()`; bọc E2ELoss như gốc."""
    make = partial(ContactSemLoss, band=block["band"], w_band=block["w_band"], w_dist=block["w_dist"])
    return E2ELoss(model, make) if is_end2end(model) else make(model)


class ContactSegTrainer(SegmentationTrainer):
    """SegmentationTrainer dựng YOLO26-seg chuẩn, đổi lớp cuối của `semseg`
    sang nc + 2 kênh, RỒI mới nạp trọng số. Cấu hình đi vào qua lớp con do
    make_trainer() tạo, vì Model.train() tự khởi tạo trainer."""

    CONTACT: dict = {}

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, None, verbose)
        expand_semseg(model, seed=int(getattr(self.args, "seed", 0) or 0))
        if weights:
            model.load(weights)
        return model


def make_trainer(cfg: dict) -> type[ContactSegTrainer]:
    return type("ContactSegTrainer", (ContactSegTrainer,), {"CONTACT": dict(cfg)})


def attach_criterion(model, block: dict):
    """Gắn hàm loss TC2 vào model đang train, giữ tiến độ ProgLoss.

    Khi --resume, `resume_training` đã dựng sẵn loss gốc và đặt tiến độ của
    ProgLoss theo epoch bắt đầu (`trainer.py:1099-1101`). Bỏ qua tiến độ đó thì
    trọng số one2many quay về 0,8 giữa chừng lượt train.
    """
    last = find_proto26(model).semseg[-1]
    nc = int(model.yaml["nc"])
    if last.out_channels != nc + 2:
        raise RuntimeError(f"semseg ra {last.out_channels} kênh, TC2 cần {nc + 2}: trainer không phải ContactSegTrainer?")
    old = getattr(model, "criterion", None)
    crit = build_criterion(model, block)
    if old is not None and hasattr(old, "updates") and hasattr(crit, "updates"):
        crit.updates, crit.o2m, crit.o2o = old.updates, old.o2m, old.o2o
    model.criterion = crit
    return crit


def _parts(crit) -> list[tuple[str, ContactSemLoss]]:
    if isinstance(crit, ContactSemLoss):
        return [("one", crit)]
    return [(name, getattr(crit, name)) for name in ("one2many",)
            if isinstance(getattr(crit, name, None), ContactSemLoss)]


class EpochLog:
    """Mỗi epoch một dòng vào mask_head_tc2.csv.

    Cột `sem_loss` của results.csv giờ là tổng ba kênh, nên từng kênh ghi ở
    đây: loss tiền cảnh (chưa nhân gain), loss dải, loss khoảng cách; xác suất
    dải trung bình trên pixel dải và trên tiền cảnh ngoài dải (hai số này tách
    nhau là dấu hiệu đầu đã học được dải); trọng số one2many DÙNG TRONG epoch
    đó. Trọng số đọc lúc bắt đầu epoch, vì trainer gọi `criterion.update()`
    trước `on_train_epoch_end`: đọc lúc cuối là đọc trọng số của epoch sau.
    Chỉ ghi nhánh one2many: nhánh one2one tính trên cùng dự đoán đã tách
    gradient nên cho đúng các số đó.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.w_o2m = float("nan")

    def start(self, crit, epoch: int) -> None:
        self.w_o2m = float(getattr(crit, "o2m", float("nan")))
        for _, part in _parts(crit):
            part.stats.zero_()

    def end(self, crit, epoch: int) -> None:
        parts = _parts(crit)
        if not parts:
            return
        row: dict = {"epoch": epoch + 1}
        for name, part in parts:
            n, s_a, s_b, s_d, p_in, n_in, p_out, n_out = part.stats.tolist()
            avg = (lambda s: round(s / n, 4)) if n else (lambda s: float("nan"))
            row.update({
                f"{name}_batches": int(n), f"{name}_fg": avg(s_a), f"{name}_band": avg(s_b),
                f"{name}_dist": avg(s_d),
                f"{name}_p_band_in": round(p_in / n_in, 4) if n_in else float("nan"),
                f"{name}_p_band_out": round(p_out / n_out, 4) if n_out else float("nan"),
            })
        row["w_o2m"] = round(self.w_o2m, 4)
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)


def install_log(yolo_model, run_dir: str | Path) -> None:
    """Gắn hàm loss TC2 và nhật ký epoch vào lượt train sắp chạy.

    Cấu hình lấy từ lớp trainer mà make_trainer() tạo. `on_train_start` bắn sau
    khi trainer đã dựng model, gắn `model.args` và (nếu --resume) dựng loss gốc,
    nhưng trước batch đầu tiên.
    """
    from ultralytics.utils.torch_utils import unwrap_model

    log = EpochLog(Path(run_dir) / "mask_head_tc2.csv")

    def on_train_start(trainer):
        block = getattr(type(trainer), "CONTACT", None)
        if not block:
            raise RuntimeError("Trainer không mang cấu hình TC2: phải tạo bằng contact_train.make_trainer().")
        m = unwrap_model(trainer.model)
        crit = attach_criterion(m, block)
        print(f"TC2: semseg ra {find_proto26(m).semseg[-1].out_channels} kênh; loss TC2 đã gắn "
              f"(band {block['band']}, w_band {block['w_band']:g}, w_dist {block['w_dist']:g}, "
              f"w_o2m hiện tại {getattr(crit, 'o2m', float('nan')):.3f})", flush=True)

    def crit(trainer):
        return getattr(unwrap_model(trainer.model), "criterion", None)

    yolo_model.add_callback("on_train_start", on_train_start)
    yolo_model.add_callback("on_train_epoch_start", lambda t: log.start(crit(t), t.epoch))
    yolo_model.add_callback("on_train_epoch_end", lambda t: log.end(crit(t), t.epoch))
