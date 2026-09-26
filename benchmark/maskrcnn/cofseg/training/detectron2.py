"""Trainer detectron2: Mask R-CNN R50-FPN, Cascade Mask R-CNN R50-FPN,
Mask2Former R50 (và PointRend nếu đưa config) — cùng benchmark/maskrcnn/train.py, cùng
khối `train:` với hai trainer kia.

    python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/maskrcnn_r50_d2.yaml --data data/export/block/f4
    python benchmark/maskrcnn/train.py --config benchmark/maskrcnn/configs/train/mask2former_r50_d2.yaml --probe

Khác biệt có chủ đích với trainer torchvision/YOLO:
- Vòng lặp là DefaultTrainer của detectron2 (hoặc Trainer trong train_net.py
  của Mask2Former); ở đây chỉ dịch epoch/batch/imgsz/lr sang khoá config và
  thêm hook giữ checkpoint tốt nhất theo mask AP trên val.
- CHỈ train và val. Không đụng tới split test: chấm là việc của evaluate.py,
  nơi có Boundary AP và chỉ số biên từng vùng mà bộ chấm của khung không có.
- Mask2Former giữ tăng cường LSJ của recipe gốc; hai model R-CNN dùng lật
  ngang/dọc + xoay 90° như trainer torchvision (ảnh nadir không có chiều trên).

CHƯA CHẠY THẬT: detectron2 không dựng được trên máy phát triển (Windows).
`build_opts` có test; prepare/probe/fit phải khói trên Linux trước khi tin.
"""

from __future__ import annotations

import contextlib
import csv
import os
import pathlib
import io
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from ..datasets.coco import CocoDataset
from ..datasets.instances import imread
from ..models.detectron2 import ARCHS, base_cfg, class_opts, require_detectron2, size_opts
from ..registry import register
from .. import progress
from . import memory
from .base import Trainer

#: Số tiến trình nạp dữ liệu. Trên Windows mỗi worker spawn một tiến trình mới
#: và nạp lại torch, mà paging file mặc định không đủ -> WinError 1455 giữa lúc
#: quét nhãn. Trên Linux thì 8 là mức hợp lý cho máy thuê. Để một con số cứng
#: rồi bắt lệnh mẫu ghi đè là cách chắc chắn để ai đó quên --workers.
DEFAULT_WORKERS = 0 if os.name == "nt" else 8

#: Cùng tên với trainer torchvision ở đâu có thể, để --imgsz/--batch/--epochs
#: nghĩa như nhau trên mọi model. lr None = theo recipe gốc của từng arch,
#: tỉ lệ tuyến tính theo batch (0.02 @16 cho R-CNN, 1e-4 @16 cho Mask2Former).
D2_DEFAULTS: dict = {
    "imgsz": 1024,
    # Đo thật trên RTX 5090: batch 16 ở imgsz 1024 dùng 12.7 GB. Mặc định cũ
    # là 4 — con số của card 8 GB ở nhà, không phải của máy sẽ chạy thật.
    "batch": 16,
    "epochs": 50,
    "lr": None,
    "weight_decay": None,
    "momentum": 0.9,
    "lr_steps": [0.7, 0.9],
    "lr_gamma": 0.1,
    # < 1 là PHẦN của lịch, >= 1 là số vòng tuyệt đối (cùng quy ước lr_steps).
    # 200 vòng cứng là 12.5% lịch khi batch 16 (500 ảnh -> 32 vòng/epoch,
    # 50 epoch -> 1600 vòng), trong khi recipe COCO warmup chưa tới 1%.
    "warmup_iters": 0.03,
    "amp": True,
    "fliplr": 0.5,
    "flipud": 0.5,
    "rot90": True,
    "val_every": 1,
    "val_conf": 0.05,
    # Batch lúc chấm val. None = theo batch train. Mặc định của detectron2 là
    # 1, và ở batch 1 thì val tốn gấp đôi train trên bộ này dù nó ít ảnh hơn
    # bốn lần — suy luận không có backward nên còn thừa rất nhiều VRAM.
    "val_batch": None,
    "max_det": 100,
    "num_queries": 100,   # chỉ Mask2Former; ảnh dày nhất có 48 tán
    # Số checkpoint ĐỊNH KỲ giữ lại trong d2/. DefaultTrainer dựng
    # PeriodicCheckpointer mà không truyền max_to_keep, nên mặc định của
    # detectron2 là giữ HẾT: một file mỗi epoch, ~350 MB với Mask R-CNN
    # (trọng số + buffer momentum) và ~530 MB với Mask2Former (AdamW giữ hai
    # moment). 50 epoch là 17 GB, 100 epoch là 53 GB, nhân 18 lượt fold.
    #
    # 1 là đủ: thứ cứu một lượt chạy bị ngắt giữa chừng là checkpoint định kỳ
    # gần nhất cộng file `last_checkpoint` trỏ vào nó — `model_final.pth`
    # (và `weights/last.pth` mà trainer chép ra từ nó) chỉ có khi train chạy
    # hết. `model_best.pth` do BestCheckpointer ghi riêng nên không bị đụng.
    "keep_ckpts": 1,
    # Đầu mask của R-CNN dự đoán ở POOLER_RESOLUTION*2 rồi phóng lên bbox.
    # 14 (-> 28x28) là recipe gốc, và với tán trung vị 129 px ở imgsz 1024 thì
    # mỗi ô mask nuốt 4.6 px — đó là TRẦN đường biên của model này, không phải
    # imgsz. Đặt 28 (-> 56x56) hạ còn 2.3 px/ô; giữ 14 làm mặc định để mốc số 0
    # đúng recipe, nhưng đây là thí nghiệm đáng chạy cho dự án lấy biên làm trọng tâm.
    "mask_resolution": 14,
    "workers": DEFAULT_WORKERS,
    "seed": 0,
    "log_every": 20,
    # verbose=True trả lại đúng cách detectron2 in mặc định: dump config, dump
    # kiến trúc model, một dòng mỗi 20 iteration. Bật khi cần soi lỗi của
    # chính khung, còn lúc train bình thường thì 71% số dòng là hai dump đó.
    "verbose": False,
}
BASE_LR = {"maskrcnn": 0.02, "cascade": 0.02, "pointrend": 0.02, "mask2former": 1e-4}
#: weight decay của recipe gốc: SGD của R-CNN dùng 1e-4, AdamW của Mask2Former 0.05.
BASE_WD = {"maskrcnn": 1e-4, "cascade": 1e-4, "pointrend": 1e-4, "mask2former": 0.05}


def build_opts(arch: str, n_train: int, args: dict, aspect: float = 9 / 16,
               names: dict | None = None, out_dir: str = "",
               backbone: dict | None = None) -> list:
    """Khối train: -> danh sách [khoá, giá trị, ...] cho cfg.merge_from_list.

    Thuần Python, không cần detectron2: đây là phần kiểm được ở nhà.

    `backbone` là ô trong BACKBONES nếu lần chạy có đổi backbone. Nó chỉ được
    phép đổi những gì recipe của backbone đó BẮT BUỘC đổi — lr/weight_decay
    gốc (ViTDet dùng AdamW 1e-4/0.1 thay SGD 0.02/1e-4) và định dạng kênh ảnh.
    Để None thì hàm này cho ra đúng danh sách như trước khi có backbone, đó là
    điều kiện để lượt r50 không đổi kết quả.
    """
    if arch not in ARCHS:
        raise ValueError(f"arch {arch!r} không có; có: {ARCHS}")
    bb = backbone or {}
    batch, epochs = int(args["batch"]), int(args["epochs"])
    if batch < 1 or epochs < 1 or n_train < 1:
        raise ValueError("batch, epochs và số ảnh train đều phải >= 1")
    per_epoch = math.ceil(n_train / batch)
    max_iter = epochs * per_epoch
    lr = args.get("lr")
    if lr is None:
        lr = float((bb.get("optim") or {}).get("lr", BASE_LR[arch])) * batch / 16
    wd = args.get("weight_decay")
    if wd is None:
        wd = (bb.get("optim") or {}).get("weight_decay", BASE_WD[arch])
    # lr_steps nhận PHẦN của lịch (0.7 = 70% số vòng) hoặc số vòng tuyệt đối,
    # phân biệt bằng < 1. Ghi theo phần thì đổi epochs không phải tính lại mốc.
    w = float(args["warmup_iters"])
    warmup = int(w * max_iter) if 0 < w < 1 else int(w)
    warmup = max(1, min(warmup, max_iter))
    steps = tuple(sorted({int(s * max_iter) if 0 < float(s) < 1 else int(s)
                          for s in (args.get("lr_steps") or ())
                          if 0 < (float(s) * max_iter if float(s) < 1 else float(s)) < max_iter}))
    names = names or {}
    opts = [
        "DATASETS.TRAIN", (names.get("train", "coffee_train"),),
        "DATASETS.TEST", (names.get("val", "coffee_val"),),
        "DATALOADER.NUM_WORKERS", int(args["workers"]),
        # Ảnh nền (đã xem, không có tán) vẫn vào train làm mẫu âm, như YOLO.
        "DATALOADER.FILTER_EMPTY_ANNOTATIONS", False,
        "SOLVER.IMS_PER_BATCH", batch,
        "SOLVER.BASE_LR", float(lr),
        "SOLVER.MAX_ITER", max_iter,
        "SOLVER.STEPS", steps,
        "SOLVER.GAMMA", float(args["lr_gamma"]),
        "SOLVER.MOMENTUM", float(args["momentum"]),
        "SOLVER.WEIGHT_DECAY", float(wd),
        "SOLVER.WARMUP_ITERS", warmup,
        "SOLVER.CHECKPOINT_PERIOD", per_epoch,
        "SOLVER.AMP.ENABLED", bool(args["amp"]),
        "TEST.EVAL_PERIOD", per_epoch * int(args["val_every"]),
        "TEST.DETECTIONS_PER_IMAGE", int(args["max_det"]),
        "SEED", int(args["seed"]),
        "OUTPUT_DIR", str(out_dir),
    ]
    opts += class_opts(arch, int(args["num_queries"]))
    opts += size_opts(int(args["imgsz"]), aspect)
    if bb.get("input_format"):
        # DatasetMapper đọc ảnh theo cfg.INPUT.FORMAT. R50 của detectron2 là
        # BGR, còn ViTDet tiền huấn luyện MAE là RGB với mean/std ImageNet.
        # Lệch chỗ này thì ảnh vào model với hai kênh đảo nhau và KHÔNG có gì
        # báo — loss vẫn giảm, chỉ là giảm từ một điểm xuất phát tệ hơn nhiều.
        opts += ["INPUT.FORMAT", str(bb["input_format"])]
    if arch == "mask2former":
        # LSJ của recipe gốc cắt ô vuông IMAGE_SIZE sau khi co giãn 0.1-2.0.
        opts += ["INPUT.IMAGE_SIZE", int(args["imgsz"])]
    else:
        opts += ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", float(args["val_conf"]),
                 "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION", int(args["mask_resolution"])]
    return opts



def lazy_model(spec: dict, num_classes: int, imgsz: int, conf: float, max_det: int):
    """`instantiate` model ViTDet từ LazyConfig của chính detectron2.

    Vì sao không tự dựng backbone bằng BACKBONE_REGISTRY: tên tham số trong
    checkpoint COCO do CÁCH DỰNG quyết định. Dựng lại bằng tay là đoán, và
    đoán sai thì DetectionCheckpointer chỉ log "Skipped loading parameter" rồi
    train tiếp với phần lệch khởi tạo ngẫu nhiên — đúng loại lỗi im lặng đã
    xảy ra một lần với R101 nạp trọng số R50. Dựng từ chính config của tác giả
    thì khớp theo cấu trúc, không theo trí nhớ.

    Trả về model đã instantiate; phần dữ liệu / solver / evaluator vẫn đi
    đường CfgNode như mọi backbone khác.
    """
    from detectron2 import model_zoo
    from detectron2.config import instantiate

    m = model_zoo.get_config(spec["lazy"]).model
    v = spec["vit"]
    net = m.backbone.net
    net.embed_dim = int(v["embed_dim"])
    net.depth = int(v["depth"])
    net.num_heads = int(v["num_heads"])
    net.drop_path_rate = float(v["drop_path_rate"])
    # window_block_indexes = mọi block TRỪ các block attention toàn cục.
    net.window_block_indexes = [i for i in range(int(v["depth"]))
                                if i not in set(v["global_at"])]
    # ViT nội suy pos_embed theo cỡ ảnh thật nên imgsz khác 1024 vẫn chạy;
    # square_pad phải đi cùng, không thì ảnh 1024x576 bị đệm về 1024x1024 và
    # gần một nửa phép tính đổ vào phần đệm.
    net.img_size = int(imgsz)
    m.backbone.square_pad = int(imgsz)
    # Một lớp: num_classes của roi_heads nội suy sang box_predictor và mask_head.
    m.roi_heads.num_classes = int(num_classes)
    m.roi_heads.box_predictor.test_score_thresh = float(conf)
    m.roi_heads.box_predictor.test_topk_per_image = int(max_det)
    return instantiate(m)


def lazy_optimizer(spec: dict, model, lr: float, weight_decay: float):
    """AdamW + layer-wise lr decay, đúng recipe ViTDet.

    Không dùng được SGD 0.02 của R-CNN ở đây: ViT tiền huấn luyện MAE cần lr
    nhỏ và bước giảm theo độ sâu (`get_vit_lr_decay_rate`), thiếu nó thì lượt
    finetune phá hỏng đặc trưng đã học ngay trong vài trăm vòng đầu.
    """
    from functools import partial

    import torch
    from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
    from detectron2.solver.build import get_default_optimizer_params

    o = spec["optim"]
    params = get_default_optimizer_params(
        model,
        base_lr=float(lr),
        weight_decay_norm=0.0,
        lr_factor_func=partial(get_vit_lr_decay_rate,
                               num_layers=int(o["num_layers"]),
                               lr_decay_rate=float(o["lr_decay_rate"])),
        overrides=None if o.get("no_pos_embed_override")
        else {"pos_embed": {"weight_decay": 0.0}},
    )
    return torch.optim.AdamW(params, lr=float(lr), betas=(0.9, 0.999),
                             weight_decay=float(weight_decay))


def weights_report(inc, model) -> dict:
    """Trọng số nào KHÔNG vào được model, tách phần chờ đợi khỏi phần đáng lo.

    Đổi 80 lớp COCO sang 1 lớp thì đầu phân loại lệch hình — điều đó là cố ý.
    Backbone lệch thì không: nó nghĩa là checkpoint không khớp kiến trúc, và
    hậu quả là một lượt train trông vẫn bình thường trên một backbone thật ra
    khởi tạo ngẫu nhiên.
    """
    thieu = list(getattr(inc, "missing_keys", None) or [])
    lech = [k for k, *_ in (getattr(inc, "incorrect_shapes", None) or [])]
    def _bb(ks):
        return sorted(k for k in ks if k.startswith("backbone."))
    ra = {
        "missing": len(thieu),
        "incorrect_shapes": len(lech),
        "backbone_missing": _bb(thieu)[:10],
        "backbone_incorrect": _bb(lech)[:10],
    }
    ra["backbone_ok"] = not ra["backbone_missing"] and not ra["backbone_incorrect"]
    return ra


def iters_per_epoch(n_train: int, batch: int) -> int:
    return math.ceil(n_train / max(1, int(batch)))


@register("trainer", "detectron2")
class Detectron2Trainer(Trainer):
    # ----------------------------------------------------------------- backbone
    #: Các bản CÓ trọng số COCO và nạp được bằng get_cfg() + merge_from_file.
    #:
    #: Bốn bản mạnh nhất của model zoo (new_baselines LSJ, mask AP 40.3-42.5)
    #: KHÔNG có ở đây: chúng là LazyConfig .py, phải chạy qua
    #: lazyconfig_train_net chứ không qua DefaultTrainer. Đáng nhớ một điều từ
    #: bảng đó: R50 + LSJ 100 epoch (40.3) hơn X101 3x (39.5) — recipe, không
    #: phải backbone.
    #:
    #: C4 và DC5 (R50/R101) đều thấp hơn FPN cùng lịch nên không đưa vào.
    BACKBONES = {
        "maskrcnn": {
            "r50": dict(
                config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                note="mốc số 0 — mask AP 37.2, 3.4 GB, 0.261 s/iter"),
            "r101": dict(
                config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                note="mask AP 38.6, 4.6 GB, 0.340 s/iter — backbone DUY NHẤT Mask2Former cũng có"),
            "x101": dict(
                config="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                note="mask AP 39.5, 7.2 GB, 0.690 s/iter — nặng nhất, dò --probe trước"),
            "r50-dconv": dict(
                config="Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml",
                note="mask AP 38.5, 3.5 GB, 0.349 s/iter — bằng R101 mà vẫn vừa batch 16"),
            "r50-gn": dict(
                config="Misc/mask_rcnn_R_50_FPN_3x_gn.yaml",
                note="mask AP 38.6, 5.6 GB, 0.309 s/iter — GroupNorm thay BatchNorm"),
        # --- ViTDet: model dựng từ LazyConfig, không phải từ yaml ---
        #
        # `lazy` là config LazyConfig trong gói detectron2 (configs/ được đóng
        # gói theo package nên không cần checkout). Trainer `instantiate` model
        # từ đó rồi giao cho DefaultTrainer; nhờ vậy tên tham số trong
        # checkpoint khớp theo cách dựng chứ không theo cách đoán.
        #
        # Ba thứ ViTDet bắt buộc đổi so với recipe R-CNN, khai ngay ở đây để
        # không chỗ nào phải nhớ hộ:
        #   input_format RGB (R50 dùng BGR — đọc sai kênh thì không ai báo),
        #   AdamW thay SGD, và layer-wise lr decay theo độ sâu.
            "vit-b": dict(
                lazy="common/models/mask_rcnn_vitdet.py",
                vit=dict(embed_dim=768, depth=12, num_heads=12,
                         drop_path_rate=0.1, global_at=(2, 5, 8, 11)),
                optim=dict(num_layers=12, lr_decay_rate=0.7, lr=1e-4, weight_decay=0.1),
                input_format="RGB",
                checkpoint="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                           "mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl",
                note="mask AP 45.9 — 86M tham số, bản DUY NHẤT trong nhóm này còn hy vọng vừa card"),
            "vit-l": dict(
                lazy="common/models/mask_rcnn_vitdet.py",
                vit=dict(embed_dim=1024, depth=24, num_heads=16,
                         drop_path_rate=0.4, global_at=(5, 11, 17, 23)),
                optim=dict(num_layers=24, lr_decay_rate=0.8, lr=1e-4, weight_decay=0.1),
                input_format="RGB",
                checkpoint="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                           "mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl",
                note="mask AP 49.2 — 304M tham số, cần batch rất nhỏ"),
            "vit-h": dict(
                lazy="common/models/mask_rcnn_vitdet.py",
                vit=dict(embed_dim=1280, depth=32, num_heads=16,
                         drop_path_rate=0.5, global_at=(7, 15, 23, 31)),
                optim=dict(num_layers=32, lr_decay_rate=0.9, lr=1e-4, weight_decay=0.1,
                           no_pos_embed_override=True),
                input_format="RGB",
                checkpoint="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                           "mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl",
                note="mask AP 50.2 — 632M tham số, ghi lại cho đủ bảng chứ khó chạy"),
        },
        "cascade": {
            "r50": dict(
                config="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
                note="mask AP 38.5, 4.0 GB, 0.328 s/iter"),
            "x152": dict(
                config="Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
                note="mask AP 44.0, 15.1 GB — IN5k + GN + dconv, KHÔNG so được với R50"),
        },
        # Repo Mask2Former không nằm trong model zoo API nên checkpoint phải
        # khai thẳng; đường dẫn config là đường dẫn BÊN TRONG upstream/.
        "mask2former": {
            "r50": dict(
                config="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
                note="mask AP 43.7 — mặc định, cùng backbone với hai model R-CNN"),
            "r101": dict(
                config="configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_R101_bs16_50ep/model_final_eba159.pkl",
                note="mask AP 44.2 — chỉ hơn R50 0.5 điểm trên COCO"),
            "swin-t": dict(
                config="configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl",
                note="mask AP 45.0 — đổi hẳn họ backbone sang transformer"),
            "swin-s": dict(
                config="configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl",
                note="mask AP 46.3"),
            "swin-b": dict(
                config="configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl",
                note="mask AP 46.7 — nặng, dò --probe trước"),
            "swin-b-in21k": dict(
                config="configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl",
                note="mask AP 48.1 — tiền huấn luyện IN21k, khác nguồn dữ liệu"),
            "swin-l-in21k": dict(
                config="configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
                checkpoint="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl",
                note="mask AP 50.1 — nặng nhất repo, gần như chắc chắn OOM ở batch 16"),
        },
    }

    def __init__(self, cfg: dict, run_dir):
        super().__init__(cfg, run_dir)
        m = cfg.get("model") or {}
        self.arch: str = m.get("arch", "maskrcnn")
        if self.arch not in ARCHS:
            raise ValueError(f"model.arch {self.arch!r} không có; có: {ARCHS}")
        self.repo = m.get("repo")
        self.config_file = m.get("config_file")
        self.weights = m.get("weights")
        self.backbone: str | None = m.get("backbone")
        bang = self.BACKBONES.get(self.arch, {})
        if self.backbone and self.backbone not in bang:
            raise ValueError(f"model.backbone {self.backbone!r} không có với {self.arch}; "
                             f"có: {', '.join(sorted(bang))}")
        #: Ô trong BACKBONES của lần chạy này, None khi dùng mặc định của arch.
        self.bb: dict = bang.get(self.backbone) or {}
        self.train_args: dict = {**D2_DEFAULTS, **(cfg.get("train") or {})}
        d = cfg.get("data") or {}
        self.root = Path(d.get("root", "data/export/block/f4"))
        self.splits = {k: d.get(k, k) for k in ("train", "val", "test")}
        self.min_area = float(d.get("min_area", 0.0))
        self.limit = d.get("limit")
        self.out_dir = self.run_dir / "d2"

    @classmethod
    def arch_of(cls, cfg: dict) -> str:
        m = cfg.get("model") or {}
        return str(m.get("arch") or "maskrcnn")

    @classmethod
    def apply_backbone(cls, cfg: dict, name: str) -> str:
        """Ghi TÊN backbone vào config chứ không ghi đường dẫn.

        Đường dẫn là chi tiết của bảng: bản yaml cần `config_file`, bản ViTDet
        không có yaml nào và cần `lazy` + kích thước ViT. Để trainer tra bảng
        thì thêm một backbone chỉ là thêm một dòng ở BACKBONES.
        """
        bang = cls.backbones(cfg)
        arch = cls.arch_of(cfg)
        if name not in bang:
            raise SystemExit(f"--backbone {name!r} không có với {arch}. "
                             f"Có: {', '.join(sorted(bang))}")
        m = cfg.setdefault("model", {})
        m["backbone"] = name
        # Đường dẫn/trọng số còn sót trong config sẽ đè lên bảng; bỏ đi.
        m.pop("config_file", None)
        m.pop("weights", None)
        return f"{arch}-{name}"

    # -------------------------------------------------------------------- tham số
    @classmethod
    def param_defaults(cls) -> dict | None:
        return dict(D2_DEFAULTS)

    TAG_KEYS = (("imgsz", "i"), ("batch", "b"), ("epochs", "e"))

    @classmethod
    def run_tag(cls, cfg: dict) -> str:
        args = {**D2_DEFAULTS, **(cfg.get("train") or {})}
        return "".join(f"{p}{args[k]}" for k, p in cls.TAG_KEYS)

    # ------------------------------------------------------------------ chuẩn bị
    def _dataset_names(self) -> dict[str, str]:
        fold = self.root.name
        return {sp: f"coffee_{fold}_{self.splits[sp]}" for sp in ("train", "val", "test")}

    def prepare(self) -> dict:
        """Kiểm dữ liệu bằng bộ đọc của repo (không cần detectron2), đếm ảnh
        train để đổi epoch -> iteration."""
        out = {}
        for sp in ("train", "val", "test"):
            ds = CocoDataset(self.root, self.splits[sp], min_area=self.min_area)
            out[sp] = ds.summary()
            if sp == "train":
                self.n_train = len(ds)
                first = ds.image_list()[0]
                shape = imread(first.path).shape
                self.aspect = shape[0] / shape[1]
                out["sample"] = {"file": first.file_name, "shape": list(shape)}
        if self.limit:
            self.n_train = min(self.n_train, int(self.limit))
            out["limit"] = int(self.limit)
        out["arch"] = self.arch
        out["iters_per_epoch"] = iters_per_epoch(self.n_train, self.train_args["batch"])
        self.dataset_counts = {sp: out[sp]["images"] for sp in ("train", "val", "test")}
        self.test_fields = sorted(out["test"].get("fields") or {})
        (self.run_dir / "dataset_check.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    # ------------------------------------------------------------ detectron2
    def _register(self) -> dict[str, str]:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances

        names = self._dataset_names()
        for sp, name in names.items():
            if name in DatasetCatalog.list():
                continue
            register_coco_instances(
                name, {},
                str(self.root / "annotations" / f"instances_{self.splits[sp]}.json"),
                str(self.root / "images" / self.splits[sp]),
            )
            MetadataCatalog.get(name).thing_classes = ["canopy"]
        if self.limit:
            # Khói: cắt danh sách train sau khi đăng ký, giữ nguyên val/test.
            full = DatasetCatalog.get(names["train"])
            DatasetCatalog.remove(names["train"])
            DatasetCatalog.register(names["train"], lambda d=full[: int(self.limit)]: d)
            MetadataCatalog.get(names["train"]).thing_classes = ["canopy"]
        return names

    def _cfg(self):
        require_detectron2()
        if not hasattr(self, "n_train"):
            self.prepare()
        names = self._register()
        # Bản ViTDet không có yaml: lấy CfgNode của arch làm bộ khung (dữ liệu,
        # solver, evaluator vẫn đi đường cũ) rồi thay riêng phần model ở
        # build_model. Bản yaml thì config_file quyết định cả kiến trúc.
        cfg, m2f = base_cfg(self.arch, repo=self.repo,
                            config_file=self.config_file or self.bb.get("config"),
                            weights=self.weights or self.bb.get("checkpoint"))
        cfg.merge_from_list(build_opts(self.arch, self.n_train, self.train_args,
                                       aspect=self.aspect, names=names,
                                       out_dir=str(self.out_dir), backbone=self.bb))
        cfg.freeze()
        return cfg, m2f, names

    def _epoch_reporter(self):
        """Hook vẽ thanh tiến trình trong epoch và in một dòng khi epoch xong.

        detectron2 đếm theo ITERATION chứ không theo epoch, nên chữ "epoch"
        không xuất hiện lần nào trong log gốc suốt lúc train. Hook này dịch
        ngược lại: mỗi `per_epoch` iteration là một epoch, đóng thanh, đọc
        segm/AP mà EvalHook vừa ghi vào storage, in một dòng rồi mở thanh mới.

        Một chi tiết dễ sai: EvalHook CỐ TÌNH bỏ qua iteration cuối trong
        `after_step` và chấm val trong `after_train` thay vào đó. Nên epoch
        cuối — epoch duy nhất mà người ta thật sự quan tâm — phải báo cáo ở
        `after_train`, không thì nó in ra dòng thiếu mất AP.
        """
        from detectron2.engine import HookBase

        a = self.train_args
        per_epoch = iters_per_epoch(self.n_train, a["batch"])
        epochs = int(a["epochs"])
        total = per_epoch * epochs

        def value(storage, key):
            """Số mới nhất của một khoá, None nếu khoá chưa từng được ghi."""
            got = storage.latest().get(key)
            return None if got is None else float(got[0])

        def smoothed(storage, key, window=20):
            try:
                return float(storage.history(key).median(window))
            except (KeyError, ValueError):
                return None

        def note(loss):
            return "" if loss is None else f"loss {loss:.3f}"

        def peak_memory():
            try:
                import torch

                if torch.cuda.is_available():
                    # 0 nghĩa là chưa cấp phát gì (chạy CPU, hoặc gọi quá sớm);
                    # in "0.0G" ra chỉ làm người đọc tưởng model không dùng GPU.
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    return peak or None
            except ImportError:
                pass
            return None

        class EpochReporter(HookBase):
            def __init__(self):
                self.bar = None
                self.epoch = 0
                self.best = None
                self.started = None
                self.trained = None      # giây train của epoch, chưa tính val
                self.pending = False
                # Cộng dồn cả lượt: train và val lệch nhau vài lần, nên khối
                # cuối phải nói riêng từng cái thay vì gộp thành một số.
                self.train_total = 0.0
                self.val_total = 0.0

            def _open(self):
                self.epoch += 1
                self.started = time.time()
                self.trained = None
                self.bar = progress.Bar(per_epoch, f"epoch {self.epoch}/{epochs}")

            def close_train_bar(self, fill: bool = True):
                """Đóng thanh train và chốt thời gian train của epoch này.

                Gọi từ EVALUATOR chứ không phải từ hook. Lý do: EvalHook chấm
                val ngay trong after_step của chính nó, mà hook báo cáo lại
                nằm cuối danh sách — tới lượt nó thì val đã xong. Chỗ duy
                nhất biết "val vừa bắt đầu" là evaluator, lúc reset().

                `fill=False` khi lượt chạy CHẾT GIỮA CHỪNG. `train()` của
                detectron2 gọi `after_train` trong `finally`, nên hook này
                vẫn chạy sau khi ném ngoại lệ — kéo đầy ở đó thì một lượt
                OOM ngay iteration 1 để lại dòng `30/30 [00:00, 36it/s]`,
                trông như một epoch đã xong trong 0 giây.

                Gọi nhiều lần vẫn an toàn: lần sau không làm gì.
                """
                if self.bar is not None:
                    # Kéo nốt cho đầy trước khi đóng. EvalHook chấm val NGAY
                    # TRONG after_step của iteration cuối epoch, tức là trước
                    # khi hook này kịp đếm bước đó — không kéo thì thanh đứng
                    # ở 29/30 dù epoch đã xong.
                    if fill:
                        self.bar.advance(max(0, self.bar.total - self.bar.n))
                    self.bar.close()
                    self.bar = None
                if self.trained is None and self.started is not None:
                    self.trained = time.time() - self.started

            def _report(self):
                st = self.trainer.storage
                ap = value(st, "segm/AP")
                is_best = ap is not None and (self.best is None or ap > self.best)
                if is_best:
                    self.best = ap
                whole = None if self.started is None else time.time() - self.started
                train_s = self.trained if self.trained is not None else whole
                val_s = None if (whole is None or self.trained is None) else whole - self.trained
                self.train_total += train_s or 0.0
                self.val_total += val_s or 0.0
                print(progress.epoch_line(
                    self.epoch, epochs, self.epoch * per_epoch, total,
                    loss=smoothed(st, "total_loss"), lr=value(st, "lr"),
                    mem=peak_memory(),
                    metrics={"mAP50-95": ap, "mAP50": value(st, "segm/AP50")},
                    best=is_best, seconds=train_s, val_seconds=val_s,
                ), flush=True)
                self.pending = False

            # ------------------------------------------------------ vòng đời
            def before_train(self):
                self._open()

            def after_step(self):
                done = self.trainer.iter - self.trainer.start_iter + 1
                # Thanh có thể ĐÃ bị đóng ngay trong chính after_step này:
                # EvalHook đứng trước hook này trong danh sách, nó chấm val, và
                # evaluator gọi close_train_bar() lúc reset(). Nên đây không
                # phải phép kiểm phòng xa — nó xảy ra ở MỌI cuối epoch.
                if self.bar is not None:
                    self.bar.advance(1, note(smoothed(self.trainer.storage, "total_loss")))
                if done % per_epoch:
                    return
                self.close_train_bar()
                if done >= self.trainer.max_iter - self.trainer.start_iter:
                    self.pending = True     # AP của epoch cuối chưa có, xem docstring
                else:
                    self._report()
                    self._open()

            def after_train(self):
                # `pending` chỉ bật khi after_step đã chạy đủ tới iteration
                # cuối, nên nó cũng là câu trả lời cho "lượt chạy có tới
                # đích không". Chết giữa chừng thì để thanh đứng đúng chỗ nó
                # dừng, đừng kéo cho đầy.
                self.close_train_bar(fill=self.pending)
                if self.pending:
                    self._report()

        return EpochReporter()

    def _trainer_cls(self, m2f):
        """DefaultTrainer (R-CNN) hoặc Trainer của Mask2Former, thêm hook giữ
        checkpoint tốt nhất theo segm/AP trên val và mapper tăng cường nadir."""
        from detectron2.data import DatasetMapper, build_detection_train_loader
        from detectron2.data import transforms as T
        from detectron2.engine import DefaultTrainer, hooks
        from detectron2.evaluation import COCOEvaluator
        from detectron2.utils.events import CommonMetricPrinter

        a = self.train_args
        base = m2f.Trainer if m2f is not None else DefaultTrainer
        val_batch = max(1, int(a.get("val_batch") or a["batch"]))
        keep_ckpts = max(1, int(a["keep_ckpts"]))
        bb = self.bb
        quiet = not a.get("verbose")
        reporter = None if not quiet else self._epoch_reporter()
        self._reporter = reporter

        class QuietCOCOEvaluator(COCOEvaluator):
            """Chấm im lặng, kèm thanh tiến trình riêng cho val/test.

            pycocotools in bảng 12 dòng bằng print() thẳng ra stdout. Đó là lý
            do hush() một mình không đủ: nó chỉnh module logging, còn chỗ này
            không đi qua logging. Hứng lại rồi đẩy vào logger để bảng vẫn
            xuống d2/log.txt mà không lên màn hình.

            Thanh tiến trình đặt ở ĐÂY chứ không ở hook, vì đây là chỗ duy
            nhất biết val bắt đầu và kết thúc lúc nào: `reset()` chạy trước
            vòng suy luận, `process()` một lần mỗi ảnh, `evaluate()` sau cùng.
            Hook báo cáo nằm cuối danh sách hook nên tới lượt nó thì val đã
            chấm xong từ lâu.
            """

            label = "val"
            n_images = 0
            on_start = None

            def reset(self):
                super().reset()
                if self.on_start is not None:
                    self.on_start()      # chốt giờ train, đóng thanh train
                self._bar = progress.Bar(self.n_images, self.label) if quiet else None

            def process(self, inputs, outputs):
                super().process(inputs, outputs)
                if getattr(self, "_bar", None) is not None:
                    self._bar.advance(len(inputs))

            def evaluate(self, *args, **kw):
                if getattr(self, "_bar", None) is not None:
                    self._bar.close()
                    self._bar = None
                if not quiet:
                    return super().evaluate(*args, **kw)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = super().evaluate(*args, **kw)
                text = buf.getvalue().strip()
                if text:
                    logging.getLogger("detectron2").info("bảng COCO:\n%s", text)
                return res

        class CoffeeTrainer(base):
            @classmethod
            def build_model(cls, cfg):
                """Bản yaml đi đường cũ; bản LazyConfig dựng riêng rồi trả về.

                `bb` rỗng hoặc không có khoá `lazy` nghĩa là backbone yaml —
                lúc đó hàm này KHÔNG đụng gì, nên lượt r50 chạy y như trước.
                """
                if not bb.get("lazy"):
                    return base.build_model(cfg)
                model = lazy_model(
                    bb, num_classes=1, imgsz=int(a["imgsz"]),
                    conf=float(a["val_conf"]), max_det=int(a["max_det"]),
                ).to(cfg.MODEL.DEVICE)
                logging.getLogger("detectron2").info(
                    "backbone LazyConfig %s: %.1fM tham số",
                    bb["lazy"], sum(p.numel() for p in model.parameters()) / 1e6)
                return model

            @classmethod
            def build_optimizer(cls, cfg, model):
                if not bb.get("lazy"):
                    return base.build_optimizer(cfg, model)
                return lazy_optimizer(bb, model, cfg.SOLVER.BASE_LR,
                                      cfg.SOLVER.WEIGHT_DECAY)

            def resume_or_load(self, resume=False):
                """Như bản gốc, nhưng GIỮ LẠI báo cáo khoá không khớp.

                DefaultTrainer gọi checkpointer rồi vứt kết quả đi, nên việc
                backbone không nạp được chỉ còn là một dòng WARNING lẫn trong
                log. Ở đây nó thành một mục trong summary.json và một dòng in
                ra khi có vấn đề.
                """
                inc = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS,
                                                       resume=resume)
                if resume and self.checkpointer.has_checkpoint():
                    self.start_iter = self.iter + 1
                bao = weights_report(inc, self.model)
                CoffeeTrainer.weights_loaded = bao
                if not bao["backbone_ok"]:
                    print(f"CẢNH BÁO: trọng số backbone không khớp checkpoint "
                          f"(thiếu {len(bao['backbone_missing'])}, "
                          f"lệch hình {len(bao['backbone_incorrect'])}). "
                          f"Lượt này khởi đầu từ phần lớn là ngẫu nhiên.", flush=True)
                return inc

            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if m2f is not None:
                    return base.build_evaluator(cfg, dataset_name, output_folder)
                ev = QuietCOCOEvaluator(
                    dataset_name, tasks=("segm",),
                    output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / "inference"))
                # Tổng số ảnh cho thanh tiến trình. DatasetCatalog trả danh
                # sách đã nạp sẵn nên phép đếm này không đọc lại đĩa.
                from detectron2.data import DatasetCatalog

                ev.n_images = len(DatasetCatalog.get(dataset_name))
                ev.label = "test" if dataset_name.endswith("_test") else "val"
                ev.on_start = None if reporter is None else reporter.close_train_bar
                return ev

            def build_writers(self):
                """Bỏ CommonMetricPrinter — đó chính là thứ in `iter: 19
                total_loss: ...` ra màn hình, và ta thay nó bằng thanh tiến
                trình. JSONWriter phải GIỮ: results.csv dựng từ metrics.json
                mà nó ghi ra, bỏ đi là mất cả bảng kết quả.
                """
                ws = super().build_writers()
                if not quiet:
                    return ws
                return [w for w in ws if not isinstance(w, CommonMetricPrinter)]

            @classmethod
            def build_test_loader(cls, cfg, dataset_name):
                """Như DefaultTrainer nhưng nạp nhiều ảnh một lượt.

                `build_detection_test_loader` để mặc định batch_size=1. Suy
                luận không giữ đồ thị cho backward nên cùng một batch tốn ít
                VRAM hơn hẳn lúc train; giữ 1 là bỏ không phần lớn card.
                """
                from detectron2.data import build_detection_test_loader

                return build_detection_test_loader(cfg, dataset_name,
                                                   batch_size=val_batch)

            @classmethod
            def build_train_loader(cls, cfg):
                if m2f is not None:
                    return base.build_train_loader(cfg)
                augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                             cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
                if a["fliplr"]:
                    augs.append(T.RandomFlip(prob=float(a["fliplr"]), horizontal=True, vertical=False))
                if a["flipud"]:
                    augs.append(T.RandomFlip(prob=float(a["flipud"]), horizontal=False, vertical=True))
                if a["rot90"]:
                    augs.append(T.RandomRotation([0, 90, 180, 270], sample_style="choice", expand=True))
                return build_detection_train_loader(
                    cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))

            def build_hooks(self):
                ret = super().build_hooks()
                # Giữ đúng `keep_ckpts` checkpoint định kỳ; xem D2_DEFAULTS.
                for i, h in enumerate(ret):
                    if isinstance(h, hooks.PeriodicCheckpointer):
                        ret[i] = hooks.PeriodicCheckpointer(
                            self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                            max_to_keep=keep_ckpts)
                        break
                # Sau EvalHook (đọc segm/AP nó vừa ghi), trước PeriodicWriter.
                ret.insert(-1, hooks.BestCheckpointer(
                    self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP",
                    mode="max", file_prefix="model_best"))
                # `log_every` được khai trong D2_DEFAULTS từ đầu nhưng KHÔNG
                # nối vào đâu cả: nhịp 20 trong log là mặc định cứng của
                # detectron2, trùng số nên nhìn như tham số đang có tác dụng.
                for h in ret:
                    if isinstance(h, hooks.PeriodicWriter):
                        h._period = max(1, int(a["log_every"]))
                # Cuối danh sách, sau EvalHook: hook nào đọc segm/AP cũng phải
                # chạy SAU cái hook vừa ghi ra nó.
                if reporter is not None:
                    ret.append(reporter)
                return ret

        return CoffeeTrainer

    # ---------------------------------------------------------------------- dò
    def probe(self) -> dict:
        """Hai iteration thật (forward + backward, AMP) rồi đo đỉnh bộ nhớ, như
        trainer torchvision."""
        import torch

        if not torch.cuda.is_available():
            return {"supported": False, "reason": "không có CUDA"}
        cfg, m2f, _ = self._cfg()
        cls = self._trainer_cls(m2f)
        model = cls.build_model(cfg)
        model.train()
        loader = cls.build_train_loader(cfg)
        opt = cls.build_optimizer(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.SOLVER.AMP.ENABLED)
        torch.cuda.reset_peak_memory_stats()
        t0, n_obj = time.time(), 0
        try:
            it = iter(loader)
            for _ in range(2):
                data = next(it)
                n_obj += sum(len(d["instances"]) for d in data)
                with torch.autocast("cuda", enabled=cfg.SOLVER.AMP.ENABLED):
                    losses = sum(model(data).values())
                opt.zero_grad(set_to_none=True)
                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
        finally:
            peak = torch.cuda.max_memory_allocated() / 2**30
            del model, opt, loader
            torch.cuda.empty_cache()
        wall = memory.wall_gb()
        batch = int(self.train_args["batch"])
        return {
            "supported": True, "arch": self.arch, "imgsz": int(self.train_args["imgsz"]),
            "batch": batch, "peak_gb": round(peak, 2), "under_wall": peak < wall,
            "wall_gb": wall,
            "suggested_batch": max(1, int(batch * wall / max(peak, 1e-6))),
            "objects_in_probe": n_obj, "seconds": round(time.time() - t0, 1),
            "note": "đỉnh đo trên 2 iteration; batch dày vùng hơn sẽ cao hơn một chút",
        }

    @staticmethod
    def _default_setup(default_setup, cfg, quiet: bool):
        """Gọi default_setup mà không để nó đổ nguyên config ra màn hình.

        Đo trên một lượt Mask2Former: 452 / 655 dòng (69%) là bảng môi trường
        và bản dump config, và CẢ HAI in bên trong `default_setup`. Bịt log
        sau khi nó trả về là muộn — đó chính là lỗi đã mắc ở đây trước đó.

        `setup_logger` của detectron2 tạo `StreamHandler(stream=sys.stdout)`
        và GIỮ tham chiếu tới luồng ngay lúc tạo, mà nó chạy bên trong
        `default_setup`, tức là lúc stdout đang bị hứng. Không trỏ handler về
        stdout thật thì mọi dòng về sau rơi vào bộ đệm đã bỏ đi — im lặng
        hoàn toàn, kể cả cảnh báo "Skip loading parameter ...", và người chạy
        không có dấu hiệu gì.

        Trỏ lại thì nhận diện handler bằng CHÍNH BỘ ĐỆM, không bằng lớp.
        detectron2 dựng đường ghi ra đĩa cũng bằng `StreamHandler` (xem
        `progress.is_console`), nên lọc theo lớp sẽ trỏ luôn d2/log.txt ra
        màn hình: mỗi dòng in hai lần, còn file thì rỗng. So sánh `is` với bộ
        đệm chỉ trúng đúng handler đã chộp stdout lúc bị hứng.

        Không mất gì: default_setup vẫn gắn handler ghi ra d2/log.txt, và bản
        đó nhận đủ những dòng bị hứng ở đây.
        """
        args = SimpleNamespace(config_file="", eval_only=False, opts=[], num_gpus=1)
        if not quiet:
            default_setup(cfg, args)
            return
        real, buf = sys.stdout, io.StringIO()
        with contextlib.redirect_stdout(buf):
            default_setup(cfg, args)
        for name in ("detectron2", "fvcore"):
            for h in logging.getLogger(name).handlers:
                if getattr(h, "stream", None) is buf:
                    h.setStream(real)
        progress.hush("detectron2", "fvcore")
        # train_loop bắt ngoại lệ, logger.exception rồi raise lại, nên cùng
        # một traceback ra màn hình hai lần: 90 dòng của detectron2 rồi 96
        # dòng của Python. Bản sau chứa đủ bản trước cộng khung của train.py.
        progress.drop_from_console("detectron2", name="detectron2.engine.train_loop",
                                   startswith="Exception during training")

    def _train_or_say_why(self, trainer, cfg):
        """Chạy train; hết VRAM thì nói ra phải làm gì, đừng đổ bức tường chữ.

        Thông báo của allocator kết thúc bằng 90 dòng traceback rồi một đoạn
        văn gợi ý `expandable_segments:True` — thứ không cứu nổi một lượt
        thiếu vài GiB. Chữ "batch" không xuất hiện lần nào, dù đó chính là
        thứ phải sửa.

        Không mất traceback: `train_loop` của detectron2 đã gọi
        `logger.exception` trước khi ném lại, nên bản đầy đủ nằm trong
        d2/log.txt.
        """
        import torch

        oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        try:
            trainer.train()
        except oom as err:
            print(progress.summary("Hết VRAM", progress.oom_rows(
                err, arch=self.arch, batch=int(cfg.SOLVER.IMS_PER_BATCH),
                imgsz=int(cfg.INPUT.MAX_SIZE_TRAIN),
                log_path=self.out_dir / "log.txt")), flush=True)
            raise SystemExit(1) from None

    # ----------------------------------------------------------------- huấn luyện
    def fit(self) -> dict:
        from detectron2.engine import default_setup

        cfg, m2f, _ = self._cfg()
        quiet = not self.train_args.get("verbose")
        self._default_setup(default_setup, cfg, quiet)
        if quiet:
            self.warned = progress.warnings_to_file(self.run_dir / "warnings.log")
        cls = self._trainer_cls(m2f)
        per_epoch = iters_per_epoch(self.n_train, self.train_args["batch"])
        print(f"{self.arch}: {self.train_args['epochs']} epoch x {per_epoch} iteration, "
              f"batch {cfg.SOLVER.IMS_PER_BATCH}, imgsz {cfg.INPUT.MAX_SIZE_TRAIN}, "
              f"lr {cfg.SOLVER.BASE_LR:g}, max_iter {cfg.SOLVER.MAX_ITER}")
        t0 = time.time()
        trainer = cls(cfg)
        if self.resume:
            # `last_checkpoint` là file một dòng trỏ vào checkpoint định kỳ gần
            # nhất; thiếu nó thì resume_or_load lặng lẽ nạp trọng số COCO và
            # chạy lại từ iteration 0 — một lượt tưởng là chạy tiếp mà thật ra
            # chạy lại, và nó sẽ ghi đè phần đã có.
            moc = self.out_dir / "last_checkpoint"
            if not moc.is_file():
                raise SystemExit(
                    f"Không thấy {moc}: không có gì để chạy tiếp.\n"
                    "  Lượt đã chạy xong sẽ không còn checkpoint định kỳ nào "
                    "(model_final.pth đã được chuyển sang weights/last.pth), nên "
                    "--resume chỉ dùng cho lượt bị ngắt giữa chừng.")
            ckpt = self.out_dir / moc.read_text(encoding="utf-8").strip()
            if not ckpt.is_file():
                raise SystemExit(
                    f"{moc} trỏ vào {ckpt.name} nhưng file đó không còn.\n"
                    "  Lượt này đã chạy xong hoặc thư mục d2/ đã bị dọn.")
            print(f"Chạy tiếp từ {ckpt.name}")
        trainer.resume_or_load(resume=self.resume)
        self._train_or_say_why(trainer, cfg)
        train_seconds = round(time.time() - t0, 1)

        # Trọng số về cùng bố cục với các trainer khác; d2_config.yaml cạnh
        # best.pth để Detectron2Model dựng lại đúng model không cần nhắc tham số.
        weights_dir = self.run_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        best_src = self.out_dir / "model_best.pth"
        final_src = self.out_dir / "model_final.pth"
        best = weights_dir / "best.pth"
        shutil.copy2(best_src if best_src.exists() else final_src, best)
        shutil.move(str(final_src), weights_dir / "last.pth")
        (weights_dir / "d2_config.yaml").write_text(cfg.dump(), encoding="utf-8")
        rows = self._results_csv(per_epoch)

        best_rows = [r for r in rows if r.get("segm/AP") not in (None, "")]
        best_row = max(best_rows, key=lambda r: float(r["segm/AP"])) if best_rows else None
        self._print_summary(best_row, train_seconds, best)
        return {
            "weights": {"best": str(best), "last": str(weights_dir / "last.pth")},
            "best_epoch": int(best_row["epoch"]) if best_row else -1,
            "best_val_AP": round(float(best_row["segm/AP"]) / 100, 4) if best_row else -1.0,
            "epochs_run": int(self.train_args["epochs"]),
            "train_seconds": train_seconds,
            "results_csv": str(self.run_dir / "results.csv"),
            # Backbone dùng thật + báo cáo nạp trọng số: ba tháng sau mở một
            # thư mục kết quả phải biết nó train trên backbone nào và
            # checkpoint COCO có vào được không.
            "backbone": self.backbone or "mặc định của arch",
            "weights_loaded": getattr(cls, "weights_loaded", None),
            "args": dict(self.train_args),
        }

    def _print_summary(self, best_row, seconds, weights) -> None:
        """Khối cuối lượt chạy: số nào đáng nhớ thì nằm ở đây, không phải rải
        rác giữa mấy trăm dòng log của khung.

        Chỉ có số của VAL. Test là việc của evaluate.py — chấm ở đây nữa thì
        cùng một checkpoint ra hai con số hơi khác nhau (bộ chấm của khung chỉ
        cho mAP, còn evaluate.py cho cả Boundary AP và chỉ số biên từng vùng)
        mà không ai biết nên tin con nào.

        Dùng tên cột của YOLO (`mAP50-95`, `mAP50`) cho cả bốn model; xem
        cofseg/progress.py để biết vì sao.
        """
        def pair(ap, ap50):
            return f"mAP50-95 {progress.fmt_num(ap)}   mAP50 {progress.fmt_num(ap50)}"

        epochs = int(self.train_args["epochs"])
        rows = [("epoch tốt nhất",
                 f"{best_row['epoch']}/{epochs}" if best_row else "—")]
        if best_row:
            rows.append(("val", pair(best_row.get("segm/AP"), best_row.get("segm/AP50"))))
        rows.append(("thời gian", self._time_row(seconds)))
        seen = getattr(getattr(self, "warned", None), "seen", ())
        if seen:
            rows.append(("cảnh báo", f"{len(seen)} loại  ->  warnings.log"))
        # Đường dẫn tương đối với run dir: tên run dir đã nằm ở tiêu đề, lặp
        # lại cả đường dẫn tuyệt đối chỉ kéo khung rộng ra mà không thêm tin.
        try:
            shown = pathlib.Path(weights).relative_to(self.run_dir)
        except ValueError:
            shown = weights
        rows.append(("trọng số", str(shown)))
        print(progress.summary(f"{self.arch} · {self.run_dir.name}", rows), flush=True)

    def _results_csv(self, per_epoch: int) -> list[dict]:
        """metrics.json (một JSON mỗi dòng) của detectron2 -> results.csv có cột
        epoch, cùng tinh thần với các trainer khác."""
        src = self.out_dir / "metrics.json"
        epochs = int(self.train_args["epochs"])
        rows: list[dict] = []
        if not src.exists():
            return rows
        keep = ("iteration", "total_loss", "lr", "time", "data_time",
                "segm/AP", "segm/AP50", "segm/AP75", "bbox/AP")
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "iteration" not in d:
                continue
            # Chặn trên ở số epoch thật. detectron2 tăng self.iter lên
            # max_iter TRƯỚC khi gọi after_train (để after_train phân biệt
            # được "train xong tử tế" với "chết giữa chừng"), mà EvalHook lại
            # cố tình để dành val của epoch CUỐI cho after_train. Nên AP quan
            # trọng nhất của cả lượt chạy được ghi ở iteration = max_iter, và
            # 90 // 30 + 1 ra epoch 4 của một lượt 3 epoch.
            row = {"epoch": min(d["iteration"] // per_epoch + 1, epochs)}
            row.update({k: d[k] for k in keep if k in d})
            rows.append(row)
        out = self.run_dir / "results.csv"
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("epoch", "iteration"), k))
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return rows
