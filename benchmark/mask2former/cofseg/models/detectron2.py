"""Model detectron2 (Mask R-CNN, Cascade Mask R-CNN, Mask2Former R50) qua cùng
hợp đồng SegmentationModel — và phần dựng config dùng chung với trainer.

Vì sao ba model này đi qua detectron2 chứ không phải torchvision/HF: chúng
phải chạy CÙNG MỘT MÃ và cùng recipe để chênh lệch giữa Mask R-CNN và
Cascade chỉ do ba tầng box head, giữa Mask R-CNN và Mask2Former chỉ do cơ chế
query. Trainer torchvision (maskrcnn.py) vẫn còn để chạy sớm ở nhà; vào bảng
so sánh thì dùng bản này.

detectron2 chỉ được import BÊN TRONG hàm: repo ở nhà (Windows) không cài nó,
nhưng registry, config và test phần thuần Python vẫn phải chạy.

CHƯA CHẠY THẬT: detectron2 không dựng được trên máy phát triển. Phần dựng
config có test; phần gọi detectron2 phải khói trên máy Linux trước khi tin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ..registry import register
from ..weights import local_for_url
from .base import Prediction, SegmentationModel

#: Tên kiến trúc -> (config trong model zoo hoặc trong repo Mask2Former,
#: checkpoint COCO). None = lấy từ model_zoo.get_checkpoint_url(config).
ZOO = {
    "maskrcnn": ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", None),
    "cascade": ("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", None),
    "mask2former": (
        "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
        "maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
    ),
    # PointRend nằm trong projects/ của detectron2, không có trong model zoo API:
    # phải đưa config_file (đường dẫn trong checkout detectron2) và weights.
    "pointrend": (None, None),
}
ARCHS = tuple(ZOO)


def require_detectron2():
    try:
        import detectron2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Cần detectron2 (Linux, build từ source; xem scripts/setup_env.py). "
            "Ở máy không có nó, chấm file predictions.json bằng model coco_predictions."
        ) from e


def load_mask2former(repo: str | Path):
    """Đưa repo Mask2Former vào sys.path và trả về module train_net của họ.

    Repo không phải gói pip; train_net.py của họ định nghĩa Trainer (mapper
    LSJ, evaluator, optimizer có clip gradient) và chỉ chạy gì đó dưới
    __main__, nên nạp làm module là an toàn và đỡ chép mã.
    """
    repo = Path(repo).resolve()
    entry = repo / "train_net.py"
    if not entry.exists():
        raise FileNotFoundError(f"Không thấy {entry}: model.repo phải trỏ vào checkout Mask2Former")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("mask2former_train_net", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def empty_safe_mapper(m2f):
    """Mapper LSJ của Mask2Former, vá chỗ nó gãy trên ẢNH NỀN.

    `COCOInstanceNewBaselineDatasetMapper.__call__` làm thế này:

        instances = utils.annotations_to_instances(annos, image_shape)
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()   # <- gãy
        ...
        if hasattr(instances, 'gt_masks'):                             # <- có canh

    `annotations_to_instances` chỉ gắn `gt_masks` khi `len(annos)` > 0. Ảnh
    không có vùng nào thì dòng thứ hai nổ `Cannot find field 'gt_masks'`.
    Dòng thứ tư cho thấy chính tác giả biết trường đó có thể vắng — chỉ là
    canh sót một chỗ. Upstream không lộ vì recipe COCO của họ để
    `FILTER_EMPTY_ANNOTATIONS: True`, còn ta cố ý để False.

    Bộ block/f1 có ĐÚNG 2 ảnh nền trên 470 ảnh train (val và test không có).
    0.4% đủ để giết lượt chạy ở iteration 43/235.

    Vá bằng cách bọc chứ không sửa repo con: ảnh không vùng nào thì gỡ hẳn
    khoá "annotations" để mapper cha bỏ qua cả khối đó, rồi tự gắn một
    `Instances` rỗng đúng dạng mà model chờ. Giữ nguyên ảnh nền thay vì bật
    FILTER_EMPTY_ANNOTATIONS cho riêng model này — bốn model phải ăn cùng
    một tập dữ liệu thì bảng so sánh mới có nghĩa.
    """
    import copy

    import torch
    from detectron2.structures import Boxes, Instances

    from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
        COCOInstanceNewBaselineDatasetMapper,
    )

    class EmptySafeLSJMapper(COCOInstanceNewBaselineDatasetMapper):
        @staticmethod
        def _co_vung(d) -> bool:
            return any(a.get("iscrowd", 0) == 0 for a in (d.get("annotations") or []))

        def __call__(self, dataset_dict):
            if not self.is_train or self._co_vung(dataset_dict):
                return super().__call__(dataset_dict)
            dataset_dict = copy.deepcopy(dataset_dict)
            dataset_dict.pop("annotations", None)
            out = super().__call__(dataset_dict)
            h, w = out["image"].shape[-2:]
            inst = Instances((h, w))
            inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            inst.gt_classes = torch.zeros(0, dtype=torch.int64)
            # uint8 (0, h, w): đúng dạng convert_coco_poly_to_mask trả về,
            # để prepare_targets của MaskFormer đệm được mà không phải
            # phân biệt trường hợp.
            inst.gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
            out["instances"] = inst
            return out

    return EmptySafeLSJMapper


def base_cfg(arch: str, repo: str | None = None, config_file: str | None = None,
             weights: str | None = None):
    """Config detectron2 CHƯA freeze cho `arch`, đã gộp file gốc và trọng số COCO.

    Trả thêm module train_net của Mask2Former (None với các arch khác) vì
    trainer cần lớp Trainer trong đó.
    """
    require_detectron2()
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    if arch not in ZOO:
        raise ValueError(f"arch {arch!r} không có; có: {ARCHS}")
    zoo_file, zoo_weights = ZOO[arch]
    cfg = get_cfg()
    m2f = None
    if arch == "mask2former":
        if not repo:
            raise ValueError("arch mask2former cần model.repo = đường dẫn checkout Mask2Former")
        from detectron2.projects.deeplab import add_deeplab_config

        m2f = load_mask2former(repo)
        from mask2former import add_maskformer2_config  # có sau load_mask2former

        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(str(Path(repo) / (config_file or zoo_file)))
        cfg.MODEL.WEIGHTS = weights or _checkpoint(zoo_weights)
    elif arch == "pointrend":
        from detectron2.projects.point_rend import add_pointrend_config

        if not config_file or not weights:
            raise ValueError("arch pointrend cần model.config_file và model.weights")
        add_pointrend_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
    else:
        # `ref` chứ không phải `zoo_file`: đổi backbone là đổi config_file, và
        # model_zoo có checkpoint COCO riêng cho TỪNG config. Lấy checkpoint
        # theo zoo_file trong khi kiến trúc dựng theo config_file thì R101 sẽ
        # nạp trọng số của R50 — detectron2 chỉ log shape mismatch rồi chạy
        # tiếp với phần lệch khởi tạo ngẫu nhiên, không có gì dừng lại.
        ref = config_file or zoo_file
        cfg.merge_from_file(model_zoo.get_config_file(ref))
        cfg.MODEL.WEIGHTS = weights or _checkpoint(model_zoo.get_checkpoint_url(ref))
    return cfg, m2f


def _checkpoint(url: str) -> str:
    """Bản đã tải trong weights/ (configs/weights.yaml) nếu có, không thì URL
    để detectron2 tự tải về cache."""
    local = local_for_url(url)
    return str(local) if local else url


def class_opts(arch: str, num_queries: int = 100) -> list:
    """Cặp khoá/giá trị đặt số lớp = 1 cho từng kiến trúc."""
    if arch == "mask2former":
        return ["MODEL.SEM_SEG_HEAD.NUM_CLASSES", 1,
                "MODEL.MASK_FORMER.NUM_OBJECT_QUERIES", int(num_queries)]
    opts = ["MODEL.ROI_HEADS.NUM_CLASSES", 1]
    if arch == "pointrend":
        opts += ["MODEL.POINT_HEAD.NUM_CLASSES", 1]
    return opts


def size_opts(imgsz: int, aspect: float = 9 / 16) -> list:
    """Độ phân giải theo nghĩa `imgsz` của repo = CẠNH DÀI sau khi thu.

    detectron2 đặt theo cạnh ngắn + trần cạnh dài, nên cạnh ngắn = imgsz x
    tỉ lệ ảnh (2560x1440 -> 1024x576), cùng số với YOLO và torchvision.
    """
    short = int(round(imgsz * aspect))
    return ["INPUT.MIN_SIZE_TRAIN", (short,), "INPUT.MAX_SIZE_TRAIN", int(imgsz),
            "INPUT.MIN_SIZE_TEST", short, "INPUT.MAX_SIZE_TEST", int(imgsz)]


@register("model", "detectron2")
class Detectron2Model(SegmentationModel):
    """Suy luận bằng DefaultPredictor; mặt nạ đã ở độ phân giải gốc.

    `weights` là best.pth do trainer ghi; config đầy đủ của lần chạy nằm cạnh
    nó (d2_config.yaml) nên không cần nhắc lại imgsz hay số query. `repo`
    chỉ cần cho Mask2Former.
    """

    needs_prompt = False

    def __init__(self, weights: str, arch: str, repo: str | None = None,
                 config_file: str | None = None, conf: float = 0.05,
                 max_det: int | None = 100, device: str | None = None):
        require_detectron2()
        import torch
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        self.weights, self.arch, self.repo = str(weights), arch, repo
        self.conf, self.max_det = float(conf), max_det
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        saved = Path(config_file) if config_file else Path(weights).with_name("d2_config.yaml")
        if saved.exists():
            # Bản dump của lần huấn luyện: có mọi khoá, kể cả khoá riêng của
            # Mask2Former, nên phải add_*_config trước rồi mới merge.
            cfg, _ = base_cfg(arch, repo=repo) if arch == "mask2former" else (get_cfg(), None)
            if arch == "pointrend":
                from detectron2.projects.point_rend import add_pointrend_config

                add_pointrend_config(cfg)
            cfg.merge_from_file(str(saved))
        else:
            cfg, _ = base_cfg(arch, repo=repo, config_file=config_file)
            cfg.merge_from_list(class_opts(arch))
        cfg.MODEL.WEIGHTS = self.weights
        cfg.MODEL.DEVICE = self.device
        if arch != "mask2former":
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        if max_det:
            cfg.TEST.DETECTIONS_PER_IMAGE = int(max_det)
        cfg.freeze()
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def warmup(self) -> None:
        h, w = self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST
        self.predict(np.zeros((int(h), int(w), 3), np.uint8))

    def predict(self, image: np.ndarray, boxes: np.ndarray | None = None) -> list[Prediction]:
        inst = self.predictor(image)["instances"].to("cpu")
        scores = inst.scores.numpy() if inst.has("scores") else np.ones(len(inst))
        masks = inst.pred_masks.numpy().astype(bool)
        order = np.argsort(-scores)
        preds: list[Prediction] = []
        for i in order:
            if scores[i] < self.conf:
                continue
            if self.max_det and len(preds) >= int(self.max_det):
                break
            if not masks[i].any():
                continue
            preds.append(Prediction(mask=masks[i], origin=(0, 0), score=float(scores[i])).cropped())
        return preds

    @property
    def describe(self) -> dict:
        return {**super().describe, "arch": self.arch, "weights": self.weights,
                "imgsz": int(self.cfg.INPUT.MAX_SIZE_TEST), "conf": self.conf,
                "max_det": self.max_det, "device": self.device}
