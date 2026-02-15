from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import cv2

# expects YOLOv5 code available in PYTHONPATH (e.g., third_party/yolov5 added)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

@dataclass
class DetectorConfig:
    weights: str
    img_size: int = 640
    conf_thres: float = 0.65
    iou_thres: float = 0.45
    max_det: int = 10
    device: str = ""
    half: bool = False
    data_yaml: Optional[str] = None  # optional dataset yaml

class YoloV5Detector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.model = DetectMultiBackend(cfg.weights, device=self.device, data=cfg.data_yaml, fp16=cfg.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((cfg.img_size, cfg.img_size), s=self.stride)

        # warmup
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame_bgr, self.imgsz)
        img = img[:, :, ::-1]  # BGR->RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.permute(2, 0, 1).float() / 255.0  # HWC->CHW
        img = img.unsqueeze(0)
        if self.model.fp16:
            img = img.half()
        return img

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        im = self._preprocess(frame_bgr)
        pred = self.model(im)
        pred = non_max_suppression(
            pred,
            self.cfg.conf_thres,
            self.cfg.iou_thres,
            max_det=self.cfg.max_det,
        )

        dets: List[Dict[str, Any]] = []
        if len(pred) == 0 or pred[0] is None or len(pred[0]) == 0:
            return dets

        det = pred[0]
        # scale boxes back to original frame size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame_bgr.shape).round()

        for *xyxy, conf, cls in det.tolist():
            x1, y1, x2, y2 = map(float, xyxy)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cls_i = int(cls)
            dets.append({
                "class_id": cls_i,
                "class_name": self.names.get(cls_i, str(cls_i)) if isinstance(self.names, dict) else self.names[cls_i],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
            })

        return dets
