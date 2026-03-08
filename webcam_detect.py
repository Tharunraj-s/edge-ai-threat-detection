#!/usr/bin/env python3
"""Webcam threat detection launcher for YOLOv5.

This helper script runs `detect.py` on a webcam stream and defaults to using a
weights file from the repository root.
"""

import argparse
from pathlib import Path

from detect import ROOT, run


def _find_root_weight() -> Path:
    """Return the first likely YOLO weights file found in the repository root."""
    candidates = sorted(ROOT.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No .pt weights file found in repository root. Place your model file "
            "(for example, best.pt) in the project root or pass --weights explicitly."
        )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam detection with your root-level weights file.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to weights file (.pt). Defaults to the first .pt file in repository root.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="", help="CUDA device (e.g. 0) or cpu.")
    parser.add_argument("--save", action="store_true", help="Save output video to runs/detect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights else _find_root_weight()

    run(
        weights=weights,
        source=str(args.camera),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=(args.imgsz, args.imgsz),
        device=args.device,
        view_img=True,
        nosave=not args.save,
        name="webcam_threat_detection",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
