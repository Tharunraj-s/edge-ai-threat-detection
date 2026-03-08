#!/usr/bin/env python3
"""Run real-time object/threat detection on a webcam feed."""

from __future__ import annotations

import argparse

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam detection with a YOLO model")
    parser.add_argument("--weights", default="best.pt", help="Path to YOLO weights file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    window_name = "Webcam Threat Detection"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        annotated = result.plot()

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
