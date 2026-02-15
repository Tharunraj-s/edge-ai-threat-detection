from __future__ import annotations
import yaml
import cv2
from pathlib import Path

from .detector import YoloV5Detector, DetectorConfig
from .publisher import MQTTPublisher, MQTTConfig
from .alerts import AlertManager, AlertConfig
from .utils import ensure_dir

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def draw_detections(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cx, cy = map(int, d["center"])
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
    return frame

def main():
    cfg = load_config()

    det_cfg = DetectorConfig(
        weights=cfg["model"]["weights"],
        img_size=cfg["model"]["img_size"],
        conf_thres=cfg["model"]["conf_thres"],
        iou_thres=cfg["model"]["iou_thres"],
        max_det=cfg["model"]["max_det"],
        device=cfg["model"]["device"],
        half=cfg["model"]["half"],
    )
    detector = YoloV5Detector(det_cfg)

    pub_cfg = MQTTConfig(**cfg["mqtt"])
    publisher = MQTTPublisher(pub_cfg)

    alert_cfg = AlertConfig(**cfg["alerts"])
    alerts = AlertManager(alert_cfg)

    src_type = cfg["source"]["type"]
    src_path = cfg["source"]["path"]

    cap = cv2.VideoCapture(0 if src_type == "webcam" else str(src_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    out_dir = ensure_dir(cfg["logging"]["out_dir"])
    save_video = bool(cfg["logging"]["save_video"])
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.detect(frame)

        if dets:
            # publish center of first detection (or choose max confidence)
            best = max(dets, key=lambda x: x["confidence"])
            cx, cy = best["center"]
            publisher.publish_center(cx, cy)
            alerts.trigger(f"Detection: {best['class_name']} conf={best['confidence']:.2f}")

        frame = draw_detections(frame, dets)
        cv2.imshow("Edge AI Detection", frame)

        if save_video and writer is None:
            h, w = frame.shape[:2]
            out_path = str(out_dir / "demo_output.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
