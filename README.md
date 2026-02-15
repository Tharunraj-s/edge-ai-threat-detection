# Edge AI Threat Detection & Alerting (YOLOv5 + Raspberry Pi + Arduino)

A real-time edge computer vision prototype that performs object detection, extracts bounding-box center coordinates, and publishes them via MQTT for monitoring/alerting. Built and demonstrated on Project Day with Raspberry Pi + Arduino integration.

## Demo
- Demo media: `docs/demo/`
- System block diagram: `docs/block diag.png`

## System Overview
1. Capture frames (webcam/video stream)
2. Run YOLOv5 inference
3. Compute center point of detections
4. Publish coordinates via MQTT
5. Trigger safe alerts (log / optional GPIO / Arduino receiver)

## Tech Stack
- Python, PyTorch, OpenCV
- YOLOv5
- MQTT (paho-mqtt)
- Raspberry Pi
- Arduino

## Project Structure
- `src/` — detection pipeline, MQTT publisher, alert manager
- `models/` — model config / (place trained weights here)
- `arduino/` — Arduino receiver sketch
- `docs/` — architecture + demo media

## Configuration
Edit `config.yaml` to update:
- model weights path and thresholds
- video/webcam source
- MQTT broker address + topic
- alert mode

## Notes
This repository contains a demonstration-oriented prototype. Hardware/network setup may require small configuration changes depending on environment (camera source, MQTT broker IP, Arduino connection).
