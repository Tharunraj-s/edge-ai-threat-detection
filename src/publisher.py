from __future__ import annotations
from dataclasses import dataclass
import time
import paho.mqtt.publish as publish

@dataclass
class MQTTConfig:
    enabled: bool
    server: str
    topic: str
    payload: str = "csv"      # "csv" or "bytes"
    publish_rate_hz: int = 10

class MQTTPublisher:
    def __init__(self, cfg: MQTTConfig):
        self.cfg = cfg
        self._min_interval = 1.0 / max(cfg.publish_rate_hz, 1)
        self._last_sent = 0.0

    def _rate_limit(self) -> bool:
        now = time.time()
        if now - self._last_sent < self._min_interval:
            return False
        self._last_sent = now
        return True

    def publish_center(self, cx: float, cy: float) -> None:
        if not self.cfg.enabled:
            return
        if not self._rate_limit():
            return

        if self.cfg.payload == "bytes":
            # Keep it safe/general: just quantize into bytes, no “targeting”
            payload = bytearray([max(0, min(255, int(cx))), max(0, min(255, int(cy)))])
        else:
            payload = f"{cx:.1f},{cy:.1f}"

        publish.single(self.cfg.topic, payload, hostname=self.cfg.server)
