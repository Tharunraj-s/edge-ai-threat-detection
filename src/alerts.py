from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .utils import now_ms

@dataclass
class AlertConfig:
    enabled: bool
    mode: str = "log"   # "log" | "gpio" | "none"

class AlertManager:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._gpio_ready = False
        self._pin = None

        if self.cfg.enabled and self.cfg.mode == "gpio":
            try:
                import RPi.GPIO as GPIO  # only on Raspberry Pi
                GPIO.setmode(GPIO.BCM)
                self._pin = 18
                GPIO.setup(self._pin, GPIO.OUT)
                self._gpio = GPIO
                self._gpio_ready = True
            except Exception:
                # If not on Pi, fall back to log
                self.cfg.mode = "log"

    def trigger(self, message: str) -> None:
        if not self.cfg.enabled or self.cfg.mode == "none":
            return

        ts = now_ms()
        if self.cfg.mode == "log":
            print(f"[ALERT {ts}] {message}")
        elif self.cfg.mode == "gpio" and self._gpio_ready:
            # simple blink
            self._gpio.output(self._pin, 1)
            self._gpio.output(self._pin, 0)
