from __future__ import annotations
from pathlib import Path
import time

def now_ms() -> int:
    return int(time.time() * 1000)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
