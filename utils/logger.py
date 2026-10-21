"""Logging setup shared by CineGuard modules."""

import logging
import sys
import threading
from collections import deque

_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RingHandler(logging.Handler):
    """Keep the last 200 formatted lines for the system tab."""

    def __init__(self, limit: int = 200) -> None:
        super().__init__()
        self._lines: deque[str] = deque(maxlen=limit)
        self._lock = threading.Lock()
        self.setFormatter(_FORMAT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        with self._lock:
            self._lines.append(line)

    def tail(self) -> list[str]:
        with self._lock:
            return list(self._lines)


_RING = RingHandler()


def recent_logs() -> list[str]:
    return _RING.tail()


def get_logger(name: str = "cineguard") -> logging.Logger:
    """Return a consistently formatted application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMAT)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    if _RING not in logger.handlers:
        logger.addHandler(_RING)
    return logger
