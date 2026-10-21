"""Compatibility wrapper around the live frame source."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from config import AppConfig
from sources import LiveSource
from utils.logger import get_logger


class ThreadedRTSPStream:
    """Latest-frame capture used by the diagnostic viewer in ``main.py``."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._source = LiveSource(
            config.camera_source,
            width=config.frame_width,
            height=config.frame_height,
            target_fps=config.target_fps,
            io_timeout=config.io_timeout,
        )
        self._logger = get_logger(__name__)

    @property
    def source(self) -> int | str:
        return self._source.camera

    @source.setter
    def source(self, value: int | str) -> None:
        self._source.camera = value

    def start(self) -> "ThreadedRTSPStream":
        self._source.start()
        return self

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = self._source.read()
        if frame is None:
            return False, None
        return True, frame.image

    def is_alive(self) -> bool:
        return self._source.is_alive()

    def stats(self) -> dict:
        return self._source.stats()

    def stop(self) -> None:
        self._source.stop()
