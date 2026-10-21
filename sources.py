"""Frame sources for live cameras and video files.

Engines never read the wall clock. Each ``Frame.ts`` is defined by the source:
monotonic seconds since start for live cameras, and ``frame_index / fps`` for
files so playback speed does not change event timing.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from security import mask_credentials
from utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


@dataclass
class Frame:
    """One captured image and the timestamp engines must use."""

    id: int
    ts: float
    image: np.ndarray


def positive_fps(value: float | None, fallback: float = 25.0) -> float:
    """Return a usable frame rate. Zero or missing metadata falls back to 25."""
    try:
        fps = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(fps) or fps <= 1e-3:
        return fallback
    return fps


def valid_image(image: object) -> bool:
    """True when ``image`` is a non-empty array with a real height and width."""
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return False
    height, width = image.shape[:2]
    return height > 0 and width > 0


def looks_like_file(source: int | str) -> bool:
    if isinstance(source, int):
        return False
    path = Path(source)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return True
    return path.is_file()


class FrameSource(ABC):
    """Capture interface. ``read`` returns a frame at most once."""

    @abstractmethod
    def start(self) -> None:
        """Begin capture."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release devices."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next unseen frame, or None when nothing new is ready."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Return a snapshot of source type, name, playback, and health."""

    def set_target_fps(self, fps: float) -> None:
        """Update the live processing cap. File pacing uses native fps instead."""

    def set_playback(self, *, speed: float | None = None, loop: bool | None = None) -> None:
        """Update file playback controls. Live sources ignore these."""

    def control(self, action: str, value: float | None = None) -> None:
        """Apply pause, play, seek, or step. Unsupported actions raise ValueError."""
        raise ValueError(f"unsupported control action: {action}")

    def stats(self) -> dict[str, Any]:
        return {}

    def is_alive(self) -> bool:
        return False


class LiveSource(FrameSource):
    """Latest-frame webcam or RTSP capture with reconnect and backoff.

    ``Frame.ts`` is monotonic seconds since ``start``, not the system clock.
    """

    def __init__(
        self,
        source: int | str,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        io_timeout: float = 5.0,
    ) -> None:
        self.camera: int | str = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.io_timeout = io_timeout
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._latest: Frame | None = None
        self._frame_pending = False
        self._next_id = 0
        self._started_at = 0.0
        self._last_shape: tuple[int, int] | None = None
        self._frames_captured = 0
        self._frames_dropped = 0
        self._frames_skipped = 0
        self._reconnects = 0
        self._state = "stopped"
        self._error: str | None = None
        self._resolution_changed = False
        self._logger = get_logger(__name__)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._state = "connecting"
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="cineguard-live",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.io_timeout + 1.0))
        self._release_capture()
        self._thread = None
        self._state = "stopped"

    def read(self) -> Frame | None:
        with self._frame_lock:
            if not self._frame_pending or self._latest is None:
                return None
            self._frame_pending = False
            return self._latest

    def info(self) -> dict[str, Any]:
        with self._frame_lock:
            frame_id = self._latest.id if self._latest is not None else 0
            return {
                "type": "live",
                "name": str(mask_credentials(str(self.camera))),
                "state": self._state,
                "fps": self.target_fps,
                "paused": False,
                "loop": False,
                "speed": 1.0,
                "error": self._error,
                "frame_id": frame_id,
                "resolution_changed": self._resolution_changed,
                "reconnects": self._reconnects,
            }

    def stats(self) -> dict[str, Any]:
        with self._frame_lock:
            last_ts = self._latest.ts if self._latest is not None else 0.0
            return {
                "frames_captured": self._frames_captured,
                "frames_dropped": self._frames_dropped,
                "frames_skipped": self._frames_skipped,
                "reconnects": self._reconnects,
                "last_frame_at": self._started_at + last_ts if self._latest is not None else 0.0,
            }

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set_target_fps(self, fps: float) -> None:
        self.target_fps = max(1, int(fps))

    def control(self, action: str, value: float | None = None) -> None:
        raise ValueError(f"live sources do not support {action}")

    def _open_capture(self) -> bool:
        capture = cv2.VideoCapture(self.camera)
        if not capture.isOpened():
            capture.release()
            self._state = "error"
            self._error = f"unable to open camera source {mask_credentials(str(self.camera))}"
            self._logger.warning(self._error)
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.io_timeout * 1000))
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.io_timeout * 1000))
        with self._capture_lock:
            self._capture = capture
        self._state = "open"
        self._error = None
        return True

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def _publish(self, image: np.ndarray) -> None:
        owned = image.copy()
        shape = owned.shape[:2]
        with self._frame_lock:
            if self._last_shape is not None and self._last_shape != shape:
                self._resolution_changed = True
                self._logger.warning(
                    "live resolution changed from %s to %s", self._last_shape, shape
                )
            self._last_shape = shape
            if self._frame_pending:
                self._frames_dropped += 1
            self._next_id += 1
            ts = time.monotonic() - self._started_at
            self._latest = Frame(id=self._next_id, ts=ts, image=owned)
            self._frame_pending = True
            self._frames_captured += 1

    def _reader_loop(self) -> None:
        backoff = 0.25
        max_backoff = max(2.0, self.io_timeout)
        while not self._stop_event.is_set():
            if not self._open_capture():
                with self._frame_lock:
                    self._reconnects += 1
                self._stop_event.wait(backoff)
                backoff = min(max_backoff, backoff * 2)
                continue

            backoff = 0.25
            self._logger.info("Camera source connected: %s", mask_credentials(str(self.camera)))
            while not self._stop_event.is_set():
                with self._capture_lock:
                    capture = self._capture
                if capture is None:
                    break
                try:
                    ok, frame = capture.read()
                except cv2.error:
                    self._logger.exception("camera read failed for %s", mask_credentials(str(self.camera)))
                    ok, frame = False, None
                if not ok or not valid_image(frame):
                    self._frames_skipped += 1
                    self._logger.warning("empty or corrupt frame from %s; reconnecting", mask_credentials(str(self.camera)))
                    break
                self._publish(frame)

            self._release_capture()
            if not self._stop_event.is_set():
                self._state = "connecting"
                with self._frame_lock:
                    self._reconnects += 1
                self._logger.warning("camera disconnected; retrying in %.2fs", backoff)
                self._stop_event.wait(backoff)
                backoff = min(max_backoff, backoff * 2)


class FileSource(FrameSource):
    """Video-file source paced by native fps times playback speed.

    Timestamps are ``frame_index / fps`` so tests and scenario runs stay
    deterministic. At end of file the source loops or enters ``ended``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = True,
        speed: float = 1.0,
        io_timeout: float = 5.0,
        opener: Callable[..., cv2.VideoCapture] | None = None,
        pace: bool = True,
    ) -> None:
        self.path = str(path)
        self.loop = loop
        self.speed = float(speed)
        self.pace = bool(pace)
        self.io_timeout = io_timeout
        self._opener = opener or cv2.VideoCapture
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._next_id = 0
        self._index = 0
        self._fps = 25.0
        self._paused = False
        self._step_once = False
        self._seek_index: int | None = None
        self._state = "stopped"
        self._error: str | None = None
        self._resolution_changed = False
        self._last_shape: tuple[int, int] | None = None
        self._frames_captured = 0
        self._frames_skipped = 0
        self._frame_count = 0
        self._logger = get_logger(__name__)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._state = "connecting"
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="cineguard-file",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.io_timeout + 1.0))
        self._release()
        self._thread = None
        self._state = "stopped"

    def read(self) -> Frame | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _enqueue(self, frame: Frame) -> None:
        """Block until the consumer takes the frame so file frames are not skipped."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(frame, timeout=0.05)
                return
            except queue.Full:
                continue

    def _clear_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def info(self) -> dict[str, Any]:
        with self._lock:
            frame_id = self._latest.id if self._latest is not None else 0
            return {
                "type": "file",
                "name": Path(self.path).name,
                "state": self._state,
                "fps": self._fps,
                "paused": self._paused,
                "loop": self.loop,
                "speed": self.speed,
                "error": self._error,
                "frame_id": frame_id,
                "resolution_changed": self._resolution_changed,
                "frame_count": self._frame_count,
                "duration_s": (self._frame_count / self._fps) if self._frame_count and self._fps else None,
                "position_s": (self._index / self._fps) if self._fps else 0.0,
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frames_captured": self._frames_captured,
                "frames_dropped": 0,
                "frames_skipped": self._frames_skipped,
                "reconnects": 0,
                "last_frame_at": 0.0,
            }

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set_playback(self, *, speed: float | None = None, loop: bool | None = None) -> None:
        with self._lock:
            if speed is not None:
                if not 0.25 <= float(speed) <= 4.0:
                    raise ValueError("playback_speed must be between 0.25 and 4.0")
                self.speed = float(speed)
            if loop is not None:
                self.loop = bool(loop)
        self._wake.set()

    def control(self, action: str, value: float | None = None) -> None:
        if action == "pause":
            with self._lock:
                self._paused = True
                self._state = "paused"
            self._wake.set()
            return
        if action == "play":
            with self._lock:
                self._paused = False
                if self._state == "paused":
                    self._state = "open"
            self._wake.set()
            return
        if action == "step":
            with self._lock:
                self._paused = True
                self._step_once = True
                self._state = "paused"
            self._wake.set()
            return
        if action == "seek":
            if value is None:
                raise ValueError("seek requires value in seconds")
            with self._lock:
                self._seek_index = max(0, int(round(float(value) * self._fps)))
            self._clear_pending()
            self._wake.set()
            return
        raise ValueError(f"unsupported control action: {action}")

    def _release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def _reader_loop(self) -> None:
        backoff = 0.25
        while not self._stop_event.is_set():
            if not Path(self.path).is_file():
                message = f"file not found: {self.path}"
                with self._lock:
                    self._state = "error"
                    self._error = message
                self._logger.error(message)
                self._stop_event.wait(backoff)
                backoff = min(max(2.0, self.io_timeout), backoff * 2)
                continue

            capture = self._opener(self.path)
            if not capture.isOpened():
                message = f"unable to open video file: {self.path}"
                with self._lock:
                    self._state = "error"
                    self._error = message
                self._logger.error(message)
                capture.release()
                self._stop_event.wait(backoff)
                backoff = min(max(2.0, self.io_timeout), backoff * 2)
                continue

            raw_fps = capture.get(cv2.CAP_PROP_FPS)
            fps = positive_fps(raw_fps)
            if raw_fps is None or float(raw_fps or 0) <= 1e-3:
                self._logger.warning("video fps metadata missing or zero; using %.1f", fps)
            with self._lock:
                self._fps = fps
                self._capture = capture
                self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self._state = "paused" if self._paused else "open"
                self._error = None
                self._index = 0
            self._logger.info("Opened %s at %.3f fps", self.path, fps)
            backoff = 0.25
            self._play(capture)
            self._release()
            with self._lock:
                finished = self._state == "ended" and not self.loop
            if finished or self._stop_event.is_set():
                return
            with self._lock:
                if self._state not in {"stopped", "ended"}:
                    self._state = "connecting"

    def _play(self, capture: cv2.VideoCapture) -> None:
        next_deadline = time.monotonic()
        consecutive_eof = 0
        while not self._stop_event.is_set():
            with self._lock:
                paused = self._paused
                step = self._step_once
                seek_index = self._seek_index
                self._seek_index = None
                speed = self.speed
                fps = self._fps
                if step:
                    self._step_once = False
            if seek_index is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, seek_index)
                with self._lock:
                    self._index = seek_index
                next_deadline = time.monotonic()
            if paused and not step and seek_index is None:
                self._wake.wait(0.05)
                self._wake.clear()
                next_deadline = time.monotonic()
                continue

            try:
                ok, image = capture.read()
            except cv2.error:
                self._logger.exception("failed reading %s", self.path)
                ok, image = False, None
            if not ok or not valid_image(image):
                if ok:
                    self._frames_skipped += 1
                    self._logger.warning("skipping corrupt frame in %s", self.path)
                    continue
                with self._lock:
                    should_loop = self.loop
                if should_loop:
                    consecutive_eof += 1
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self._lock:
                        self._index = 0
                    next_deadline = time.monotonic()
                    if consecutive_eof > 1:
                        self._stop_event.wait(0.2)
                    continue
                with self._lock:
                    self._state = "ended"
                    self._paused = True
                self._logger.info("end of file: %s", self.path)
                while not self._stop_event.is_set():
                    with self._lock:
                        if self.loop or self._seek_index is not None:
                            break
                    self._wake.wait(0.1)
                    self._wake.clear()
                with self._lock:
                    resume = self.loop or self._seek_index is not None
                    if self.loop:
                        self._paused = False
                        self._state = "open"
                if resume:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self._lock:
                        self._index = 0
                    continue
                return

            consecutive_eof = 0
            shape = image.shape[:2]
            with self._lock:
                if self._last_shape is not None and self._last_shape != shape:
                    self._resolution_changed = True
                    self._logger.warning(
                        "file resolution changed from %s to %s", self._last_shape, shape
                    )
                self._last_shape = shape
                index = self._index
                self._index += 1
                self._next_id += 1
                frame = Frame(id=self._next_id, ts=index / fps, image=image.copy())
                self._latest = frame
                self._frames_captured += 1
                paused_now = self._paused
            self._enqueue(frame)

            if paused_now or not self.pace:
                next_deadline = time.monotonic()
                continue
            interval = 1.0 / (fps * max(speed, 0.25))
            next_deadline += interval
            delay = next_deadline - time.monotonic()
            if delay > 0 and self._stop_event.wait(delay):
                return


def open_source(
    camera_source: int | str,
    *,
    loop: bool,
    speed: float,
    width: int,
    height: int,
    target_fps: int,
    io_timeout: float,
) -> FrameSource:
    """Build a live or file source from a config camera value."""
    if looks_like_file(camera_source):
        return FileSource(
            str(camera_source),
            loop=loop,
            speed=speed,
            io_timeout=io_timeout,
        )
    return LiveSource(
        camera_source,
        width=width,
        height=height,
        target_fps=target_fps,
        io_timeout=io_timeout,
    )
