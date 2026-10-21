"""Clean-frame evidence clips. The live stream is never annotated.

Keyframe JPEGs carry a timestamp, a seat caption, and a SIMULATED NIR watermark
when the incident was raised in simulated mode. Clip frames stay clean.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

_CODEC_ORDER = (("avc1", "mp4", "mp4"), ("VP80", "webm", "webm"), ("mp4v", "mp4", "mp4"))
_PLAYABLE_CODEC: tuple[str, str, str] | None = None
_CODEC_PROBED = False


def _playable_codec() -> tuple[str, str, str] | None:
    """First codec in the fallback order that OpenCV can write and read back."""
    global _PLAYABLE_CODEC, _CODEC_PROBED
    if _CODEC_PROBED:
        return _PLAYABLE_CODEC
    _CODEC_PROBED = True
    scratch = Path(tempfile.mkdtemp(prefix="cineguard-codec-"))
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        for fourcc, ext, container in _CODEC_ORDER:
            path = scratch / f"probe.{ext}"
            handle = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (32, 32))
            if not handle.isOpened():
                handle.release()
                continue
            handle.write(blank)
            handle.release()
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()
            capture.release()
            if ok and frame is not None and frame.size:
                _PLAYABLE_CODEC = (fourcc, ext, container)
                logger.info("evidence clip codec %s", fourcc)
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _PLAYABLE_CODEC


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str) -> Path | None:
    """Resolve ``relative`` and reject anything that escapes ``root``."""
    base = root.resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        return None
    return candidate


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray | None:
    if image is None or image.ndim < 2 or image.size == 0:
        return None
    frame = image
    height, width = frame.shape[:2]
    if width > max_width and width > 0:
        scale = max_width / float(width)
        frame = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    if even_h < 2 or even_w < 2:
        return None
    if even_h != height or even_w != width:
        frame = frame[:even_h, :even_w]
    return frame.copy()


def _annotate(image: np.ndarray, ts: float, seat: str, nir_mode: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (40, 40, 40), -1)
    label = f"t={ts:.2f}s  {seat}"
    if nir_mode == "simulated":
        label += "  SIMULATED NIR"
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


@dataclass
class _Writer:
    incident_id: int
    path: Path
    container: str
    handle: Any
    started_ts: float
    stop_after: float | None = None
    frames: int = 0
    max_span: float = 30.0
    done: bool = False
    width: int = 0
    height: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class EvidenceStore:
    """Ring buffer plus one clip writer per open incident."""

    def __init__(
        self,
        root: str | Path,
        *,
        preroll_s: float = 8.0,
        postroll_s: float = 5.0,
        clip_max_s: float = 30.0,
        max_width: int = 960,
        max_bytes: int = 32_000_000,
        fps: float = 10.0,
        on_update: Callable[[int, dict[str, Any]], None] | None = None,
        save_bytes: Callable[[Path, bytes], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.preroll_s = preroll_s
        self.postroll_s = postroll_s
        self.clip_max_s = clip_max_s
        self.max_width = max_width
        self.max_bytes = max_bytes
        self.fps = fps
        self.on_update = on_update
        self._save_bytes = save_bytes or (lambda path, data: Path(path).write_bytes(data))
        self._lock = threading.Lock()
        self._ring: deque[tuple[float, np.ndarray]] = deque()
        self._bytes = 0
        self._writers: dict[int, _Writer] = {}
        self.degraded = False
        self.error: str | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.degraded = True
            self.error = f"evidence directory unavailable: {exc}"
            logger.exception("failed to create evidence dir %s", self.root)

    def configure(self, preroll_s: float, postroll_s: float, clip_max_s: float) -> None:
        self.preroll_s = preroll_s
        self.postroll_s = postroll_s
        self.clip_max_s = clip_max_s

    def push_frame(self, ts: float, image: np.ndarray) -> None:
        frame = _downscale(image, self.max_width)
        if frame is None:
            return
        with self._lock:
            self._ring.append((ts, frame))
            self._bytes += int(frame.nbytes)
            self._trim(ts)
            writers = list(self._writers.values())
        for writer in writers:
            self._write_frame(writer, ts, frame)

    def begin(self, incident_id: int, ts: float, seat: str, nir_mode: str) -> dict[str, Any]:
        """Save the keyframe and open a clip that already contains the pre-roll."""
        with self._lock:
            preroll = [(stamp, frame.copy()) for stamp, frame in self._ring if ts - stamp <= self.preroll_s + 1e-6]
        meta: dict[str, Any] = {"evidence_status": "failed"}
        if not preroll:
            self._emit(incident_id, meta)
            return meta
        directory = self.root / str(int(incident_id))
        try:
            directory.mkdir(parents=True, exist_ok=True)
            keyframe = _annotate(preroll[-1][1], ts, seat, nir_mode)
            key_path = directory / "keyframe.jpg"
            ok, encoded = cv2.imencode(".jpg", keyframe, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError("jpeg encode failed")
            self._save_bytes(key_path, encoded.tobytes())
            meta["keyframe_path"] = str(key_path)
            meta["keyframe_sha256"] = sha256_file(key_path)
        except OSError as exc:
            self.degraded = True
            self.error = f"keyframe write failed: {exc}"
            logger.exception("keyframe write failed for incident %s", incident_id)
            meta["evidence_status"] = "failed"
            self._emit(incident_id, meta)
            return meta

        height, width = preroll[-1][1].shape[:2]
        opened = self._open_writer(directory, width, height)
        if opened is None:
            meta["evidence_status"] = "failed"
            self.degraded = True
            self.error = "no clip codec could be opened"
            logger.error("clip writer failed for incident %s", incident_id)
            self._emit(incident_id, meta)
            return meta
        handle, path, container = opened
        writer = _Writer(
            incident_id=incident_id,
            path=path,
            container=container,
            handle=handle,
            started_ts=preroll[0][0],
            max_span=self.clip_max_s,
            meta=meta,
        )
        writer.width = width
        writer.height = height
        for stamp, frame in preroll:
            self._write_frame(writer, stamp, frame)
        meta["evidence_status"] = "recording"
        meta["clip_container"] = container
        meta["clip_path"] = str(path)
        with self._lock:
            self._writers[incident_id] = writer
        self._emit(incident_id, meta)
        return meta

    def finish(self, incident_id: int, ts: float) -> None:
        with self._lock:
            writer = self._writers.get(incident_id)
        if writer is None:
            return
        writer.stop_after = ts + self.postroll_s
        if writer.frames and ts >= writer.stop_after:
            self._close_writer(writer)

    def finalize_all(self) -> None:
        with self._lock:
            writers = list(self._writers.values())
        for writer in writers:
            self._close_writer(writer)

    def cleanup(self, retention_days: int) -> int:
        """Delete evidence directories older than the retention window. Uses file mtimes."""
        if retention_days < 1 or not self.root.exists():
            return 0
        import time
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime >= cutoff:
                    continue
                for item in child.iterdir():
                    item.unlink(missing_ok=True)
                child.rmdir()
                removed += 1
            except OSError:
                logger.exception("evidence retention failed for %s", child)
        return removed

    def file_for(self, incident_id: int, kind: str) -> tuple[Path | None, str | None]:
        folder = resolve_under(self.root, str(int(incident_id)))
        if folder is None or not folder.is_dir():
            return None, None
        if kind == "keyframe":
            path = resolve_under(self.root, f"{int(incident_id)}/keyframe.jpg")
            return path if path and path.is_file() else None, "image/jpeg"
        if kind != "clip":
            return None, None
        for name, media in (("clip.mp4", "video/mp4"), ("clip.webm", "video/webm")):
            path = resolve_under(self.root, f"{int(incident_id)}/{name}")
            if path is not None and path.is_file():
                return path, media
        return None, None

    def _trim(self, now: float) -> None:
        while self._ring:
            stale = now - self._ring[0][0] > self.preroll_s
            overweight = self._bytes > self.max_bytes and len(self._ring) > 1
            if not stale and not overweight:
                break
            _stamp, frame = self._ring.popleft()
            self._bytes -= int(frame.nbytes)

    def _open_writer(self, directory: Path, width: int, height: int) -> tuple[Any, Path, str] | None:
        chosen = _playable_codec()
        if chosen is None:
            return None
        fourcc, ext, container = chosen
        path = directory / f"clip.{ext}"
        if path.exists():
            path.unlink()
        handle = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, (width, height))
        if handle.isOpened():
            return handle, path, container
        handle.release()
        return None

    def _write_frame(self, writer: _Writer, ts: float, frame: np.ndarray) -> None:
        if writer.done:
            return
        try:
            if writer.width and writer.height and (frame.shape[1] != writer.width or frame.shape[0] != writer.height):
                frame = cv2.resize(frame, (writer.width, writer.height))
            writer.handle.write(frame)
            writer.frames += 1
        except Exception:
            logger.exception("clip frame write failed for incident %s", writer.incident_id)
            writer.meta["evidence_status"] = "failed"
            self.degraded = True
            self._close_writer(writer)
            return
        span = ts - writer.started_ts
        if span >= writer.max_span or (writer.stop_after is not None and ts >= writer.stop_after):
            self._close_writer(writer)

    def _close_writer(self, writer: _Writer) -> None:
        if writer.done:
            return
        writer.done = True
        try:
            writer.handle.release()
        except Exception:
            logger.exception("clip release failed for incident %s", writer.incident_id)
        with self._lock:
            self._writers.pop(writer.incident_id, None)
        meta = dict(writer.meta)
        try:
            if writer.path.is_file() and writer.frames > 0:
                meta["clip_sha256"] = sha256_file(writer.path)
                meta["clip_path"] = str(writer.path)
                meta["clip_container"] = writer.container
                if meta.get("evidence_status") != "failed":
                    meta["evidence_status"] = "ready"
            else:
                meta["evidence_status"] = "failed"
        except OSError as exc:
            meta["evidence_status"] = "failed"
            self.degraded = True
            self.error = f"clip hash failed: {exc}"
            logger.exception("clip hash failed for incident %s", writer.incident_id)
        self._emit(writer.incident_id, meta)

    def _emit(self, incident_id: int, meta: dict[str, Any]) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(incident_id, meta)
        except Exception:
            logger.exception("evidence metadata callback failed for incident %s", incident_id)
