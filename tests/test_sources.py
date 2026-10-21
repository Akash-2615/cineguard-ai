"""File source timestamps, loop, pause, seek, and failure reporting."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from sources import FileSource, LiveSource, positive_fps, valid_image


def _write_video(path: Path, frames: int = 6, fps: float = 25.0, size: tuple[int, int] = (64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((size[1], size[0], 3), (index * 20) % 256, dtype=np.uint8)
        writer.write(image)
    writer.release()


def _collect(source: FileSource, count: int, timeout: float = 5.0) -> list:
    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < count and time.monotonic() < deadline:
        frame = source.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frames.append(frame)
    assert len(frames) == count
    return frames


def test_positive_fps_fallback() -> None:
    assert positive_fps(0) == 25.0
    assert positive_fps(None) == 25.0
    assert positive_fps(30) == 30.0


def test_valid_image_rejects_empty() -> None:
    assert valid_image(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not valid_image(np.zeros((0, 4, 3), dtype=np.uint8))
    assert not valid_image(None)


def test_file_timestamps_are_index_over_fps(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    _write_video(path, frames=4, fps=25.0)
    source = FileSource(path, loop=False, speed=4.0)
    source.start()
    try:
        frames = _collect(source, 4)
        fps = source.info()["fps"]
        assert fps == pytest.approx(25.0)
        for index, frame in enumerate(frames):
            assert frame.ts == pytest.approx(index / fps)
            assert frame.id == index + 1
    finally:
        source.stop()


def test_file_loop_restarts_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "loop.avi"
    _write_video(path, frames=4, fps=25.0)
    source = FileSource(path, loop=True, speed=4.0)
    source.start()
    try:
        frames = _collect(source, 6)
        fps = source.info()["fps"]
        expected = [0, 1, 2, 3, 0, 1]
        assert [frame.id for frame in frames] == [1, 2, 3, 4, 5, 6]
        for frame, index in zip(frames, expected):
            assert frame.ts == pytest.approx(index / fps)
    finally:
        source.stop()


def test_pause_stops_new_frames(tmp_path: Path) -> None:
    path = tmp_path / "pause.avi"
    _write_video(path, frames=30, fps=25.0)
    source = FileSource(path, loop=True, speed=4.0)
    source.start()
    try:
        _collect(source, 1)
        source.control("pause")
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            source.read()
            time.sleep(0.02)
        assert source.info()["paused"] is True
        assert source.read() is None
    finally:
        source.stop()


def test_seek_sets_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "seek.avi"
    _write_video(path, frames=10, fps=25.0)
    source = FileSource(path, loop=False, speed=4.0)
    source.start()
    try:
        _collect(source, 1)
        source.control("seek", 2 / 25.0)
        deadline = time.monotonic() + 2.0
        found = None
        while time.monotonic() < deadline:
            frame = source.read()
            if frame is not None and frame.ts == pytest.approx(2 / 25.0):
                found = frame
                break
            time.sleep(0.01)
        assert found is not None
    finally:
        source.stop()


def test_missing_file_reports_error(tmp_path: Path) -> None:
    source = FileSource(tmp_path / "missing.mp4", loop=False)
    source.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and source.info()["state"] != "error":
            time.sleep(0.05)
        info = source.info()
        assert info["state"] == "error"
        assert info["error"] is not None
        assert "not found" in info["error"].lower()
    finally:
        source.stop()


class _ScriptedCapture:
    def __init__(self, frames: list[np.ndarray], fps: float) -> None:
        self._frames = frames
        self._fps = fps
        self._index = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        if frame is None:
            return True, np.zeros((0, 0, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._open = False


def test_zero_fps_and_resolution_change(tmp_path: Path) -> None:
    path = tmp_path / "odd.mp4"
    path.write_bytes(b"placeholder")
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        None,
        np.zeros((12, 16, 3), dtype=np.uint8),
    ]
    source = FileSource(
        path,
        loop=False,
        speed=4.0,
        opener=lambda _path: _ScriptedCapture(frames, fps=0.0),
    )
    source.start()
    try:
        collected = _collect(source, 2)
        assert source.info()["fps"] == pytest.approx(25.0)
        assert collected[0].ts == pytest.approx(0.0)
        assert collected[1].ts == pytest.approx(1 / 25.0)
        assert source.info()["resolution_changed"] is True
    finally:
        source.stop()


def test_live_open_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dead:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: _Dead())
    source = LiveSource(99, io_timeout=0.5)
    source.start()
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and source.stats()["reconnects"] < 1:
            time.sleep(0.05)
        info = source.info()
        assert info["state"] == "error"
        assert info["error"]
        assert source.stats()["reconnects"] >= 1
    finally:
        source.stop()
