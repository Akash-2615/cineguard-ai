"""MJPEG and WebSocket smoke test with a fake source and no model."""

from __future__ import annotations

import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from app import CommandCenterRuntime, create_app
from config import AppConfig
from sources import Frame


class SequenceSource:
    """Emits a new frame about 20 times a second."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = False
        self._latest: Frame | None = None
        self._next_id = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._next_id += 1
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            image[:, :] = (self._next_id * 3) % 255
            frame = Frame(self._next_id, (self._next_id - 1) / 25.0, image)
            with self._lock:
                self._latest = frame
                self._pending = True
            self._stop.wait(0.05)

    def read(self) -> Frame | None:
        with self._lock:
            if not self._pending or self._latest is None:
                return None
            self._pending = False
            return self._latest

    def info(self) -> dict:
        with self._lock:
            frame_id = self._latest.id if self._latest else 0
        return {
            "type": "file",
            "name": "fake.mp4",
            "state": "open",
            "fps": 25.0,
            "paused": False,
            "loop": True,
            "speed": 1.0,
            "error": None,
            "frame_id": frame_id,
        }

    def set_target_fps(self, fps: float) -> None:
        return None

    def set_playback(self, *, speed: float | None = None, loop: bool | None = None) -> None:
        return None

    def control(self, action: str, value: float | None = None) -> None:
        if action not in {"pause", "play", "seek", "step"}:
            raise ValueError(f"unsupported control action: {action}")

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


def _runtime() -> CommandCenterRuntime:
    config = AppConfig(_env_file=None, pose_model_path="models/does-not-exist.pt", target_fps=20)
    return CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=True)


def test_health_degraded_without_model_and_telemetry_frame_ids() -> None:
    app = create_app(_runtime())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["model_loaded"] is False
        assert body["degraded"] is True
        assert body["errors"]
        with client.websocket_connect("/ws/live_telemetry") as websocket:
            first = websocket.receive_json()
            second = websocket.receive_json()
            deadline = time.monotonic() + 3.0
            while second["frame_id"] <= first["frame_id"] and time.monotonic() < deadline:
                second = websocket.receive_json()
        assert second["frame_id"] > first["frame_id"]
        assert second["system"]["degraded"] is True
        assert second["nir"]["mode"] == "simulated"
        assert "frame_snapshot_base64" not in str(second)
        assert second["system"]["gpu"] is None or isinstance(second["system"]["gpu"], (int, float))


def test_mjpeg_emits_a_jpeg_part() -> None:
    app = create_app(_runtime())
    with TestClient(app) as client:
        response = client.get("/api/v1/stream?frames=1")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content
    assert b"image/jpeg" in response.content
    assert b"\xff\xd8" in response.content


def test_invalid_config_returns_422() -> None:
    app = create_app(_runtime())
    with TestClient(app) as client:
        response = client.post("/api/v1/config", json={"nir_mode": "banana"})
        assert response.status_code == 422
        ok = client.post("/api/v1/config", json={"conf_threshold": 0.2})
        assert ok.status_code == 200
        assert ok.json()["conf_threshold"] == 0.2
        assert "api_token" not in ok.json()
