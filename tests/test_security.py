"""Bearer auth, headers, body cap, rate limit, and masked credentials."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_api_smoke import SequenceSource

from app import CommandCenterRuntime, create_app
from config import AppConfig


def _app(tmp_path: Path, **overrides) -> TestClient:
    values = dict(
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "cineguard.db"),
        evidence_dir=str(tmp_path / "evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
        cors_origins="http://127.0.0.1:8000",
    )
    values.update(overrides)
    config = AppConfig(_env_file=None, **values)
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    return TestClient(create_app(runtime))


def test_open_mode_allows_post_and_hides_the_token(tmp_path: Path) -> None:
    with _app(tmp_path) as client:
        health = client.get("/health")
        assert health.headers["x-content-type-options"] == "nosniff"
        assert health.headers["x-frame-options"] == "DENY"
        assert client.get("/api/v1/config").json()["auth_enabled"] is False
        updated = client.post("/api/v1/config", json={"conf_threshold": 0.4})
        assert updated.status_code == 200


def test_token_rejects_unauthorized_post_evidence_and_socket(tmp_path: Path) -> None:
    with _app(tmp_path, api_token="secret") as client:
        denied = client.post("/api/v1/config", json={"conf_threshold": 0.4})
        assert denied.status_code == 401
        assert "secret" not in denied.text
        allowed = client.post(
            "/api/v1/config",
            json={"conf_threshold": 0.4},
            headers={"Authorization": "Bearer secret"},
        )
        assert allowed.status_code == 200
        assert "api_token" not in allowed.json()
        evidence = client.get("/api/v1/evidence/1/keyframe.jpg")
        assert evidence.status_code == 401
        missing = client.get("/api/v1/evidence/1/keyframe.jpg?token=secret")
        assert missing.status_code == 404
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/live_telemetry") as socket:
                socket.receive_json()
        with client.websocket_connect("/ws/live_telemetry?token=secret") as socket:
            payload = socket.receive_json()
            assert "tracks" in payload


def test_cors_body_limit_and_rate_limit(tmp_path: Path) -> None:
    with _app(tmp_path, api_token="secret", max_body_bytes=64, rate_limit_per_minute=2) as client:
        allowed = client.get("/health", headers={"Origin": "http://127.0.0.1:8000"})
        assert allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"
        blocked = client.get("/health", headers={"Origin": "http://evil.example"})
        assert blocked.headers.get("access-control-allow-origin") is None
        huge = client.post(
            "/api/v1/config",
            content=b'{"conf_threshold": 0.2, "note": "' + b"x" * 200 + b'"}',
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        assert huge.status_code == 413
        first = client.post(
            "/api/v1/config",
            json={"conf_threshold": 0.3},
            headers={"Authorization": "Bearer secret"},
        )
        second = client.post(
            "/api/v1/config",
            json={"conf_threshold": 0.31},
            headers={"Authorization": "Bearer secret"},
        )
        third = client.post(
            "/api/v1/config",
            json={"conf_threshold": 0.32},
            headers={"Authorization": "Bearer secret"},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429


def test_rtsp_password_is_masked() -> None:
    config = AppConfig(_env_file=None, camera_source="rtsp://operator:s3cret@10.0.0.8/live", api_token="secret")
    public = config.public_dict()
    assert "s3cret" not in str(public)
    assert public["camera_source"] == "rtsp://operator:***@10.0.0.8/live"
    assert public["auth_enabled"] is True
