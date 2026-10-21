"""The dashboard shell and its modules are served."""

from pathlib import Path

from fastapi.testclient import TestClient
from test_api_smoke import SequenceSource

from app import ROOT, CommandCenterRuntime, create_app
from config import AppConfig

ASSETS = [
    "/static/css/app.css",
    "/static/js/main.js",
    "/static/js/api.js",
    "/static/js/ws.js",
    "/static/js/overlay.js",
    "/static/js/charts.js",
    "/static/js/tabs/live.js",
    "/static/js/tabs/nir.js",
    "/static/js/tabs/seats.js",
    "/static/js/tabs/incidents.js",
    "/static/js/tabs/scenarios.js",
    "/static/js/tabs/system.js",
]


def test_dashboard_references_existing_assets(tmp_path: Path) -> None:
    config = AppConfig(
        _env_file=None,
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "cineguard.db"),
        evidence_dir=str(tmp_path / "evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
    )
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    app = create_app(runtime)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "/static/js/main.js" in page.text
        assert "NIR" in page.text
        for asset in ASSETS:
            response = client.get(asset)
            assert response.status_code == 200, asset
            assert response.text.strip()
        logs = client.get("/api/v1/logs")
        assert logs.status_code == 200
        assert "lines" in logs.json()
        clips = client.get("/api/v1/demo_clips")
        assert clips.status_code == 200
    missing = [asset for asset in ASSETS if not (ROOT / asset.removeprefix("/")).is_file()]
    assert missing == []
