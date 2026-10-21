"""Scenario runner pass/fail, determinism, and the printable report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from test_api_smoke import SequenceSource

from app import ROOT, CommandCenterRuntime, create_app
from config import AppConfig
from scenarios import render_report, run_manifest
from tools.run_scenarios import main


def _clip(path: Path, frames: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), (index * 20) % 256, dtype=np.uint8))
    writer.release()


def _manifest(folder: Path) -> Path:
    _clip(folder / "quiet.avi")
    _clip(folder / "should_alert.avi")
    path = folder / "manifest.json"
    path.write_text(json.dumps({"clips": [
        {"id": "quiet", "file": "quiet.avi", "label": "Quiet", "expected": "no_alert", "description": "empty room", "synthetic_glint": False, "notes": ""},
        {"id": "raised", "file": "should_alert.avi", "label": "Raised", "expected": "alert", "description": "phone up", "synthetic_glint": True, "notes": "no person in this synthetic file"},
    ]}), encoding="utf-8")
    return path


def test_runner_marks_pass_and_fail_and_repeats(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = AppConfig(
        _env_file=None,
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "ignored.db"),
        evidence_dir=str(tmp_path / "ignored-evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
    )
    first = run_manifest(manifest, config, None)
    second = run_manifest(manifest, config, None)
    assert first == second
    by_id = {clip["id"]: clip for clip in first["clips"]}
    assert by_id["quiet"]["passed"] is True
    assert by_id["quiet"]["outcome"] == "no_alert"
    assert by_id["raised"]["passed"] is False
    assert by_id["raised"]["outcome"] == "no_alert"
    assert first["metrics"]["accuracy"] == 0.5
    report = render_report(first)
    assert "SIMULATED NIR" in report
    assert "Limitations" in report
    assert "pass" in report and "fail" in report


def test_cli_writes_report(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "reports"
    assert main(["--manifest", str(manifest), "--out", str(out)]) == 0
    assert (out / "results.json").is_file()
    text = (out / "report.html").read_text(encoding="utf-8")
    assert "Limitations" in text
    assert "window.print" in text


def test_scenario_api_job(tmp_path: Path) -> None:
    folder = ROOT / "demo_clips"
    video = folder / "_quiet_test.avi"
    manifest = folder / "_test_manifest.json"
    _clip(video)
    manifest.write_text(json.dumps({"clips": [
        {"id": "quiet", "file": "_quiet_test.avi", "label": "Quiet", "expected": "no_alert", "synthetic_glint": False},
    ]}), encoding="utf-8")
    config = AppConfig(
        _env_file=None,
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "cineguard.db"),
        evidence_dir=str(tmp_path / "evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
    )
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    app = create_app(runtime)
    try:
        with TestClient(app) as client:
            started = client.post("/api/v1/scenarios/run", json={"manifest": "demo_clips/_test_manifest.json"})
            assert started.status_code == 200
            body = started.json()
            for _ in range(50):
                body = client.get("/api/v1/scenarios/results").json()
                if body["status"] in {"done", "error"}:
                    break
                time.sleep(0.05)
            assert body["status"] == "done"
            assert body["results"]["clips"][0]["passed"] is True
            page = client.get("/report")
            assert page.status_code == 200
            assert "SIMULATED NIR" in page.text
            assert "Limitations" in page.text
    finally:
        video.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
