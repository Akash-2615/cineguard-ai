"""Incident lifecycle, evidence files, and the review API."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api_smoke import SequenceSource

from app import CommandCenterRuntime, create_app
from config import AppConfig
from engine.pose_detector import PoseCandidate
from engine.tracker import Track
from evidence import EvidenceStore, resolve_under, sha256_file
from incidents import IncidentStore
from sources import Frame


def _runtime(tmp_path: Path, **overrides) -> CommandCenterRuntime:
    values = dict(
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "cineguard.db"),
        evidence_dir=str(tmp_path / "evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
        piracy_threshold_sec=0.4,
        preroll_s=1.0,
        postroll_s=0.3,
        clip_max_s=30.0,
        max_frame_dt=5.0,
        alert_grace_sec=2.0,
    )
    values.update(overrides)
    config = AppConfig(_env_file=None, **values)
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    runtime.prepare()
    return runtime


def _step(runtime: CommandCenterRuntime, ts: float, present: bool = True, elevated: bool = True, glint: bool = True, track_id: int = 1) -> None:
    image = np.full((96, 128, 3), 30, dtype=np.uint8)
    bbox = (20, 10, 50, 90)
    candidate = PoseCandidate(
        person_id=0, hand_bbox=bbox, elevation_confidence=0.9, is_elevated=elevated,
        keypoints_dict={}, person_bbox=bbox, raw_elevated=elevated,
    )
    track = Track(track_id=track_id, bbox=bbox, confidence=0.9, source_detection_id=0)
    glints = {}
    if glint:
        from types import SimpleNamespace
        glints[track_id] = SimpleNamespace(
            confirmed=True, best_score=0.91, ema_score=0.9,
            accepted=[SimpleNamespace(synthetic=True, bbox=(30, 20, 36, 26), score=0.91, peak=255, area=12)],
        )
    runtime._update_states(
        Frame(int(ts * 1000) + track_id, ts, image),
        runtime.config,
        [candidate] if present else [],
        [track] if present else [],
        [],
        128,
        96,
        glints,
    )


def test_recording_creates_one_incident_and_restart_keeps_it(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for ts in (0.0, 0.25, 0.5, 0.7):
        _step(runtime, ts)
    _step(runtime, 4.0, elevated=False, glint=False)
    _step(runtime, 4.5, elevated=False, glint=False)
    page = runtime.incidents.list_incidents()
    assert page["total"] == 1
    item = page["items"][0]
    assert item["synthetic"] is True
    assert item["nir_mode"] == "simulated"
    assert item["evidence_type"] == "glint"
    assert item["ended_ts"] is not None
    assert item["evidence_status"] == "ready"
    assert item["max_score"] >= item["mean_score"]
    keyframe = Path(item["keyframe_path"])
    clip = Path(item["clip_path"])
    assert sha256_file(keyframe) == item["keyframe_sha256"]
    assert sha256_file(clip) == item["clip_sha256"]
    image = cv2.imread(str(keyframe))
    assert image is not None and tuple(int(value) for value in image[2, 2]) == (40, 40, 40)
    capture = cv2.VideoCapture(str(clip))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    reopened = IncidentStore(runtime.incidents.path)
    assert reopened.list_incidents()["total"] == 1


def test_glance_does_not_create_an_incident(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, piracy_threshold_sec=30.0)
    _step(runtime, 0.0)
    _step(runtime, 0.2)
    _step(runtime, 4.0, elevated=False, glint=False)
    assert runtime.incidents.list_incidents()["total"] == 0


def test_same_seat_merges_and_false_positive_suppresses(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for ts in (0.0, 0.5):
        _step(runtime, ts)
    _step(runtime, 4.0, elevated=False, glint=False)
    first = runtime.incidents.list_incidents()["items"][0]["id"]
    for ts in (6.0, 6.5):
        _step(runtime, ts)
    assert runtime.incidents.list_incidents()["total"] == 1
    assert runtime.incidents.list_incidents()["items"][0]["id"] == first
    runtime._last_frame_ts = 6.5
    reviewed = runtime.incidents.review(first, "false_positive", "phone down", 6.5, 60.0)
    assert reviewed is not None and reviewed["status"] == "false_positive"
    for ts in (10.0, 10.5):
        _step(runtime, ts)
    assert runtime.incidents.list_incidents()["total"] == 1
    for ts in (80.0, 80.5):
        _step(runtime, ts)
    assert runtime.incidents.list_incidents()["total"] == 2


def test_pagination_review_and_db_failure(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "page.db")
    for seat in ("Row A, Seat 1", "Row A, Seat 2", "Row A, Seat 3"):
        store.open_or_update(
            now=1.0, track_id=1, seat=seat, score=0.7, state="PIRACY_ALERT",
            breakdown={"pose": 1}, evidence_type="glint", nir_mode="simulated",
            synthetic=False, config_snapshot={"nir_mode": "simulated"}, merge_window_s=120,
        )
    page = store.list_incidents(limit=1, offset=1)
    assert page["total"] == 3
    assert len(page["items"]) == 1
    with pytest.raises(ValueError):
        store.list_incidents(limit=0)
    csv_text = store.export_csv()
    assert "seat" in csv_text.splitlines()[0]
    assert "Row A, Seat 1" in csv_text
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    failed = IncidentStore(blocked / "cineguard.db")
    assert failed.degraded is True
    assert failed.open_or_update(
        now=1, track_id=1, seat="Row A, Seat 1", score=1, state="PIRACY_ALERT",
        breakdown={}, evidence_type="glint", nir_mode="simulated", synthetic=False,
        config_snapshot={}, merge_window_s=10,
    ) is None


def test_disk_full_keeps_the_incident(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "cineguard.db")
    incident_id = store.open_or_update(
        now=1, track_id=4, seat="Row B, Seat 2", score=0.8, state="PIRACY_ALERT",
        breakdown={"glint": 1}, evidence_type="glint", nir_mode="simulated", synthetic=True,
        config_snapshot={}, merge_window_s=10,
    )
    assert incident_id is not None

    def fail_write(_path: Path, _data: bytes) -> None:
        raise OSError(28, "No space left on device")

    evidence = EvidenceStore(
        tmp_path / "evidence",
        save_bytes=fail_write,
        on_update=lambda incident_id, meta: store.set_evidence(incident_id, **meta),
    )
    evidence.push_frame(1.0, np.full((48, 64, 3), 20, dtype=np.uint8))
    meta = evidence.begin(incident_id, 1.0, "Row B, Seat 2", "simulated")
    assert meta["evidence_status"] == "failed"
    assert evidence.degraded is True
    row = store.get(incident_id)
    assert row is not None
    assert row["evidence_status"] == "failed"
    assert row["seat"] == "Row B, Seat 2"


def test_review_endpoint_and_traversal(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    incident_id = runtime.incidents.open_or_update(
        now=2, track_id=1, seat="Row A, Seat 1", score=0.5, state="PIRACY_ALERT",
        breakdown={}, evidence_type="glint", nir_mode="simulated", synthetic=False,
        config_snapshot={}, merge_window_s=10,
    )
    app = create_app(runtime)
    with TestClient(app) as client:
        rejected = client.post(f"/api/v1/incidents/{incident_id}/review", json={"status": "expired"})
        assert rejected.status_code == 422
        missing = client.post("/api/v1/incidents/999/review", json={"status": "confirmed", "note": "x"})
        assert missing.status_code == 404
        ok = client.post(f"/api/v1/incidents/{incident_id}/review", json={"status": "acknowledged", "note": "checked"})
        assert ok.status_code == 200
        assert ok.json()["status"] == "acknowledged"
        exported = client.get("/api/v1/incidents/export.csv")
        assert exported.status_code == 200
        assert "acknowledged" in exported.text
    assert resolve_under(tmp_path, "../secrets.txt") is None
