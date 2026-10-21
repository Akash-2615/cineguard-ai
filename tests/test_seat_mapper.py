"""Seat homography, aisles, and the calibration API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api_smoke import SequenceSource

from app import CommandCenterRuntime, create_app
from calibration import AuditoriumCalibration, LayoutSpec, load_calibration, save_calibration
from config import AppConfig
from engine.seat_mapper import AuditoriumSeatMapper, foot_pixel


def _grid(**layout) -> AuditoriumCalibration:
    spec = dict(rows=2, seats_per_row=2, row_labels="AB", seat_numbering="left_to_right", aisle_after_seats=[])
    spec.update(layout)
    return AuditoriumCalibration(
        name="hall",
        camera_id="cam-1",
        image_size=(200, 100),
        image_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        layout=LayoutSpec(**spec),
        tolerance=0.05,
    )


def test_identity_perspective_and_out_of_bounds() -> None:
    mapper = AuditoriumSeatMapper(_grid(), 200, 100)
    assert mapper.map_normalized(0.25, 0.25).label == "Row A, Seat 1"
    assert mapper.map_normalized(0.75, 0.75).label == "Row B, Seat 2"
    assert mapper.map_normalized(0.25, 0.25).in_bounds is True
    outside = mapper.map_normalized(0.5, 1.4)
    assert outside.label == "Unmapped"
    assert outside.in_bounds is False

    skewed = _grid().model_copy(update={"image_points": [(0.2, 0.1), (0.8, 0.15), (0.9, 0.95), (0.1, 0.85)]})
    perspective = AuditoriumSeatMapper(skewed, 200, 100)
    assert perspective.map_normalized(0.2, 0.1).seat == 1
    assert perspective.map_normalized(0.0, 0.0).label == "Unmapped"


def test_aisle_numbering_offsets_and_resolution() -> None:
    aisle = AuditoriumSeatMapper(_grid(rows=1, seats_per_row=4, row_labels="A", aisle_after_seats=[2]), 100, 100)
    assert aisle.map_normalized(0.1, 0.5).label == "Row A, Seat 1"
    assert aisle.map_normalized(0.5, 0.5).label == "Unmapped"
    assert aisle.map_normalized(0.7, 0.5).label == "Row A, Seat 3"
    flipped = AuditoriumSeatMapper(
        _grid(rows=1, seats_per_row=4, row_labels="A", seat_numbering="right_to_left"),
        100,
        100,
    )
    assert flipped.map_normalized(0.125, 0.5).seat == 4
    raked = AuditoriumSeatMapper(_grid(rows=2, seats_per_row=1, row_labels="AB", row_y_offsets=[0.0, 1.0]), 90, 90)
    assert raked.map_normalized(0.5, 2.0 / 3.0).label == "Row B, Seat 1"
    small = AuditoriumSeatMapper(_grid(), 100, 80)
    large = AuditoriumSeatMapper(_grid(), 640, 480)
    assert small.map_normalized(0.75, 0.75).label == large.map_normalized(0.75, 0.75).label


def test_foot_point_uses_hips_when_ankles_are_weak() -> None:
    bbox = (0, 0, 100, 200)
    hips = {"left_hip": (0.4, 0.6), "right_hip": (0.6, 0.6)}
    confidence = {"left_ankle": 0.1, "right_ankle": 0.1, "left_hip": 0.9, "right_hip": 0.9}
    x, y = foot_pixel(bbox, hips, confidence, 100, 100)
    assert x == pytest.approx(50)
    assert y == pytest.approx(60)
    ankles = {"left_ankle": (0.2, 0.9), "right_ankle": (0.4, 0.9), **hips}
    strong = {"left_ankle": 0.9, "right_ankle": 0.9, "left_hip": 0.9, "right_hip": 0.9}
    ax, ay = foot_pixel(bbox, ankles, strong, 100, 100)
    assert ay == pytest.approx(90)


def test_degenerate_points_rejected_and_round_trip(tmp_path: Path) -> None:
    def reject(points: list[tuple[float, float]], match: str) -> None:
        payload = _grid().model_dump()
        payload["image_points"] = points
        with pytest.raises(ValidationError, match=match):
            AuditoriumCalibration.model_validate(payload)

    reject([(0.0, 0.2), (1.0, 0.2), (0.4, 0.2), (0.8, 0.2)], "collinear")
    reject([(0.1, 0.1), (0.9, 0.8), (0.1, 0.9), (0.8, 0.2)], "self-intersecting")
    reject([(0.0, 0.0), (1.2, 0.0), (1.0, 1.0), (0.0, 1.0)], "outside")
    path = tmp_path / "auditorium.json"
    original = _grid()
    save_calibration(path, original)
    loaded = load_calibration(path)
    assert loaded is not None
    assert AuditoriumSeatMapper(loaded, 200, 100).map_normalized(0.25, 0.25).label == "Row A, Seat 1"


def test_uncalibrated_default_and_probe_after_save(tmp_path: Path) -> None:
    config = AppConfig(
        _env_file=None,
        pose_model_path="models/does-not-exist.pt",
        db_path=str(tmp_path / "cineguard.db"),
        evidence_dir=str(tmp_path / "evidence"),
        calibration_path=str(tmp_path / "auditorium.json"),
    )
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    runtime.prepare()
    app = create_app(runtime)
    with TestClient(app) as client:
        current = client.get("/api/v1/calibration")
        assert current.status_code == 200
        assert current.json()["uncalibrated"] is True
        probe = client.post("/api/v1/calibration/probe", json={"x": 0.05, "y": 0.1})
        assert probe.status_code == 200
        assert probe.json()["label"].endswith("(uncalibrated)")
        saved = _grid().model_dump()
        put = client.put("/api/v1/calibration", json=saved)
        assert put.status_code == 200
        assert put.json()["uncalibrated"] is False
        checked = client.post("/api/v1/calibration/probe", json={"x": 0.25, "y": 0.25})
        assert checked.json()["label"] == "Row A, Seat 1"
        bad = client.put("/api/v1/calibration", json={**saved, "image_points": [[0, 0], [1, 0], [0.5, 0], [0.2, 0]]})
        assert bad.status_code == 422
