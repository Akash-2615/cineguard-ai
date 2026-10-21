"""Synthetic-image tests for the SIMULATED NIR glint detector."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import CommandCenterRuntime, create_app
from config import AppConfig
from engine.nir import GlintDetector, NirParams, NIRSimulator
from engine.pose_detector import PoseCandidate
from engine.tracker import Track
from sources import Frame
from tests.test_api_smoke import SequenceSource


def _dot(size: int = 96, sigma: float = 1.6, peak: float = 240.0, center: tuple[int, int] | None = None) -> np.ndarray:
    image = np.full((size, size), 12, dtype=np.float32)
    cy, cx = center if center is not None else (size // 2, size // 2)
    yy, xx = np.mgrid[0:size, 0:size]
    image += peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return np.clip(image, 0, 255).astype(np.uint8)


def test_gaussian_dot_scores_high() -> None:
    detector = GlintDetector()
    result = detector.detect(_dot())
    assert result.accepted
    assert result.best_score > 0.55
    assert result.accepted[0].peak > 180


def test_large_bright_rectangle_is_rejected() -> None:
    image = np.full((96, 96), 15, dtype=np.uint8)
    image[18:78, 16:82] = 220
    result = GlintDetector().detect(image)
    assert result.accepted == []
    assert result.best_score == 0.0


def test_noise_only_has_no_candidates() -> None:
    image = np.random.default_rng(0).integers(0, 45, size=(96, 96), dtype=np.uint8)
    result = GlintDetector().detect(image)
    assert result.accepted == []


def test_saturated_tiny_dot_is_detected() -> None:
    image = np.full((64, 64), 8, dtype=np.uint8)
    image[30:34, 30:34] = 255
    result = GlintDetector().detect(image)
    assert result.accepted
    assert result.accepted[0].saturated


def test_two_dots_pick_the_better_score() -> None:
    image = _dot(size=110, sigma=1.4, peak=255, center=(32, 36))
    dim = _dot(size=110, sigma=2.2, peak=90, center=(78, 70))
    combined = np.maximum(image, dim)
    result = GlintDetector().detect(combined)
    assert len(result.accepted) >= 2
    assert result.accepted[0].score == result.best_score
    assert result.accepted[0].peak >= result.accepted[1].peak


def test_empty_and_nan_frames_are_safe() -> None:
    detector = GlintDetector()
    empty = detector.detect(np.zeros((0, 0), dtype=np.uint8))
    tiny = detector.detect(np.zeros((2, 2), dtype=np.uint8))
    missing = detector.detect(np.full((16, 16), np.nan))
    assert empty.candidates == []
    assert tiny.accepted == []
    assert missing.candidates == []
    simulator = NIRSimulator()
    assert simulator.convert(np.full((8, 8, 3), np.nan), NirParams()) is None


def test_persistence_needs_three_of_five() -> None:
    detector = GlintDetector()
    dot = _dot()
    dark = np.full_like(dot, 10)
    flash = detector.detect(dot, track_id=3)
    assert flash.confirmed is False
    for _ in range(4):
        assert detector.detect(dark, track_id=3).confirmed is False

    detector.reset()
    pattern = [dot, dark, dot, dot, dark]
    results = [detector.detect(image, track_id=3) for image in pattern]
    assert results[0].confirmed is False
    assert results[3].confirmed is True


def test_synthetic_flag_propagates_for_a_raised_hand() -> None:
    frame = np.full((120, 160, 3), (18, 16, 14), dtype=np.uint8)
    candidate = PoseCandidate(
        person_id=0,
        hand_bbox=(70, 40, 92, 62),
        elevation_confidence=0.9,
        is_elevated=True,
        keypoints_dict={},
        person_bbox=(40, 10, 120, 110),
        raw_elevated=True,
    )
    track = Track(track_id=4, bbox=candidate.person_bbox, confidence=0.9, source_detection_id=0)
    from engine.fusion_engine import NIRLensFusionEngine

    engine = NIRLensFusionEngine()
    engine.update_params({"synthetic_glint": True, "synthetic_radius": 3.5, "synthetic_intensity": 1.0})
    result = None
    for _ in range(4):
        found = engine.analyze(frame, [candidate], [track], "simulated")
        result = found[4]
    assert result is not None
    assert result.confirmed
    assert result.accepted
    assert all(glint.synthetic for glint in result.accepted)
    screen = np.full((96, 96, 3), (20, 20, 20), dtype=np.uint8)
    screen[18:78, 16:82] = (230, 230, 230)
    screened = GlintDetector().detect(NIRSimulator().convert(screen, NirParams()) )
    assert screened.accepted == []


def test_red_channel_dominates_simulation_and_wavelength_is_a_label() -> None:
    simulator = NIRSimulator()
    params = NirParams(gamma=1.0, intensity=1.0, vignette=0.0, gain=1.0, noise_floor=0.0)
    red = np.zeros((12, 12, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    blue = np.zeros((12, 12, 3), dtype=np.uint8)
    blue[:, :, 0] = 255
    red_mono = simulator.convert(red, params)
    blue_mono = simulator.convert(blue, params)
    assert red_mono is not None and blue_mono is not None
    assert float(red_mono.mean()) > float(blue_mono.mean())
    other = params.model_copy(update={"wavelength_nm": 940})
    again = simulator.convert(red, other)
    assert np.array_equal(red_mono, again)


def test_mode_off_disables_glint_score() -> None:
    result = GlintDetector().detect(_dot(), mode="off")
    assert result.best_score == 0.0
    assert result.confirmed is False
    assert result.candidates == []


def test_invalid_params_are_rejected() -> None:
    with pytest.raises(ValidationError):
        NirParams(wavelength_nm=700)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        NirParams(persistence_m=6, persistence_n=3)
    with pytest.raises(ValidationError):
        NirParams(gamma=8)


def test_params_change_behavior_live_without_restart() -> None:
    config = AppConfig(_env_file=None, pose_model_path="models/does-not-exist.pt")
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    app = create_app(runtime)
    weak = _dot(sigma=2.4, peak=70)
    with TestClient(app) as client:
        before = client.get("/api/v1/nir/params")
        assert before.status_code == 200
        assert before.json()["simulated"] is True
        sensitive = client.post("/api/v1/nir/params", json={"preset": "sensitive"})
        assert sensitive.status_code == 200
        assert sensitive.json()["preset"] == "sensitive"
        runtime.nir.detector.reset()
        seen = runtime.nir.detector.detect(weak)
        assert seen.accepted
        conservative = client.post(
            "/api/v1/nir/params",
            json={
                "preset": "conservative",
                "circularity_min": 0.95,
                "contrast_min": 0.9,
                "min_area": 40,
                "max_area": 60,
            },
        )
        assert conservative.status_code == 200
        runtime.nir.detector.reset()
        hidden = runtime.nir.detector.detect(weak)
        assert seen.best_score >= hidden.best_score
        assert hidden.accepted == []
        rejected = client.post("/api/v1/nir/params", json={"wavelength_nm": 700})
        assert rejected.status_code == 422
        frame = np.full((48, 64, 3), 20, dtype=np.uint8)
        runtime.nir.analyze(frame, [], [], "simulated")
        view = client.get("/api/v1/nir/frame.jpg?view=heat")
        assert view.status_code == 200
        assert view.content[:2] == b"\xff\xd8"
        missing = client.get("/api/v1/nir/frame.jpg?view=nope")
        assert missing.status_code == 422


def test_processing_loop_labels_synthetic_glints() -> None:
    config = AppConfig(_env_file=None, pose_model_path="models/does-not-exist.pt", nir_mode="simulated")
    runtime = CommandCenterRuntime(config=config, source=SequenceSource(), autoload_models=False)
    runtime.prepare()
    runtime.nir.update_params({"synthetic_glint": True})
    candidate = PoseCandidate(
        person_id=0,
        hand_bbox=(20, 12, 40, 32),
        elevation_confidence=0.92,
        is_elevated=True,
        keypoints_dict={},
        person_bbox=(8, 4, 56, 44),
        raw_elevated=True,
    )
    track = Track(track_id=1, bbox=candidate.person_bbox, confidence=0.9, source_detection_id=0)
    image = np.full((48, 64, 3), (16, 14, 12), dtype=np.uint8)
    confirmed = False
    for index in range(4):
        frame = Frame(index + 1, index / 25.0, image)
        glints = runtime.nir.analyze(image, [candidate], [track], "simulated")
        confirmed = glints[1].confirmed
        runtime._update_states(frame, runtime.config, [candidate], [track], [], 64, 48, glints)
    assert confirmed
    row = runtime._telemetry["tracks"][0]
    assert row["glints"]
    assert all(item["synthetic"] for item in row["glints"])
    assert runtime._telemetry["nir"]["mode"] == "simulated"
    assert runtime._telemetry["nir"]["synthetic_glint"] is True
