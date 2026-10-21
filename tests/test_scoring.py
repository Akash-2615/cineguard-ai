"""Behavior labels and the pose/glint hard gate."""

from types import SimpleNamespace

from engine.pose_detector import PoseCandidate
from engine.scoring import ScoreWeights, score_track


def _candidate(**overrides) -> PoseCandidate:
    payload = dict(
        person_id=0,
        hand_bbox=(0, 0, 10, 10),
        elevation_confidence=0.8,
        is_elevated=False,
        keypoints_dict={},
        person_bbox=(0, 0, 40, 80),
        raw_elevated=False,
        keypoint_confidence={},
    )
    payload.update(overrides)
    return PoseCandidate(**payload)


def _glint(confirmed: bool = True, synthetic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        confirmed=confirmed,
        best_score=0.9 if confirmed else 0.0,
        accepted=[SimpleNamespace(synthetic=synthetic)] if confirmed else [],
    )


def test_recording_candidate_requires_confirmed_glint() -> None:
    weights = ScoreWeights()
    raised = _candidate(is_elevated=True, raw_elevated=True)
    blocked = score_track(
        raised, _glint(False), active_seconds=10, threshold_sec=30,
        nir_mode="simulated", weights=weights,
    )
    assert blocked.alert_eligible is False
    assert blocked.behavior == "idle"
    ready = score_track(
        raised, _glint(True, synthetic=True), active_seconds=10, threshold_sec=30,
        nir_mode="simulated", weights=weights,
    )
    assert ready.alert_eligible is True
    assert ready.behavior == "recording_candidate"
    assert ready.synthetic is True
    assert set(ready.breakdown) == {"pose", "sightline", "glint", "persistence", "stability"}
    assert 0.0 <= ready.score <= 1.0


def test_pose_only_is_capped_until_explicitly_allowed() -> None:
    weights = ScoreWeights()
    raised = _candidate(is_elevated=True, elevation_confidence=1.0)
    capped = score_track(
        raised, None, active_seconds=30, threshold_sec=30,
        nir_mode="off", weights=weights, allow_pose_only_alert=False, pose_only_score_cap=0.6,
    )
    assert capped.evidence_type == "pose_only"
    assert capped.alert_eligible is False
    assert capped.score <= 0.6
    allowed = score_track(
        raised, None, active_seconds=30, threshold_sec=30,
        nir_mode="off", weights=weights, allow_pose_only_alert=True,
    )
    assert allowed.alert_eligible is True
    assert allowed.evidence_type == "pose_only"


def test_behavior_rules() -> None:
    weights = ScoreWeights()
    eating = _candidate(keypoints_dict={"nose": (0.5, 0.3), "left_wrist": (0.52, 0.32)})
    assert score_track(eating, None, active_seconds=0, threshold_sec=30, nir_mode="simulated", weights=weights).behavior == "eating_like"
    texting = _candidate(keypoints_dict={
        "left_eye": (0.45, 0.2), "right_eye": (0.55, 0.2), "nose": (0.5, 0.36),
        "left_shoulder": (0.4, 0.45), "right_shoulder": (0.6, 0.45),
        "left_wrist": (0.42, 0.7), "right_wrist": (0.58, 0.72),
        "left_elbow": (0.4, 0.55), "right_elbow": (0.6, 0.55),
    })
    assert score_track(texting, None, active_seconds=1, threshold_sec=30, nir_mode="simulated", weights=weights).behavior == "texting_like"
    stretch = _candidate(is_elevated=True, keypoints_dict={
        "left_shoulder": (0.4, 0.5), "right_shoulder": (0.6, 0.5),
        "left_wrist": (0.38, 0.3), "right_wrist": (0.62, 0.3),
        "left_elbow": (0.38, 0.4), "right_elbow": (0.62, 0.4),
    })
    assert score_track(stretch, None, active_seconds=0.4, threshold_sec=30, nir_mode="simulated", weights=weights).behavior == "stretching_like"
    assert score_track(stretch, _glint(True), active_seconds=0.4, threshold_sec=30, nir_mode="simulated", weights=weights).behavior == "recording_candidate"
