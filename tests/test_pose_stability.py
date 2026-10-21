"""Pose stability history is keyed by track id."""

from collections import defaultdict, deque

from engine.pose_detector import PoseVectorEngine


def _engine() -> PoseVectorEngine:
    engine = PoseVectorEngine.__new__(PoseVectorEngine)
    engine.stability_window = 5
    engine.stable_frames = 3
    engine._stability = defaultdict(lambda: deque(maxlen=5))
    return engine


def test_stability_is_per_track_id() -> None:
    engine = _engine()
    assert engine.apply_track_stability(7, True) is False
    assert engine.apply_track_stability(7, True) is False
    assert engine.apply_track_stability(7, True) is True
    assert engine.apply_track_stability(8, True) is False


def test_forget_tracks_drops_retired_ids() -> None:
    engine = _engine()
    engine.apply_track_stability(1, True)
    engine.apply_track_stability(2, True)
    engine.forget_tracks({1})
    assert 2 not in engine._stability
    assert 1 in engine._stability
