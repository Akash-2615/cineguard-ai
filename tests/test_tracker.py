"""Tracker identity through occlusion and nearby people."""

from engine.tracker import SpatialProximityTracker, TrackDetection


def test_id_persists_through_short_occlusion() -> None:
    tracker = SpatialProximityTracker(max_missed_frames=5)
    first = tracker.update([TrackDetection(0, (0, 0, 20, 60), 0.9)])
    track_id = first[0].track_id
    for _ in range(3):
        hidden = tracker.update([])
        assert any(track.track_id == track_id for track in hidden)
    again = tracker.update([TrackDetection(1, (4, 0, 24, 60), 0.9)])
    matched = [track for track in again if track.track_id == track_id and track.missed == 0]
    assert len(matched) == 1


def test_nearby_people_do_not_swap_ids() -> None:
    tracker = SpatialProximityTracker()
    left = (0, 0, 40, 80)
    right = (300, 0, 340, 80)
    first = tracker.update(
        [TrackDetection(0, left, 0.9), TrackDetection(1, right, 0.9)]
    )
    ids = {track.bbox: track.track_id for track in first}
    second = tracker.update(
        [
            TrackDetection(0, (10, 0, 50, 80), 0.9),
            TrackDetection(1, (310, 0, 350, 80), 0.9),
        ]
    )
    by_id = {track.track_id: track.bbox for track in second}
    assert by_id[ids[left]][0] < 80
    assert by_id[ids[right]][0] > 200
