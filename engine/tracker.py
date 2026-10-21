"""Spatially stable target tracking across dropped frames and short occlusions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger


@dataclass(frozen=True)
class TrackDetection:
    """A detection consumed by the spatial tracker."""

    detection_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0


@dataclass(frozen=True)
class Track:
    """Persistent local track state."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int = 1
    missed: int = 0
    source_detection_id: Optional[int] = None


class SpatialProximityTracker:
    """Greedy proximity tracker with a 30-frame occlusion grace period.

    Matching uses center distance normalized by detection height and prefers
    IoU-compatible assignments, preventing nearby audience members from
    swapping IDs during brief drops.
    """

    def __init__(self, max_missed_frames: int = 30, max_distance_ratio: float = 1.5) -> None:
        if max_missed_frames < 1 or max_distance_ratio <= 0:
            raise ValueError("invalid tracker limits")
        self.max_missed_frames = max_missed_frames
        self.max_distance_ratio = max_distance_ratio
        self._next_id = 0
        self._tracks: Dict[int, Track] = {}
        self._logger = get_logger(__name__)

    def update(self, detections: List[TrackDetection]) -> List[Track]:
        """Update tracks and return active tracks, including occluded tracks."""
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        assignments: List[Tuple[int, int]] = []
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                distance = self._center_distance(track.bbox, detection.bbox)
                scale = max(1.0, self._bbox_height(track.bbox), self._bbox_height(detection.bbox))
                iou = self._iou(track.bbox, detection.bbox)
                if distance <= scale * self.max_distance_ratio or iou > 0.05:
                    candidates.append((distance / scale - iou, track_id, index))
        for _, track_id, index in sorted(candidates):
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            assignments.append((track_id, index))
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        updated: Dict[int, Track] = {}
        for track_id, index in assignments:
            detection = detections[index]
            previous = self._tracks[track_id]
            updated[track_id] = Track(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                age=previous.age + 1,
                missed=0,
                source_detection_id=detection.detection_id,
            )
        for track_id in unmatched_tracks:
            previous = self._tracks[track_id]
            if previous.missed + 1 <= self.max_missed_frames:
                updated[track_id] = replace(
                    previous, age=previous.age + 1, missed=previous.missed + 1,
                    source_detection_id=None,
                )
        for index in unmatched_detections:
            detection = detections[index]
            updated[self._next_id] = Track(
                track_id=self._next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                source_detection_id=detection.detection_id,
            )
            self._next_id += 1
        self._tracks = updated
        return list(sorted(updated.values(), key=lambda track: track.track_id))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    @staticmethod
    def _center_distance(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        return float(np.linalg.norm(np.subtract(first_center, second_center)))

    @staticmethod
    def _bbox_height(bbox: Tuple[int, int, int, int]) -> int:
        return max(1, abs(bbox[3] - bbox[1]))

    @staticmethod
    def _iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
        left, top = max(first[0], second[0]), max(first[1], second[1])
        right, bottom = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0
