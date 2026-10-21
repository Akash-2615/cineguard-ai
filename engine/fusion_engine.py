"""Pose and SIMULATED-NIR glint fusion.

Lens evidence comes only from ``engine.nir``. A bright phone screen is not
accepted as a lens. Real NIR suppresses screens as well, because they emit
little near-infrared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.nir import NIRLensPipeline
from engine.pose_detector import PoseCandidate


@dataclass(frozen=True)
class VerifiedTarget:
    """A pose candidate corroborated by a confirmed glint."""

    person_id: int
    bounding_box: tuple[int, int, int, int]
    fusion_confidence: float
    has_lens_reflection: bool
    synthetic: bool = False


class NIRLensFusionEngine(NIRLensPipeline):
    """Compatibility wrapper around the simulated NIR pipeline."""

    def __init__(self) -> None:
        super().__init__()

    def fuse_modalities(
        self,
        pose_candidates: list[PoseCandidate],
        frame: np.ndarray,
        tracks: list[Any] | None = None,
        mode: str = "simulated",
    ) -> list[VerifiedTarget]:
        """Score elevated poses against glints on the NIR image."""
        if tracks is None:
            tracks = _tracks_from_candidates(pose_candidates)
        results = self.analyze(frame, pose_candidates, tracks, mode)
        verified: list[VerifiedTarget] = []
        for track in tracks:
            candidate = _candidate(pose_candidates, track)
            if candidate is None or not candidate.is_elevated:
                continue
            result = results.get(track.track_id)
            if result is None or mode == "off":
                continue
            has_glint = bool(result.confirmed and result.best_score > 0)
            if not has_glint:
                continue
            synthetic = any(glint.synthetic for glint in result.accepted)
            pose_score = float(np.clip(candidate.elevation_confidence, 0.0, 1.0))
            fusion_confidence = float(np.clip(0.55 * pose_score + 0.45 * result.ema_score, 0.0, 1.0))
            verified.append(
                VerifiedTarget(
                    person_id=candidate.person_id,
                    bounding_box=result.roi or candidate.hand_bbox,
                    fusion_confidence=fusion_confidence,
                    has_lens_reflection=True,
                    synthetic=synthetic,
                )
            )
        return verified


def _tracks_from_candidates(candidates: list[PoseCandidate]) -> list[Any]:
    from engine.tracker import Track

    return [
        Track(
            track_id=index,
            bbox=candidate.person_bbox,
            confidence=candidate.elevation_confidence,
            source_detection_id=index,
        )
        for index, candidate in enumerate(candidates)
    ]


def _candidate(candidates: list[PoseCandidate], track: Any) -> PoseCandidate | None:
    index = getattr(track, "source_detection_id", None)
    if index is None or index >= len(candidates):
        return None
    return candidates[index]
