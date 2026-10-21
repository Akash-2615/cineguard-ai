"""Rule-based recording score and behavior labels.

Components, all in ``[0, 1]``:

- pose: wrist above the shoulder line, forearm angle, wrist not at the mouth
- sightline: wrist stays in the eye-to-torso corridor
- glint: confirmed retroreflection score (0 when NIR is off or unconfirmed)
- persistence: fraction of the piracy threshold already accumulated
- stability: mean keypoint confidence

Behavior labels, first match wins:

- ``eating_like``: a wrist is within 0.08 of the nose
- ``texting_like``: both wrists are below the shoulders and the nose is below the eyes
- ``stretching_like``: both wrists are above the shoulders, held for less than
  ``stretch_brief_s``, and there is no confirmed glint
- ``recording_candidate``: the stable pose is elevated and the hard gate passes
- ``idle``: everything else

Hard gate: with NIR on, a score is alert-eligible only when the pose is elevated
and the glint is confirmed. With NIR off, the score is pose-only, capped, and
cannot open an alert unless ``allow_pose_only_alert`` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.pose_detector import PoseVectorEngine


@dataclass(frozen=True)
class ScoreWeights:
    pose: float = 0.25
    sightline: float = 0.15
    glint: float = 0.35
    persistence: float = 0.15
    stability: float = 0.10

    @classmethod
    def from_mapping(cls, weights: dict[str, float]) -> "ScoreWeights":
        return cls(
            pose=float(weights.get("pose", 0.25)),
            sightline=float(weights.get("sightline", 0.15)),
            glint=float(weights.get("glint", 0.35)),
            persistence=float(weights.get("persistence", 0.15)),
            stability=float(weights.get("stability", 0.10)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "pose": self.pose,
            "sightline": self.sightline,
            "glint": self.glint,
            "persistence": self.persistence,
            "stability": self.stability,
        }


@dataclass(frozen=True)
class ScoreResult:
    score: float
    breakdown: dict[str, float]
    alert_eligible: bool
    evidence_type: str
    behavior: str
    reason: str
    synthetic: bool

    def public_breakdown(self) -> dict[str, float | bool | str]:
        payload: dict[str, float | bool | str] = dict(self.breakdown)
        payload["score"] = self.score
        payload["alert_eligible"] = self.alert_eligible
        payload["evidence_type"] = self.evidence_type
        payload["behavior"] = self.behavior
        return payload


def score_track(
    candidate: Any,
    glint: Any,
    *,
    active_seconds: float,
    threshold_sec: float,
    nir_mode: str,
    weights: ScoreWeights,
    allow_pose_only_alert: bool = False,
    pose_only_score_cap: float = 0.6,
    stretch_brief_s: float = 2.0,
) -> ScoreResult:
    """Score one track. ``active_seconds`` comes from the frame-timestamp state machine."""
    points = dict(getattr(candidate, "keypoints_dict", None) or {})
    confidence = dict(getattr(candidate, "keypoint_confidence", None) or {})
    elevated = bool(getattr(candidate, "is_elevated", False))
    confirmed = bool(glint is not None and getattr(glint, "confirmed", False) and getattr(glint, "best_score", 0) > 0)
    glint_score = float(getattr(glint, "best_score", 0.0) or 0.0) if confirmed else 0.0
    synthetic = bool(glint is not None and any(getattr(item, "synthetic", False) for item in getattr(glint, "accepted", []) or []))

    pose = _pose_component(points, confidence, elevated)
    sightline = _sightline_component(points, confidence)
    stability = _stability_component(confidence, getattr(candidate, "elevation_confidence", 0.0))
    persistence = 0.0
    if threshold_sec > 0:
        persistence = float(np.clip(active_seconds / threshold_sec, 0.0, 1.0))
    glint_component = float(np.clip(glint_score, 0.0, 1.0)) if nir_mode != "off" else 0.0

    total = weights.pose + weights.sightline + weights.glint + weights.persistence + weights.stability
    raw = (
        weights.pose * pose
        + weights.sightline * sightline
        + weights.glint * glint_component
        + weights.persistence * persistence
        + weights.stability * stability
    ) / total
    score = float(np.clip(raw, 0.0, 1.0))

    if nir_mode == "off":
        evidence_type = "pose_only"
        alert_eligible = bool(elevated and allow_pose_only_alert)
        if not allow_pose_only_alert:
            score = min(score, float(pose_only_score_cap))
    else:
        evidence_type = "glint"
        alert_eligible = bool(elevated and confirmed)

    behavior, reason = _behavior(
        points,
        confidence,
        elevated=elevated,
        confirmed=confirmed and nir_mode != "off",
        alert_eligible=alert_eligible,
        active_seconds=active_seconds,
        stretch_brief_s=stretch_brief_s,
        nir_mode=nir_mode,
    )
    breakdown = {
        "pose": round(pose, 4),
        "sightline": round(sightline, 4),
        "glint": round(glint_component, 4),
        "persistence": round(persistence, 4),
        "stability": round(stability, 4),
    }
    return ScoreResult(
        score=round(score, 4),
        breakdown=breakdown,
        alert_eligible=alert_eligible,
        evidence_type=evidence_type,
        behavior=behavior,
        reason=reason,
        synthetic=synthetic,
    )


def _pose_component(points: dict, confidence: dict, elevated: bool) -> float:
    wrist_name, elbow_name = _best_arm(points, confidence)
    shoulders = _mean_point(points, confidence, ("left_shoulder", "right_shoulder"))
    if wrist_name is None or shoulders is None or elbow_name not in points:
        return 1.0 if elevated else 0.0
    wrist = points[wrist_name]
    elbow = points[elbow_name]
    above = 1.0 if wrist[1] < shoulders[1] - 0.02 else 0.0
    dy = elbow[1] - wrist[1]
    dx = abs(wrist[0] - elbow[0])
    angle = math.degrees(math.atan2(max(dy, 0.0), dx + 1e-6))
    angle_score = float(np.clip(angle / 90.0, 0.0, 1.0))
    near_mouth = _near_nose(points, wrist)
    clear = 0.0 if near_mouth else 1.0
    return float(np.clip((above + angle_score + clear) / 3.0, 0.0, 1.0))


def _sightline_component(points: dict, confidence: dict) -> float:
    wrist_name, _elbow = _best_arm(points, confidence)
    eyes = _mean_point(points, confidence, ("left_eye", "right_eye", "nose"))
    shoulders = _mean_point(points, confidence, ("left_shoulder", "right_shoulder"))
    if wrist_name is None or eyes is None or shoulders is None:
        return 0.0
    return float(np.clip(PoseVectorEngine._sightline_score(points[wrist_name], eyes, shoulders), 0.0, 1.0))


def _stability_component(confidence: dict, elevation_confidence: float) -> float:
    if confidence:
        values = [float(value) for value in confidence.values()]
        return float(np.clip(sum(values) / len(values), 0.0, 1.0))
    return float(np.clip(elevation_confidence or 0.0, 0.0, 1.0))


def _behavior(
    points: dict,
    confidence: dict,
    *,
    elevated: bool,
    confirmed: bool,
    alert_eligible: bool,
    active_seconds: float,
    stretch_brief_s: float,
    nir_mode: str,
) -> tuple[str, str]:
    wrists = [points[name] for name in ("left_wrist", "right_wrist") if name in points]
    if any(_near_nose(points, wrist) for wrist in wrists):
        return "eating_like", "wrist is near the nose"
    shoulders = _mean_point(points, confidence, ("left_shoulder", "right_shoulder"))
    eyes = _mean_point(points, confidence, ("left_eye", "right_eye"))
    nose = points.get("nose")
    both_low = (
        shoulders is not None
        and len(wrists) == 2
        and all(wrist[1] > shoulders[1] + 0.02 for wrist in wrists)
    )
    head_down = nose is not None and eyes is not None and nose[1] > eyes[1] + 0.04
    if both_low and head_down:
        return "texting_like", "both hands are low and the head is tilted down"
    both_high = (
        shoulders is not None
        and len(wrists) == 2
        and all(wrist[1] < shoulders[1] - 0.02 for wrist in wrists)
    )
    if both_high and active_seconds < stretch_brief_s and not confirmed:
        return "stretching_like", "both arms are raised briefly without a glint"
    if alert_eligible or (elevated and confirmed):
        return "recording_candidate", "elevated pose with a confirmed glint" if nir_mode != "off" else "elevated pose in pose-only mode"
    if elevated and nir_mode != "off" and not confirmed:
        return "idle", "elevated pose without a confirmed glint"
    return "idle", "no recording pose"


def _best_arm(points: dict, confidence: dict) -> tuple[str | None, str | None]:
    best_name = None
    best_score = -1.0
    for wrist_name, elbow_name in (("left_wrist", "left_elbow"), ("right_wrist", "right_elbow")):
        if wrist_name not in points or elbow_name not in points:
            continue
        score = float(confidence.get(wrist_name, 1.0))
        if score > best_score:
            best_score = score
            best_name = wrist_name
    if best_name is None:
        return None, None
    elbow = "left_elbow" if best_name == "left_wrist" else "right_elbow"
    return best_name, elbow


def _mean_point(points: dict, confidence: dict, names: tuple[str, ...]) -> tuple[float, float] | None:
    valid = []
    for name in names:
        if name not in points:
            continue
        if confidence and float(confidence.get(name, 0.0)) < 0.5 and name in confidence:
            continue
        valid.append(points[name])
    if not valid:
        return None
    return float(np.mean([point[0] for point in valid])), float(np.mean([point[1] for point in valid]))


def _near_nose(points: dict, wrist: tuple[float, float]) -> bool:
    nose = points.get("nose")
    if nose is None:
        return False
    return math.dist(wrist, nose) < 0.08
