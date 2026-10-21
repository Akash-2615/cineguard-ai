"""CineGuard inference engines."""

from engine.fusion_engine import NIRLensFusionEngine, VerifiedTarget
from engine.nir import NIRSimulator
from engine.pose_detector import PoseCandidate, PoseVectorEngine
from engine.seat_mapper import AuditoriumSeatMapper
from engine.state_machine import PiracyState, PiracyStateMachine
from engine.tracker import SpatialProximityTracker, Track, TrackDetection

__all__ = [
    "NIRLensFusionEngine",
    "AuditoriumSeatMapper",
    "NIRSimulator",
    "PoseCandidate",
    "PoseVectorEngine",
    "PiracyState",
    "PiracyStateMachine",
    "SpatialProximityTracker",
    "Track",
    "TrackDetection",
    "VerifiedTarget",
]
