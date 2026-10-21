"""Temporal piracy state machine.

State timing uses the ``now`` value supplied by the caller (a frame timestamp).
An optional clock exists only so tests can inject time. Production calls pass
``frame.ts`` and do not read the wall clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class PiracyState(str, Enum):
    IDLE = "IDLE"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT_PENDING = "ALERT_PENDING"
    PIRACY_ALERT = "PIRACY_ALERT"


class PiracyStateMachine:
    """Stateful evidence timer with a soft three-second loss grace period."""

    def __init__(
        self,
        piracy_threshold_sec: float = 30.0,
        soft_decay_sec: float = 3.0,
        clock: Callable[[], float] | None = None,
        max_step_sec: float | None = None,
    ) -> None:
        if piracy_threshold_sec <= 0 or soft_decay_sec <= 0:
            raise ValueError("state-machine durations must be positive")
        if max_step_sec is not None and max_step_sec <= 0:
            raise ValueError("max_step_sec must be positive")
        self.piracy_threshold_sec = piracy_threshold_sec
        self.soft_decay_sec = soft_decay_sec
        self.max_step_sec = max_step_sec
        self._clock = clock
        self.state = PiracyState.IDLE
        self.active_seconds = 0.0
        self.lowered_seconds = 0.0
        self.last_update: float | None = None

    def _resolve_now(self, now: Optional[float]) -> float:
        if now is not None:
            return float(now)
        if self._clock is not None:
            return float(self._clock())
        raise ValueError("frame timestamp is required")

    def update(self, is_elevated: bool, has_lens_glare: bool, now: Optional[float] = None) -> PiracyState:
        current = self._resolve_now(now)
        if self.last_update is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, current - self.last_update)
        # A seek or pause in a file can jump the timestamp by minutes. Clamp
        # that gap so it cannot be counted as continuous recording time.
        if self.max_step_sec is not None:
            elapsed = min(elapsed, self.max_step_sec)
        self.last_update = current
        corroborated = bool(is_elevated and has_lens_glare)
        if corroborated:
            self.active_seconds += elapsed
            self.lowered_seconds = 0.0
            if self.active_seconds >= self.piracy_threshold_sec:
                self.state = PiracyState.PIRACY_ALERT
            elif self.active_seconds > 0.0:
                self.state = (
                    PiracyState.ALERT_PENDING
                    if self.active_seconds >= self.piracy_threshold_sec * 0.8
                    else PiracyState.SUSPICIOUS
                )
        else:
            self.lowered_seconds += elapsed
            if self.lowered_seconds <= self.soft_decay_sec:
                self.active_seconds = max(0.0, self.active_seconds - elapsed * 0.25)
                if self.state != PiracyState.PIRACY_ALERT:
                    self.state = PiracyState.SUSPICIOUS if self.active_seconds > 0 else PiracyState.IDLE
            else:
                self.reset(now=current)
        return self.state

    def reset(self, now: Optional[float] = None) -> None:
        self.state = PiracyState.IDLE
        self.active_seconds = 0.0
        self.lowered_seconds = 0.0
        if now is not None:
            self.last_update = float(now)
        elif self._clock is not None:
            self.last_update = float(self._clock())
        else:
            self.last_update = None

    def finalize(self, now: Optional[float] = None) -> PiracyState:
        """Drop a lost track back to idle so it cannot keep accumulating time."""
        self.reset(now=now)
        return self.state

    def render_overlay(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        seat_label: Optional[str] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """Render amber progress or flashing red alert graphics in-place."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
        x1, y1, x2, y2 = self._clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if self.state in (PiracyState.SUSPICIOUS, PiracyState.ALERT_PENDING):
            amber = (0, 191, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), amber, 3, cv2.LINE_AA)
            center = (x1 + 28, y1 + 28)
            radius = 18
            progress = min(1.0, self.active_seconds / self.piracy_threshold_sec)
            cv2.ellipse(frame, center, (radius, radius), -90, 0, 360, (35, 35, 35), 4, cv2.LINE_AA)
            cv2.ellipse(frame, center, (radius, radius), -90, 0, 360 * progress, amber, 4, cv2.LINE_AA)
            cv2.putText(frame, f"{self.active_seconds:.1f}s / {self.piracy_threshold_sec:.0f}s",
                        (x1, max(22, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.65, amber, 2, cv2.LINE_AA)
        elif self.state == PiracyState.PIRACY_ALERT:
            stamp = 0.0 if now is None and self._clock is None else self._resolve_now(now)
            flash = int(stamp * 4) % 2 == 0
            red = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, 5, cv2.LINE_AA)
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 70, 255), 2, cv2.LINE_AA)
            if flash:
                label = "CRITICAL: PIRACY DETECTED"
                if seat_label:
                    label += f" - {seat_label}"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 150), -1)
                cv2.putText(frame, label, (14, 29), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _clamp_bbox(bbox, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return max(0, min(width - 1, x1)), max(0, min(height - 1, y1)), max(0, min(width - 1, x2)), max(0, min(height - 1, y2))


@dataclass
class RecentTrack:
    """A track that left the frame, kept so a new id at the same seat can merge."""

    track_id: int
    seat: str
    foot: Tuple[float, float] | None
    ts: float
    incident_id: int | None
    state: PiracyState


class TrackStateManager:
    """One piracy state machine per track id.

    A lost ``PIRACY_ALERT`` is held for ``grace_sec`` before ``finalize()``.
    Other lost tracks are finalized on the next frame. Reappearance is decided
    by seat label, or by a nearby foot point when the seat is unmapped.
    """

    def __init__(
        self,
        machines: dict[int, PiracyStateMachine],
        piracy_threshold_sec: float = 30.0,
        max_step_sec: float = 1.0,
        grace_sec: float = 2.0,
        merge_window_s: float = 120.0,
        soft_decay_sec: float = 3.0,
    ) -> None:
        self.machines = machines
        self.piracy_threshold_sec = piracy_threshold_sec
        self.max_step_sec = max_step_sec
        self.grace_sec = grace_sec
        self.merge_window_s = merge_window_s
        self.soft_decay_sec = soft_decay_sec
        self.missing_since: dict[int, float] = {}
        self.recent: list[RecentTrack] = []

    def configure(
        self,
        piracy_threshold_sec: float,
        max_step_sec: float,
        grace_sec: float,
        merge_window_s: float,
    ) -> None:
        self.piracy_threshold_sec = piracy_threshold_sec
        self.max_step_sec = max_step_sec
        self.grace_sec = grace_sec
        self.merge_window_s = merge_window_s
        for machine in self.machines.values():
            machine.piracy_threshold_sec = piracy_threshold_sec
            machine.max_step_sec = max_step_sec

    def machine_for(self, track_id: int) -> PiracyStateMachine:
        self.missing_since.pop(track_id, None)
        machine = self.machines.get(track_id)
        if machine is None:
            machine = PiracyStateMachine(
                self.piracy_threshold_sec,
                soft_decay_sec=self.soft_decay_sec,
                max_step_sec=self.max_step_sec,
            )
            self.machines[track_id] = machine
        machine.piracy_threshold_sec = self.piracy_threshold_sec
        machine.max_step_sec = self.max_step_sec
        return machine

    def remember(
        self,
        track_id: int,
        seat: str,
        foot: Tuple[float, float] | None,
        ts: float,
        incident_id: int | None,
    ) -> None:
        machine = self.machines.get(track_id)
        state = machine.state if machine is not None else PiracyState.IDLE
        self.recent = [item for item in self.recent if item.track_id != track_id and ts - item.ts <= self.merge_window_s]
        self.recent.append(RecentTrack(track_id, seat, foot, ts, incident_id, state))

    def match_recent(
        self,
        track_id: int,
        seat: str,
        foot: Tuple[float, float] | None,
        ts: float,
    ) -> int | None:
        """Return an incident id when this track continues someone who just left."""
        best: int | None = None
        for item in self.recent:
            if item.track_id == track_id or item.incident_id is None:
                continue
            if ts - item.ts > self.merge_window_s:
                continue
            same_seat = seat != "Unmapped" and item.seat == seat
            near = False
            if foot is not None and item.foot is not None:
                near = abs(foot[0] - item.foot[0]) <= 0.08 and abs(foot[1] - item.foot[1]) <= 0.08
            if same_seat or near:
                best = item.incident_id
        return best

    def retire(self, active_ids: set[int], now: float) -> list[int]:
        """Finalize tracks that are gone. Alerts wait out the grace period first."""
        finalized: list[int] = []
        for track_id in list(self.machines):
            if track_id in active_ids:
                continue
            machine = self.machines[track_id]
            if machine.state == PiracyState.PIRACY_ALERT:
                started = self.missing_since.setdefault(track_id, now)
                if now - started < self.grace_sec:
                    continue
            self.missing_since.pop(track_id, None)
            machine.finalize(now=now)
            del self.machines[track_id]
            finalized.append(track_id)
        return finalized

