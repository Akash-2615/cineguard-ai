"""CineGuard AI command center API and processing runtime."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncGenerator

import cv2
import numpy as np
import psutil
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from calibration import (
    AuditoriumCalibration,
    default_calibration,
    load_calibration,
    save_calibration,
)
from config import AppConfig
from engine.fusion_engine import NIRLensFusionEngine, VerifiedTarget
from engine.pose_detector import PoseVectorEngine
from engine.scoring import ScoreResult, ScoreWeights, score_track
from engine.seat_mapper import UNMAPPED, AuditoriumSeatMapper, SeatResult, foot_pixel
from engine.state_machine import PiracyState, PiracyStateMachine, TrackStateManager
from engine.tracker import SpatialProximityTracker, TrackDetection
from evidence import EvidenceStore
from incidents import IncidentStore
from scenarios import ScenarioService, render_report
from security import SecurityMiddleware, tokens_match
from sources import Frame, FrameSource, open_source
from utils.logger import get_logger, recent_logs

logger = get_logger(__name__)

MAX_RESTARTS_PER_MINUTE = 5
APP_VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parent


@dataclass
class ModelBundle:
    """Pose and fusion models owned by the processing thread after startup."""

    pose: PoseVectorEngine | None
    fusion: NIRLensFusionEngine | None
    tracker: SpatialProximityTracker
    loaded: bool
    error: str | None

    @classmethod
    def unavailable(cls, message: str) -> "ModelBundle":
        return cls(
            pose=None,
            fusion=None,
            tracker=SpatialProximityTracker(max_missed_frames=30),
            loaded=False,
            error=message,
        )

    @classmethod
    def load(cls, config: AppConfig) -> "ModelBundle":
        tracker = SpatialProximityTracker(max_missed_frames=30)
        path = resolve_pose_model(config.pose_model_path)
        try:
            if path is None:
                raise FileNotFoundError(
                    f"pose model not found: {config.pose_model_path}"
                )
            pose = PoseVectorEngine(
                model_path=str(path),
                confidence_threshold=config.conf_threshold,
            )
            logger.info("pose model ready: %s", path)
            return cls(pose=pose, fusion=None, tracker=tracker, loaded=True, error=None)
        except Exception as exc:
            logger.exception("failed to load pose model from %s", config.pose_model_path)
            return cls(
                pose=None,
                fusion=None,
                tracker=tracker,
                loaded=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def resolve_pose_model(configured: str) -> Path | None:
    """Find a local weights file. Missing weights are not downloaded."""
    configured_path = Path(configured).expanduser()
    candidates = [configured_path]
    if not configured_path.is_absolute():
        candidates.append(ROOT / configured_path)
    if configured_path.name == "yolov8n-pose.pt":
        candidates.extend((ROOT / "models" / "yolov8n-pose.pt", ROOT / "yolov8n-pose.pt"))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


class ConfigUpdate(BaseModel):
    piracy_threshold_sec: float | None = Field(default=None, gt=0)
    conf_threshold: float | None = Field(default=None, ge=0, le=1)
    target_fps: int | None = Field(default=None, gt=0)
    nir_mode: str | None = None
    loop_video: bool | None = None
    playback_speed: float | None = Field(default=None, ge=0.25, le=4.0)
    demo_mode: bool | None = None
    camera_source: int | str | None = None
    allow_pose_only_alert: bool | None = None
    max_frame_dt: float | None = Field(default=None, gt=0)
    alert_grace_sec: float | None = Field(default=None, gt=0)
    incident_merge_window_s: float | None = Field(default=None, ge=0)
    suppress_after_dismiss_s: float | None = Field(default=None, ge=0)
    weight_pose: float | None = Field(default=None, ge=0, le=1)
    weight_sightline: float | None = Field(default=None, ge=0, le=1)
    weight_glint: float | None = Field(default=None, ge=0, le=1)
    weight_persistence: float | None = Field(default=None, ge=0, le=1)
    weight_stability: float | None = Field(default=None, ge=0, le=1)


class ReviewBody(BaseModel):
    status: str
    note: str = ""


class ProbeBody(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class SourceRequest(BaseModel):
    type: str
    uri: str
    loop: bool | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


class SourceControl(BaseModel):
    action: str
    value: float | None = None


class FrameBus:
    """One JPEG per processed frame. Clients wait until the sequence changes."""

    def __init__(self, initial: bytes) -> None:
        self._cond = threading.Condition()
        self._jpeg = initial
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()
            loop = self._loop
            event = self._event
        if loop is not None and event is not None:
            loop.call_soon_threadsafe(event.set)

    def latest(self) -> bytes:
        with self._cond:
            return self._jpeg

    def snapshot(self) -> tuple[int, bytes]:
        with self._cond:
            return self._seq, self._jpeg

    def wait_next(self, last_seq: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._jpeg

    async def wait_change(self, last_seq: int, timeout: float = 1.0) -> tuple[int, bytes]:
        seq, jpeg = self.snapshot()
        if seq != last_seq:
            return seq, jpeg
        event = self._event
        if event is None:
            await asyncio.sleep(min(timeout, 0.05))
            return self.snapshot()
        event.clear()
        seq, jpeg = self.snapshot()
        if seq != last_seq:
            return seq, jpeg
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        return self.snapshot()


class TelemetryHub:
    """Fan out immutable telemetry snapshots. Each client keeps only the latest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._clients: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._latest = snapshot
            loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._dispatch)

    def _dispatch(self) -> None:
        with self._lock:
            snapshot = self._latest
            clients = list(self._clients)
        if snapshot is None:
            return
        for queue in clients:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._clients.append(queue)
            latest = self._latest
        if latest is not None:
            queue.put_nowait(latest)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._clients:
                self._clients.remove(queue)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest


class CommandCenterRuntime:
    """Owns capture, the supervised processing loop, and published snapshots."""

    def __init__(
        self,
        config: AppConfig | None = None,
        source: FrameSource | None = None,
        models: ModelBundle | None = None,
        autoload_models: bool = True,
    ) -> None:
        self.config = config or AppConfig()
        self.source = source or open_source(
            self.config.camera_source,
            loop=self.config.loop_video,
            speed=self.config.playback_speed,
            width=self.config.frame_width,
            height=self.config.frame_height,
            target_fps=self.config.target_fps,
            io_timeout=self.config.io_timeout,
        )
        self.models = models
        self._autoload = autoload_models and models is None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cineguard-cv")
        self.frames = FrameBus(_placeholder_jpeg())
        self.telemetry_hub = TelemetryHub()
        self._stop = threading.Event()
        self._hold = threading.Event()
        self._restart_count = 0
        self._lock = threading.RLock()
        self._states: dict[int, PiracyStateMachine] = {}
        self._alerts: dict[int, dict[str, Any]] = {}
        self._incident_by_track: dict[int, int] = {}
        self._seat_mapper: AuditoriumSeatMapper | None = None
        self._seat_shape: tuple[int, int] | None = None
        self._seat_generation: int | None = None
        self._last_frame_ts = 0.0
        self._errors: list[str] = []
        self._degraded = False
        self._supervisor_halted = False
        self._future = None
        self._started = False
        self._proc_times: list[float] = []
        self._last_processed_id = 0
        self._last_live_ts: float | None = None
        self._gpu_reader: Any = False
        self._telemetry: dict[str, Any] = _empty_telemetry(self.config)
        self.nir = NIRLensFusionEngine()
        self.incidents = IncidentStore(_repo_path(self.config.db_path))
        self.evidence = EvidenceStore(
            _repo_path(self.config.evidence_dir),
            preroll_s=self.config.preroll_s,
            postroll_s=self.config.postroll_s,
            clip_max_s=self.config.clip_max_s,
            on_update=self._on_evidence,
        )
        self.calibration = self._load_calibration()
        self.state_manager = TrackStateManager(
            self._states,
            piracy_threshold_sec=self.config.piracy_threshold_sec,
            max_step_sec=self.config.max_frame_dt,
            grace_sec=self.config.alert_grace_sec,
            merge_window_s=self.config.incident_merge_window_s,
        )
        if self.incidents.degraded or self.evidence.degraded:
            self._degraded = True

    def prepare(self) -> None:
        """Load models on the server lifespan, before the processing thread starts."""
        if self.models is None and self._autoload:
            self.models = ModelBundle.load(self.config)
        elif self.models is None:
            self.models = ModelBundle.unavailable("model loading skipped")
        if not self.models.loaded:
            self._degraded = True
            if self.models.error:
                self._remember_error(self.models.error)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.telemetry_hub.bind_loop(loop)
        self.frames.bind_loop(loop)

    def start(self) -> None:
        if self._started:
            return
        if self.models is None:
            self.prepare()
        self._stop.clear()
        self.source.start()
        self._future = self.executor.submit(self._supervise)
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._future is not None:
            try:
                self._future.result(timeout=max(2.0, self.config.io_timeout + 2.0))
            except Exception:
                logger.exception("processing thread did not stop cleanly")
        self.source.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.evidence.finalize_all()
        self._started = False

    def health(self) -> dict[str, Any]:
        with self._lock:
            model_loaded = bool(self.models and self.models.loaded)
            degraded = self._degraded or self.incidents.degraded or self.evidence.degraded or not model_loaded
            return {
                "status": "degraded" if degraded else "ok",
                "model_loaded": model_loaded,
                "degraded": degraded,
                "errors": list(self._errors),
                "supervisor_halted": self._supervisor_halted,
                "restarts": self._restart_count,
                "version": APP_VERSION,
                "source": self.source.info(),
            }

    def hold_processing(self, held: bool) -> None:
        """Pause the live loop while a scenario job uses the shared pose model."""
        if held:
            self._hold.set()
        else:
            self._hold.clear()

    def apply_config(self, update: dict[str, Any]) -> dict[str, Any]:
        """Validate, swap config, and push values into live engines and the source."""
        cleaned = {key: value for key, value in update.items() if value is not None}
        with self._lock:
            merged = self.config.model_dump()
            merged.update(cleaned)
            merged.pop("auth_enabled", None)
            new_config = AppConfig.model_validate(merged)
            pose = self.models.pose if self.models else None
            source = self.source
            previous_source = self.config.camera_source
            self.config = new_config
        if pose is not None:
            pose.set_confidence_threshold(new_config.conf_threshold)
        self.state_manager.configure(
            new_config.piracy_threshold_sec,
            new_config.max_frame_dt,
            new_config.alert_grace_sec,
            new_config.incident_merge_window_s,
        )
        self.evidence.configure(new_config.preroll_s, new_config.postroll_s, new_config.clip_max_s)
        source.set_target_fps(new_config.target_fps)
        source.set_playback(speed=new_config.playback_speed, loop=new_config.loop_video)
        self.nir.set_mode(new_config.nir_mode)
        if "camera_source" in cleaned and cleaned["camera_source"] != previous_source:
            self.replace_source(
                open_source(
                    new_config.camera_source,
                    loop=new_config.loop_video,
                    speed=new_config.playback_speed,
                    width=new_config.frame_width,
                    height=new_config.frame_height,
                    target_fps=new_config.target_fps,
                    io_timeout=new_config.io_timeout,
                )
            )
        return new_config.public_dict()

    def update_nir_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate SIMULATED NIR tunables and apply them on the live detector."""
        public = self.nir.update_params(payload)
        if payload.get("mode") is not None:
            with self._lock:
                self.config = self.config.model_copy(update={"nir_mode": public["mode"]})
        return public

    def replace_source(self, source: FrameSource) -> None:
        source.start()
        with self._lock:
            previous = self.source
            self.source = source
            self._last_live_ts = None
            self._seat_mapper = None
            self._seat_shape = None
        previous.stop()

    def control_source(self, action: str, value: float | None = None) -> dict[str, Any]:
        try:
            self.source.control(action, value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return self.source.info()

    def _remember_error(self, message: str) -> None:
        self._errors.append(message)
        del self._errors[:-20]

    def _supervise(self) -> None:
        restarts: list[float] = []
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._process_frames()
                return
            except Exception as exc:
                logger.exception("processing loop failed")
                summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                with self._lock:
                    self._remember_error(summary)
                now = time.monotonic()
                restarts = [stamp for stamp in restarts if now - stamp < 60.0]
                restarts.append(now)
                self._restart_count += 1
                if len(restarts) > MAX_RESTARTS_PER_MINUTE:
                    halt = "processing supervisor halted after repeated failures"
                    logger.error(halt)
                    with self._lock:
                        self._remember_error(halt)
                        self._degraded = True
                        self._supervisor_halted = True
                    return
                if self._stop.wait(backoff):
                    return
                backoff = min(5.0, backoff * 2.0)

    def _process_frames(self) -> None:
        while not self._stop.is_set():
            if self._hold.is_set():
                self._stop.wait(0.02)
                continue
            with self._lock:
                source = self.source
                config = self.config
            frame = source.read()
            if frame is None:
                self._stop.wait(0.005)
                continue
            if frame.id == self._last_processed_id:
                continue
            if source.info().get("type") == "live" and self._last_live_ts is not None:
                min_interval = 1.0 / max(config.target_fps, 1)
                if frame.ts - self._last_live_ts < min_interval:
                    continue
            self._last_processed_id = frame.id
            if source.info().get("type") == "live":
                self._last_live_ts = frame.ts
            started = time.perf_counter()
            self._consume_frame(frame, config)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._publish_frame(frame, latency_ms)

    def _consume_frame(self, frame: Frame, config: AppConfig) -> None:
        image = frame.image
        height, width = image.shape[:2]
        self._ensure_seat_mapper(height, width)
        models = self.models
        candidates: list[Any] = []
        tracks: list[Any] = []
        targets: list[Any] = []
        if models and models.loaded and models.pose:
            candidates = models.pose.detect_recording_postures(image)
            tracks = models.tracker.update(
                [
                    TrackDetection(
                        index,
                        candidate.person_bbox,
                        max(candidate.elevation_confidence, 0.01),
                    )
                    for index, candidate in enumerate(candidates)
                ]
            )
            candidates = self._stabilize(models.pose, candidates, tracks)
            models.pose.forget_tracks({track.track_id for track in tracks})
        elif models and models.tracker:
            tracks = models.tracker.update([])
        glint_by_track = self.nir.analyze(image, candidates, tracks, config.nir_mode)
        targets = _verified_targets(candidates, tracks, glint_by_track, config.nir_mode)
        self._update_states(frame, config, candidates, tracks, targets, width, height, glint_by_track)

    def _stabilize(self, pose: PoseVectorEngine, candidates: list[Any], tracks: list[Any]) -> list[Any]:
        updated = list(candidates)
        for track in tracks:
            index = track.source_detection_id
            if index is None or index >= len(updated):
                continue
            candidate = updated[index]
            stable = pose.apply_track_stability(track.track_id, candidate.raw_elevated)
            updated[index] = replace(candidate, is_elevated=stable)
        return updated

    def _load_calibration(self) -> AuditoriumCalibration:
        path = _repo_path(self.config.calibration_path)
        try:
            loaded = load_calibration(path)
        except Exception as exc:
            logger.exception("calibration file rejected: %s", path)
            self._degraded = True
            self._remember_error(f"calibration file rejected: {exc}")
            return default_calibration(self.config.frame_width, self.config.frame_height)
        if loaded is None:
            return default_calibration(self.config.frame_width, self.config.frame_height)
        return loaded

    def _on_evidence(self, incident_id: int, meta: dict[str, Any]) -> None:
        self.incidents.set_evidence(incident_id, **meta)
        if self.evidence.degraded and self.evidence.error:
            self._degraded = True
            self._remember_error(self.evidence.error)

    def _ensure_seat_mapper(self, height: int, width: int) -> None:
        shape = (height, width)
        generation = id(self.calibration)
        if self._seat_shape == shape and self._seat_generation == generation and self._seat_mapper is not None:
            return
        if self._seat_shape is not None and self._seat_shape != shape:
            logger.warning("rebuilding seat mapper after resolution change to %s", shape)
        try:
            self._seat_mapper = AuditoriumSeatMapper(self.calibration, width, height)
        except Exception:
            logger.exception("seat mapper failed for frame %s", shape)
            self._degraded = True
            self._remember_error(f"seat mapper failed for {shape}")
            self._seat_mapper = None
        self._seat_shape = shape
        self._seat_generation = generation

    def put_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = AuditoriumCalibration.model_validate(payload).model_copy(update={"uncalibrated": False})
        save_calibration(_repo_path(self.config.calibration_path), model)
        with self._lock:
            self.calibration = model
            self._seat_mapper = None
            self._seat_shape = None
            self._seat_generation = None
        return model.model_dump()

    def probe_seat(self, x: float, y: float) -> dict[str, Any]:
        with self._lock:
            shape = self._seat_shape or (self.config.frame_height, self.config.frame_width)
        self._ensure_seat_mapper(shape[0], shape[1])
        if self._seat_mapper is None:
            return {"label": "Unmapped", "row": None, "seat": None, "confidence": 0.0, "in_bounds": False, "uncalibrated": True}
        result = self._seat_mapper.map_normalized(x, y)
        return _seat_payload(result)

    def _update_states(
        self,
        frame: Frame,
        config: AppConfig,
        candidates: list[Any],
        tracks: list[Any],
        targets: list[Any],
        width: int,
        height: int,
        glint_by_track: dict[int, Any] | None = None,
    ) -> None:
        self._last_frame_ts = frame.ts
        self._ensure_seat_mapper(height, width)
        self.evidence.push_frame(frame.ts, frame.image)
        active_ids = {track.track_id for track in tracks}
        track_rows: list[dict[str, Any]] = []
        openings: list[dict[str, Any]] = []
        closings: list[int] = []
        weights = ScoreWeights.from_mapping(config.score_weights())
        with self._lock:
            for track in tracks:
                candidate = _candidate_for_track(candidates, track)
                machine = self.state_manager.machine_for(track.track_id)
                glint = (glint_by_track or {}).get(track.track_id)
                seat_result = self._seat_for(track, candidate, width, height)
                foot = _foot_norm(track, candidate, width, height)
                suppressed = self.incidents.is_suppressed(seat_result.label, frame.ts)
                preview = score_track(
                    candidate, glint,
                    active_seconds=machine.active_seconds,
                    threshold_sec=config.piracy_threshold_sec,
                    nir_mode=config.nir_mode,
                    weights=weights,
                    allow_pose_only_alert=config.allow_pose_only_alert,
                    pose_only_score_cap=config.pose_only_score_cap,
                    stretch_brief_s=config.stretch_brief_s,
                )
                eligible = preview.alert_eligible and not suppressed
                machine.update(eligible, eligible, now=frame.ts)
                scored = score_track(
                    candidate, glint,
                    active_seconds=machine.active_seconds,
                    threshold_sec=config.piracy_threshold_sec,
                    nir_mode=config.nir_mode,
                    weights=weights,
                    allow_pose_only_alert=config.allow_pose_only_alert,
                    pose_only_score_cap=config.pose_only_score_cap,
                    stretch_brief_s=config.stretch_brief_s,
                )
                incident_id = self._incident_by_track.get(track.track_id)
                self.state_manager.remember(track.track_id, seat_result.label, foot, frame.ts, incident_id)
                if machine.state == PiracyState.PIRACY_ALERT and scored.alert_eligible and not suppressed:
                    openings.append({
                        "track_id": track.track_id,
                        "seat": seat_result.label,
                        "foot": foot,
                        "score": scored,
                        "state": machine.state.value,
                    })
                elif machine.state == PiracyState.IDLE and incident_id is not None:
                    closings.append(incident_id)
                    self._incident_by_track.pop(track.track_id, None)
                if machine.state == PiracyState.IDLE:
                    self._alerts.pop(track.track_id, None)
                else:
                    previous = self._alerts.get(track.track_id)
                    row = _alert_row(track.track_id, machine, seat_result.label, candidate, frame.ts, scored)
                    if previous is not None:
                        row["started_at"] = previous["started_at"]
                        row["since"] = previous["started_at"]
                    self._alerts[track.track_id] = row
                track_rows.append(
                    _track_row(track, candidate, machine, seat_result.label, width, height, config, glint, scored)
                )
            for track_id in self.state_manager.retire(active_ids, frame.ts):
                incident_id = self._incident_by_track.pop(track_id, None)
                self._alerts.pop(track_id, None)
                if incident_id is not None:
                    closings.append(incident_id)
            alerts = list(self._alerts.values())
            self._telemetry = self._build_telemetry(
                frame, config, track_rows, alerts, len(candidates), len(targets)
            )
        for item in openings:
            self._sync_open_incident(frame, config, item)
        for incident_id in closings:
            self.incidents.close(incident_id, frame.ts)
            self.evidence.finish(incident_id, frame.ts)
        if self.incidents.degraded:
            self._degraded = True
            if self.incidents.error:
                self._remember_error(self.incidents.error)
        with self._lock:
            listed = self.incidents.list_incidents(limit=20)
            self._telemetry["evidence"] = listed["items"]
            self._telemetry["calibration"] = {
                "name": self.calibration.name,
                "uncalibrated": self.calibration.uncalibrated,
            }
            self._telemetry["score_weights"] = config.score_weights()
            if self._degraded:
                self._telemetry["system"]["degraded"] = True

    def _sync_open_incident(self, frame: Frame, config: AppConfig, item: dict[str, Any]) -> None:
        scored: ScoreResult = item["score"]
        prefer = self._incident_by_track.get(item["track_id"])
        if prefer is None:
            prefer = self.state_manager.match_recent(item["track_id"], item["seat"], item["foot"], frame.ts)
        incident_id = self.incidents.open_or_update(
            now=frame.ts,
            track_id=item["track_id"],
            seat=item["seat"],
            score=scored.score,
            state=item["state"],
            breakdown=scored.breakdown,
            evidence_type=scored.evidence_type,
            nir_mode=config.nir_mode,
            synthetic=scored.synthetic,
            config_snapshot=config.public_dict(),
            merge_window_s=config.incident_merge_window_s,
            prefer_id=prefer,
        )
        if incident_id is None:
            return
        self._incident_by_track[item["track_id"]] = incident_id
        self.state_manager.remember(item["track_id"], item["seat"], item["foot"], frame.ts, incident_id)
        current = self.incidents.get(incident_id)
        if current is not None and current.get("evidence_status") in {None, "pending"}:
            self.evidence.begin(incident_id, frame.ts, item["seat"], config.nir_mode)

    def _seat_for(self, track: Any, candidate: Any, width: int, height: int) -> SeatResult:
        if self._seat_mapper is None:
            return UNMAPPED
        keypoints = getattr(candidate, "keypoints_dict", None) if candidate is not None else None
        confidence = getattr(candidate, "keypoint_confidence", None) if candidate is not None else None
        return self._seat_mapper.map_person(track.bbox, keypoints, confidence)

    def _publish_frame(self, frame: Frame, latency_ms: float) -> None:
        encoded = _encode_jpeg(frame.image)
        if not encoded:
            logger.error("jpeg encode failed for frame %s", frame.id)
            return
        now = time.perf_counter()
        self._proc_times.append(now)
        self._proc_times = [stamp for stamp in self._proc_times if now - stamp <= 2.0]
        proc_fps = round(len(self._proc_times) / 2.0, 1)
        with self._lock:
            system = self._telemetry["system"]
            system["proc_fps"] = proc_fps
            system["fps"] = proc_fps
            system["latency_ms"] = round(latency_ms, 1)
            metrics = _system_metrics(self)
            system["cpu"] = metrics["cpu"]
            system["cpu_percent"] = metrics["cpu"]
            system["mem"] = metrics["mem"]
            system["memory_percent"] = metrics["mem"]
            system["gpu"] = metrics["gpu"]
            system["gpu_percent"] = metrics["gpu"]
            snapshot = self._telemetry
        self.frames.publish(encoded)
        self.telemetry_hub.publish(snapshot)

    def _build_telemetry(
        self,
        frame: Frame,
        config: AppConfig,
        tracks: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
        pose_count: int,
        fused_count: int,
    ) -> dict[str, Any]:
        info = self.source.info()
        model_loaded = bool(self.models and self.models.loaded)
        system = {
            "proc_fps": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "cpu": None,
            "cpu_percent": None,
            "mem": None,
            "memory_percent": None,
            "gpu": None,
            "gpu_percent": None,
            "model_loaded": model_loaded,
            "degraded": self._degraded,
            "errors": list(self._errors),
        }
        return {
            "frame_id": frame.id,
            "ts": frame.ts,
            "source": {
                "type": info.get("type"),
                "name": info.get("name"),
                "fps": info.get("fps"),
                "paused": info.get("paused", False),
                "state": info.get("state"),
            },
            "system": system,
            "nir": self.nir.telemetry_block(),
            "tracks": tracks,
            "alerts": alerts,
            "active_tracking_targets": len(tracks),
            "active_alerts": alerts,
            "diagnostics": {
                "pose_candidates": pose_count,
                "fused_targets": fused_count,
                "processing_error": self._errors[-1] if self._errors else "",
                "model_loaded": model_loaded,
            },
            "evidence": [],
            "calibration": {
                "name": self.calibration.name,
                "uncalibrated": self.calibration.uncalibrated,
            },
            "score_weights": config.score_weights(),
            "timestamp": frame.ts,
        }


def _candidate_for_track(candidates: list[Any], track: Any) -> Any:
    index = track.source_detection_id
    if index is None or index >= len(candidates):
        return None
    return candidates[index]


def _alert_row(
    track_id: int,
    state: PiracyStateMachine,
    seat: str,
    candidate: Any,
    ts: float,
    scored: ScoreResult | None = None,
) -> dict[str, Any]:
    confidence = float(scored.score) if scored is not None else (
        float(candidate.elevation_confidence) if candidate is not None else 0.0
    )
    return {
        "track_id": track_id,
        "id": track_id,
        "state": state.state.value,
        "seat": seat,
        "since": ts,
        "started_at": ts,
        "score": round(confidence, 3),
        "confidence": round(confidence, 3),
        "duration": round(state.active_seconds, 1),
        "timestamp": ts,
        "behavior": scored.behavior if scored is not None else "idle",
        "synthetic": bool(scored.synthetic) if scored is not None else False,
        "evidence_type": scored.evidence_type if scored is not None else "",
    }


def _verified_targets(
    candidates: list[Any],
    tracks: list[Any],
    glint_by_track: dict[int, Any],
    mode: str,
) -> list[VerifiedTarget]:
    if mode == "off":
        return []
    verified: list[VerifiedTarget] = []
    for track in tracks:
        candidate = _candidate_for_track(candidates, track)
        result = glint_by_track.get(track.track_id)
        if candidate is None or not candidate.is_elevated or result is None:
            continue
        if not result.confirmed or result.best_score <= 0:
            continue
        verified.append(
            VerifiedTarget(
                person_id=candidate.person_id,
                bounding_box=result.roi or candidate.hand_bbox,
                fusion_confidence=float(np.clip(0.55 * candidate.elevation_confidence + 0.45 * result.ema_score, 0.0, 1.0)),
                has_lens_reflection=True,
                synthetic=any(glint.synthetic for glint in result.accepted),
            )
        )
    return verified


def _glint_payload(result: Any, width: int, height: int) -> list[dict[str, Any]]:
    if result is None:
        return []
    payload = []
    for glint in result.accepted:
        x1, y1, x2, y2 = glint.bbox
        payload.append({
            "bbox": [
                float(np.clip(x1 / width, 0.0, 1.0)) if width else 0.0,
                float(np.clip(y1 / height, 0.0, 1.0)) if height else 0.0,
                float(np.clip(x2 / width, 0.0, 1.0)) if width else 0.0,
                float(np.clip(y2 / height, 0.0, 1.0)) if height else 0.0,
            ],
            "score": round(float(glint.score), 3),
            "peak": int(glint.peak),
            "area": round(float(glint.area), 2),
            "synthetic": bool(glint.synthetic),
        })
    return payload


def _track_row(
    track: Any,
    candidate: Any,
    state: PiracyStateMachine,
    seat: str,
    width: int,
    height: int,
    config: AppConfig,
    glint: Any = None,
    scored: ScoreResult | None = None,
) -> dict[str, Any]:
    keypoints: dict[str, list[float]] = {}
    if candidate is not None:
        confidence = candidate.keypoint_confidence or {}
        for name, point in candidate.keypoints_dict.items():
            score = float(confidence.get(name, 0.0))
            keypoints[name] = [float(point[0]), float(point[1]), score]
    return {
        "id": track.track_id,
        "bbox": _normalize_bbox(track.bbox, width, height),
        "keypoints": keypoints,
        "state": state.state.value,
        "active_s": round(state.active_seconds, 3),
        "threshold_s": config.piracy_threshold_sec,
        "score": round(scored.score, 3) if scored is not None else (
            round(float(candidate.elevation_confidence), 3) if candidate is not None else 0.0
        ),
        "breakdown": scored.breakdown if scored is not None else {},
        "behavior": scored.behavior if scored is not None else "idle",
        "behavior_reason": scored.reason if scored is not None else "",
        "evidence_type": scored.evidence_type if scored is not None else "",
        "glints": _glint_payload(glint, width, height),
        "hand_bbox": _normalize_bbox(candidate.hand_bbox, width, height) if candidate is not None else None,
        "seat": seat,
    }


def _normalize_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    ]


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return b""
    return encoded.tobytes()


def _placeholder_jpeg() -> bytes:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "CINEGUARD CONNECTING",
        (150, 190),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (6, 182, 212),
        2,
    )
    return _encode_jpeg(image)


def _empty_telemetry(config: AppConfig) -> dict[str, Any]:
    return {
        "frame_id": 0,
        "ts": 0.0,
        "source": {"type": None, "name": None, "fps": None, "paused": False, "state": "stopped"},
        "system": {
            "proc_fps": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "cpu": None,
            "mem": None,
            "gpu": None,
            "model_loaded": False,
            "degraded": False,
            "errors": [],
        },
        "nir": {"mode": config.nir_mode, "simulated": config.nir_mode == "simulated"},
        "tracks": [],
        "alerts": [],
        "active_tracking_targets": 0,
        "active_alerts": [],
        "diagnostics": {"model_loaded": False, "pose_candidates": 0, "fused_targets": 0, "processing_error": ""},
        "evidence": [],
        "calibration": {"name": "uncalibrated", "uncalibrated": True},
        "score_weights": config.score_weights(),
        "timestamp": 0.0,
    }


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _seat_payload(result: SeatResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "row": result.row,
        "seat": result.seat,
        "confidence": round(result.confidence, 3),
        "in_bounds": result.in_bounds,
        "uncalibrated": result.uncalibrated,
    }


def _foot_norm(track: Any, candidate: Any, width: int, height: int) -> tuple[float, float] | None:
    if width <= 0 or height <= 0:
        return None
    keypoints = getattr(candidate, "keypoints_dict", None) if candidate is not None else None
    confidence = getattr(candidate, "keypoint_confidence", None) if candidate is not None else None
    x, y = foot_pixel(track.bbox, keypoints, confidence, width, height)
    return float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))


def _system_metrics(runtime: CommandCenterRuntime) -> dict[str, Any]:
    metrics: dict[str, Any] = {"cpu": None, "mem": None, "gpu": None}
    try:
        metrics["cpu"] = round(psutil.cpu_percent(interval=None), 1)
        metrics["mem"] = round(psutil.virtual_memory().percent, 1)
    except Exception:
        logger.exception("failed to read cpu or memory metrics")
    metrics["gpu"] = _gpu_utilization(runtime)
    return metrics


def _gpu_utilization(runtime: CommandCenterRuntime) -> float | None:
    """Return device utilization when the runtime can measure it, otherwise null."""
    if runtime._gpu_reader is False:
        runtime._gpu_reader = _bind_gpu_reader()
    reader = runtime._gpu_reader
    if reader is None:
        return None
    try:
        value = reader()
    except Exception:
        logger.warning("GPU utilization read failed; reporting null", exc_info=True)
        runtime._gpu_reader = None
        return None
    if value is None:
        return None
    return round(float(value), 1)


def _bind_gpu_reader() -> Any:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    utilization = getattr(torch.cuda, "utilization", None)
    if not callable(utilization):
        return None

    def read() -> float | None:
        value = utilization()
        return None if value is None else float(value)

    return read


def create_app(runtime: CommandCenterRuntime | None = None) -> FastAPI:
    """Build the ASGI app. ``uvicorn app:app`` uses the default runtime."""
    rt = runtime or CommandCenterRuntime()
    scenarios = ScenarioService(rt)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        rt.prepare()
        rt.bind_loop(asyncio.get_running_loop())
        rt.start()
        retention = asyncio.create_task(_retain_evidence(rt))
        try:
            yield
        finally:
            retention.cancel()
            rt.stop()

    application = FastAPI(title="CineGuard AI Command Center", version=APP_VERSION, lifespan=lifespan)
    application.state.runtime = rt

    @application.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return rt.health()

    @application.get("/api/v1/stream")
    async def stream_video(frames: int | None = None) -> StreamingResponse:
        if frames is not None and not 1 <= frames <= 30:
            raise HTTPException(status_code=422, detail="frames must be between 1 and 30")

        async def generator() -> AsyncGenerator[bytes, None]:
            last_seq = -1
            sent = 0
            while frames is None or sent < frames:
                seq, frame = await rt.frames.wait_change(last_seq, 1.0)
                if seq == last_seq:
                    continue
                last_seq = seq
                sent += 1
                header = (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                )
                yield header + frame + b"\r\n"

        return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @application.get("/api/v1/frame.jpg")
    async def latest_frame() -> Response:
        return Response(content=rt.frames.latest(), media_type="image/jpeg")

    @application.websocket("/ws/live_telemetry")
    async def live_telemetry(websocket: WebSocket) -> None:
        if rt.config.api_token and not tokens_match(rt.config.api_token, websocket.query_params.get("token")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        queue = rt.telemetry_hub.subscribe()
        try:
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            rt.telemetry_hub.unsubscribe(queue)

    @application.get("/api/v1/config")
    async def get_config() -> dict[str, Any]:
        return rt.config.public_dict()

    @application.post("/api/v1/config")
    async def update_config(update: ConfigUpdate) -> dict[str, Any]:
        try:
            return rt.apply_config(update.model_dump(exclude_none=True))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/source")
    async def set_source(request: SourceRequest) -> dict[str, Any]:
        kind = request.type.strip().lower()
        if kind not in {"live", "file", "webcam", "rtsp"}:
            raise HTTPException(status_code=422, detail="type must be live, file, webcam, or rtsp")
        loop = rt.config.loop_video if request.loop is None else request.loop
        speed = rt.config.playback_speed if request.speed is None else request.speed
        uri: int | str = request.uri
        if kind in {"live", "webcam"} and request.uri.strip().lstrip("-").isdigit():
            uri = int(request.uri.strip())
        if kind == "file":
            uri = request.uri
        try:
            config_update: dict[str, Any] = {
                "camera_source": uri,
                "loop_video": loop,
                "playback_speed": speed,
            }
            return rt.apply_config(config_update)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/source/control")
    async def control_source(request: SourceControl) -> dict[str, Any]:
        action = request.action.strip().lower()
        if action not in {"pause", "play", "seek", "step"}:
            raise HTTPException(status_code=422, detail="action must be pause, play, seek, or step")
        return rt.control_source(action, request.value)

    @application.get("/api/v1/nir/params")
    async def get_nir_params() -> dict[str, Any]:
        return rt.nir.public_params()

    @application.post("/api/v1/nir/params")
    async def update_nir_params(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return rt.update_nir_params(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/nir/frame.jpg")
    async def nir_frame(view: str = "nir", track: int | None = None) -> Response:
        if view not in {"nir", "mask", "heat"}:
            raise HTTPException(status_code=422, detail="view must be nir, mask, or heat")
        payload, error = rt.nir.frame_jpeg(view, track)
        if payload is None:
            raise HTTPException(status_code=404, detail=error or "no NIR frame yet")
        return Response(content=payload, media_type="image/jpeg")

    @application.get("/api/v1/nir/candidates")
    async def nir_candidates() -> dict[str, Any]:
        return rt.nir.candidates_payload()

    @application.get("/api/v1/nir/candidates/{index}.jpg")
    async def nir_candidate_thumbnail(index: int) -> Response:
        payload = rt.nir.thumbnail(index)
        if payload is None:
            raise HTTPException(status_code=404, detail="candidate thumbnail not found")
        return Response(content=payload, media_type="image/jpeg")

    @application.get("/api/v1/incidents/export.csv")
    async def export_incidents() -> Response:
        return Response(content=rt.incidents.export_csv(), media_type="text/csv")

    @application.get("/api/v1/incidents")
    async def list_incidents(
        status: str | None = None,
        seat: str | None = None,
        since: float | None = None,
        until: float | None = None,
        synthetic: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        try:
            return rt.incidents.list_incidents(
                status=status or None, seat=seat or None, since=since, until=until,
                synthetic=synthetic, limit=limit, offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/incidents/{incident_id}")
    async def get_incident(incident_id: int) -> dict[str, Any]:
        row = rt.incidents.get(incident_id)
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return row

    @application.post("/api/v1/incidents/{incident_id}/review")
    async def review_incident(incident_id: int, body: ReviewBody) -> dict[str, Any]:
        try:
            row = rt.incidents.review(
                incident_id,
                body.status.strip().lower(),
                body.note.strip(),
                rt._last_frame_ts,
                rt.config.suppress_after_dismiss_s,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return row

    @application.get("/api/v1/evidence/{incident_id}/keyframe.jpg")
    async def evidence_keyframe(incident_id: int) -> FileResponse:
        path, media = rt.evidence.file_for(incident_id, "keyframe")
        if path is None or media is None:
            raise HTTPException(status_code=404, detail="keyframe not found")
        return FileResponse(path, media_type=media)

    @application.get("/api/v1/evidence/{incident_id}/clip")
    async def evidence_clip(incident_id: int, download: int = 0) -> FileResponse:
        path, media = rt.evidence.file_for(incident_id, "clip")
        if path is None or media is None:
            raise HTTPException(status_code=404, detail="clip not found")
        filename = f"incident-{incident_id}{path.suffix}"
        disposition = "attachment" if download else "inline"
        return FileResponse(path, media_type=media, filename=filename, content_disposition_type=disposition)

    @application.get("/api/v1/evidence")
    async def get_evidence() -> list[dict[str, Any]]:
        return rt.incidents.list_incidents(limit=50)["items"]

    @application.get("/api/v1/calibration")
    async def get_calibration() -> dict[str, Any]:
        return rt.calibration.model_dump()

    @application.put("/api/v1/calibration")
    async def put_calibration(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return rt.put_calibration(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_context=False)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/api/v1/calibration/probe")
    async def probe_calibration(body: ProbeBody) -> dict[str, Any]:
        return rt.probe_seat(body.x, body.y)

    @application.get("/api/v1/demo_clips")
    async def demo_clips() -> dict[str, Any]:
        folder = ROOT / "demo_clips"
        names = []
        if folder.is_dir():
            names = sorted(
                path.name for path in folder.iterdir()
                if path.suffix.lower() in {".mp4", ".avi", ".webm", ".mov", ".mkv"}
            )
        return {"files": names}

    @application.get("/api/v1/logs")
    async def logs() -> dict[str, Any]:
        return {"lines": recent_logs()[-200:]}

    @application.get("/api/v1/scenarios/manifest")
    async def scenario_manifest() -> dict[str, Any]:
        file = ROOT / "demo_clips" / "manifest.json"
        if not file.is_file():
            raise HTTPException(status_code=404, detail="manifest not found")
        return json.loads(file.read_text(encoding="utf-8"))

    @application.post("/api/v1/scenarios/run")
    async def run_scenarios(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        raw = payload.get("manifest") or "demo_clips/manifest.json"
        try:
            path = _manifest_path(str(raw))
            return scenarios.start(path)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/scenarios/results")
    async def scenario_results() -> dict[str, Any]:
        return scenarios.snapshot()

    @application.get("/report", response_class=HTMLResponse)
    async def report() -> str:
        snapshot = scenarios.snapshot()
        payload = snapshot["results"] or {
            "manifest": "",
            "disclaimer": "SIMULATED NIR. No scenario run yet. This page is not a measurement.",
            "clips": [],
            "metrics": {},
            "metrics_recorded": {},
            "metrics_synthetic": {},
            "config": rt.config.public_dict(),
            "limitations": [],
            "future_work": [],
        }
        return render_report(payload)

    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    application.add_middleware(SecurityMiddleware, runtime=rt)
    origins = [item.strip() for item in rt.config.cors_origins.split(",") if item.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    return application


def _manifest_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(ROOT.resolve()):
        raise ValueError("manifest must stay inside the project")
    if candidate.suffix.lower() != ".json" or not candidate.is_file():
        raise ValueError("manifest not found")
    return candidate


async def _retain_evidence(runtime: CommandCenterRuntime) -> None:
    try:
        while True:
            await asyncio.to_thread(runtime.evidence.cleanup, runtime.config.evidence_retention_days)
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        return


app = create_app()
