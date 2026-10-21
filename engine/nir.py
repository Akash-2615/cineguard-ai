"""SIMULATED near-infrared stage and retroreflection glint detector.

There is no NIR camera in this proof of concept. ``NIRSimulator`` builds a
NIR-like mono image from a colour frame so the rest of the pipeline can be
demonstrated. ``mode="real"`` skips that conversion and expects an already
NIR or mono frame. Nothing produced here is a real-world accuracy measurement.

A lit phone screen is rejected on purpose. Real NIR also suppresses screens,
because a display emits very little near-infrared compared with a
retroreflector returning the illuminator beam.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from math import hypot, pi
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.logger import get_logger

logger = get_logger(__name__)

NirMode = Literal["off", "simulated", "real"]
Wavelength = Literal[850, 940]


class GlintWeights(BaseModel):
    """Weights for the retroreflection score. They are normalised at use time."""

    model_config = ConfigDict(extra="forbid")

    circularity: float = Field(default=0.25, ge=0.0, le=1.0)
    peak: float = Field(default=0.20, ge=0.0, le=1.0)
    contrast: float = Field(default=0.35, ge=0.0, le=1.0)
    convexity: float = Field(default=0.10, ge=0.0, le=1.0)
    aspect: float = Field(default=0.10, ge=0.0, le=1.0)

    def as_unit(self) -> dict[str, float]:
        raw = {
            "circularity": self.circularity,
            "peak": self.peak,
            "contrast": self.contrast,
            "convexity": self.convexity,
            "aspect": self.aspect,
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("at least one glint weight must be positive")
        return {key: value / total for key, value in raw.items()}


class NirParams(BaseModel):
    """Live tunables for the simulated illuminator and the glint detector."""

    model_config = ConfigDict(extra="forbid")

    mode: NirMode = "simulated"
    gamma: float = Field(default=1.3, ge=0.3, le=3.0)
    noise_floor: float = Field(default=0.0, ge=0.0, le=0.3)
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)
    wavelength_nm: Wavelength = 850
    vignette: float = Field(default=0.35, ge=0.0, le=1.0)
    gain: float = Field(default=1.0, ge=0.1, le=4.0)
    percentile: float = Field(default=98.5, ge=90.0, le=99.9)
    min_area: float = Field(default=3.0, gt=0.0, le=500.0)
    max_area: float = Field(default=80.0, gt=0.0, le=5000.0)
    max_area_ratio: float = Field(default=0.12, gt=0.0, le=0.5)
    circularity_min: float = Field(default=0.45, ge=0.0, le=1.0)
    contrast_min: float = Field(default=0.18, ge=0.0, le=1.0)
    clahe: bool = True
    clahe_clip: float = Field(default=2.0, ge=0.5, le=8.0)
    persistence_m: int = Field(default=3, ge=1, le=15)
    persistence_n: int = Field(default=5, ge=1, le=15)
    distance_gate: float = Field(default=0.5, ge=0.05, le=1.5)
    weights: GlintWeights = Field(default_factory=GlintWeights)
    synthetic_glint: bool = False
    synthetic_radius: float = Field(default=3.5, ge=1.0, le=30.0)
    synthetic_intensity: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(self) -> "NirParams":
        if self.persistence_m > self.persistence_n:
            raise ValueError("persistence_m cannot exceed persistence_n")
        if self.min_area > self.max_area:
            raise ValueError("min_area cannot exceed max_area")
        self.weights.as_unit()
        return self


PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "percentile": 99.4,
        "min_area": 6.0,
        "max_area": 40.0,
        "circularity_min": 0.7,
        "contrast_min": 0.4,
        "persistence_m": 4,
        "persistence_n": 5,
        "gamma": 1.5,
        "intensity": 0.55,
        "clahe": False,
    },
    "balanced": {
        "percentile": 98.5,
        "min_area": 3.0,
        "max_area": 80.0,
        "max_area_ratio": 0.12,
        "circularity_min": 0.45,
        "contrast_min": 0.18,
        "persistence_m": 3,
        "persistence_n": 5,
        "gamma": 1.3,
        "intensity": 0.7,
        "vignette": 0.35,
        "gain": 1.0,
        "clahe": True,
        "clahe_clip": 2.0,
        "weights": {
            "circularity": 0.25,
            "peak": 0.20,
            "contrast": 0.35,
            "convexity": 0.10,
            "aspect": 0.10,
        },
    },
    "sensitive": {
        "percentile": 96.0,
        "min_area": 2.0,
        "max_area": 140.0,
        "circularity_min": 0.32,
        "contrast_min": 0.1,
        "persistence_m": 2,
        "persistence_n": 4,
        "gamma": 1.0,
        "intensity": 0.85,
        "clahe": True,
        "clahe_clip": 3.0,
    },
}


@dataclass
class GlintCandidate:
    """One bright spot inside a hand ROI, accepted or rejected."""

    bbox: tuple[int, int, int, int]
    score: float
    peak: int
    area: float
    saturated: bool
    circularity: float
    convexity: float
    aspect: float
    contrast: float
    distance_to_hand: float
    size_ratio: float
    synthetic: bool
    accepted: bool
    rejection_reason: str | None = None

    def as_dict(self, width: int, height: int, index: int, track_id: int | None) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "index": index,
            "track_id": track_id,
            "bbox": _normalise_box(self.bbox, width, height),
            "score": round(self.score, 3),
            "peak": self.peak,
            "area": round(self.area, 2),
            "saturated": self.saturated,
            "circularity": round(self.circularity, 3),
            "convexity": round(self.convexity, 3),
            "aspect": round(self.aspect, 3),
            "contrast": round(self.contrast, 3),
            "distance_to_hand": round(self.distance_to_hand, 2),
            "size_ratio": round(self.size_ratio, 4),
            "synthetic": self.synthetic,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class GlintResult:
    """Detector output for one track on one frame."""

    candidates: list[GlintCandidate] = field(default_factory=list)
    best_score: float = 0.0
    confirmed: bool = False
    ema_score: float = 0.0
    mask: np.ndarray | None = None
    heat: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None

    @property
    def accepted(self) -> list[GlintCandidate]:
        return [candidate for candidate in self.candidates if candidate.accepted]


def empty_glint_result() -> GlintResult:
    return GlintResult()


class NIRSimulator:
    """Build a uint8 NIR-like image. Wavelength is a label, not a spectral model."""

    def convert(self, frame: np.ndarray, params: NirParams) -> np.ndarray | None:
        mono = _to_mono(frame, params.mode)
        if mono is None:
            return None
        if params.mode == "real":
            return mono
        if params.mode == "off":
            return mono
        worked = mono.astype(np.float32)
        # IR-pass sensors keep mostly the red channel. This is an illustration,
        # not a camera calibration.
        worked = np.clip(worked, 0.0, 255.0) / 255.0
        worked = np.power(worked, float(params.gamma))
        if params.noise_floor > 0:
            noise = np.random.default_rng(0).normal(0.0, params.noise_floor, worked.shape)
            worked = np.clip(worked + noise, 0.0, 1.0)
        worked = self._illuminate(worked, params)
        return np.clip(worked * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _illuminate(image: np.ndarray, params: NirParams) -> np.ndarray:
        """Radial falloff stands in for a ring of IR LEDs around the lens."""
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius /= max(float(radius.max()), 1.0)
        falloff = 1.0 - float(params.vignette) * radius
        level = (0.35 + 0.65 * float(params.intensity)) * float(params.gain)
        return np.clip(image * falloff * level, 0.0, 1.0)

    @staticmethod
    def inject(
        image: np.ndarray,
        points: list[tuple[float, float]],
        radius: float,
        intensity: float,
    ) -> np.ndarray:
        """Paint SIMULATED gaussian spots. Callers must mark detections synthetic."""
        if image is None or image.size == 0 or not points or intensity <= 0:
            return image
        canvas = image.astype(np.float32)
        height, width = canvas.shape[:2]
        yy, xx = np.ogrid[0:height, 0:width]
        sigma = max(float(radius), 1.0)
        for x, y in points:
            blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
            canvas += blob * float(intensity) * 255.0
        return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


class GlintDetector:
    """Find compact retroreflective spots inside one hand ROI at a time."""

    def __init__(self, params: NirParams | None = None) -> None:
        self.params = params or NirParams()
        self._history: dict[int, deque[bool]] = {}
        self._ema: dict[int, float] = {}

    def reset(self) -> None:
        self._history.clear()
        self._ema.clear()

    def forget_tracks(self, active_ids: set[int]) -> None:
        for track_id in list(self._history):
            if track_id not in active_ids:
                self._history.pop(track_id, None)
                self._ema.pop(track_id, None)

    def note_absent(self, track_id: int) -> GlintResult:
        confirmed, ema = self._advance(track_id, False, 0.0)
        return GlintResult(confirmed=confirmed, ema_score=ema)

    def detect(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        *,
        track_id: int = 0,
        mode: NirMode = "simulated",
        person_bbox: tuple[int, int, int, int] | None = None,
        injection_points: list[tuple[float, float]] | None = None,
        params: NirParams | None = None,
    ) -> GlintResult:
        settings = params or self.params
        if mode == "off":
            confirmed, ema = self._advance(track_id, False, 0.0)
            return GlintResult(confirmed=confirmed and False, ema_score=0.0, best_score=0.0)

        if not _finite_image(image):
            confirmed, ema = self._advance(track_id, False, 0.0)
            return GlintResult(confirmed=confirmed, ema_score=ema)

        height, width = image.shape[:2]
        bounds = roi if roi is not None else (0, 0, width, height)
        left, top, right, bottom = _clamp_box(bounds, width, height)
        if right - left < 3 or bottom - top < 3:
            confirmed, ema = self._advance(track_id, False, 0.0)
            return GlintResult(confirmed=confirmed, ema_score=ema, roi=(left, top, right, bottom))

        crop = image[top:bottom, left:right]
        heat, mask, candidates = self._extract(
            crop, settings, person_bbox, (left, top), injection_points or []
        )
        accepted = [candidate for candidate in candidates if candidate.accepted]
        best = max((candidate.score for candidate in accepted), default=0.0)
        near = False
        if accepted:
            hand_x = (left + right) / 2.0
            hand_y = (top + bottom) / 2.0
            gate = settings.distance_gate * hypot(right - left, bottom - top)
            near = any(
                hypot(
                    (candidate.bbox[0] + candidate.bbox[2]) / 2.0 - hand_x,
                    (candidate.bbox[1] + candidate.bbox[3]) / 2.0 - hand_y,
                )
                <= gate
                for candidate in accepted
            )
        confirmed, ema = self._advance(track_id, bool(accepted) and near, best)
        return GlintResult(
            candidates=candidates,
            best_score=best,
            confirmed=confirmed,
            ema_score=ema,
            mask=mask,
            heat=heat,
            roi=(left, top, right, bottom),
        )

    def _advance(self, track_id: int, present: bool, score: float) -> tuple[bool, float]:
        window = max(1, int(self.params.persistence_n))
        history = self._history.get(track_id)
        if history is None or history.maxlen != window:
            history = deque(maxlen=window)
            self._history[track_id] = history
        history.append(bool(present))
        previous = self._ema.get(track_id, 0.0)
        ema = 0.45 * float(score) + 0.55 * previous
        self._ema[track_id] = ema
        confirmed = sum(history) >= int(self.params.persistence_m) and len(history) >= int(self.params.persistence_m)
        return confirmed, ema

    def _extract(
        self,
        crop: np.ndarray,
        params: NirParams,
        person_bbox: tuple[int, int, int, int] | None,
        origin: tuple[int, int],
        injection_points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, list[GlintCandidate]]:
        original = crop if crop.ndim == 2 else crop[:, :, 0]
        gray = original
        if params.clahe and min(original.shape[:2]) >= 8:
            tile = 8 if min(original.shape[:2]) >= 16 else 2
            clahe = cv2.createCLAHE(clipLimit=float(params.clahe_clip), tileGridSize=(tile, tile))
            gray = clahe.apply(original)
        heat = _difference_of_gaussians(gray)
        peak_heat = float(heat.max()) if heat.size else 0.0
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if peak_heat < 4.0:
            return heat, mask, []
        percentile = float(np.percentile(heat, params.percentile))
        threshold = float(np.clip(min(percentile, peak_heat * 0.45), 6.0, 80.0))
        binary = np.where(heat >= threshold, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(gray.shape[0] * gray.shape[1])
        person_area = _box_area(person_bbox) if person_bbox is not None else roi_area
        hand_center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
        weights = params.weights.as_unit()
        found: list[GlintCandidate] = []
        for contour in contours:
            candidate = _score_contour(
                contour,
                original,
                params,
                weights,
                roi_area,
                person_area,
                hand_center,
                origin,
                injection_points,
            )
            if candidate is not None:
                found.append(candidate)
        found.sort(key=lambda item: item.score, reverse=True)
        return heat, mask, found


@dataclass
class NirDebug:
    nir: np.ndarray
    mask: np.ndarray
    heat: np.ndarray
    rois: dict[int, tuple[int, int, int, int]]
    candidates: list[dict[str, Any]]
    thumbnails: list[bytes]


class NIRLensPipeline:
    """SIMULATED NIR conversion plus per-track glint detection.

    The processing thread calls ``analyze``. API handlers only read the latest
    debug snapshot and replace ``params`` through ``update_params``.
    """

    def __init__(self) -> None:
        self.simulator = NIRSimulator()
        self.detector = GlintDetector()
        self._params = NirParams()
        self.detector.params = self._params
        self._lock = threading.Lock()
        self._debug: NirDebug | None = None
        self._logger = logger

    @property
    def params(self) -> NirParams:
        return self._params

    def set_mode(self, mode: str) -> None:
        if mode not in {"off", "simulated", "real"}:
            raise ValueError("nir mode must be off, simulated, or real")
        if self._params.mode == mode:
            return
        with self._lock:
            self._params = self._params.model_copy(update={"mode": mode})
            self.detector.params = self._params

    def update_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset_name = payload.get("preset")
        overrides = {key: value for key, value in payload.items() if key != "preset" and value is not None}
        with self._lock:
            current = self._params.model_dump()
            if preset_name is not None:
                if preset_name not in PRESETS:
                    raise ValueError("preset must be conservative, balanced, or sensitive")
                current.update(PRESETS[preset_name])
            current.update(overrides)
            updated = NirParams.model_validate(current)
            self._params = updated
            self.detector.params = updated
            self.detector.reset()
        self._logger.info("updated SIMULATED NIR params (mode=%s, preset=%s)", updated.mode, preset_name or "custom")
        return self.public_params(preset_name if not overrides else None)

    def public_params(self, preset: str | None = None) -> dict[str, Any]:
        data = self._params.model_dump()
        data["preset"] = preset or _matching_preset(self._params)
        data["simulated"] = self._params.mode == "simulated"
        data["note"] = "SIMULATED NIR is a software stand-in, not a calibrated camera."
        return data

    def telemetry_block(self) -> dict[str, Any]:
        params = self._params
        return {
            "mode": params.mode,
            "simulated": params.mode == "simulated",
            "synthetic_glint": params.synthetic_glint,
            "illuminator": {
                "intensity": params.intensity,
                "wavelength_nm": params.wavelength_nm,
                "vignette": params.vignette,
            },
        }

    def analyze(
        self,
        frame: np.ndarray,
        candidates: list[Any],
        tracks: list[Any],
        mode: str,
    ) -> dict[int, GlintResult]:
        """Convert one frame and score each tracked hand. ``mode`` comes from config."""
        with self._lock:
            if mode in {"off", "simulated", "real"} and self._params.mode != mode:
                self._params = self._params.model_copy(update={"mode": mode})
                self.detector.params = self._params
            params = self._params
            return self._analyze(frame, candidates, tracks, params)

    def _analyze(
        self,
        frame: np.ndarray,
        candidates: list[Any],
        tracks: list[Any],
        params: NirParams,
    ) -> dict[int, GlintResult]:
        nir = self.simulator.convert(frame, params)
        if nir is None:
            self._debug = None
            return {}
        injections: list[tuple[float, float]] = []
        if params.synthetic_glint and params.mode == "simulated":
            for track in tracks:
                candidate = _candidate_for(candidates, track)
                if candidate is not None and candidate.is_elevated:
                    injections.append(_box_center(candidate.hand_bbox))
            if injections:
                nir = self.simulator.inject(
                    nir, injections, params.synthetic_radius, params.synthetic_intensity
                )
        height, width = nir.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_heat = np.zeros((height, width), dtype=np.float32)
        rois: dict[int, tuple[int, int, int, int]] = {}
        results: dict[int, GlintResult] = {}
        public: list[dict[str, Any]] = []
        thumbnails: list[bytes] = []
        active: set[int] = set()
        for track in tracks:
            active.add(track.track_id)
            candidate = _candidate_for(candidates, track)
            if candidate is None:
                results[track.track_id] = self.detector.note_absent(track.track_id)
                continue
            roi = padded_roi(candidate.hand_bbox, width, height)
            result = self.detector.detect(
                nir,
                roi,
                track_id=track.track_id,
                mode=params.mode,
                person_bbox=candidate.person_bbox,
                injection_points=injections if candidate.is_elevated else [],
                params=params,
            )
            results[track.track_id] = result
            if result.roi is None:
                continue
            left, top, right, bottom = result.roi
            rois[track.track_id] = result.roi
            if result.mask is not None and result.mask.shape == (bottom - top, right - left):
                full_mask[top:bottom, left:right] = np.maximum(full_mask[top:bottom, left:right], result.mask)
            if result.heat is not None and result.heat.shape[:2] == (bottom - top, right - left):
                full_heat[top:bottom, left:right] = np.maximum(full_heat[top:bottom, left:right], result.heat)
            for glint in result.candidates:
                public.append(glint.as_dict(width, height, len(public), track.track_id))
                thumbnails.append(_thumbnail(nir, glint.bbox))
        self.detector.forget_tracks(active)
        self._debug = NirDebug(
            nir=nir,
            mask=full_mask,
            heat=full_heat,
            rois=rois,
            candidates=public,
            thumbnails=thumbnails,
        )
        return results

    def frame_jpeg(self, view: str, track_id: int | None = None) -> tuple[bytes | None, str | None]:
        with self._lock:
            debug = self._debug
        if debug is None:
            return None, "no NIR frame yet"
        image = {"nir": debug.nir, "mask": debug.mask, "heat": debug.heat}.get(view)
        if image is None:
            return None, "view must be nir, mask, or heat"
        if track_id is not None:
            roi = debug.rois.get(track_id)
            if roi is None:
                return None, f"track {track_id} not found"
            left, top, right, bottom = roi
            image = image[top:bottom, left:right]
        encoded = _encode_view(view, image)
        if not encoded:
            return None, "failed to encode NIR view"
        return encoded, None

    def candidates_payload(self) -> dict[str, Any]:
        with self._lock:
            debug = self._debug
        if debug is None:
            return {"candidates": [], "simulated": self._params.mode == "simulated"}
        return {
            "candidates": list(debug.candidates),
            "simulated": self._params.mode == "simulated",
            "mode": self._params.mode,
        }

    def thumbnail(self, index: int) -> bytes | None:
        with self._lock:
            debug = self._debug
        if debug is None or index < 0 or index >= len(debug.thumbnails):
            return None
        return debug.thumbnails[index]


def padded_roi(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    max_side: int = 160,
) -> tuple[int, int, int, int]:
    """Pad a hand box, then cap and clamp it so a huge detection cannot cover the frame."""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    width = max(1, right - left)
    height = max(1, bottom - top)
    pad_x = max(int(round(width * 0.15)), int(round(width * 0.75)))
    pad_y = max(int(round(height * 0.15)), int(round(height * 0.75)))
    left -= pad_x
    right += pad_x
    top -= pad_y
    bottom += pad_y
    cap_w = max(3, min(max_side, int(frame_width * 0.4)))
    cap_h = max(3, min(max_side, int(frame_height * 0.4)))
    if right - left > cap_w:
        center = (left + right) / 2.0
        left = int(round(center - cap_w / 2.0))
        right = left + cap_w
    if bottom - top > cap_h:
        center = (top + bottom) / 2.0
        top = int(round(center - cap_h / 2.0))
        bottom = top + cap_h
    return _clamp_box((left, top, right, bottom), frame_width, frame_height)


def _to_mono(frame: np.ndarray, mode: NirMode) -> np.ndarray | None:
    if not _finite_image(frame):
        return None
    if frame.ndim == 2:
        return _as_uint8(frame)
    if frame.ndim != 3 or frame.shape[2] == 1:
        return _as_uint8(frame[:, :, 0] if frame.ndim == 3 else frame)
    if mode == "simulated":
        blue = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        red = frame[:, :, 2].astype(np.float32)
        mixed = 0.05 * blue + 0.15 * green + 0.80 * red
        return np.clip(mixed, 0.0, 255.0).astype(np.uint8)
    # Real NIR frames are already mono. A 3-channel buffer is reduced to gray.
    gray = cv2.cvtColor(_as_uint8(frame), cv2.COLOR_BGR2GRAY)
    return gray


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    finite = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    max_value = float(np.max(finite)) if finite.size else 0.0
    if max_value <= 1.0:
        finite *= 255.0
    return np.clip(finite, 0.0, 255.0).astype(np.uint8)


def _finite_image(image: object) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return False
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        return False
    if not np.issubdtype(image.dtype, np.number):
        return False
    return bool(np.isfinite(image).all())


def _centre_surround(gray: np.ndarray, x: float, y: float) -> float:
    """Contrast of a small disk against a ring farther out, in 0..1."""
    height, width = gray.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    inner = gray[max(0, cy - 1): min(height, cy + 2), max(0, cx - 1): min(width, cx + 2)]
    y0, y1 = max(0, cy - 7), min(height, cy + 8)
    x0, x1 = max(0, cx - 7), min(width, cx + 8)
    outer = gray[y0:y1, x0:x1]
    if inner.size == 0 or outer.size == 0:
        return 0.0
    ring = np.ones(outer.shape, dtype=bool)
    iy, ix = max(0, cy - 1) - y0, max(0, cx - 1) - x0
    ring[iy: iy + inner.shape[0], ix: ix + inner.shape[1]] = False
    surround = outer[ring]
    if surround.size == 0:
        return 0.0
    return float(np.clip((float(inner.mean()) - float(surround.mean())) / 255.0, 0.0, 1.0))


def _difference_of_gaussians(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32)
    small = cv2.GaussianBlur(values, (0, 0), 0.8)
    large = cv2.GaussianBlur(values, (0, 0), 2.4)
    return np.clip(small - large, 0.0, None)


def _score_contour(
    contour: np.ndarray,
    gray: np.ndarray,
    params: NirParams,
    weights: dict[str, float],
    roi_area: float,
    person_area: float,
    hand_center: tuple[float, float],
    origin: tuple[int, int],
    injection_points: list[tuple[float, float]],
) -> GlintCandidate | None:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0 or area < 1.0:
        return None
    circularity = float(4.0 * pi * area / (perimeter * perimeter))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity = area / hull_area if hull_area > 0 else 0.0
    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    short_side = max(min(rect_w, rect_h), 1e-3)
    aspect = max(rect_w, rect_h) / short_side
    rectangularity = area / max(rect_w * rect_h, 1e-3)
    local = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(local, [contour], -1, 255, thickness=-1)
    pixels = gray[local > 0]
    if pixels.size == 0:
        return None
    peak = int(pixels.max())
    inner = float(pixels.mean())
    flatness = float(pixels.std()) / max(inner, 1.0)
    moments = cv2.moments(contour)
    if moments["m00"]:
        local_x = float(moments["m10"] / moments["m00"])
        local_y = float(moments["m01"] / moments["m00"])
    else:
        local_x, local_y = rect[0]
    contrast = _centre_surround(gray, local_x, local_y)
    bright_fraction = _bright_fraction(gray, local_x, local_y, peak)
    distance = hypot(local_x - hand_center[0], local_y - hand_center[1])
    box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
    frame_box = (
        origin[0] + box_x,
        origin[1] + box_y,
        origin[0] + box_x + box_w,
        origin[1] + box_y + box_h,
    )
    center_x = origin[0] + local_x
    center_y = origin[1] + local_y
    synthetic = any(hypot(center_x - point[0], center_y - point[1]) <= params.synthetic_radius * 2.5 for point in injection_points)
    reason = _rejection_reason(
        area,
        roi_area,
        circularity,
        convexity,
        aspect,
        rectangularity,
        flatness,
        contrast,
        bright_fraction,
        params,
    )
    aspect_score = float(np.clip(1.0 - (aspect - 1.0) / 2.5, 0.0, 1.0))
    score = float(np.clip(
        weights["circularity"] * min(circularity, 1.0)
        + weights["peak"] * (peak / 255.0)
        + weights["contrast"] * contrast
        + weights["convexity"] * min(convexity, 1.0)
        + weights["aspect"] * aspect_score,
        0.0,
        1.0,
    ))
    return GlintCandidate(
        bbox=frame_box,
        score=score,
        peak=peak,
        area=area,
        saturated=peak >= 250,
        circularity=circularity,
        convexity=convexity,
        aspect=aspect,
        contrast=contrast,
        distance_to_hand=distance,
        size_ratio=area / max(person_area, 1.0),
        synthetic=synthetic,
        accepted=reason is None,
        rejection_reason=reason,
    )


def _bright_fraction(gray: np.ndarray, x: float, y: float, peak: int) -> float:
    """Share of a wide window that is nearly as bright as the spot.

    A lens glint is a point. A phone-screen corner lights up a whole quadrant,
    so this fraction stays high. Real NIR suppresses screens for the same
    reason: the panel is an extended emitter, not a retroreflector.
    """
    height, width = gray.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    window = gray[max(0, cy - 10): min(height, cy + 11), max(0, cx - 10): min(width, cx + 11)]
    if window.size == 0 or peak <= 0:
        return 0.0
    level = max(float(peak) * 0.72, 40.0)
    return float(np.mean(window.astype(np.float32) >= level))


def _rejection_reason(
    area: float,
    roi_area: float,
    circularity: float,
    convexity: float,
    aspect: float,
    rectangularity: float,
    flatness: float,
    contrast: float,
    bright_fraction: float,
    params: NirParams,
) -> str | None:
    """Reject screens and stray shapes. There is no screen_like accept path."""
    if area < params.min_area:
        return "too_small"
    if bright_fraction > 0.12:
        return "bright_region"
    if area > params.max_area or area > roi_area * params.max_area_ratio:
        return "too_large"
    if aspect > 3.0:
        return "bad_aspect"
    if rectangularity >= 0.82 and circularity < 0.7:
        return "rectangular"
    if flatness < 0.06 and area >= 20:
        return "flat_intensity"
    if contrast < params.contrast_min:
        return "low_contrast"
    if circularity < params.circularity_min or convexity < 0.7:
        return "low_circularity"
    return None


def _matching_preset(params: NirParams) -> str:
    current = params.model_dump()
    for name, overrides in PRESETS.items():
        if all(current.get(key) == value or _same_model(current.get(key), value) for key, value in overrides.items()):
            return name
    return "custom"


def _same_model(current: Any, expected: Any) -> bool:
    if isinstance(expected, BaseModel):
        return current == expected.model_dump()
    return current == expected


def _candidate_for(candidates: list[Any], track: Any) -> Any:
    index = getattr(track, "source_detection_id", None)
    if index is None or index >= len(candidates):
        return None
    return candidates[index]


def _box_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_area(bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None:
        return 1.0
    return float(max(1, abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])))


def _clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = (int(value) for value in box)
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _normalise_box(box: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = box
    return [
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    ]


def _thumbnail(image: np.ndarray, box: tuple[int, int, int, int]) -> bytes:
    height, width = image.shape[:2]
    left, top, right, bottom = _clamp_box(box, width, height)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        crop = np.zeros((8, 8), dtype=np.uint8)
    return _encode_view("nir", crop)


def _encode_view(view: str, image: np.ndarray) -> bytes:
    if image is None or image.size == 0:
        return b""
    if view == "heat":
        finite = np.nan_to_num(image.astype(np.float32), nan=0.0)
        peak = float(finite.max()) if finite.size else 0.0
        scaled = np.zeros(finite.shape, dtype=np.uint8) if peak <= 0 else np.clip(finite / peak * 255.0, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        ok, encoded = cv2.imencode(".jpg", colour, [cv2.IMWRITE_JPEG_QUALITY, 82])
    elif view == "mask":
        gray = image if image.ndim == 2 else image[:, :, 0]
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 82])
    else:
        gray = image if image.ndim == 2 else image[:, :, 0]
        if gray.dtype != np.uint8:
            gray = _as_uint8(gray)
        ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return b""
    return encoded.tobytes()
