"""Vector-based elevated-device pose detection for auditorium footage."""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger


@dataclass(frozen=True)
class PoseCandidate:
    """A structured elevated-hand candidate emitted by the pose engine.

    ``raw_elevated`` is the single-frame pose decision. ``is_elevated`` becomes
    the track-stable decision only after ``apply_track_stability``.
    """

    person_id: int
    hand_bbox: Tuple[int, int, int, int]
    elevation_confidence: float
    is_elevated: bool
    keypoints_dict: Dict[str, Tuple[float, float]]
    person_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    raw_elevated: bool = False
    keypoint_confidence: Dict[str, float] | None = None


class PoseVectorEngine:
    """YOLO pose inference plus confidence-gated 2D pose geometry.

    The screen plane is represented by the image horizontal axis. This is a
    deliberately conservative 2D heuristic: without camera calibration or a
    screen landmark, screen-facing alignment cannot be recovered metrically.
    """

    KEYPOINT_NAMES = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    REQUIRED_CONFIDENCE = 0.5
    _WRIST_INDICES = (9, 10)
    _ELBOW_INDICES = (7, 8)
    _SHOULDER_INDICES = (5, 6)

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.5,
        stability_window: int = 5,
        stable_frames: int = 3,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if stability_window < 1 or stable_frames < 1 or stable_frames > stability_window:
            raise ValueError("stable_frames must be within stability_window")

        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.stable_frames = stable_frames
        self._logger = get_logger(__name__)
        self._inference_lock = threading.Lock()
        self._stability: Dict[int, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.stability_window)
        )
        self.device = "cpu"
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        """Push a live config change into the next inference call."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold

    def apply_track_stability(self, track_id: int, raw_positive: bool) -> bool:
        """Record one frame of pose evidence for a track id and return stability.

        Histories are keyed by tracker id. A detection index must never be used
        here, because that index changes whenever people enter or leave.
        """
        history = self._stability[int(track_id)]
        history.append(bool(raw_positive))
        return sum(history) >= self.stable_frames and len(history) >= self.stable_frames

    def clear_stability(self) -> None:
        """Drop every track history so a new clip does not inherit the previous one."""
        self._stability.clear()

    def forget_tracks(self, active_ids: set[int]) -> None:
        """Drop stability history for track ids the tracker has retired."""
        for track_id in list(self._stability):
            if track_id not in active_ids:
                del self._stability[track_id]

    def _load_model(self, model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to use PoseVectorEngine") from exc

        candidates = self._device_candidates()
        last_error: Optional[Exception] = None
        for device in candidates:
            try:
                model = YOLO(model_path)
                if device == "openvino":
                    model = self._load_openvino_model(model, model_path)
                else:
                    model.to(device)
                self.device = device
                self._logger.info("Loaded pose model %s on %s", model_path, device)
                return model
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                self._logger.warning("Pose backend %s unavailable: %s", device, exc)

        raise RuntimeError(f"Unable to load pose model {model_path}") from last_error

    @staticmethod
    def _device_candidates() -> List[str]:
        candidates: List[str] = []
        try:
            import torch

            if torch.cuda.is_available():
                candidates.append("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                candidates.append("mps")
        except ImportError:
            pass
        try:
            import openvino  # noqa: F401
            candidates.append("openvino")
        except ImportError:
            pass
        candidates.append("cpu")
        return candidates

    @staticmethod
    def _load_openvino_model(model: Any, model_path: str) -> Any:
        exported = Path(model_path).with_suffix("")
        openvino_path = Path(f"{exported}_openvino_model")
        if not openvino_path.exists():
            exported_result = model.export(format="openvino")
            openvino_path = Path(str(exported_result))
        from ultralytics import YOLO

        return YOLO(str(openvino_path))

    def detect_recording_postures(self, frame: np.ndarray) -> List[PoseCandidate]:
        """Detect hands held steadily above shoulders and aligned with sightline."""
        if not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return []

        with self._inference_lock:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                # Ultralytics selects the OpenVINO runtime from the model
                # itself; passing "openvino" to torch device selection fails.
                device="cpu" if self.device == "openvino" else self.device,
                verbose=False,
            )

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        box_data = self._as_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
        box_conf = self._as_numpy(getattr(boxes, "conf", np.ones(len(box_data))))
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "data", None) is None:
            data = np.empty((0, 17, 3))
        else:
            data = self._as_numpy(keypoints.data)
        if data.ndim != 3 or data.shape[1] < 17 or data.shape[2] < 2:
            data = np.empty((0, 17, 3))

        candidates: List[PoseCandidate] = []
        person_count = max(len(box_data), len(data))
        for person_id in range(person_count):
            person_points = data[person_id] if person_id < len(data) else np.empty((17, 3))
            person_bbox = self._box_tuple(box_data[person_id], width, height) if person_id < len(box_data) else (0, 0, width, height)
            candidate = self._candidate_for_person(
                person_id, person_points, width, height, person_bbox,
                float(box_conf[person_id]) if person_id < len(box_conf) else 0.0,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _candidate_for_person(
        self, person_id: int, points: np.ndarray, width: int, height: int,
        person_bbox: Tuple[int, int, int, int], person_confidence: float,
    ) -> Optional[PoseCandidate]:
        normalized: Dict[str, Tuple[float, float]] = {}
        confidence: Dict[str, float] = {}
        for index, name in enumerate(self.KEYPOINT_NAMES):
            if index >= len(points):
                continue
            x, y = float(points[index][0]), float(points[index][1])
            score = float(points[index][2]) if points.shape[1] > 2 else 1.0
            if not np.isfinite([x, y, score]).all():
                continue
            normalized[name] = (
                float(np.clip(x / width, 0.0, 1.0)),
                float(np.clip(y / height, 0.0, 1.0)),
            )
            confidence[name] = score

        shoulders = self._mean_point(
            normalized, confidence, ("left_shoulder", "right_shoulder")
        )
        eyes = self._mean_point(
            normalized, confidence, ("left_eye", "right_eye", "nose")
        )
        if shoulders is None or eyes is None:
            return PoseCandidate(
                person_id, person_bbox, 0.0, False, normalized, person_bbox,
                raw_elevated=False, keypoint_confidence=confidence,
            )

        wrist_candidates: List[Tuple[float, str]] = []
        for wrist_name, elbow_name in zip(
            ("left_wrist", "right_wrist"), ("left_elbow", "right_elbow")
        ):
            if (
                wrist_name in normalized
                and elbow_name in normalized
                and confidence[wrist_name] >= self.REQUIRED_CONFIDENCE
                and confidence[elbow_name] >= self.REQUIRED_CONFIDENCE
            ):
                wrist_candidates.append((confidence[wrist_name], wrist_name))
        if not wrist_candidates:
            return PoseCandidate(
                person_id, person_bbox, float(np.clip(person_confidence, 0.0, 1.0)),
                False, normalized, person_bbox,
                raw_elevated=False, keypoint_confidence=confidence,
            )

        _, wrist_name = max(wrist_candidates)
        elbow_name = "left_elbow" if wrist_name == "left_wrist" else "right_elbow"
        wrist = normalized[wrist_name]
        elbow = normalized[elbow_name]
        wrist_score = confidence[wrist_name]
        shoulder_y = shoulders[1]
        dx, dy = wrist[0] - elbow[0], elbow[1] - wrist[1]
        angle = math.degrees(math.atan2(max(dy, 0.0), abs(dx) + 1e-6))
        above_shoulder = wrist[1] < shoulder_y - 0.02
        angle_score = float(np.clip(angle / 90.0, 0.0, 1.0))
        sightline_score = self._sightline_score(wrist, eyes, shoulders)
        mouth_or_head = self._near_mouth_or_head(wrist, normalized, shoulders)
        likely_chair_rest = wrist[1] >= shoulder_y - 0.02 or angle < 20.0
        raw_positive = (
            above_shoulder
            and angle >= 25.0
            and sightline_score >= 0.45
            and not mouth_or_head
            and not likely_chair_rest
        )
        elevation_confidence = float(
            np.clip(
                0.35 * wrist_score
                + 0.25 * angle_score
                + 0.25 * sightline_score
                + 0.15 * (1.0 if above_shoulder else 0.0),
                0.0,
                1.0,
            )
        )
        bbox = self._hand_bbox(wrist, elbow, width, height)
        return PoseCandidate(
            person_id=person_id,
            hand_bbox=bbox,
            elevation_confidence=elevation_confidence,
            is_elevated=raw_positive,
            keypoints_dict=normalized,
            person_bbox=person_bbox,
            raw_elevated=raw_positive,
            keypoint_confidence=confidence,
        )

    @staticmethod
    def _box_tuple(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        values = np.asarray(box, dtype=float).reshape(-1)[:4]
        if len(values) < 4 or not np.isfinite(values).all():
            return 0, 0, width, height
        x1, y1, x2, y2 = values
        return (
            max(0, min(width - 1, int(x1))),
            max(0, min(height - 1, int(y1))),
            max(0, min(width, int(x2))),
            max(0, min(height, int(y2))),
        )

    @staticmethod
    def _mean_point(
        points: Dict[str, Tuple[float, float]],
        confidence: Dict[str, float],
        names: Tuple[str, ...],
    ) -> Optional[Tuple[float, float]]:
        valid = [points[name] for name in names if name in points and confidence[name] >= 0.5]
        if not valid:
            return None
        return tuple(np.mean(valid, axis=0).tolist())  # type: ignore[return-value]

    @staticmethod
    def _sightline_score(
        wrist: Tuple[float, float],
        eyes: Tuple[float, float],
        shoulders: Tuple[float, float],
    ) -> float:
        # In the camera view, a held device should remain near the eye-to-screen
        # corridor and not be far outside the person's torso width.
        eye_line = (wrist[0] - eyes[0], wrist[1] - eyes[1])
        dx = abs(eye_line[0])
        vertical = max(0.0, -eye_line[1])
        torso_width = max(0.06, abs(shoulders[0] - eyes[0]) * 2.0)
        horizontal_score = float(np.clip(1.0 - dx / (torso_width * 2.0), 0.0, 1.0))
        vertical_score = float(np.clip(vertical / 0.18, 0.0, 1.0))
        return 0.55 * horizontal_score + 0.45 * vertical_score

    @staticmethod
    def _near_mouth_or_head(
        wrist: Tuple[float, float],
        points: Dict[str, Tuple[float, float]],
        shoulders: Tuple[float, float],
    ) -> bool:
        nose = points.get("nose")
        if nose is not None and math.dist(wrist, nose) < 0.07:
            return True
        head_y = min((points[name][1] for name in ("nose", "left_eye", "right_eye") if name in points), default=0.0)
        shoulder_width = max(0.06, abs(shoulders[0] - points.get("nose", shoulders)[0]) * 2.0)
        return wrist[1] < head_y + 0.03 and abs(wrist[0] - shoulders[0]) < shoulder_width

    @staticmethod
    def _hand_bbox(
        wrist: Tuple[float, float],
        elbow: Tuple[float, float],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        radius_x = max(12, int(abs(wrist[0] - elbow[0]) * width * 0.35))
        radius_y = max(12, int(abs(wrist[1] - elbow[1]) * height * 0.35))
        x = int(wrist[0] * width)
        y = int(wrist[1] * height)
        left = max(0, x - radius_x)
        top = max(0, y - radius_y)
        right = min(width - 1, x + radius_x)
        bottom = min(height - 1, y + radius_y)
        return left, top, right, bottom
