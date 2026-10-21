"""Application configuration for CineGuard AI."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NirMode = Literal["off", "simulated", "real"]


class AppConfig(BaseSettings):
    """Validated runtime configuration loaded from the environment and .env.

    ``nir_mode`` selects the NIR path. ``simulated`` is the default because this
    proof of concept has no NIR camera. Simulated output must never be reported
    as real-world accuracy.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    camera_source: Union[int, str] = 0
    target_fps: int = Field(default=30, gt=0)
    frame_width: int = Field(default=1280, gt=0)
    frame_height: int = Field(default=720, gt=0)
    piracy_threshold_sec: float = Field(default=30.0, gt=0)
    conf_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    io_timeout: float = Field(default=5.0, gt=0)
    nir_mode: NirMode = "simulated"
    loop_video: bool = True
    playback_speed: float = Field(default=1.0, ge=0.25, le=4.0)
    api_token: str | None = None
    demo_mode: bool = False
    pose_model_path: str = "models/yolov8n-pose.pt"
    allow_pose_only_alert: bool = False
    pose_only_score_cap: float = Field(default=0.6, ge=0.0, le=1.0)
    weight_pose: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_sightline: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_glint: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_persistence: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_stability: float = Field(default=0.10, ge=0.0, le=1.0)
    max_frame_dt: float = Field(default=1.0, gt=0)
    alert_grace_sec: float = Field(default=2.0, gt=0)
    stretch_brief_s: float = Field(default=2.0, gt=0)
    incident_merge_window_s: float = Field(default=120.0, ge=0)
    suppress_after_dismiss_s: float = Field(default=60.0, ge=0)
    preroll_s: float = Field(default=8.0, ge=0)
    postroll_s: float = Field(default=5.0, ge=0)
    clip_max_s: float = Field(default=30.0, gt=0)
    evidence_retention_days: int = Field(default=7, ge=1)
    evidence_dir: str = "evidence"
    db_path: str = "data/cineguard.db"
    calibration_path: str = "data/auditorium.json"
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000"
    max_body_bytes: int = Field(default=1_048_576, ge=0)
    rate_limit_per_minute: int = Field(default=120, ge=0)

    @field_validator("camera_source", mode="before")
    @classmethod
    def parse_camera_source(cls, value: object) -> object:
        """Keep RTSP and file paths as strings while converting numeric camera IDs."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)
            return stripped
        return value

    @field_validator("nir_mode", mode="before")
    @classmethod
    def normalize_nir_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("api_token", mode="before")
    @classmethod
    def empty_token_is_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def weights_must_be_positive(self) -> "AppConfig":
        total = (
            self.weight_pose
            + self.weight_sightline
            + self.weight_glint
            + self.weight_persistence
            + self.weight_stability
        )
        if total <= 0:
            raise ValueError("score weights must sum to a positive value")
        return self

    def score_weights(self) -> dict[str, float]:
        return {
            "pose": self.weight_pose,
            "sightline": self.weight_sightline,
            "glint": self.weight_glint,
            "persistence": self.weight_persistence,
            "stability": self.weight_stability,
        }

    def public_dict(self) -> dict:
        """Config safe to return from the API. The token itself is never included."""
        from security import mask_credentials
        data = self.model_dump()
        token = data.pop("api_token", None)
        data["auth_enabled"] = bool(token)
        data["camera_source"] = mask_credentials(data.get("camera_source"))
        return data
