"""Auditorium calibration loaded from ``data/auditorium.json``.

``image_points`` are normalized (0–1) in order top-left, top-right, bottom-right,
bottom-left. Pixel size is applied at map time so a resolution change does not
invalidate the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class LayoutSpec(BaseModel):
    rows: int = Field(ge=1, le=40)
    seats_per_row: int = Field(ge=1, le=60)
    row_labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seat_numbering: Literal["left_to_right", "right_to_left"] = "left_to_right"
    aisle_after_seats: list[int] = Field(default_factory=list)
    row_y_offsets: list[float] = Field(default_factory=list)

    @field_validator("row_labels")
    @classmethod
    def labels_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("row_labels must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def check_aisles_and_offsets(self) -> "LayoutSpec":
        seen: set[int] = set()
        for seat in self.aisle_after_seats:
            if seat < 1 or seat >= self.seats_per_row:
                raise ValueError("aisle_after_seats entries must be between 1 and seats_per_row - 1")
            if seat in seen:
                raise ValueError("aisle_after_seats contains a duplicate")
            seen.add(seat)
        self.aisle_after_seats = sorted(seen)
        if self.row_y_offsets and len(self.row_y_offsets) != self.rows:
            raise ValueError("row_y_offsets must be empty or one value per row")
        if any(offset < 0 for offset in self.row_y_offsets):
            raise ValueError("row_y_offsets must be >= 0")
        return self


class AuditoriumCalibration(BaseModel):
    name: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    image_size: tuple[int, int]
    image_points: list[tuple[float, float]]
    layout: LayoutSpec
    tolerance: float = Field(default=0.08, ge=0.0, le=0.5)
    uncalibrated: bool = False

    @field_validator("image_size")
    @classmethod
    def size_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if width < 2 or height < 2:
            raise ValueError("image_size must be at least 2x2")
        return int(width), int(height)

    @model_validator(mode="after")
    def points_form_a_quad(self) -> "AuditoriumCalibration":
        points = self.image_points
        if len(points) != 4:
            raise ValueError("image_points must contain exactly 4 points")
        for x, y in points:
            if x < -1e-6 or y < -1e-6 or x > 1 + 1e-6 or y > 1 + 1e-6:
                raise ValueError("image_points are outside the image; use normalized coordinates in 0..1")
        if _polygon_area(points) < 1e-4:
            raise ValueError("image_points are degenerate (collinear or zero area)")
        if _edges_cross(points):
            raise ValueError("image_points are self-intersecting")
        return self


def default_calibration(width: int = 1280, height: int = 720) -> AuditoriumCalibration:
    """Full-frame 4x10 grid used until a real calibration is saved."""
    return AuditoriumCalibration(
        name="uncalibrated",
        camera_id="default",
        image_size=(width, height),
        image_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        layout=LayoutSpec(rows=4, seats_per_row=10, row_labels="ABCD"),
        tolerance=0.02,
        uncalibrated=True,
    )


def load_calibration(path: str | Path) -> AuditoriumCalibration | None:
    file = Path(path)
    if not file.is_file():
        return None
    payload = json.loads(file.read_text(encoding="utf-8"))
    model = AuditoriumCalibration.model_validate(payload)
    return model.model_copy(update={"uncalibrated": False})


def save_calibration(path: str | Path, model: AuditoriumCalibration) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    stored = model.model_copy(update={"uncalibrated": False})
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(stored.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(file)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _edges_cross(points: list[tuple[float, float]]) -> bool:
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            if len({a, b, c, d}) < 4:
                continue
            d1 = _cross(points[a], points[b], points[c])
            d2 = _cross(points[a], points[b], points[d])
            d3 = _cross(points[c], points[d], points[a])
            d4 = _cross(points[c], points[d], points[b])
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False
