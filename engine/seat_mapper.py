"""Map a person's foot point into a calibrated seat grid.

There is no silent clamping. A point that falls outside the grid by more than
``tolerance`` (a fraction of the grid width or height) is ``Unmapped``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from calibration import AuditoriumCalibration, LayoutSpec


@dataclass(frozen=True)
class SeatResult:
    label: str
    row: int | None
    seat: int | None
    confidence: float
    in_bounds: bool
    uncalibrated: bool = False


UNMAPPED = SeatResult("Unmapped", None, None, 0.0, False)


class AuditoriumSeatMapper:
    """Homography from normalized quad corners onto the seat grid."""

    def __init__(self, calibration: AuditoriumCalibration, width: int, height: int) -> None:
        if width < 2 or height < 2:
            raise ValueError("frame size must be at least 2x2")
        self.calibration = calibration
        self.width = int(width)
        self.height = int(height)
        self.layout = calibration.layout
        self.columns = _columns(self.layout)
        self.row_edges = _row_edges(self.layout)
        self.grid_width = float(len(self.columns))
        self.grid_height = float(self.row_edges[-1])
        source = np.array(
            [[x * self.width, y * self.height] for x, y in calibration.image_points],
            dtype=np.float32,
        )
        destination = np.array(
            [
                [0.0, 0.0],
                [self.grid_width, 0.0],
                [self.grid_width, self.grid_height],
                [0.0, self.grid_height],
            ],
            dtype=np.float32,
        )
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None or matrix.shape != (3, 3):
            raise ValueError("unable to compute homography")
        self.matrix = matrix.astype(np.float64)

    def map_person(
        self,
        bbox: tuple[int, int, int, int],
        keypoints: dict[str, tuple[float, float]] | None = None,
        confidence: dict[str, float] | None = None,
    ) -> SeatResult:
        x, y = foot_pixel(bbox, keypoints, confidence, self.width, self.height)
        return self._from_pixel(x, y)

    def map_normalized(self, x: float, y: float) -> SeatResult:
        return self._from_pixel(x * self.width, y * self.height)

    def map_bbox(self, bbox: tuple[int, int, int, int]) -> str:
        return self.map_person(bbox).label

    def _from_pixel(self, x: float, y: float) -> SeatResult:
        point = np.asarray([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.matrix.astype(np.float32))[0, 0]
        gx, gy = float(mapped[0]), float(mapped[1])
        tol_x = self.calibration.tolerance * self.grid_width
        tol_y = self.calibration.tolerance * self.grid_height
        if gx < -tol_x or gy < -tol_y or gx > self.grid_width + tol_x or gy > self.grid_height + tol_y:
            return self._unmapped()
        gx = min(max(gx, 0.0), self.grid_width - 1e-6)
        gy = min(max(gy, 0.0), self.grid_height - 1e-6)
        column = min(int(gx), len(self.columns) - 1)
        row = _row_index(self.row_edges, gy)
        seat_number = self.columns[column]
        if seat_number is None:
            return self._unmapped()
        if self.layout.seat_numbering == "right_to_left":
            seat_number = self.layout.seats_per_row + 1 - seat_number
        letter = self.layout.row_labels[row] if row < len(self.layout.row_labels) else str(row + 1)
        label = f"Row {letter}, Seat {seat_number}"
        if self.calibration.uncalibrated:
            label += " (uncalibrated)"
        return SeatResult(
            label=label,
            row=row,
            seat=seat_number,
            confidence=_cell_confidence(gx, gy, column, row, self.row_edges),
            in_bounds=True,
            uncalibrated=self.calibration.uncalibrated,
        )

    def _unmapped(self) -> SeatResult:
        return SeatResult("Unmapped", None, None, 0.0, False, uncalibrated=self.calibration.uncalibrated)


def foot_pixel(
    bbox: tuple[int, int, int, int],
    keypoints: dict[str, tuple[float, float]] | None,
    confidence: dict[str, float] | None,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Ankle midpoint when both ankles are confident, otherwise the hips, otherwise the bbox bottom-centre."""
    points = keypoints or {}
    scores = confidence or {}
    left = scores.get("left_ankle", 0.0)
    right = scores.get("right_ankle", 0.0)
    if left >= 0.5 and right >= 0.5 and "left_ankle" in points and "right_ankle" in points:
        return _pixel_mid(points["left_ankle"], points["right_ankle"], width, height)
    ankles_low = left < 0.5 and right < 0.5
    if (
        ankles_low
        and scores.get("left_hip", 0.0) >= 0.5
        and scores.get("right_hip", 0.0) >= 0.5
        and "left_hip" in points
        and "right_hip" in points
    ):
        return _pixel_mid(points["left_hip"], points["right_hip"], width, height)
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def _pixel_mid(
    a: tuple[float, float],
    b: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    return (a[0] + b[0]) * 0.5 * width, (a[1] + b[1]) * 0.5 * height


def _columns(layout: LayoutSpec) -> list[int | None]:
    """Seat number per column. ``None`` is an aisle inserted after the given seat."""
    columns: list[int | None] = []
    for seat in range(1, layout.seats_per_row + 1):
        columns.append(seat)
        if seat in layout.aisle_after_seats:
            columns.append(None)
    return columns


def _row_edges(layout: LayoutSpec) -> list[float]:
    edges = [0.0]
    for index in range(layout.rows):
        extra = layout.row_y_offsets[index] if layout.row_y_offsets else 0.0
        edges.append(edges[-1] + 1.0 + extra)
    return edges


def _row_index(edges: list[float], y: float) -> int:
    for index in range(len(edges) - 1):
        if y < edges[index + 1]:
            return index
    return len(edges) - 2


def _cell_confidence(x: float, y: float, column: int, row: int, edges: list[float]) -> float:
    center_x = column + 0.5
    center_y = (edges[row] + edges[row + 1]) / 2.0
    half_w = 0.5
    half_h = max(1e-6, (edges[row + 1] - edges[row]) / 2.0)
    dx = min(1.0, abs(x - center_x) / half_w)
    dy = min(1.0, abs(y - center_y) / half_h)
    return float(max(0.0, 1.0 - 0.5 * max(dx, dy)))


def calibration_payload(model: AuditoriumCalibration) -> dict[str, Any]:
    return model.model_dump()
