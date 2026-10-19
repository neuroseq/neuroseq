"""Geometry of AOI shapes.

AOI coordinates are image pixel coordinates with a top-left origin (x = column,
y = row). Image views flip their camera (`FPLWidget.show_images_top_down`), so
world coordinates equal these and need no conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuroseq.importers.importer_cellpose import CircleAOI
    from neuroseq.importers.importer_cellpose import SquareAOI


def square_corners(square: SquareAOI) -> list[tuple[float, float]]:
    """Return square corners ordered TL, TR, BR, BL."""
    x0 = float(square.x)
    y0 = float(square.y)
    x1 = x0 + float(square.side)
    y1 = y0 + float(square.side)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def square_corner_points(square: SquareAOI, *, z: float = 220.0) -> np.ndarray:
    """Return square corners (TL, TR, BR, BL) as 3D points."""
    return np.asarray([(x, y, z) for x, y in square_corners(square)], dtype=np.float32)


def square_polyline(square: SquareAOI, *, z: float = 220.0) -> np.ndarray:
    """Return a closed square polyline."""
    corners = square_corner_points(square, z=z)
    return np.vstack([corners, corners[:1]])


def circle_polyline(
    circle: CircleAOI, *, z: float = 220.0, n_points: int = 72
) -> np.ndarray:
    """Return a closed circle polyline."""
    radius = float(circle.diameter) / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_points, endpoint=False)
    points = np.column_stack(
        [
            (float(circle.cx) + radius * np.cos(theta)).astype(np.float32),
            (float(circle.cy) + radius * np.sin(theta)).astype(np.float32),
            np.full(n_points, z, dtype=np.float32),
        ]
    )
    return np.vstack([points, points[:1]])


def circle_handles(circle: CircleAOI, *, z: float = 220.0) -> np.ndarray:
    """Return the cardinal handle points of a circle as 3D points."""
    radius = float(circle.diameter) / 2.0
    cx, cy = float(circle.cx), float(circle.cy)
    handle_points = [
        (cx - radius, cy),
        (cx, cy - radius),
        (cx + radius, cy),
        (cx, cy + radius),
    ]
    return np.asarray([(x, y, z) for x, y in handle_points], dtype=np.float32)
