from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuroseq.importer_cellpose import CircleAOI
    from neuroseq.importer_cellpose import SquareAOI


def user_to_world_point(
    user_x: float, user_y: float, image_height: int
) -> tuple[float, float]:
    """Convert top-left-origin user coordinates into bottom-left world coordinates."""
    return float(user_x), float(image_height - user_y)


def world_to_user_point(
    world_x: float, world_y: float, image_height: int
) -> tuple[float, float]:
    """Convert bottom-left-origin world coordinates into top-left user coordinates."""
    return float(world_x), float(image_height - world_y)


def square_user_corners(square: SquareAOI) -> list[tuple[float, float]]:
    """Return square corners in user coordinates ordered TL, TR, BR, BL."""
    x0 = float(square.x)
    y0 = float(square.y)
    x1 = x0 + float(square.side)
    y1 = y0 + float(square.side)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def square_world_polyline(
    square: SquareAOI, image_height: int, *, z: float = 220.0
) -> np.ndarray:
    """Return a closed square polyline in world coordinates."""
    corners = square_world_corners(square, image_height, z=z)
    return np.vstack([corners, corners[:1]])


def square_world_corners(
    square: SquareAOI, image_height: int, *, z: float = 220.0
) -> np.ndarray:
    """Return square corners in world coordinates ordered TL, TR, BR, BL."""
    corners = [
        (*user_to_world_point(x, y, image_height), z)
        for x, y in square_user_corners(square)
    ]
    return np.asarray(corners, dtype=np.float32)


def circle_world_polyline(
    circle: CircleAOI,
    image_height: int,
    *,
    z: float = 220.0,
    n_points: int = 72,
) -> np.ndarray:
    """Return a closed circle polyline in world coordinates."""
    radius = float(circle.diameter) / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_points, endpoint=False)
    user_x = float(circle.cx) + radius * np.cos(theta)
    user_y = float(circle.cy) + radius * np.sin(theta)
    world_y = image_height - user_y
    points = np.column_stack(
        [
            user_x.astype(np.float32),
            world_y.astype(np.float32),
            np.full(n_points, z, dtype=np.float32),
        ]
    )
    return np.vstack([points, points[:1]])


def circle_world_handles(
    circle: CircleAOI, image_height: int, *, z: float = 220.0
) -> np.ndarray:
    """Return the cardinal handle points for a circle in world coordinates."""
    radius = float(circle.diameter) / 2.0
    handle_points = [
        (float(circle.cx) - radius, float(circle.cy)),
        (float(circle.cx), float(circle.cy) - radius),
        (float(circle.cx) + radius, float(circle.cy)),
        (float(circle.cx), float(circle.cy) + radius),
    ]
    return np.asarray(
        [(*user_to_world_point(x, y, image_height), z) for x, y in handle_points],
        dtype=np.float32,
    )
