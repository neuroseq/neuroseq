from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def generate_cellpose_test_grid(
    *,
    rows: int = 10,
    cols: int = 10,
    radius: int = 10,
    gap: int = 10,
    square_size: int | None = None,
    preserve_radius_gap: bool = False,
    preserve_radius_auto_gap: bool = False,
    edge_padding: int | None = None,
    background_intensity: int = 0,
    cell_intensity: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic grid of circular Cellpose targets.

    Returns:
        Tuple of ``(image, segmentation)`` where:
        - ``image`` is a uint8 grayscale image
        - ``segmentation`` is an int32 mask with 0 as background and
          row-major cell IDs starting at 1
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if square_size is not None and square_size <= 0:
        raise ValueError("square_size must be positive")
    if edge_padding is None:
        edge_padding = radius
    if edge_padding < 0:
        raise ValueError("edge_padding must be >= 0")
    if not 0 <= background_intensity <= 255:
        raise ValueError("background_intensity must be in [0, 255]")
    if not 0 <= cell_intensity <= 255:
        raise ValueError("cell_intensity must be in [0, 255]")
    if background_intensity == cell_intensity:
        raise ValueError("background_intensity and cell_intensity must differ")
    if preserve_radius_gap and preserve_radius_auto_gap:
        raise ValueError(
            "preserve_radius_gap and preserve_radius_auto_gap are mutually exclusive"
        )

    if square_size is None:
        pitch = (2 * radius) + gap
        height = (2 * edge_padding) + (2 * radius) + ((rows - 1) * pitch)
        width = (2 * edge_padding) + (2 * radius) + ((cols - 1) * pitch)
        top_padding = edge_padding
        left_padding = edge_padding
    else:
        if preserve_radius_gap:
            top_padding, left_padding = _center_fixed_grid_geometry_for_square_canvas(
                square_size,
                rows=rows,
                cols=cols,
                radius=radius,
                gap=gap,
            )
        elif preserve_radius_auto_gap:
            (
                fitted_gap,
                top_padding,
                left_padding,
            ) = _fit_gap_for_fixed_radius_square_canvas(
                square_size,
                rows=rows,
                cols=cols,
                radius=radius,
            )
            gap = fitted_gap
        else:
            (
                fitted_radius,
                fitted_gap,
                top_padding,
                left_padding,
            ) = _fit_grid_geometry_for_square_canvas(
                square_size,
                rows=rows,
                cols=cols,
                preferred_radius=radius,
                preferred_gap=gap,
            )
            radius = fitted_radius
            gap = fitted_gap
        pitch = (2 * radius) + gap
        height = square_size
        width = square_size

    image = np.full((height, width), background_intensity, dtype=np.uint8)
    segmentation = np.zeros((height, width), dtype=np.int32)

    yy, xx = np.ogrid[:height, :width]

    cell_id = 1
    for row in range(rows):
        center_y = top_padding + radius + (row * pitch)
        for col in range(cols):
            center_x = left_padding + radius + (col * pitch)
            mask = ((yy - center_y) ** 2) + ((xx - center_x) ** 2) <= (radius**2)
            image[mask] = cell_intensity
            segmentation[mask] = cell_id
            cell_id += 1

    return image, segmentation


def _fit_grid_geometry_for_square_canvas(
    square_size: int,
    *,
    rows: int,
    cols: int,
    preferred_radius: int,
    preferred_gap: int,
) -> tuple[int, int, int, int]:
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if preferred_radius <= 0:
        raise ValueError("preferred_radius must be positive")
    if preferred_gap < 0:
        raise ValueError("preferred_gap must be >= 0")

    max_cells_axis = max(rows, cols)
    if square_size < 2 * max_cells_axis:
        raise ValueError(
            "square_size is too small for the requested rows/cols "
            f"({rows}x{cols}) with at least radius=1"
        )

    preferred_gap_ratio = preferred_gap / float(2 * preferred_radius)
    max_radius = square_size // (2 * max_cells_axis)
    if max_radius < 1:
        raise ValueError("square_size is too small to fit requested grid")

    fitted_radius: int | None = None
    fitted_gap: int | None = None
    for candidate_radius in range(max_radius, 0, -1):
        if max_cells_axis == 1:
            max_gap_for_radius = 0
        else:
            max_gap_for_radius = (
                square_size - (2 * candidate_radius * max_cells_axis)
            ) // (max_cells_axis - 1)
        if max_gap_for_radius < 0:
            continue
        preferred_gap_for_radius = int(
            round((2 * candidate_radius) * preferred_gap_ratio)
        )
        candidate_gap = min(preferred_gap_for_radius, max_gap_for_radius)
        candidate_gap = max(0, candidate_gap)
        fitted_radius = candidate_radius
        fitted_gap = candidate_gap
        break

    if fitted_radius is None or fitted_gap is None:
        raise ValueError("Could not fit requested grid into square canvas")

    x_extent = (2 * fitted_radius * cols) + (fitted_gap * (cols - 1))
    y_extent = (2 * fitted_radius * rows) + (fitted_gap * (rows - 1))
    left_padding = (square_size - x_extent) // 2
    top_padding = (square_size - y_extent) // 2
    if left_padding < 0 or top_padding < 0:
        raise ValueError("Internal error while fitting square-grid geometry")
    return fitted_radius, fitted_gap, top_padding, left_padding


def _center_fixed_grid_geometry_for_square_canvas(
    square_size: int,
    *,
    rows: int,
    cols: int,
    radius: int,
    gap: int,
) -> tuple[int, int]:
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    x_extent = (2 * radius * cols) + (gap * (cols - 1))
    y_extent = (2 * radius * rows) + (gap * (rows - 1))
    if x_extent > square_size or y_extent > square_size:
        raise ValueError(
            "square_size is too small for requested fixed geometry "
            f"({rows}x{cols}, radius={radius}, gap={gap})"
        )

    left_padding = (square_size - x_extent) // 2
    top_padding = (square_size - y_extent) // 2
    return top_padding, left_padding


def _fit_gap_for_fixed_radius_square_canvas(
    square_size: int,
    *,
    rows: int,
    cols: int,
    radius: int,
) -> tuple[int, int, int]:
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    x_base_extent = 2 * radius * cols
    y_base_extent = 2 * radius * rows
    if x_base_extent > square_size or y_base_extent > square_size:
        raise ValueError(
            "square_size is too small for requested fixed radius geometry "
            f"({rows}x{cols}, radius={radius})"
        )

    gap_candidates: list[int] = []
    if cols > 1:
        gap_candidates.append((square_size - x_base_extent) // (cols - 1))
    if rows > 1:
        gap_candidates.append((square_size - y_base_extent) // (rows - 1))
    fitted_gap = min(gap_candidates) if gap_candidates else 0
    fitted_gap = max(0, fitted_gap)

    x_extent = x_base_extent + (fitted_gap * (cols - 1))
    y_extent = y_base_extent + (fitted_gap * (rows - 1))
    left_padding = (square_size - x_extent) // 2
    top_padding = (square_size - y_extent) // 2
    if left_padding < 0 or top_padding < 0:
        raise ValueError("Internal error while fitting gap for fixed-radius geometry")
    return fitted_gap, top_padding, left_padding


def save_cellpose_test_data(
    image_path: str | Path,
    *,
    segmentation_path: str | Path | None = None,
    rows: int = 10,
    cols: int = 10,
    radius: int = 10,
    gap: int = 10,
    square_size: int | None = None,
    preserve_radius_gap: bool = False,
    preserve_radius_auto_gap: bool = False,
) -> tuple[Path, Path]:
    """Generate and save synthetic Cellpose test data to disk."""
    image_path = Path(image_path)
    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("image_path must use .tif or .tiff extension")

    if segmentation_path is None:
        segmentation_path = image_path.with_name(f"{image_path.stem}_segmentation.npy")
    else:
        segmentation_path = Path(segmentation_path)

    image_array, segmentation_array = generate_cellpose_test_grid(
        rows=rows,
        cols=cols,
        radius=radius,
        gap=gap,
        square_size=square_size,
        preserve_radius_gap=preserve_radius_gap,
        preserve_radius_auto_gap=preserve_radius_auto_gap,
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_array).save(image_path, format="TIFF")
    np.save(segmentation_path, segmentation_array)

    return image_path, segmentation_path
