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

    pitch = (2 * radius) + gap
    height = (2 * edge_padding) + (2 * radius) + ((rows - 1) * pitch)
    width = (2 * edge_padding) + (2 * radius) + ((cols - 1) * pitch)

    image = np.full((height, width), background_intensity, dtype=np.uint8)
    segmentation = np.zeros((height, width), dtype=np.int32)

    yy, xx = np.ogrid[:height, :width]

    cell_id = 1
    for row in range(rows):
        center_y = edge_padding + radius + (row * pitch)
        for col in range(cols):
            center_x = edge_padding + radius + (col * pitch)
            mask = ((yy - center_y) ** 2) + ((xx - center_x) ** 2) <= (radius**2)
            image[mask] = cell_intensity
            segmentation[mask] = cell_id
            cell_id += 1

    return image, segmentation


def save_cellpose_test_data(
    image_path: str | Path,
    *,
    segmentation_path: str | Path | None = None,
    rows: int = 10,
    cols: int = 10,
    radius: int = 10,
    gap: int = 10,
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
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_array).save(image_path, format="TIFF")
    np.save(segmentation_path, segmentation_array)

    return image_path, segmentation_path
