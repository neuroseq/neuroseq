from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from neuroseq.cellpose_test_data import generate_cellpose_test_grid
from neuroseq.cellpose_test_data import save_cellpose_test_data
from neuroseq.importer_cellpose import CellPoseImporter


def test_generate_cellpose_test_grid_default() -> None:
    image_array, segmentation_array = generate_cellpose_test_grid()

    assert image_array.dtype == np.uint8
    assert segmentation_array.dtype == np.int32
    assert image_array.shape == segmentation_array.shape

    unique_ids = np.unique(segmentation_array)
    assert unique_ids[0] == 0
    assert unique_ids[-1] == 100
    assert unique_ids.shape[0] == 101

    first_cell_pixels = np.count_nonzero(segmentation_array == 1)
    assert first_cell_pixels > 0


def test_generate_cellpose_test_grid_spacing() -> None:
    _, segmentation_array = generate_cellpose_test_grid(
        rows=2,
        cols=2,
        radius=10,
        gap=10,
    )

    y1, x1 = np.where(segmentation_array == 1)
    y2, x2 = np.where(segmentation_array == 2)
    y3, x3 = np.where(segmentation_array == 3)

    center_1 = (x1.mean(), y1.mean())
    center_2 = (x2.mean(), y2.mean())
    center_3 = (x3.mean(), y3.mean())

    assert np.isclose(center_2[0] - center_1[0], 30.0)
    assert np.isclose(center_3[1] - center_1[1], 30.0)


def test_save_cellpose_test_data_roundtrip(tmp_path: Path) -> None:
    image_path = tmp_path / "grid.tif"
    returned_image_path, returned_segmentation_path = save_cellpose_test_data(
        image_path,
        rows=3,
        cols=4,
        radius=6,
        gap=4,
    )

    assert returned_image_path == image_path
    assert returned_image_path.exists()
    assert returned_segmentation_path.exists()

    image_array = np.array(Image.open(returned_image_path))
    segmentation_array = np.load(returned_segmentation_path)

    assert image_array.shape == segmentation_array.shape
    assert segmentation_array.max() == 12


def test_saved_cellpose_test_data_imports_with_cellpose_importer(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "grid.tif"
    _, segmentation_path = save_cellpose_test_data(
        image_path,
        rows=10,
        cols=10,
        radius=10,
        gap=10,
    )

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(segmentation_path)
    importer.validate_dimensions()
    importer.generate_overlay()

    assert importer.image_data is not None
    assert importer.segmentation_data is not None
    assert importer.overlay_image is not None
    assert importer.segmentation_data.max() == 99
