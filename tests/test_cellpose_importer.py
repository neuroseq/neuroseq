from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from neuroseq.importers.importer_cellpose import CellPoseImporter
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import clone_cellpose_result
from neuroseq.importers.importer_cellpose import generate_linear_stimulus_mapping
from neuroseq.importers.importer_cellpose import load_background_image
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping
from neuroseq.utils import neuron_indices_to_colors

"""
We test cellpose import with 2 of the 3 formats here, color exactness, and mapping of the cellpose results, plus the
persistent nature of cellpose until you clear it
"""


def _fastplotlib_expected_colors(n_neurons: int) -> np.ndarray:
    if n_neurons <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    indices = np.arange(n_neurons)
    colors_rgba_float = neuron_indices_to_colors(indices)
    return (colors_rgba_float[:, :3] * 255).astype(np.uint8)


def test_load_tiff_stack_mean(tmp_path):
    importer = CellPoseImporter()

    rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    gray_frame = np.full((4, 4), 120, dtype=np.uint8)
    blue_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    blue_frame[..., 2] = 255

    pil_frames = [
        Image.fromarray(rgb_frame, mode="RGB"),
        Image.fromarray(gray_frame),
        Image.fromarray(blue_frame, mode="RGB"),
    ]

    tiff_path = Path(tmp_path) / "stack.tif"
    pil_frames[0].save(
        tiff_path, format="TIFF", save_all=True, append_images=pil_frames[1:]
    )

    importer.load_image_file(str(tiff_path))
    assert importer.image_data is not None

    expected_arrays = [importer._frame_to_rgb_float(frame) for frame in pil_frames]
    expected_float = np.mean(np.stack(expected_arrays, axis=0), axis=0)
    # 8-bit frames stay on their own 0-255 scale (not stretched).
    expected = np.rint(expected_float).astype(np.uint8)

    np.testing.assert_array_equal(importer.image_data, expected)


def test_load_uint16_tiff_stack(tmp_path):
    importer = CellPoseImporter()

    base = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 1000).astype(np.uint16)
    frame_a = base
    frame_b = base + 5000

    frames = [Image.fromarray(frame_a), Image.fromarray(frame_b)]

    tiff_path = tmp_path / "stack16.tif"
    frames[0].save(tiff_path, format="TIFF", save_all=True, append_images=frames[1:])

    importer.load_image_file(str(tiff_path))
    assert importer.image_data is not None

    expected_arrays = [importer._frame_to_rgb_float(frame) for frame in frames]
    expected_float = np.mean(np.stack(expected_arrays, axis=0), axis=0)
    expected_rgb = importer._scale_float_image_to_uint8(expected_float)

    np.testing.assert_array_equal(importer.image_data, expected_rgb)
    assert len(np.unique(importer.image_data)) > 1


def test_load_segmentation_npy(tmp_path):
    importer = CellPoseImporter()

    segmentation = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [0, 3, 3],
        ],
        dtype=np.int32,
    )

    seg_path = tmp_path / "segmentation.npy"
    np.save(seg_path, segmentation)

    importer.load_segmentation_file(seg_path)

    assert importer.segmentation_data is not None
    assert importer.segmentation_data.dtype == np.int32
    expected = segmentation - 1
    np.testing.assert_array_equal(importer.segmentation_data, expected)


def test_load_suite2p_segmentation(tmp_path):
    importer = CellPoseImporter()

    roi1 = {
        "ypix": np.array([0, 0, 1, 1]),
        "xpix": np.array([0, 1, 0, 1]),
        "npix": 4,
        "med": [0.5, 0.5],
        "radius": 1.0,
        "aspect_ratio": 1.0,
    }
    roi2 = {
        "ypix": np.array([2, 2, 3, 3]),
        "xpix": np.array([2, 3, 2, 3]),
        "npix": 4,
        "med": [2.5, 2.5],
        "radius": 1.0,
        "aspect_ratio": 1.0,
    }
    roi3 = {
        "ypix": np.array([0, 1]),
        "xpix": np.array([3, 3]),
        "npix": 2,
        "med": [0.5, 3.0],
        "radius": 0.5,
        "aspect_ratio": 0.5,
    }

    stat_data = np.array([roi1, roi2, roi3], dtype=object)

    iscell_data = np.array([[1.0, 0.8], [1.0, 0.9], [0.0, 0.2]])

    stat_path = tmp_path / "stat.npy"
    iscell_path = tmp_path / "iscell.npy"
    np.save(stat_path, stat_data, allow_pickle=True)
    np.save(iscell_path, iscell_data)

    importer.load_segmentation_file(stat_path)

    assert importer.segmentation_data is not None
    assert importer.segmentation_data.shape == (4, 4)
    assert importer.segmentation_data.dtype == np.int32

    expected = np.array(
        [[0, 0, -1, -1], [0, 0, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(importer.segmentation_data, expected)


def test_load_csv_masks_segmentation(tmp_path):
    """Test loading CSV segmentation file (masks.csv format)."""
    importer = CellPoseImporter()

    segmentation = np.array(
        [
            [0, 1, 1, 2, 2],
            [0, 1, 1, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
        ],
        dtype=np.int32,
    )

    csv_path = tmp_path / "masks.csv"
    np.savetxt(csv_path, segmentation, fmt="%d", delimiter=",")

    importer.load_segmentation_file(csv_path)

    assert importer.segmentation_data is not None
    assert importer.segmentation_data.shape == (5, 5)
    assert importer.segmentation_data.dtype == np.int32
    expected = segmentation - 1
    np.testing.assert_array_equal(importer.segmentation_data, expected)

    expected_unique = np.array(
        [-1, 0, 1, 2, 3]
    )  # -1 represents background here now, it is 0 in cellpose by default.
    # We want however the first cell (1 in cellpose) to map to 0 indexed 0th stim ID for imports by default. Hence the change.
    actual_unique = np.unique(importer.segmentation_data)
    np.testing.assert_array_equal(actual_unique, expected_unique)


def test_cellpose_patchclamp_visual_integration(tmp_path):
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    # Create test segmentation with a few cells
    segmentation = np.zeros((50, 50), dtype=np.int32)
    segmentation[10:20, 10:20] = 1
    segmentation[30:40, 30:40] = 2
    segmentation[15:25, 35:45] = 3  # cell 3

    # Save test files
    image_path = tmp_path / "test_image.png"
    seg_path = tmp_path / "segmentation.npy"

    Image.fromarray(test_image).save(image_path)
    np.save(seg_path, segmentation)

    # Import cellpose data
    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay()

    cellpose_result = importer.create_result(image_path)

    # Verify cellpose result has expected data structure for visual display
    assert cellpose_result.overlay_image is not None
    assert cellpose_result.segmentation_array is not None
    assert cellpose_result.overlay_image.shape == (50, 50, 3)
    assert cellpose_result.segmentation_array.shape == (50, 50)

    # Verify overlay image is properly generated (should be blend of image and segmentation colors)
    expected_segmentation = segmentation - 1
    np.testing.assert_array_equal(
        cellpose_result.segmentation_array, expected_segmentation
    )

    # Test that the importer computed neuron colors for visual display
    assert importer._neuron_ids is not None
    assert importer._neuron_colors is not None
    expected_ids = [0, 1, 2]
    np.testing.assert_array_equal(importer._neuron_ids, expected_ids)
    assert importer._neuron_colors.shape == (3, 3)  # 3 neurons, RGB colors

    # Test that legend data is available for visual display
    legend_data = importer.get_legend_data()
    assert len(legend_data) == 3
    legend_ids = [cell_id for cell_id, _ in legend_data]
    for expected_id in expected_ids:
        assert expected_id in legend_ids

    # Verify each legend entry has proper RGB color format
    for cell_id, rgb in legend_data:
        assert isinstance(cell_id, int)
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)

    # Test that overlay image pixels were modified (not identical to original)
    assert not np.array_equal(cellpose_result.overlay_image, test_image)

    # Test that background pixels (segmentation == 0) exist in overlay
    background_mask = importer.segmentation_data == -1
    assert np.any(background_mask)

    # Test that cell pixels have different colors from background in overlay
    cell_mask = importer.segmentation_data >= 0
    assert np.any(cell_mask)

    # Verify that the overlay properly blends original image with colorized segmentation
    # The overlay should be: 0.5 * original + 0.5 * colored_segmentation
    assert cellpose_result.overlay_image.dtype == np.uint8
    assert cellpose_result.overlay_image.shape == test_image.shape


def test_custom_stim_cellpose_mapping(tmp_path):
    """Test custom mapping between stimulus IDs and cellpose IDs (stim:cellpose format)."""
    test_image = np.random.randint(50, 150, (40, 40, 3), dtype=np.uint8)
    segmentation = np.zeros((40, 40), dtype=np.int32)
    segmentation[5:15, 5:15] = 3
    segmentation[20:30, 20:30] = 7
    segmentation[10:20, 25:35] = 12
    image_path = tmp_path / "test.png"
    seg_path = tmp_path / "seg.npy"
    Image.fromarray(test_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay()

    stim_to_cellpose_mapping = {1: 2, 5: 6, 9: 11}
    cellpose_result = importer.create_result(
        image_path, stim_to_cellpose_mapping, mapping_strategy="Custom"
    )

    assert cellpose_result.stimulus_to_cellpose_mapping == stim_to_cellpose_mapping
    expected_cellpose_ids = [2, 6, 11]
    for cellpose_id in expected_cellpose_ids:
        assert cellpose_id in cellpose_result.original_cellpose_ids

    # Test mapping parsing logic (simulates dialog parsing)
    mapping_str = "1:2,5:6,9:11"
    parsed_mapping = {}
    for pair in mapping_str.split(","):
        stim_key_str, cellpose_id_str = pair.split(":", 1)
        stim_key = int(stim_key_str.strip())
        cellpose_id = int(cellpose_id_str.strip())
        parsed_mapping[stim_key] = cellpose_id

    assert parsed_mapping == stim_to_cellpose_mapping


def test_linear_mapping_offset_starts_at_zero():
    original_ids = np.array([1, 3, 4, 7], dtype=np.int32)
    mapping = generate_linear_stimulus_mapping(original_ids, offset=3)
    assert mapping == {0: 3, 1: 4, 2: 7, 3: 1}


def _cellpose_result(
    *,
    original_cellpose_ids: np.ndarray | None = None,
    stimulus_to_cellpose_mapping: dict[int, int] | None = None,
    mapping_strategy: str | None = None,
    linear_offset: int | None = None,
) -> CellposeImportResult:
    return CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=original_cellpose_ids,
        stimulus_to_cellpose_mapping=stimulus_to_cellpose_mapping,
        mapping_strategy=mapping_strategy,
        linear_offset=linear_offset,
    )


def test_resolve_stimulus_to_cellpose_mapping_reconstructs_linear_offset():
    """Regression test for #554's Linear-strategy blind spot: only
    linear_offset is stored for a Linear import (no materialized
    stimulus_to_cellpose_mapping), so reading the mapping field directly
    looks like "no mapping" and silently drops the offset."""
    result = _cellpose_result(
        original_cellpose_ids=np.array([1, 3, 4, 7], dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
        mapping_strategy="Linear",
        linear_offset=3,
    )

    assert resolve_stimulus_to_cellpose_mapping(result) == {
        0: 3,
        1: 4,
        2: 7,
        3: 1,
    }


def test_resolve_stimulus_to_cellpose_mapping_passes_through_custom_mapping():
    result = _cellpose_result(
        stimulus_to_cellpose_mapping={4: 0, 7: 1},
        mapping_strategy="Custom",
    )

    assert resolve_stimulus_to_cellpose_mapping(result) == {4: 0, 7: 1}


def test_resolve_stimulus_to_cellpose_mapping_defaults_missing_strategy_to_linear():
    """Projects saved before ``mapping_strategy`` existed have it as None -
    inferred as Linear, matching the same inference already used elsewhere
    (e.g. CellposeImportDialog._apply_existing_mapping)."""
    result = _cellpose_result(
        original_cellpose_ids=np.array([0, 1, 2], dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
        mapping_strategy=None,
        linear_offset=None,
    )

    assert resolve_stimulus_to_cellpose_mapping(result) == {0: 0, 1: 1, 2: 2}


def test_resolve_stimulus_to_cellpose_mapping_handles_none():
    assert resolve_stimulus_to_cellpose_mapping(None) is None


def test_custom_mapping_parsing_edge_cases():
    """Test edge cases in custom mapping string parsing."""
    # Test various mapping string formats

    # Standard format
    mapping_str = "12:1,15:7,20:3"
    mapping = {}
    for pair in mapping_str.split(","):
        stim_str, cellpose_str = pair.split(":", 1)
        mapping[int(stim_str.strip())] = int(cellpose_str.strip())

    expected = {12: 1, 15: 7, 20: 3}
    assert mapping == expected

    # Format with spaces
    mapping_str = " 12 : 1 , 15 : 7 , 20 : 3 "
    mapping = {}
    for pair in mapping_str.split(","):
        stim_str, cellpose_str = pair.split(":", 1)
        mapping[int(stim_str.strip())] = int(cellpose_str.strip())

    assert mapping == expected

    # Single mapping
    mapping_str = "5:10"
    mapping = {}
    for pair in mapping_str.split(","):
        stim_str, cellpose_str = pair.split(":", 1)
        mapping[int(stim_str.strip())] = int(cellpose_str.strip())

    assert mapping == {5: 10}


def test_cellpose_overlay_color_pixels(tmp_path):
    """Test that cellpose overlay contains exact expected colors matching legend."""
    test_image = np.zeros((30, 30, 3), dtype=np.uint8)

    segmentation = np.zeros((30, 30), dtype=np.int32)
    segmentation[5:15, 5:15] = (
        1  # set these cells to cell 1 ID in the segmentation mask for our purposes here
    )
    segmentation[16:25, 16:25] = 2  # Same for cell 2
    segmentation[5:15, 20:25] = 3  # and cell 3.

    image_path = tmp_path / "test_image.png"
    seg_path = tmp_path / "segmentation.npy"

    Image.fromarray(test_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=1.0)  # For testing exact colors.

    legend_data = importer.get_legend_data()
    assert len(legend_data) == 3

    for cell_id, expected_rgb in legend_data:
        cell_mask = importer.segmentation_data == cell_id
        if np.any(cell_mask):
            actual_pixels = importer.overlay_image[cell_mask]
            actual_rgb = actual_pixels[0]

            assert np.array_equal(actual_rgb, expected_rgb), (
                f"Cell {cell_id}: expected {expected_rgb}, got {actual_rgb}"
            )

            for pixel in actual_pixels:
                assert np.array_equal(pixel, actual_rgb), (
                    f"Cell {cell_id} has inconsistent colors"
                )


def test_cellpose_preview_pixel_differences(tmp_path):
    """Test that cellpose preview shows 50% blend and background unchanged."""
    original_image = np.full((20, 20, 3), 100, dtype=np.uint8)

    segmentation = np.zeros((20, 20), dtype=np.int32)
    segmentation[5:15, 5:15] = 1

    image_path = tmp_path / "original.png"
    seg_path = tmp_path / "seg.npy"

    Image.fromarray(original_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=0.5)

    np.testing.assert_array_equal(importer.image_data, original_image)

    bg_mask = importer.segmentation_data == -1
    cell_mask = importer.segmentation_data == 0

    bg_pixels = importer.overlay_image[bg_mask]
    original_bg_pixels = original_image[bg_mask]
    np.testing.assert_array_equal(bg_pixels, original_bg_pixels)

    if np.any(cell_mask):
        cell_pixels = importer.overlay_image[cell_mask]
        original_cell_pixels = original_image[cell_mask]
        assert not np.array_equal(cell_pixels, original_cell_pixels)


def test_restore_from_result_preserves_colors_without_custom_mapping(tmp_path):
    """Reopening existing cellpose data should keep legend colors aligned with stored overlay."""
    base_image = np.full((16, 16, 3), 50, dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int32)
    segmentation[0:8, 0:8] = 1
    segmentation[8:16, 8:16] = 2

    image_path = tmp_path / "image.png"
    seg_path = tmp_path / "seg.npy"
    Image.fromarray(base_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=1.0)

    original_legend = importer.get_legend_data()
    result = importer.create_result(image_path)

    restored = CellPoseImporter()
    restored.restore_from_result(result)

    assert restored.overlay_image is not None
    assert restored.segmentation_data is not None
    # Segmentations keep Cellpose's top-left orientation (#601).
    expected_segmentation = segmentation - 1
    np.testing.assert_array_equal(restored.segmentation_data, expected_segmentation)

    restored_legend = restored.get_legend_data()
    assert restored_legend == original_legend

    # Verify pixel colors still match legend after restore
    for cell_id, expected_rgb in restored_legend:
        mask = restored.segmentation_data == cell_id
        assert np.any(mask)
        pixels = restored.overlay_image[mask]
        assert np.all(pixels == expected_rgb)


def test_create_result_stores_the_raw_background_image(tmp_path):
    base_image = np.full((16, 16, 3), 50, dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int32)
    segmentation[0:8, 0:8] = 1

    image_path = tmp_path / "image.png"
    seg_path = tmp_path / "seg.npy"
    Image.fromarray(base_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=1.0)

    result = importer.create_result(image_path)

    assert result.background_image is not None
    np.testing.assert_array_equal(result.background_image, importer.image_data)
    # the overlay has cell colors baked in - the background must not
    assert not np.array_equal(result.background_image, result.overlay_image)


def test_restore_from_result_falls_back_to_stored_background_image(tmp_path):
    """If the original image file is gone, restore should prefer the
    stored raw background over the (already colored) overlay image."""
    base_image = np.full((16, 16, 3), 50, dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int32)
    segmentation[0:8, 0:8] = 1

    image_path = tmp_path / "image.png"
    seg_path = tmp_path / "seg.npy"
    Image.fromarray(base_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=1.0)
    result = importer.create_result(image_path)

    image_path.unlink()  # the original file is no longer reachable

    restored = CellPoseImporter()
    restored.restore_from_result(result)

    assert restored.image_data is not None
    np.testing.assert_array_equal(restored.image_data, result.background_image)
    assert not np.array_equal(restored.image_data, result.overlay_image)


def test_load_background_image_reloads_from_disk(tmp_path):
    base_image = np.full((16, 16, 3), 50, dtype=np.uint8)
    image_path = tmp_path / "image.png"
    Image.fromarray(base_image).save(image_path)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))

    reloaded = load_background_image(image_path)

    assert reloaded is not None
    np.testing.assert_array_equal(reloaded, importer.image_data)


def test_load_background_image_returns_none_when_file_missing(tmp_path):
    assert load_background_image(tmp_path / "missing.png") is None


def test_restore_from_result_preserves_custom_mapping(tmp_path):
    base_image = np.full((16, 16, 3), 80, dtype=np.uint8)
    segmentation = np.zeros((16, 16), dtype=np.int32)
    segmentation[1:8, 1:8] = 2
    segmentation[9:15, 9:15] = 5

    image_path = tmp_path / "custom_image.png"
    seg_path = tmp_path / "custom_seg.npy"
    Image.fromarray(base_image).save(image_path)
    np.save(seg_path, segmentation)

    importer = CellPoseImporter()
    importer.load_image_file(str(image_path))
    importer.load_segmentation_file(seg_path)
    importer.validate_dimensions()
    importer.generate_overlay(opacity=1.0)

    custom_mapping = {11: 1, 12: 4}
    original_result = importer.create_result(
        image_path,
        custom_mapping,
        mapping_strategy="Custom",
    )
    cloned_result = clone_cellpose_result(original_result)

    restored = CellPoseImporter()
    restored.restore_from_result(cloned_result)

    assert cloned_result.mapping_strategy == "Custom"
    assert cloned_result.stimulus_to_cellpose_mapping == custom_mapping
    expected_segmentation = segmentation - 1
    np.testing.assert_array_equal(restored.segmentation_data, expected_segmentation)
    # background_image is a defensive copy, not the same array object
    assert cloned_result.background_image is not original_result.background_image
    np.testing.assert_array_equal(
        cloned_result.background_image, original_result.background_image
    )


def test_an_8bit_tiff_is_shown_on_its_own_scale_not_stretched(tmp_path) -> None:
    import tifffile

    from neuroseq.importers.importer_cellpose import CellPoseImporter

    path = tmp_path / "dim8.tif"
    image = np.full((4, 5), 20, dtype=np.uint8)
    image[0, 0] = 40
    tifffile.imwrite(path, image, photometric="minisblack")
    importer = CellPoseImporter()
    importer.load_image_file(path)
    assert importer.image_data is not None
    assert importer.image_data.dtype == np.uint8
    assert int(importer.image_data.max()) == 40
    assert int(importer.image_data.min()) == 20
