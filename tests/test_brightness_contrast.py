from __future__ import annotations

from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from neuroseq.brightness_contrast import BrightnessContrastDialog
from neuroseq.brightness_contrast import DisplayRange
from neuroseq.brightness_contrast import apply_display_range
from neuroseq.brightness_contrast import auto_display_range
from neuroseq.brightness_contrast import build_lut
from neuroseq.brightness_contrast import luminance_histogram
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.visualization.spatial_footprint_widget import SpatialFootprintWidget


def test_identity_range_returns_image_unchanged() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert apply_display_range(image, DisplayRange()) is image


def test_lut_stretches_window_to_full_range() -> None:
    lut = build_lut(DisplayRange(50, 150))
    assert lut[50] == 0
    assert lut[150] == 255
    assert lut[0] == 0
    assert lut[255] == 255
    assert 120 <= lut[100] <= 135


def test_brightness_contrast_round_trip() -> None:
    window = DisplayRange(40, 180)
    rebuilt = DisplayRange.from_brightness_contrast(window.brightness, window.contrast)
    assert abs(rebuilt.minimum - 40) <= 1
    assert abs(rebuilt.maximum - 180) <= 1


def test_clamped_keeps_minimum_below_maximum() -> None:
    window = DisplayRange.clamped(200, 100)
    assert window.minimum < window.maximum


def test_auto_ignores_hot_pixels() -> None:
    image = np.tile(np.linspace(60, 100, 100).astype(np.uint8), (100, 1))
    image[0, 0] = 255  # a single hot pixel must not stretch the window
    window = auto_display_range(luminance_histogram(image))
    assert window.minimum == 60
    assert window.maximum == 100


def test_auto_on_empty_histogram_is_identity() -> None:
    assert auto_display_range(np.zeros(256, dtype=np.int64)).is_identity


def test_dialog_sliders_emit_and_sync(qtbot: QtBot) -> None:
    dialog = BrightnessContrastDialog()
    qtbot.addWidget(dialog)
    seen: list[DisplayRange] = []
    dialog.range_changed.connect(seen.append)

    dialog.minimum_slider.setValue(30)
    dialog.maximum_slider.setValue(200)
    assert dialog.display_range == DisplayRange(30, 200)
    assert seen[-1] == DisplayRange(30, 200)

    dialog.reset()
    assert dialog.display_range.is_identity
    assert dialog.minimum_slider.value() == 0
    assert dialog.maximum_slider.value() == 255


def test_dialog_auto_uses_loaded_image(qtbot: QtBot) -> None:
    dialog = BrightnessContrastDialog()
    qtbot.addWidget(dialog)
    ramp = np.tile(np.linspace(60, 100, 100).astype(np.uint8), (100, 1))
    image = np.repeat(ramp[..., np.newaxis], 3, axis=2)
    dialog.set_image(image)
    dialog.auto()
    assert dialog.display_range == DisplayRange(60, 100)


def _result() -> CellposeImportResult:
    background = np.full((4, 4, 3), 100, dtype=np.uint8)
    segmentation = np.zeros((4, 4), dtype=np.int32)
    segmentation[0, 0] = 1
    overlay = background.copy()
    overlay[0, 0] = (0.5 * 100 + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return CellposeImportResult(
        image_path=Path("fov.png"),
        segmentation_array=segmentation,
        overlay_image=overlay,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping=None,
        background_image=background,
    )


def test_footprint_panel_adjusts_fov_but_not_cell_colors(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_result())
    original = widget._composed_overlay_image().copy()

    widget._show_brightness_contrast_dialog()
    dialog = widget._brightness_contrast_dialog
    assert dialog is not None
    dialog.set_display_range(DisplayRange(50, 150))

    adjusted = widget._composed_overlay_image()
    # FOV pixel 100 maps to ~128 under the 50-150 window.
    assert 120 <= int(adjusted[3, 3, 0]) <= 135
    # Inside a cell the cell color's half is preserved: only the FOV half moved.
    assert adjusted[0, 0, 0] != original[0, 0, 0]
    assert adjusted[0, 0, 1] < 100  # no green leaked into the red cell

    dialog.reset()
    assert np.array_equal(widget._composed_overlay_image(), original)
    widget.close()


def test_display_range_is_stored_on_shared_data_and_synced_across_views(
    qtbot: QtBot,
) -> None:
    from neuroseq.spatial_footprint_state import SpatialFootprintState

    state = SpatialFootprintState()
    first = SpatialFootprintWidget(state=state)
    second = SpatialFootprintWidget(state=state)
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    result = _result()
    state.set_data(result)

    first._show_brightness_contrast_dialog()
    dialog = first._brightness_contrast_dialog
    assert dialog is not None
    dialog.set_display_range(DisplayRange(50, 150))

    # Lands on the object the project saves, and the other view re-renders.
    assert result.display_range == (50, 150)
    assert 120 <= int(second._composed_overlay_image()[3, 3, 0]) <= 135

    dialog.reset()
    assert result.display_range is None
    first.close()
    second.close()


def test_loaded_display_range_is_applied_and_shown_in_dialog(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    result = _result()
    result.display_range = (50, 150)
    widget.set_data(result)

    assert 120 <= int(widget._composed_overlay_image()[3, 3, 0]) <= 135
    widget._show_brightness_contrast_dialog()
    assert widget._brightness_contrast_dialog is not None
    assert widget._brightness_contrast_dialog.display_range == DisplayRange(50, 150)
    widget.close()
