from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import brian2.numpy_ as np
from PIL import Image
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.aoi_shapes import square_world_corners
from neuroseq.aoi_shapes import user_to_world_point
from neuroseq.cell_overlay_widget import STRIPED_CELL_ALPHA
from neuroseq.cell_overlay_widget import STRIPED_CELL_GREY
from neuroseq.cell_overlay_widget import STRIPED_CELL_WHITE
from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cell_overlay_widget import OverlayView
from neuroseq.importer_cellpose import AOIRole
from neuroseq.importer_cellpose import AreasOfInterestConfig
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import CircleAOI
from neuroseq.importer_cellpose import SquareAOI
from neuroseq.importer_cellpose import clone_aoi_config
from neuroseq.spatial_footprint_state import SpatialFootprintState
from neuroseq.spatial_footprint_widget import ACTIVE_STIM_BRIGHTNESS_BOOST
from neuroseq.spatial_footprint_widget import HOVERED_CELL_BRIGHTNESS_BOOST
from neuroseq.spatial_footprint_widget import SELECTION_OUTLINE_BLACK
from neuroseq.spatial_footprint_widget import SELECTION_OUTLINE_RED
from neuroseq.spatial_footprint_widget import UNSELECTED_CELL_BRIGHTNESS
from neuroseq.spatial_footprint_widget import SpatialFootprintWidget


def _right_click_event() -> SimpleNamespace:
    """A stand-in pointer event for _cell_selection_pointer_down - only
    its .button is read (pygfx numbering: 1=left, 2=right, 3=middle);
    map_screen_to_world (which would read x/y) is monkeypatched in these
    tests to ignore the event object entirely."""
    return SimpleNamespace(button=2)


def _cellpose_result(
    mapping: dict[int, int] | None = None,
    aoi_config: AreasOfInterestConfig | None = None,
) -> CellposeImportResult:
    segmentation = np.array(
        [
            [0, 0, 1],
            [0, 2, 2],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    )
    return CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((3, 3, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1, 2], dtype=np.int32),
        stimulus_to_cellpose_mapping=mapping,
        aoi_config=aoi_config,
    )


def _cellpose_result_with_unmapped_cells() -> CellposeImportResult:
    """8 segmented cells (ids 0-7): only 5,6,7 are mapped to neurons 0,1,2.
    Ids 0-4 are unmapped stray cells - and crucially, their numbers collide
    with the real neuron indices 0,1,2 (regression for issue: selecting a
    cell hiding it right back because an unrelated unmapped cell happens to
    share its neuron index)."""
    segmentation = -1 * np.ones((2, 16), dtype=np.int32)
    for cellpose_id in range(8):
        segmentation[:, cellpose_id * 2 : cellpose_id * 2 + 2] = cellpose_id
    return CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((2, 16, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array(range(8), dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 5, 1: 6, 2: 7},
    )


def test_set_data_populates_overlay_and_clears_on_none(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    assert not widget.empty_state_widget.isHidden()
    assert widget.overlay_widget.isHidden()
    assert widget.checkbox_container.isHidden()

    widget.set_data(_cellpose_result())
    assert widget.overlay_widget.segmentation is not None
    assert widget.empty_state_widget.isHidden()
    assert not widget.overlay_widget.isHidden()
    assert not widget.checkbox_container.isHidden()

    widget.set_data(None)
    assert widget.overlay_widget.overlay_image is None
    assert widget.overlay_widget.segmentation is None
    assert not widget.empty_state_widget.isHidden()
    assert widget.overlay_widget.isHidden()
    assert widget.checkbox_container.isHidden()
    widget.close()


def _aoi_config(show_shapes: bool = True) -> AreasOfInterestConfig:
    return AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=0.5, y=0.5, side=1.0),
        circle=CircleAOI(True, AOIRole.EXCLUSION, cx=1.5, cy=1.5, diameter=1.0),
        show_shapes=show_shapes,
        coordinate_mode="pixels",
    )


def _wide_square_aoi_config() -> AreasOfInterestConfig:
    """A square whose body center sits well outside the corner-handle hit
    tolerance, so a click there unambiguously hits the body, not a handle."""
    return AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=2.0, y=2.0, side=16.0),
        circle=CircleAOI(False, AOIRole.EXCLUSION, cx=1.5, cy=1.5, diameter=1.0),
        show_shapes=True,
        coordinate_mode="pixels",
    )


def test_set_data_shows_persisted_aoi_shapes_as_a_read_only_reference(
    qtbot: QtBot,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    widget.set_data(_cellpose_result(aoi_config=_aoi_config()))

    overlay_view = widget.overlay_widget.overlay_view
    assert overlay_view._aoi_config == _aoi_config()
    assert overlay_view._aoi_visible
    # shown for reference only - dragging must not be possible here (that's
    # the Cell Segmentation import dialog's job)
    assert not overlay_view.aoi_editable

    widget.set_data(None)
    assert overlay_view._aoi_config is None


def test_set_data_hides_aoi_shapes_when_show_shapes_is_false(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    widget.set_data(_cellpose_result(aoi_config=_aoi_config(show_shapes=False)))

    assert not widget.overlay_widget.overlay_view._aoi_visible
    widget.close()


def test_set_data_with_no_aoi_config_shows_no_aoi_shapes(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    widget.set_data(_cellpose_result())

    assert widget.overlay_widget.overlay_view._aoi_config is None
    widget.close()


def test_show_labels_checkbox_toggles_stimulus_id_labels(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    assert widget.overlay_widget.show_stimulus_ids
    assert not widget.overlay_widget.show_cellpose_ids

    widget.show_labels_checkbox.setChecked(False)

    assert not widget.overlay_widget.show_stimulus_ids
    assert not widget.overlay_widget.show_cellpose_ids  # never toggled on here
    widget.close()


def test_show_segmentation_checkbox_toggles_the_segmentation_layer(
    qtbot: QtBot,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result())
    segmentation_graphic = widget.overlay_widget.overlay_view._segmentation_graphic
    assert segmentation_graphic is not None
    assert segmentation_graphic.visible

    widget.show_segmentation_checkbox.setChecked(False)

    assert widget.overlay_widget.show_segmentation is False
    assert not segmentation_graphic.visible
    widget.close()


def test_show_segmentation_checkbox_reveals_the_stored_background_image(
    qtbot: QtBot,
) -> None:
    """Regression: unchecking "Show segmentation" used to have no visible
    effect, because the composed image (overlay_image) always has cell
    colors baked in and there was nowhere else to fall back to - it must
    now reveal the stored, uncolored background_image instead."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 200)
    overlay_image[:, 4:] = (200, 0, 0)
    background_image = np.full((8, 8, 3), 128, dtype=np.uint8)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        background_image=background_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    before_image = np.asarray(image_graphic.data.value).copy()
    np.testing.assert_array_equal(before_image, overlay_image)

    widget.show_segmentation_checkbox.setChecked(False)

    after_image = np.asarray(image_graphic.data.value)
    np.testing.assert_array_equal(after_image, background_image)
    assert not np.array_equal(after_image, before_image)
    widget.close()


def test_show_segmentation_checkbox_reloads_background_from_disk_when_missing(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Regression: projects saved before `background_image` existed have
    `None` there even though the original image file may still be reachable
    on disk - the widget must reload it lazily (same fallback the Cell
    Segmentation import dialog already has via
    `CellPoseImporter.restore_from_result`), not just show the colored
    overlay regardless of the checkbox."""
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 200)
    overlay_image[:, 4:] = (200, 0, 0)
    background_image = np.full((8, 8, 3), 128, dtype=np.uint8)
    image_path = tmp_path / "cellpose.png"
    Image.fromarray(background_image).save(image_path)

    result = CellposeImportResult(
        image_path=image_path,
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        background_image=None,  # as an old, pre-existing project would load
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )

    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(result)

    assert result.background_image is not None

    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    np.testing.assert_array_equal(np.asarray(image_graphic.data.value), overlay_image)

    widget.show_segmentation_checkbox.setChecked(False)

    after_image = np.asarray(image_graphic.data.value)
    np.testing.assert_array_equal(after_image, result.background_image)
    assert not np.array_equal(after_image, overlay_image)
    widget.close()


def test_set_data_leaves_background_image_none_when_file_unreachable(
    qtbot: QtBot,
) -> None:
    """No original file on disk (e.g. `_cellpose_result()`'s placeholder
    path) - the lazy reload must fail silently, not raise."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    result = _cellpose_result()
    widget.set_data(result)

    assert result.background_image is None
    widget.close()


def test_show_shapes_checkbox_toggles_aoi_overlay_live(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(aoi_config=_aoi_config()))
    assert widget.show_shapes_checkbox.isEnabled()
    assert widget.show_shapes_checkbox.isChecked()
    assert widget.overlay_widget.overlay_view._aoi_visible

    widget.show_shapes_checkbox.setChecked(False)

    assert not widget.overlay_widget.overlay_view._aoi_visible
    widget.close()


def test_show_shapes_checkbox_disabled_without_an_aoi_config(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    widget.set_data(_cellpose_result())

    assert not widget.show_shapes_checkbox.isEnabled()
    widget.close()


def test_import_button_emits_import_requested(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.import_requested):
        widget.import_button.click()

    widget.close()


def test_cell_overlay_resize_keeps_embedded_view_layout_driven(qtbot: QtBot) -> None:
    widget = CellOverlayWidget()
    qtbot.addWidget(widget)
    result = _cellpose_result()

    widget.resize(320, 240)
    widget.set_overlay(
        result.overlay_image,
        result.segmentation_array,
        result.original_cellpose_ids,
        result.stimulus_to_cellpose_mapping,
    )
    qtbot.wait(10)

    assert widget.overlay_view.maximumWidth() > widget.overlay_view.minimumWidth()
    assert widget.overlay_view.maximumHeight() > widget.overlay_view.minimumHeight()
    widget.close()


def test_set_overlay_image_swaps_pixels_without_touching_segmentation_or_labels(
    qtbot: QtBot,
) -> None:
    widget = CellOverlayWidget()
    qtbot.addWidget(widget)
    result = _cellpose_result()
    widget.set_overlay(
        result.overlay_image,
        result.segmentation_array,
        result.original_cellpose_ids,
        result.stimulus_to_cellpose_mapping,
    )
    segmentation_graphic = widget.overlay_view._segmentation_graphic
    new_image = np.full_like(result.overlay_image, 200)

    widget.set_overlay_image(new_image)

    image_graphic = widget.overlay_view._image_graphic
    assert image_graphic is not None
    np.testing.assert_array_equal(np.asarray(image_graphic.data.value), new_image)
    # the segmentation color layer is untouched - same graphic, not rebuilt
    assert widget.overlay_view._segmentation_graphic is segmentation_graphic
    widget.close()


def test_set_overlay_image_is_a_noop_with_a_mismatched_shape_or_no_graphic_yet(
    qtbot: QtBot,
) -> None:
    widget = CellOverlayWidget()
    qtbot.addWidget(widget)

    # no overlay set yet - nothing to swap into
    widget.set_overlay_image(np.zeros((3, 3, 3), dtype=np.uint8))
    assert widget.overlay_view._image_graphic is None

    result = _cellpose_result()
    widget.set_overlay(
        result.overlay_image,
        result.segmentation_array,
        result.original_cellpose_ids,
        result.stimulus_to_cellpose_mapping,
    )
    image_graphic = widget.overlay_view._image_graphic
    assert image_graphic is not None
    before = np.asarray(image_graphic.data.value).copy()

    widget.set_overlay_image(np.zeros((5, 5, 3), dtype=np.uint8))  # wrong shape

    np.testing.assert_array_equal(np.asarray(image_graphic.data.value), before)
    widget.close()


def test_checkbox_row_sits_at_the_top_of_the_widget(qtbot: QtBot) -> None:
    """The display-toggle checkboxes are the first thing in the layout, so
    they read as a fixed toolbar above the view rather than floating below
    it."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    layout = widget.layout()
    assert layout is not None
    assert layout.indexOf(widget.checkbox_container) == 0
    assert layout.indexOf(widget.checkbox_container) < layout.indexOf(
        widget.empty_state_widget
    )
    assert layout.indexOf(widget.checkbox_container) < layout.indexOf(
        widget.overlay_widget
    )
    widget.close()


def test_checkbox_row_is_half_its_default_height(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    default_height = widget.checkbox_container.sizeHint().height()
    assert widget.checkbox_container.height() == round(default_height / 2)
    widget.close()


def test_clicking_a_cell_selects_it_and_greys_the_rest(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 2, 1: 1, 2: 0}))
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(2)

    assert widget._selected_cellpose_ids == {2}
    assert widget._greyed_cellpose_ids == {0, 1}
    # cellpose 0 -> stim 2, cellpose 1 -> stim 1 (mapping inverted)
    assert sorted(emitted) == [[1, True], [2, True]]
    widget.close()


def test_clicking_same_cell_twice_deselects_and_shows_everyone_again(
    qtbot: QtBot,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result())
    widget.overlay_widget.neuron_clicked.emit(1)
    assert widget._greyed_cellpose_ids == {0, 2}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(1)

    assert widget._selected_cellpose_ids == set()
    assert widget._greyed_cellpose_ids == set()
    assert sorted(emitted) == [[0, False], [2, False]]
    widget.close()


def test_selecting_a_second_cell_only_shows_both_selected_cells(
    qtbot: QtBot,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget.overlay_widget.neuron_clicked.emit(0)
    assert widget._greyed_cellpose_ids == {1, 2}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(1)

    assert widget._selected_cellpose_ids == {0, 1}
    assert widget._greyed_cellpose_ids == {2}
    assert emitted == [[1, False]]
    widget.close()


def test_clear_selection_shows_all_cells_again(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 2, 1: 1, 2: 0}))
    widget.overlay_widget.neuron_clicked.emit(2)
    widget.overlay_widget.neuron_clicked.emit(1)
    assert widget._selected_cellpose_ids == {1, 2}
    assert widget._greyed_cellpose_ids == {0}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.clear_selection()

    assert widget._selected_cellpose_ids == set()
    assert widget._greyed_cellpose_ids == set()
    assert emitted == [[2, False]]
    widget.close()


def test_clear_selection_button_is_a_noop_with_nothing_selected(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result())
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.clear_selection()

    assert emitted == []
    widget.close()


def test_identity_mapping_when_none_present(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping=None))
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(2)

    assert sorted(emitted) == [[0, True], [1, True]]
    widget.close()


def test_selecting_a_cell_dims_the_other_unselected_cells(qtbot: QtBot) -> None:
    """A cell that becomes hidden from trace/stim is dimmed in this view
    too, to make the selected cell(s) stand out."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 255)
    overlay_image[:, 4:] = (255, 0, 0)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    before_image = np.asarray(image_graphic.data.value).copy()

    # select cell 0 - cell 1 becomes hidden from trace/stim, and dimmed here
    widget.overlay_widget.neuron_clicked.emit(0)

    after_image = np.asarray(image_graphic.data.value)
    np.testing.assert_allclose(
        after_image[result.segmentation_array == 1].astype(np.float32),
        before_image[result.segmentation_array == 1].astype(np.float32)
        * UNSELECTED_CELL_BRIGHTNESS,
        atol=1.0,
    )
    widget.close()


def test_clearing_the_selection_restores_full_brightness(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    original_image = np.asarray(image_graphic.data.value).copy()

    widget.overlay_widget.neuron_clicked.emit(0)
    assert not np.array_equal(np.asarray(image_graphic.data.value), original_image)

    widget.clear_selection()

    np.testing.assert_array_equal(np.asarray(image_graphic.data.value), original_image)
    widget.close()


def test_selecting_a_previously_hidden_cell_shows_it_again(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget.overlay_widget.neuron_clicked.emit(0)  # select 0, hides 1 and 2
    assert widget._greyed_cellpose_ids == {1, 2}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(1)  # also select 1

    assert widget._selected_cellpose_ids == {0, 1}
    assert widget._greyed_cellpose_ids == {2}
    assert emitted == [[1, False]]
    widget.close()


def test_cellpose_to_stim_id_returns_none_for_unmapped_cell() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result_with_unmapped_cells())

    # 5, 6, 7 are the real mapped cells (-> neurons 0, 1, 2)
    assert state._cellpose_to_stim_id(5) == 0
    assert state._cellpose_to_stim_id(6) == 1
    assert state._cellpose_to_stim_id(7) == 2
    # 0, 1, 2, 3, 4 are unmapped strays - must not alias onto neuron 0/1/2
    for unmapped_id in range(5):
        assert state._cellpose_to_stim_id(unmapped_id) is None


def test_hovered_cellpose_id_resolves_via_mapping() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result(mapping={0: 2, 1: 1, 2: 0}))

    state.set_hovered_stim_id(0)
    assert state.hovered_cellpose_id == 2

    state.set_hovered_stim_id(None)
    assert state.hovered_cellpose_id is None


def test_hovered_cellpose_id_is_identity_when_no_mapping() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result(mapping=None))

    state.set_hovered_stim_id(1)
    assert state.hovered_cellpose_id == 1


def test_set_hovered_stim_id_is_a_noop_when_unchanged() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result())
    emitted: list[int | None] = []
    state.hovered_stim_id_changed.connect(lambda: emitted.append(state.hovered_stim_id))

    state.set_hovered_stim_id(1)
    state.set_hovered_stim_id(1)

    assert emitted == [1]


def test_active_cellpose_ids_resolves_via_mapping() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result(mapping={0: 2, 1: 1, 2: 0}))

    state.set_active_stim_ids({0, 1})

    assert state.active_cellpose_ids == {2, 1}


def test_active_cellpose_ids_skips_stim_ids_with_no_mapping_entry() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result_with_unmapped_cells())

    state.set_active_stim_ids({0, 99})  # 99 has no mapping entry at all

    assert state.active_cellpose_ids == {5}


def test_set_active_stim_ids_is_a_noop_when_unchanged() -> None:
    state = SpatialFootprintState()
    state.set_data(_cellpose_result())
    emitted = []
    state.active_stim_ids_changed.connect(lambda: emitted.append(state.active_stim_ids))

    state.set_active_stim_ids({1, 2})
    state.set_active_stim_ids({2, 1})  # same set, different order

    assert emitted == [frozenset({1, 2})]


def test_selecting_a_cell_does_not_hide_it_when_unmapped_cells_collide_with_its_id(
    qtbot: QtBot,
) -> None:
    """Regression: Cellpose finding more cells than are mapped to neurons
    used to hide the very cell the user selected, because an unrelated
    unmapped cell's id happened to equal that neuron's real index (see
    _cellpose_to_stim_id's identity-fallback bug)."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result_with_unmapped_cells())
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    # select only cellpose cell 7 (-> neuron 2)
    widget.overlay_widget.neuron_clicked.emit(7)

    assert widget._greyed_cellpose_ids == {0, 1, 2, 3, 4, 5, 6}
    # neuron 2 (the selected cell) must never be told it's greyed
    assert [2, True] not in emitted
    widget.close()


def test_selected_cell_renders_red_black_ring(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((10, 10), dtype=np.int32)
    segmentation[2:8, 2:8] = 1
    overlay_image = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay_image[:, :] = (0, 0, 255)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None

    widget.overlay_widget.neuron_clicked.emit(1)

    after_image = np.asarray(image_graphic.data.value)
    cell_colors = after_image[segmentation == 1][:, :3].astype(np.float32) / 255.0
    assert np.any(
        np.all(np.isclose(cell_colors, SELECTION_OUTLINE_RED, atol=1 / 255), axis=1)
    )
    assert np.any(
        np.all(np.isclose(cell_colors, SELECTION_OUTLINE_BLACK, atol=1 / 255), axis=1)
    )
    # the cell interior is not touched, only its boundary
    assert np.any(
        np.all(np.isclose(cell_colors, np.array([0.0, 0.0, 1.0]), atol=1 / 255), axis=1)
    )
    widget.close()


def test_hovering_a_cell_boosts_its_brightness(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 100)
    overlay_image[:, 4:] = (100, 0, 0)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    before_image = np.asarray(image_graphic.data.value).copy()

    widget._state.set_hovered_stim_id(1)  # -> cellpose cell 1

    after_image = np.asarray(image_graphic.data.value)
    np.testing.assert_allclose(
        after_image[segmentation == 1].astype(np.float32),
        np.clip(
            before_image[segmentation == 1].astype(np.float32)
            * HOVERED_CELL_BRIGHTNESS_BOOST,
            0,
            255,
        ),
        atol=1.0,
    )
    # the other, unhovered cell is untouched
    np.testing.assert_array_equal(
        after_image[segmentation == 0], before_image[segmentation == 0]
    )
    widget.close()


def test_hovering_a_dimmed_unselected_cell_restores_full_brightness(
    qtbot: QtBot,
) -> None:
    """A cell that's currently dimmed (hidden from trace/stim) still gets
    boosted to a bright highlight when hovered."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 100)
    overlay_image[:, 4:] = (100, 0, 0)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    widget.overlay_widget.neuron_clicked.emit(0)  # selects 0, dims 1
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    dimmed_image = np.asarray(image_graphic.data.value).copy()

    widget._state.set_hovered_stim_id(1)  # hover the dimmed cell 1

    hovered_image = np.asarray(image_graphic.data.value)
    assert not np.array_equal(
        hovered_image[segmentation == 1], dimmed_image[segmentation == 1]
    )
    widget.close()


def test_active_stim_id_boosts_the_cell_brightness(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    segmentation = np.zeros((8, 8), dtype=np.int32)
    segmentation[:, 4:] = 1
    overlay_image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay_image[:, :4] = (0, 0, 100)
    overlay_image[:, 4:] = (100, 0, 0)
    result = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=segmentation,
        overlay_image=overlay_image,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 0, 1: 1},
    )
    widget.set_data(result)
    image_graphic = widget.overlay_widget.overlay_view._image_graphic
    assert image_graphic is not None
    before_image = np.asarray(image_graphic.data.value).copy()

    widget._state.set_active_stim_ids({1})  # -> cellpose cell 1

    after_image = np.asarray(image_graphic.data.value)
    np.testing.assert_allclose(
        after_image[segmentation == 1].astype(np.float32),
        np.clip(
            before_image[segmentation == 1].astype(np.float32)
            * ACTIVE_STIM_BRIGHTNESS_BOOST,
            0,
            255,
        ),
        atol=1.0,
    )
    # the other, non-stimulated cell is untouched
    np.testing.assert_array_equal(
        after_image[segmentation == 0], before_image[segmentation == 0]
    )
    widget.close()
    widget.close()


def test_active_stim_ids_cleared_when_data_changes(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget._state.set_active_stim_ids({0})
    assert widget._state.active_stim_ids == frozenset({0})

    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))

    assert widget._state.active_stim_ids == frozenset()
    widget.close()


def test_hover_update_does_not_trigger_a_full_overlay_rebuild(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """A hover can fire at pointer-move frequency - it must only recolor
    the existing image, never pay for set_overlay's legend/label rebuild."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))

    full_rebuild_calls = 0
    original_set_overlay = widget.overlay_widget.set_overlay

    def _counting_set_overlay(*args: Any, **kwargs: Any) -> None:
        nonlocal full_rebuild_calls
        full_rebuild_calls += 1
        original_set_overlay(*args, **kwargs)

    monkeypatch.setattr(widget.overlay_widget, "set_overlay", _counting_set_overlay)

    widget._state.set_hovered_stim_id(1)

    assert full_rebuild_calls == 0
    widget.close()


def test_lasso_selection_adds_cells_and_hides_the_rest(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_lasso_selected.emit([1, 2])

    assert widget._selected_cellpose_ids == {1, 2}
    assert widget._greyed_cellpose_ids == {0}
    assert emitted == [[0, True]]
    widget.close()


def test_lasso_selection_grows_when_enclosing_only_new_cells(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget.overlay_widget.neuron_lasso_selected.emit([1])
    assert widget._greyed_cellpose_ids == {0, 2}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_lasso_selected.emit([2])

    assert widget._selected_cellpose_ids == {1, 2}
    assert widget._greyed_cellpose_ids == {0}
    assert emitted == [[2, False]]
    widget.close()


def test_lasso_selection_toggles_cells_already_selected(qtbot: QtBot) -> None:
    """Regression: a lasso used to be purely additive - enclosing an
    already-selected cell alongside new ones now deselects it instead,
    same as re-clicking it directly would."""
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget.overlay_widget.neuron_lasso_selected.emit([1])
    assert widget._selected_cellpose_ids == {1}

    widget.overlay_widget.neuron_lasso_selected.emit([1, 2])

    assert widget._selected_cellpose_ids == {2}
    assert widget._greyed_cellpose_ids == {0, 1}
    widget.close()
    widget.close()


def test_overlay_view_lasso_selects_only_enclosed_cells(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.zeros((10, 10), dtype=np.int32)
    overlay_view._label_centers = {1: (2.0, 2.0), 2: (8.0, 8.0)}
    positions = iter(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([6.0, 0.0, 0.0]),
            np.array([6.0, 6.0, 0.0]),
            np.array([0.0, 6.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: next(positions),
    )

    with qtbot.waitSignal(overlay_view.neuron_lasso_selected) as blocker:
        overlay_view._cell_selection_pointer_down(_right_click_event())
        overlay_view.pointer_move(None)  # crosses tolerance -> becomes a lasso
        overlay_view.pointer_move(None)
        overlay_view.pointer_move(None)
        overlay_view._cell_selection_pointer_up(None)

    assert blocker.args == [[1]]
    assert overlay_view._lasso_points == []
    assert overlay_view._lasso_line is None
    overlay_view.close()


def test_overlay_view_lasso_ignores_loop_enclosing_nothing(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.zeros((10, 10), dtype=np.int32)
    overlay_view._label_centers = {1: (8.0, 8.0)}
    positions = iter(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 2.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: next(positions),
    )
    emitted: list[list[int]] = []
    overlay_view.neuron_lasso_selected.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(_right_click_event())
    overlay_view.pointer_move(None)
    overlay_view.pointer_move(None)
    overlay_view.pointer_move(None)
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == []
    overlay_view.close()


def test_overlay_view_plain_click_does_not_start_a_lasso(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """A click with no (or minimal) movement stays a plain click/select,
    never a lasso - lasso only kicks in once the drag crosses tolerance."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.array([[1]], dtype=np.int32)
    overlay_view._label_centers = {1: (0.0, 0.0)}
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),
    )
    lasso_emitted: list[list[int]] = []
    overlay_view.neuron_lasso_selected.connect(lasso_emitted.append)

    with qtbot.waitSignal(overlay_view.neuron_clicked) as blocker:
        overlay_view._cell_selection_pointer_down(_right_click_event())
        overlay_view._cell_selection_pointer_up(None)
        qtbot.wait(10)

    assert blocker.args == [1]
    assert lasso_emitted == []
    overlay_view.close()


def test_overlay_view_renders_striped_cell_colors(qtbot: QtBot) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    segmentation = np.ones((8, 8), dtype=np.int32)

    colors = overlay_view._build_segmentation_colors(
        segmentation,
        legend_data=None,
        striped_cell_ids={1},
    )
    cell_colors = colors[segmentation == 1]

    assert np.any(np.all(np.isclose(cell_colors[:, :3], STRIPED_CELL_GREY), axis=1))
    assert np.any(np.all(np.isclose(cell_colors[:, :3], STRIPED_CELL_WHITE), axis=1))
    assert np.allclose(cell_colors[:, 3], STRIPED_CELL_ALPHA)
    overlay_view.close()


def test_overlay_view_emits_neuron_clicked_on_plain_click(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.array(
        [[0, 0, 1], [0, 0, 1], [-1, -1, -1]],
        dtype=np.int32,
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([2.0, 0.0, 0.0]),
    )

    with qtbot.waitSignal(overlay_view.neuron_clicked) as blocker:
        overlay_view._cell_selection_pointer_down(_right_click_event())
        overlay_view._cell_selection_pointer_up(None)

    assert blocker.args == [1]
    overlay_view.close()


def test_overlay_view_ignores_drag_past_tolerance_with_no_move_event(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """Releasing far from the down position without any pointer_move in
    between never became a lasso (that transition only happens in
    pointer_move), so it's still treated as an over-tolerance click and
    ignored."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.array([[0, 1]], dtype=np.int32)
    positions = iter(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([overlay_view._world_tolerance() + 10.0, 0.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: next(positions),
    )
    emitted: list[int] = []
    overlay_view.neuron_clicked.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(_right_click_event())
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == []
    overlay_view.close()


def test_overlay_view_ignores_click_when_selection_disabled(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.segmentation = np.array([[1]], dtype=np.int32)
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),
    )
    emitted: list[int] = []
    overlay_view.neuron_clicked.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(_right_click_event())
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == []
    overlay_view.close()


def test_overlay_view_left_click_selects_right_click_does_not(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """SpatialFootprintWidget restricts plain-click selection to the left
    button and lasso to the right button - verify the underlying
    OverlayView actually honors that split."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.cell_click_button = 1
    overlay_view.cell_lasso_button = 2
    overlay_view.segmentation = np.array([[1]], dtype=np.int32)
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),
    )
    emitted: list[int] = []
    overlay_view.neuron_clicked.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(_right_click_event())
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)
    assert emitted == []  # a plain right click does not select

    overlay_view._cell_selection_pointer_down(SimpleNamespace(button=1))  # left
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)
    assert emitted == [1]
    overlay_view.close()


def test_overlay_view_shift_left_click_without_a_drag_still_just_selects(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """Shift only matters if the pointer actually moves past tolerance
    (see test_overlay_view_shift_left_drag_lassos) - a plain Shift+left
    click with no drag still selects normally, it doesn't need Shift held
    to do nothing differently."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.cell_click_button = 1
    overlay_view.cell_lasso_button = 2
    overlay_view.segmentation = np.array([[1]], dtype=np.int32)
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),
    )
    emitted: list[int] = []
    overlay_view.neuron_clicked.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(
        SimpleNamespace(button=1, modifiers=("Shift",))
    )
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == [1]
    overlay_view.close()


def test_overlay_view_left_drag_past_tolerance_abandons_the_gesture(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """A left drag never becomes a lasso and never selects either - it's
    left entirely to the camera controller to pan, uninterrupted."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.cell_click_button = 1
    overlay_view.cell_lasso_button = 2
    overlay_view.segmentation = np.array([[0, 1]], dtype=np.int32)
    positions = iter(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([overlay_view._world_tolerance() + 10.0, 0.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: next(positions),
    )
    lasso_emitted: list[list[int]] = []
    overlay_view.neuron_lasso_selected.connect(lasso_emitted.append)

    overlay_view._cell_selection_pointer_down(SimpleNamespace(button=1))  # left
    overlay_view.pointer_move(None)  # crosses tolerance

    assert overlay_view._cell_click_down_world is None
    assert overlay_view._lasso_points == []
    assert overlay_view.subplot.controller.enabled  # never touched
    assert lasso_emitted == []
    overlay_view.close()


def test_overlay_view_shift_left_drag_lassos(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """Holding Shift turns a left drag into a lasso too, without needing
    the dedicated right-button lasso gesture."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.cell_click_button = 1
    overlay_view.cell_lasso_button = 2
    overlay_view.segmentation = np.array([[0, 1]], dtype=np.int32)
    positions = iter(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([overlay_view._world_tolerance() + 10.0, 0.0, 0.0]),
        ]
    )
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: next(positions),
    )

    overlay_view._cell_selection_pointer_down(
        SimpleNamespace(button=1, modifiers=("Shift",))
    )
    overlay_view.pointer_move(None)  # crosses tolerance

    assert overlay_view._lasso_points != []
    assert not overlay_view.subplot.controller.enabled  # disabled for the lasso drag
    overlay_view.close()


def test_spatial_footprint_widget_click_and_lasso_use_different_buttons(
    qtbot: QtBot,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    assert widget.overlay_widget.overlay_view.cell_click_button == 1
    assert widget.overlay_widget.overlay_view.cell_lasso_button == 2
    widget.close()


def test_overlay_view_context_menu_event_does_not_bubble_to_parent(
    qtbot: QtBot,
) -> None:
    """Regression: a right click on the actual cell-drawing canvas used to
    still pop open an ancestor's "Detach panel..." context menu alongside
    starting the lasso. Excluding the canvas from the CustomContextMenu
    target list (gui.py's _install_detach_context_menus) wasn't enough on
    its own - Qt's default policy (NoContextMenu) explicitly *defers an
    unhandled QContextMenuEvent to the parent widget*, so the event still
    reached the ancestor's CustomContextMenu handler. Only PreventContextMenu
    blocks that deferral outright."""
    parent = QtWidgets.QWidget()
    parent.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    requests: list[QtCore.QPoint] = []
    parent.customContextMenuRequested.connect(requests.append)
    layout = QtWidgets.QVBoxLayout(parent)
    overlay_view = OverlayView(parent)
    layout.addWidget(overlay_view)
    qtbot.addWidget(parent)

    for target in (overlay_view, overlay_view.widget):
        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse, QtCore.QPoint(1, 1)
        )
        QtWidgets.QApplication.sendEvent(target, event)

    assert requests == []


def test_overlay_view_background_click_emits_background_clicked(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.enable_cell_selection = True
    overlay_view.segmentation = np.array([[-1]], dtype=np.int32)
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: np.array([0.0, 0.0, 0.0]),
    )
    clicked: list[int] = []
    overlay_view.neuron_clicked.connect(clicked.append)

    with qtbot.waitSignal(overlay_view.background_clicked):
        overlay_view._cell_selection_pointer_down(_right_click_event())
        overlay_view._cell_selection_pointer_up(None)

    assert clicked == []
    overlay_view.close()


def test_background_click_clears_the_whole_selection(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    widget.overlay_widget.neuron_clicked.emit(1)
    assert widget._greyed_cellpose_ids == {0, 2}
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.background_clicked.emit()

    assert widget._selected_cellpose_ids == set()
    assert widget._greyed_cellpose_ids == set()
    assert sorted(emitted) == [[0, False], [2, False]]
    widget.close()


def test_overlay_view_ignores_aoi_drag_when_not_editable(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.overlay_image = np.zeros((10, 10, 3), dtype=np.uint8)
    config = _aoi_config()
    overlay_view._aoi_config = config
    overlay_view._aoi_visible = True
    overlay_view.aoi_editable = False
    corner = square_world_corners(config.square, overlay_view.overlay_image.shape[0])[0]
    monkeypatch.setattr(
        overlay_view.subplot,
        "map_screen_to_world",
        lambda ev: corner,
    )

    overlay_view._aoi_pointer_down(None)

    assert overlay_view._drag_state is None

    overlay_view.aoi_editable = True
    overlay_view._aoi_pointer_down(SimpleNamespace(stop_propagation=lambda: None))

    assert overlay_view._drag_state is not None
    overlay_view.close()


def test_overlay_view_body_drag_requires_shift(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    """Plain click-and-drag inside a shape's body must not hijack panning the
    FOV (issue #531) - only Shift+drag starts a body move. Corner/handle hits
    (see test_overlay_view_ignores_aoi_drag_when_not_editable) stay
    always-active regardless of Shift."""
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.overlay_image = np.zeros((20, 20, 3), dtype=np.uint8)
    config = _wide_square_aoi_config()
    overlay_view._aoi_config = config
    overlay_view._aoi_visible = True
    overlay_view.aoi_editable = True
    square = config.square
    body_user = (square.x + square.side / 2, square.y + square.side / 2)
    body_world = user_to_world_point(*body_user, overlay_view.overlay_image.shape[0])
    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: body_world
    )

    overlay_view._aoi_pointer_down(
        SimpleNamespace(stop_propagation=lambda: None, modifiers=())
    )
    assert overlay_view._drag_state is None

    overlay_view._aoi_pointer_down(
        SimpleNamespace(stop_propagation=lambda: None, modifiers=("Shift",))
    )
    assert overlay_view._drag_state is not None
    assert overlay_view._drag_state.part == "body"
    overlay_view.close()


def test_aoi_undo_redo_noop_with_empty_history(qtbot: QtBot) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)

    assert not overlay_view.can_undo_aoi_edit()
    assert not overlay_view.can_redo_aoi_edit()
    assert not overlay_view.undo_aoi_edit()
    assert not overlay_view.redo_aoi_edit()
    overlay_view.close()


def test_aoi_click_without_movement_does_not_add_undo_entry(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.overlay_image = np.zeros((20, 20, 3), dtype=np.uint8)
    config = _wide_square_aoi_config()
    overlay_view._aoi_config = config
    overlay_view._aoi_visible = True
    overlay_view.aoi_editable = True
    square = config.square
    body_user = (square.x + square.side / 2, square.y + square.side / 2)
    body_world = user_to_world_point(*body_user, overlay_view.overlay_image.shape[0])
    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: body_world
    )

    overlay_view._aoi_pointer_down(
        SimpleNamespace(stop_propagation=lambda: None, modifiers=("Shift",))
    )
    overlay_view._aoi_pointer_up(SimpleNamespace(stop_propagation=lambda: None))

    assert not overlay_view.can_undo_aoi_edit()


def test_aoi_body_drag_can_be_undone_and_redone(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.overlay_image = np.zeros((20, 20, 3), dtype=np.uint8)
    image_height = overlay_view.overlay_image.shape[0]
    original = _wide_square_aoi_config()
    original_snapshot = clone_aoi_config(original)
    overlay_view._aoi_config = original
    overlay_view._aoi_visible = True
    overlay_view.aoi_editable = True
    square = original.square
    body_user = (square.x + square.side / 2, square.y + square.side / 2)
    body_world = user_to_world_point(*body_user, image_height)
    moved_world = user_to_world_point(body_user[0] + 2.0, body_user[1], image_height)

    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: body_world
    )
    overlay_view._aoi_pointer_down(
        SimpleNamespace(stop_propagation=lambda: None, modifiers=("Shift",))
    )
    assert overlay_view._drag_state is not None

    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: moved_world
    )
    overlay_view.pointer_move(SimpleNamespace(stop_propagation=lambda: None))
    overlay_view._aoi_pointer_up(SimpleNamespace(stop_propagation=lambda: None))

    assert overlay_view._aoi_config is not None
    assert overlay_view._aoi_config != original_snapshot
    moved_x = overlay_view._aoi_config.square.x
    assert overlay_view.can_undo_aoi_edit()
    assert not overlay_view.can_redo_aoi_edit()

    assert overlay_view.undo_aoi_edit()
    assert overlay_view._aoi_config == original_snapshot
    assert not overlay_view.can_undo_aoi_edit()
    assert overlay_view.can_redo_aoi_edit()

    assert overlay_view.redo_aoi_edit()
    assert overlay_view._aoi_config is not None
    assert overlay_view._aoi_config.square.x == moved_x
    overlay_view.close()


def test_aoi_undo_redo_history_resets_on_new_image(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    overlay_view = OverlayView()
    qtbot.addWidget(overlay_view)
    overlay_view.overlay_image = np.zeros((20, 20, 3), dtype=np.uint8)
    original = _wide_square_aoi_config()
    overlay_view._aoi_config = original
    overlay_view._aoi_visible = True
    overlay_view.aoi_editable = True
    square = original.square
    body_user = (square.x + square.side / 2, square.y + square.side / 2)
    body_world = user_to_world_point(*body_user, overlay_view.overlay_image.shape[0])
    moved_world = user_to_world_point(
        body_user[0] + 2.0, body_user[1], overlay_view.overlay_image.shape[0]
    )
    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: body_world
    )
    overlay_view._aoi_pointer_down(
        SimpleNamespace(stop_propagation=lambda: None, modifiers=("Shift",))
    )
    monkeypatch.setattr(
        overlay_view.subplot, "map_screen_to_world", lambda ev: moved_world
    )
    overlay_view.pointer_move(SimpleNamespace(stop_propagation=lambda: None))
    overlay_view._aoi_pointer_up(SimpleNamespace(stop_propagation=lambda: None))
    assert overlay_view.can_undo_aoi_edit()

    overlay_view.add_overlay_data(np.zeros((20, 20, 3), dtype=np.uint8), None)

    assert not overlay_view.can_undo_aoi_edit()
    assert not overlay_view.can_redo_aoi_edit()
    overlay_view.close()


def test_two_views_sharing_state_stay_in_sync_on_click(qtbot: QtBot) -> None:
    state = SpatialFootprintState()
    view_a = SpatialFootprintWidget(state=state)
    view_b = SpatialFootprintWidget(state=state)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)
    view_a.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))

    # data set via one view is visible in the other immediately
    assert view_b._cellpose_data is not None
    assert not view_b.empty_state_widget.isVisible()

    view_a.overlay_widget.neuron_clicked.emit(1)

    assert view_a._greyed_cellpose_ids == {0, 2}
    assert view_b._greyed_cellpose_ids == {0, 2}

    view_b.clear_selection()

    assert view_a._greyed_cellpose_ids == set()
    view_a.close()
    view_b.close()


def test_two_views_sharing_state_stay_in_sync_on_lasso(qtbot: QtBot) -> None:
    state = SpatialFootprintState()
    view_a = SpatialFootprintWidget(state=state)
    view_b = SpatialFootprintWidget(state=state)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)
    view_a.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))

    view_a.overlay_widget.neuron_lasso_selected.emit([1])

    assert view_a._selected_cellpose_ids == {1}
    assert view_b._selected_cellpose_ids == {1}
    assert view_a._greyed_cellpose_ids == {0, 2}
    assert view_b._greyed_cellpose_ids == {0, 2}
    view_a.close()
    view_b.close()


def test_widget_with_no_state_arg_gets_a_private_state(qtbot: QtBot) -> None:
    view_a = SpatialFootprintWidget()
    view_b = SpatialFootprintWidget()
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    view_a.set_data(_cellpose_result())
    view_a.overlay_widget.neuron_clicked.emit(1)

    assert view_a._greyed_cellpose_ids == {0, 2}
    assert view_b._greyed_cellpose_ids == set()
    assert view_b._cellpose_data is None
    view_a.close()
    view_b.close()


def test_close_fully_disconnects_from_shared_state(qtbot: QtBot) -> None:
    """Regression: closing a view left one connection (the signal-to-signal
    forwarding of neuron_greyed_out_changed) still wired to the shared state,
    since disconnecting by re-passing the callable doesn't work for a
    signal-to-signal connection - only disconnecting the exact handle
    .connect() returned does. A stale connection to a closed (and
    eventually garbage-collected) view crashes the next time any other view
    mutates the shared state (see issue: repeated detach/close cycles)."""
    state = SpatialFootprintState()
    view_a = SpatialFootprintWidget(state=state)
    qtbot.addWidget(view_a)
    view_a.set_data(_cellpose_result())
    baseline_data = state.receivers(state.data_changed)
    baseline_selection = state.receivers(state.selection_changed)
    baseline_greyed = state.receivers(state.neuron_greyed_out_changed)

    view_b = SpatialFootprintWidget(state=state)
    assert state.receivers(state.data_changed) == baseline_data + 1
    assert state.receivers(state.selection_changed) == baseline_selection + 1
    assert state.receivers(state.neuron_greyed_out_changed) == baseline_greyed + 1

    view_b.close()

    assert state.receivers(state.data_changed) == baseline_data
    assert state.receivers(state.selection_changed) == baseline_selection
    assert state.receivers(state.neuron_greyed_out_changed) == baseline_greyed
    view_a.close()
