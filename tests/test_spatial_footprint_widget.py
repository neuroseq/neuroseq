from __future__ import annotations

from pathlib import Path
from typing import Any

import brian2.numpy_ as np
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot

from neuroseq.cell_overlay_widget import STRIPED_CELL_ALPHA
from neuroseq.cell_overlay_widget import STRIPED_CELL_GREY
from neuroseq.cell_overlay_widget import STRIPED_CELL_WHITE
from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cell_overlay_widget import OverlayView
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spatial_footprint_widget import SpatialFootprintWidget


def _cellpose_result(
    mapping: dict[int, int] | None = None,
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
    )


def test_set_data_populates_overlay_and_clears_on_none(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)

    assert not widget.empty_state_widget.isHidden()
    assert widget.overlay_widget.isHidden()

    widget.set_data(_cellpose_result())
    assert widget.overlay_widget.segmentation is not None
    assert widget.empty_state_widget.isHidden()
    assert not widget.overlay_widget.isHidden()

    widget.set_data(None)
    assert widget.overlay_widget.overlay_image is None
    assert widget.overlay_widget.segmentation is None
    assert not widget.empty_state_widget.isHidden()
    assert widget.overlay_widget.isHidden()
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


def test_clicking_a_cell_emits_stim_id_via_mapping(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 2, 1: 1, 2: 0}))

    with qtbot.waitSignal(widget.neuron_greyed_out_changed) as blocker:
        widget.overlay_widget.neuron_clicked.emit(2)

    assert blocker.args == [0, True]
    widget.close()


def test_clicking_same_cell_twice_toggles_back(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result())
    emitted: list[list[Any]] = []
    widget.neuron_greyed_out_changed.connect(lambda *args: emitted.append(list(args)))

    widget.overlay_widget.neuron_clicked.emit(1)
    widget.overlay_widget.neuron_clicked.emit(1)

    assert emitted[-1] == [1, False]
    assert widget._greyed_cellpose_ids == set()
    widget.close()


def test_identity_mapping_when_none_present(qtbot: QtBot) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping=None))

    with qtbot.waitSignal(widget.neuron_greyed_out_changed) as blocker:
        widget.overlay_widget.neuron_clicked.emit(2)

    assert blocker.args == [2, True]
    widget.close()


def test_greyed_cell_uses_striped_overlay_and_black_label(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
    widget = SpatialFootprintWidget()
    qtbot.addWidget(widget)
    widget.set_data(_cellpose_result(mapping={0: 0, 1: 1, 2: 2}))
    captured_striped_cell_ids: list[set[int]] = []

    original_set_overlay = widget.overlay_widget.set_overlay

    def capture_set_overlay(*args: Any, **kwargs: Any) -> None:
        captured_striped_cell_ids.append(set(kwargs.get("striped_cell_ids") or []))
        original_set_overlay(*args, **kwargs)

    monkeypatch.setattr(widget.overlay_widget, "set_overlay", capture_set_overlay)

    widget.overlay_widget.neuron_clicked.emit(1)

    assert captured_striped_cell_ids[-1] == {1}
    stim_label = widget.overlay_widget.overlay_view._label_graphics[1]["stim"]
    assert stim_label.visible
    assert stim_label.text == "1"
    assert stim_label.face_color == (0.0, 0.0, 0.0, 1.0)
    widget.close()


def test_clicking_cell_greys_rendered_spatial_footprint(qtbot: QtBot) -> None:
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
    graphic = widget.overlay_widget.overlay_view._segmentation_graphic
    assert image_graphic is not None
    assert graphic is not None
    before_image = np.asarray(image_graphic.data.value).copy()
    before_segmentation = np.asarray(graphic.data.value).copy()

    widget.overlay_widget.neuron_clicked.emit(1)

    after_image = np.asarray(image_graphic.data.value)
    after_segmentation = np.asarray(graphic.data.value)
    cell_image_colors = after_image[result.segmentation_array == 1]
    cell_image_rgb = cell_image_colors[:, :3].astype(np.float32) / 255.0
    assert np.any(
        np.all(np.isclose(cell_image_rgb, STRIPED_CELL_GREY, atol=1 / 255), axis=1)
    )
    assert np.any(
        np.all(np.isclose(cell_image_rgb, STRIPED_CELL_WHITE, atol=1 / 255), axis=1)
    )
    assert not np.allclose(
        after_image[result.segmentation_array == 1],
        before_image[result.segmentation_array == 1],
    )
    assert np.allclose(
        after_image[result.segmentation_array == 0],
        before_image[result.segmentation_array == 0],
    )
    stripe_overlay_colors = after_segmentation[result.segmentation_array == 1]
    assert np.allclose(stripe_overlay_colors[:, 3], STRIPED_CELL_ALPHA)
    assert np.allclose(
        after_segmentation[result.segmentation_array == 0],
        before_segmentation[result.segmentation_array == 0],
    )

    widget.overlay_widget.neuron_clicked.emit(1)
    assert np.allclose(np.asarray(image_graphic.data.value), before_image)
    assert np.allclose(np.asarray(graphic.data.value), before_segmentation)
    widget.close()


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
        overlay_view._cell_selection_pointer_down(None)
        overlay_view._cell_selection_pointer_up(None)

    assert blocker.args == [1]
    overlay_view.close()


def test_overlay_view_ignores_drag_past_tolerance(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
) -> None:
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

    overlay_view._cell_selection_pointer_down(None)
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

    overlay_view._cell_selection_pointer_down(None)
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == []
    overlay_view.close()


def test_overlay_view_ignores_background_click(
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
    emitted: list[int] = []
    overlay_view.neuron_clicked.connect(emitted.append)

    overlay_view._cell_selection_pointer_down(None)
    overlay_view._cell_selection_pointer_up(None)
    qtbot.wait(10)

    assert emitted == []
    overlay_view.close()
