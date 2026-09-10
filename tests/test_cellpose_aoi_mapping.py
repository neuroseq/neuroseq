from __future__ import annotations

import zipfile
from pathlib import Path

import brian2.numpy_ as np
import pytest
from PIL import Image
from pytestqt.qtbot import QtBot

from neuroseq import io
from neuroseq.aoi_shapes import user_to_world_point
from neuroseq.cellpose_import_dialog import AOI_PERCENT_MODE
from neuroseq.cellpose_import_dialog import CUSTOM_AOI_STRATEGY
from neuroseq.cellpose_import_dialog import CellposeImportDialog
from neuroseq.importer_cellpose import AOIRole
from neuroseq.importer_cellpose import AreasOfInterestConfig
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import CircleAOI
from neuroseq.importer_cellpose import SquareAOI
from neuroseq.importer_cellpose import default_aoi_config
from neuroseq.importer_cellpose import generate_aoi_stimulus_mapping
from neuroseq.importer_cellpose import point_in_circle
from neuroseq.importer_cellpose import point_in_square
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions


def _save_and_reload_json(data: dict, tmp_path: Path) -> dict:
    tmp_file = tmp_path / "aoi_roundtrip.nseq"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED) as zf:
        io.append_to_zip(data, "data", zf)
    with zipfile.ZipFile(tmp_file, "r", zipfile.ZIP_DEFLATED) as zf:
        return io.load_from_zip("data", zf)


def _world_center(
    user_x: float, user_y: float, image_height: int
) -> tuple[float, float]:
    return user_to_world_point(user_x, user_y, image_height)


def _make_dialog(tmp_path: Path, qtbot: QtBot) -> CellposeImportDialog:
    dialog = CellposeImportDialog()
    qtbot.addWidget(dialog)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    segmentation = np.zeros((20, 20), dtype=np.int32)
    segmentation[1:3, 1:3] = 1
    segmentation[7:9, 7:9] = 2
    segmentation[9:11, 9:11] = 3

    image_path = tmp_path / "cellpose_aoi_image.tif"
    seg_path = tmp_path / "cellpose_aoi_segmentation.npy"
    Image.fromarray(image).save(image_path)
    np.save(seg_path, segmentation)

    dialog._load_image_file(str(image_path))
    dialog._load_segmentation_file(str(seg_path))
    return dialog


def _select_aoi_strategy(dialog: CellposeImportDialog) -> None:
    for button in dialog.mapping_buttons.buttons():
        if button.text() == CUSTOM_AOI_STRATEGY:
            button.setChecked(True)
            break


def test_default_aoi_config_uses_centred_50_percent_square() -> None:
    cfg = default_aoi_config((400, 600))
    assert cfg.circle.diameter == 38
    assert cfg.square.side == 200
    assert cfg.circle.cx == 300
    assert cfg.circle.cy == 200
    assert cfg.square.x == 200
    assert cfg.square.y == 100
    assert cfg.square.role == AOIRole.PREFERENTIAL
    assert cfg.circle.role == AOIRole.EXCLUSION
    assert cfg.coordinate_mode == "percent"


def test_point_in_square_inclusive_edges() -> None:
    sq = SquareAOI(True, AOIRole.PREFERENTIAL, x=10, y=10, side=20)
    assert point_in_square(10, 10, sq)
    assert point_in_square(30, 30, sq)
    assert not point_in_square(31, 20, sq)


def test_point_in_circle() -> None:
    circle = CircleAOI(True, AOIRole.EXCLUSION, cx=50, cy=50, diameter=20)
    assert point_in_circle(50, 50, circle)
    assert point_in_circle(60, 50, circle)
    assert not point_in_circle(61, 50, circle)


def test_aoi_mapping_preferential_only() -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=0, y=0, side=50),
        circle=CircleAOI(False, AOIRole.EXCLUSION, cx=0, cy=0, diameter=10),
    )
    image_height = 100
    label_centers = {
        1: _world_center(10, 10, image_height),
        2: _world_center(60, 60, image_height),
        3: _world_center(40, 40, image_height),
        4: _world_center(90, 90, image_height),
    }
    mapping = generate_aoi_stimulus_mapping(
        np.array([1, 2, 3, 4], dtype=np.int32),
        label_centers,
        config,
        image_height,
    )
    assert mapping == {0: 1, 1: 3, 2: 2, 3: 4}


def test_aoi_mapping_exclusion_only() -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(False, AOIRole.PREFERENTIAL, x=0, y=0, side=10),
        circle=CircleAOI(True, AOIRole.EXCLUSION, cx=50, cy=50, diameter=30),
    )
    image_height = 100
    label_centers = {
        1: _world_center(45, 50, image_height),
        2: _world_center(55, 50, image_height),
        3: _world_center(10, 10, image_height),
        4: _world_center(90, 90, image_height),
    }
    mapping = generate_aoi_stimulus_mapping(
        np.array([1, 2, 3, 4], dtype=np.int32),
        label_centers,
        config,
        image_height,
    )
    assert mapping == {0: 3, 1: 4, 2: 1, 3: 2}


def test_aoi_mapping_overlap_exclusion_wins() -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=20, y=20, side=40),
        circle=CircleAOI(True, AOIRole.EXCLUSION, cx=50, cy=50, diameter=20),
    )
    image_height = 100
    label_centers = {
        1: _world_center(25, 25, image_height),
        2: _world_center(50, 50, image_height),
    }
    mapping = generate_aoi_stimulus_mapping(
        np.array([1, 2], dtype=np.int32),
        label_centers,
        config,
        image_height,
    )
    assert mapping == {0: 1, 1: 2}


def test_aoi_mapping_disabled_shape_ignored() -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=0, y=0, side=40),
        circle=CircleAOI(False, AOIRole.EXCLUSION, cx=70, cy=70, diameter=20),
    )
    image_height = 100
    label_centers = {
        1: _world_center(10, 10, image_height),
        2: _world_center(70, 70, image_height),
    }
    mapping = generate_aoi_stimulus_mapping(
        np.array([1, 2], dtype=np.int32),
        label_centers,
        config,
        image_height,
    )
    assert mapping == {0: 1, 1: 2}


def test_aoi_mapping_sorted_by_cellpose_id_within_tier() -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, x=0, y=0, side=40),
        circle=CircleAOI(False, AOIRole.EXCLUSION, cx=0, cy=0, diameter=10),
    )
    image_height = 100
    label_centers = {
        5: _world_center(10, 10, image_height),
        2: _world_center(20, 20, image_height),
        8: _world_center(30, 30, image_height),
    }
    mapping = generate_aoi_stimulus_mapping(
        np.array([5, 2, 8], dtype=np.int32),
        label_centers,
        config,
        image_height,
    )
    assert mapping == {0: 2, 1: 5, 2: 8}


def test_aoi_config_round_trips_through_io(tmp_path: Path) -> None:
    config = AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.EXCLUSION, x=12.5, y=23.5, side=44.0),
        circle=CircleAOI(False, AOIRole.PREFERENTIAL, cx=50.0, cy=60.0, diameter=18.5),
        show_shapes=False,
        coordinate_mode="percent",
    )
    result = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        aoi_config=config,
        segmentation_path=tmp_path / "segmentation.npy",
    )

    reloaded = _save_and_reload_json({"cellpose": result}, tmp_path)
    loaded_result = reloaded["cellpose"]
    assert loaded_result.aoi_config == config
    assert loaded_result.segmentation_path == result.segmentation_path


def test_background_image_round_trips_through_io(tmp_path: Path) -> None:
    background = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    result = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        background_image=background,
    )

    reloaded = _save_and_reload_json({"cellpose": result}, tmp_path)
    loaded_result = reloaded["cellpose"]

    np.testing.assert_array_equal(loaded_result.background_image, background)


def test_missing_background_image_round_trips_as_none(tmp_path: Path) -> None:
    result = CellposeImportResult(
        image_path=tmp_path / "image.png",
        segmentation_array=np.zeros((2, 2), dtype=np.int32),
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
    )

    reloaded = _save_and_reload_json({"cellpose": result}, tmp_path)

    assert reloaded["cellpose"].background_image is None


def test_dialog_selects_aoi_strategy_shows_subgroup(
    tmp_path: Path, qtbot: QtBot
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    assert not dialog.aoi_group.isHidden()
    assert not dialog.show_shapes_checkbox.isHidden()
    assert dialog.aoi_coordinate_mode_combo.currentText() == AOI_PERCENT_MODE
    assert dialog.square_x_spin.value() == 50
    assert dialog.square_y_spin.value() == 50
    assert dialog.square_side_spin.value() == 50
    dialog.close()


def test_import_button_is_default_after_files_load(
    tmp_path: Path, qtbot: QtBot
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)

    assert dialog.import_button.isDefault()
    assert not dialog.open_fiji_button.autoDefault()
    assert not dialog.run_cellpose_button.autoDefault()
    dialog.close()


def test_dialog_aoi_autofills_custom_mapping_line(tmp_path: Path, qtbot: QtBot) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    text = dialog.custom_mapping_line.text()
    assert text != ""
    parsed = dialog._parse_custom_mapping(text)
    assert parsed
    dialog.close()


def test_dialog_aoi_shows_region_summary(tmp_path: Path, qtbot: QtBot) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    summary = dialog.aoi_summary_browser.toPlainText()
    assert not dialog.aoi_summary_browser.isHidden()
    assert "Areas of interest summary" in summary
    assert "Square" in summary
    assert "Circle" in summary
    assert "preferential" in summary
    assert "exclusion" in summary
    dialog.close()


def test_dialog_aoi_changes_remap_on_spinbox_edit(tmp_path: Path, qtbot: QtBot) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    before = dialog._generate_stimulus_mapping(CUSTOM_AOI_STRATEGY)
    dialog.square_x_spin.setValue(0.0)
    dialog.square_y_spin.setValue(0.0)
    after = dialog._generate_stimulus_mapping(CUSTOM_AOI_STRATEGY)

    assert before != after
    assert dialog.custom_mapping_line.text() == ""
    dialog.close()


def test_dialog_aoi_percentage_mode_uses_centres_and_fov_size(
    tmp_path: Path, qtbot: QtBot
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)
    dialog.aoi_coordinate_mode_combo.setCurrentText(AOI_PERCENT_MODE)
    dialog.square_x_spin.setValue(50.0)
    dialog.square_y_spin.setValue(50.0)
    dialog.square_side_spin.setValue(70.0)
    dialog.circle_cx_spin.setValue(25.0)
    dialog.circle_cy_spin.setValue(75.0)
    dialog.circle_diameter_spin.setValue(20.0)

    config = dialog._current_aoi_config()
    assert config is not None
    assert config.coordinate_mode == "percent"
    assert config.square.x == 3.0
    assert config.square.y == 3.0
    assert config.square.side == 14.0
    assert config.circle.cx == 5.0
    assert config.circle.cy == 15.0
    assert config.circle.diameter == 4.0
    dialog.close()


def test_dialog_aoi_pixel_mode_uses_square_centre(tmp_path: Path, qtbot: QtBot) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)
    dialog.aoi_coordinate_mode_combo.setCurrentIndex(0)
    dialog.square_x_spin.setValue(10.0)
    dialog.square_y_spin.setValue(12.0)
    dialog.square_side_spin.setValue(6.0)

    config = dialog._current_aoi_config()
    assert config is not None
    assert config.square.x == 7.0
    assert config.square.y == 9.0
    assert config.square.side == 6.0

    dialog._set_aoi_controls(config)
    assert dialog.square_x_spin.value() == 10.0
    assert dialog.square_y_spin.value() == 12.0
    dialog.close()


def test_dialog_aoi_percent_mode_round_trips_through_set_aoi_controls(
    tmp_path: Path, qtbot: QtBot
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)
    dialog.aoi_coordinate_mode_combo.setCurrentText(AOI_PERCENT_MODE)
    original = {
        "square_x": 40.0,
        "square_y": 45.0,
        "square_side": 30.0,
        "circle_cx": 70.0,
        "circle_cy": 25.0,
        "circle_diameter": 15.0,
    }
    dialog.square_x_spin.setValue(original["square_x"])
    dialog.square_y_spin.setValue(original["square_y"])
    dialog.square_side_spin.setValue(original["square_side"])
    dialog.circle_cx_spin.setValue(original["circle_cx"])
    dialog.circle_cy_spin.setValue(original["circle_cy"])
    dialog.circle_diameter_spin.setValue(original["circle_diameter"])

    config = dialog._current_aoi_config()
    assert config is not None
    dialog._set_aoi_controls(config)

    assert dialog.square_x_spin.value() == pytest.approx(original["square_x"])
    assert dialog.square_y_spin.value() == pytest.approx(original["square_y"])
    assert dialog.square_side_spin.value() == pytest.approx(original["square_side"])
    assert dialog.circle_cx_spin.value() == pytest.approx(original["circle_cx"])
    assert dialog.circle_cy_spin.value() == pytest.approx(original["circle_cy"])
    assert dialog.circle_diameter_spin.value() == pytest.approx(
        original["circle_diameter"]
    )
    dialog.close()


def test_dialog_aoi_import_updates_region_summary(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    dialog._import_data()

    assert dialog.get_imported_data() is not None
    summary = dialog.aoi_summary_browser.toPlainText()
    assert "Stim IDs:" in summary
    assert "Cellpose IDs:" in summary
    assert any(call["name"] == "information" for call in qmessagebox_calls)
    dialog.close()


def test_dialog_aoi_persists_through_reopen(tmp_path: Path, qtbot: QtBot) -> None:
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)
    dialog.square_x_spin.setValue(1.5)
    dialog.circle_diameter_spin.setValue(6.5)
    dialog.show_shapes_checkbox.setChecked(False)

    mapping = dialog._generate_stimulus_mapping(CUSTOM_AOI_STRATEGY)
    assert dialog.image_path is not None
    result = dialog.importer.create_result(
        dialog.image_path,
        mapping,
        mapping_strategy=CUSTOM_AOI_STRATEGY,
        aoi_config=dialog._current_aoi_config(),
    )

    reopened = CellposeImportDialog(existing_data=result)
    qtbot.addWidget(reopened)
    assert reopened._get_mapping_strategy() == CUSTOM_AOI_STRATEGY
    assert reopened.square_x_spin.value() == 1.5
    assert reopened.circle_diameter_spin.value() == 6.5
    assert not reopened.show_shapes_checkbox.isChecked()
    dialog.close()
    reopened.close()


def test_aoi_drag_import_persists_through_project_save_reload(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Reproduces the real user flow behind issue #518: drag an AOI shape,
    click Import, apply the result to a project (as gui.py does), save the
    project to a .nseq archive and reload it - the dragged mapping/config
    must come back unchanged."""
    dialog = _make_dialog(tmp_path, qtbot)
    _select_aoi_strategy(dialog)

    baseline_mapping = dialog._generate_stimulus_mapping(CUSTOM_AOI_STRATEGY)

    # Simulate the geometry-changed signal that OverlayView emits after a
    # real mouse drag (cell_overlay_widget.py: _aoi_pointer_up/_apply_drag).
    dragged_config = dialog._current_aoi_config()
    assert dragged_config is not None
    dragged_config.square.x = 0.0
    dragged_config.square.y = 0.0
    dragged_config.square.role = AOIRole.EXCLUSION
    dialog._on_overlay_aoi_geometry_changed(dragged_config)

    dragged_mapping = dialog._generate_stimulus_mapping(CUSTOM_AOI_STRATEGY)
    assert dragged_mapping != baseline_mapping

    dialog._import_data()
    imported = dialog.get_imported_data()
    assert imported is not None
    assert imported.aoi_config is not None
    assert imported.aoi_config.square.role == AOIRole.EXCLUSION
    assert imported.stimulus_to_cellpose_mapping == dragged_mapping

    # Mirrors gui.py's _apply_cellpose_import_result, which unconditionally
    # assigns the dialog's result onto the project.
    project = Project()
    project.cellpose_data = imported

    tmp_file = tmp_path / "project_with_dragged_aoi.nseq"
    project.save(tmp_file, SaveLoadOptions(True, True, False, True))

    reloaded = Project(tmp_file)
    assert reloaded.cellpose_data is not None
    assert reloaded.cellpose_data.mapping_strategy == CUSTOM_AOI_STRATEGY
    assert reloaded.cellpose_data.stimulus_to_cellpose_mapping == dragged_mapping
    assert reloaded.cellpose_data.aoi_config == dragged_config
    assert reloaded.cellpose_data.segmentation_path == imported.segmentation_path

    dialog.close()
