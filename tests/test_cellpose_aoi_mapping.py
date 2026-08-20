from __future__ import annotations

import zipfile
from pathlib import Path

import brian2.numpy_ as np
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


def test_default_aoi_config_uses_centred_70_percent_square() -> None:
    cfg = default_aoi_config((400, 600))
    assert cfg.circle.diameter == 50
    assert cfg.square.side == 280
    assert cfg.circle.cx == 300
    assert cfg.circle.cy == 200
    assert cfg.square.x == 160
    assert cfg.square.y == 60
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
    )

    reloaded = _save_and_reload_json({"cellpose": result}, tmp_path)
    loaded_result = reloaded["cellpose"]
    assert loaded_result.aoi_config == config


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
    assert dialog.square_side_spin.value() == 70
    dialog.close()


def test_import_button_is_default_after_files_load(
    tmp_path: Path, qtbot: QtBot
) -> None:
    dialog = _make_dialog(tmp_path, qtbot)

    assert dialog.import_button.isDefault()
    assert not dialog.open_fiji_button.autoDefault()
    assert not dialog.change_fiji_button.autoDefault()
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
