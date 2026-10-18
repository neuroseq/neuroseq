"""Cellpose data keeps the raw top-left orientation; only the view flips (#601)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pytestqt.qtbot import QtBot

from neuroseq.export.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.importers.cellpose_import_dialog import CUSTOM_AOI_STRATEGY
from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
from neuroseq.importers.importer_cellpose import generate_aoi_stimulus_mapping
from neuroseq.io import CELLPOSE_ORIENTATION
from neuroseq.io import load_from_zip
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget
from neuroseq.visualization.cell_overlay_widget import OverlayView
from neuroseq.visualization.cell_overlay_widget import _pixel_at
from neuroseq.visualization.cell_overlay_widget import compute_cell_centroids
from tests.test_exporter_syscon2 import _template_header

# Saved and exported by NeuroSeq before #601; see make_fixture.py and
# make_aoi_fixture.py there.
LEGACY = Path(__file__).parent / "data" / "legacy_cellpose_v1"
SAVE_ALL = SaveLoadOptions(True, True, False, True)


def _legacy_project() -> Project:
    return Project(LEGACY / "legacy.nseq")


def test_legacy_project_loads_cellpose_data_in_raw_orientation() -> None:
    result = _legacy_project().cellpose_data
    assert result is not None
    np.testing.assert_array_equal(
        result.segmentation_array, np.load(LEGACY / "masks.npy") - 1
    )
    raw_image = np.asarray(Image.open(LEGACY / "image.png").convert("RGB"))
    np.testing.assert_array_equal(result.background_image, raw_image)
    # The overlay is the image tinted under the masks: background pixels match.
    background = result.segmentation_array < 0
    np.testing.assert_array_equal(
        result.overlay_image[background], raw_image[background]
    )


def test_legacy_project_exports_a_byte_identical_seq() -> None:
    project = _legacy_project()
    seq_text, _ = generate_syscon2_sequence_text(
        project.stim_arrays["stim"],
        project.cellpose_data,
        template_text=_template_header(),
        spiralling=False,
        stimulation_spots_per_roi=3,
    )
    assert seq_text == (LEGACY / "legacy_export.seq").read_text()


def test_legacy_aoi_mapping_is_reproduced() -> None:
    result = _legacy_project().cellpose_data
    assert result is not None and result.aoi_config is not None
    ids = np.unique(result.segmentation_array)
    mapping = generate_aoi_stimulus_mapping(
        ids[ids >= 0],
        compute_cell_centroids(result.segmentation_array),
        result.aoi_config,
    )
    legacy = json.loads((LEGACY / "aoi_mapping.json").read_text())
    assert mapping == {int(k): v for k, v in legacy.items()}
    assert result.stimulus_to_cellpose_mapping == mapping


def test_resaving_does_not_flip_again_and_hides_data_from_old_readers(
    tmp_path: Path,
) -> None:
    project = _legacy_project()
    first = tmp_path / "first.nseq"
    project.save(first, SAVE_ALL)
    second = tmp_path / "second.nseq"
    Project(first).save(second, SAVE_ALL)

    reloaded = Project(second).cellpose_data
    assert reloaded is not None
    np.testing.assert_array_equal(
        reloaded.segmentation_array, np.load(LEGACY / "masks.npy") - 1
    )
    with zipfile.ZipFile(second) as zf:
        names = zf.namelist()
        blob = json.loads(zf.read("cellpose_data_v2.json"))
        # NeuroSeq before #601 reads "cellpose_data": it finds nothing rather
        # than raw arrays it would treat as flipped (mirrored SysCon targets).
        assert load_from_zip("cellpose_data", zf) == {}
    assert "cellpose_data.json" not in names
    assert blob["cellpose_import"]["orientation"] == CELLPOSE_ORIENTATION


def test_image_views_flip_the_camera_not_the_data(qtbot: QtBot) -> None:
    overlay = OverlayView()
    qtbot.addWidget(overlay)
    image = np.zeros((6, 10, 3), dtype=np.uint8)
    segmentation = np.full((6, 10), -1, dtype=np.int32)
    segmentation[1:3, 1:4] = 0
    overlay.add_overlay_data(image, segmentation)
    assert overlay.subplot.camera.local.scale_y < 0
    overlay.show_images_top_down()  # idempotent
    assert overlay.subplot.camera.local.scale_y < 0
    # World-space labels are mirrored back so they read upright.
    for graphics in overlay._label_graphics.values():
        for key in ("cellpose", "stim", "stim_dash"):
            assert graphics[key].world_object.local.scale_y < 0
    # Pixel centres sit at integers: centre the camera on the image itself.
    position = overlay.subplot.camera.get_state()["position"]
    assert tuple(position[:2]) == (4.5, 2.5)

    video = CalciumVideoWidget()
    qtbot.addWidget(video)
    video.current_frame = image
    video._draw_current_frame()
    assert video.subplot.camera.local.scale_y < 0


def test_pointer_positions_round_to_the_nearest_pixel() -> None:
    assert _pixel_at(2.49, 0.51) == (2, 1)
    assert _pixel_at(-0.49, 3.0) == (0, 3)
    assert _pixel_at(-0.51, 0.0) == (-1, 0)


def test_reopened_legacy_aoi_project_keeps_shapes_and_mapping(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """AOI shapes saved through the dialog before #601 come back unchanged.

    AOI coordinates were always stored top-left, like the raw image; only the
    masks were flipped. Reopening the dialog must show the saved shapes over
    the same cells and reproduce the saved mapping, and re-importing and
    re-saving must not move anything or flip the masks again.
    """
    project = Project(LEGACY / "legacy_aoi_dialog.nseq")
    saved = project.cellpose_data
    assert saved is not None and saved.aoi_config is not None
    masks = np.load(LEGACY / "masks.npy") - 1
    np.testing.assert_array_equal(saved.segmentation_array, masks)
    assert saved.mapping_strategy == CUSTOM_AOI_STRATEGY
    assert saved.stimulus_to_cellpose_mapping == {0: 0, 1: 2, 2: 1}
    square, circle = saved.aoi_config.square, saved.aoi_config.circle
    assert (square.x, square.y, square.side) == (0.0, 0.0, 12.0)
    assert (circle.cx, circle.cy, circle.diameter) == (30.0, 17.0, 12.0)

    dialog = CellposeImportDialog(existing_data=saved)
    qtbot.addWidget(dialog)
    checked = [b.text() for b in dialog.mapping_buttons.buttons() if b.isChecked()]
    assert checked == [CUSTOM_AOI_STRATEGY]
    shown = dialog._current_aoi_config()
    assert shown is not None
    assert (shown.square.x, shown.square.y, shown.square.side) == (0.0, 0.0, 12.0)
    assert (shown.circle.cx, shown.circle.cy, shown.circle.diameter) == (
        30.0,
        17.0,
        12.0,
    )
    # Cell 0 (top left) lies in the preferential square and cell 1 (lower
    # right) in the exclusion circle, exactly as when the shapes were saved.
    assert dialog._compute_aoi_mapping() == saved.stimulus_to_cellpose_mapping

    dialog.import_button.click()
    reimported = dialog.get_imported_data()
    assert reimported is not None
    project.cellpose_data = reimported
    resaved = tmp_path / "resaved.nseq"
    project.save(resaved, SAVE_ALL)
    again = Project(resaved).cellpose_data
    assert again is not None and again.aoi_config is not None
    np.testing.assert_array_equal(again.segmentation_array, masks)
    assert again.stimulus_to_cellpose_mapping == saved.stimulus_to_cellpose_mapping
    assert again.aoi_config.square == saved.aoi_config.square
    assert again.aoi_config.circle == saved.aoi_config.circle
