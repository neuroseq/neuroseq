"""Saving a frame range of the imaging data (issue #609)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.calcium_video_widget import CalciumVideoWidget
from neuroseq.save_imaging_dialog import SaveImagingDialog


@pytest.fixture
def video(qtbot: QtBot, tmp_path: Path) -> CalciumVideoWidget:
    # Ten pages, each holding its own index; two interleaved channels.
    path = tmp_path / "movie.tif"
    pages = np.stack(
        [np.full((4, 5), 100 + page, dtype=np.uint16) for page in range(10)]
    )
    tifffile.imwrite(path, pages)
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths([path], 20.0, warn_if_missing_cellpose=False)
    return widget


def test_original_substack_keeps_pixels_and_frame_rate(
    video: CalciumVideoWidget, tmp_path: Path
) -> None:
    export = tmp_path / "substack.tif"
    video.export_labeled_video(export, first=3, last=6, original=True)

    with tifffile.TiffFile(export) as tif:
        frames = tif.asarray()
        assert tif.is_imagej
        assert tif.imagej_metadata["finterval"] == pytest.approx(1 / 20)
    assert frames.dtype == np.uint16
    assert frames.shape == (4, 4, 5)
    np.testing.assert_array_equal(frames[:, 0, 0], [103, 104, 105, 106])


def test_substack_of_one_channel(video: CalciumVideoWidget, tmp_path: Path) -> None:
    video.set_channels(2, 2)  # pages 1, 3, 5, 7, 9
    export = tmp_path / "channel2.tif"
    video.export_labeled_video(export, first=1, last=3, original=True)
    np.testing.assert_array_equal(tifffile.imread(export)[:, 0, 0], [103, 105, 107])


@pytest.mark.parametrize("suffix", [".tif", ".gif"])
def test_marked_export_saves_only_the_range(
    video: CalciumVideoWidget, tmp_path: Path, suffix: str
) -> None:
    from PIL import Image

    export = tmp_path / f"marked{suffix}"
    video.export_labeled_video(export, first=2, last=4)
    if suffix == ".gif":
        with Image.open(export) as image:
            assert image.n_frames == 3
    else:
        with tifffile.TiffFile(export) as tif:
            assert len(tif.pages) == 3
            assert tif.pages[0].shape == (4, 5, 3)  # RGB, as displayed


def test_dialog_range_and_current_frame(qtbot: QtBot) -> None:
    dialog = SaveImagingDialog(30, current_frame=12)
    qtbot.addWidget(dialog)
    assert dialog.frame_range == (0, 29)
    assert not dialog.original
    assert dialog.count_label.text() == "30 frames"

    buttons = dialog.findChildren(QtWidgets.QPushButton)
    first_current = next(b for b in buttons if b.text() == "Current frame")
    all_frames = next(b for b in buttons if b.text() == "All frames")
    first_current.click()
    assert dialog.frame_range == (11, 29)
    dialog.last_spin.setValue(5)  # before the first frame: first follows
    assert dialog.frame_range == (4, 4)
    assert dialog.count_label.text() == "1 frame"
    all_frames.click()
    assert dialog.frame_range == (0, 29)

    dialog.original_radio.setChecked(True)
    assert dialog.original


def test_save_button_asks_for_the_range_first(
    video: CalciumVideoWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.set_time_ms(250)  # frame 6 at 20 frames per second
    export = tmp_path / "from_button.tif"

    def _choose(dialog: SaveImagingDialog) -> QtWidgets.QDialog.DialogCode:
        assert dialog._current_frame == 6
        dialog.first_spin.setValue(6)
        dialog.last_spin.setValue(7)
        dialog.original_radio.setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(SaveImagingDialog, "exec", _choose)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(export), ""),
    )
    video.save_button.click()
    np.testing.assert_array_equal(tifffile.imread(export)[:, 0, 0], [105, 106])
    assert video._save_original  # remembered for the next save


def test_saved_frames_share_one_grey_scale(qtbot: QtBot, tmp_path: Path) -> None:
    """Issue #618: scaling each saved frame to its own minimum and maximum
    made the background flicker; one hot pixel changed a whole frame."""
    rng = np.random.default_rng(0)
    frames = rng.poisson(100, (6, 32, 32)).astype(np.uint16)
    frames[2, 0, 0] = 60000  # a hot pixel in one frame only
    path = tmp_path / "noisy.tif"
    tifffile.imwrite(path, frames)
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths([path], 10.0, warn_if_missing_cellpose=False)

    export = tmp_path / "marked.tif"
    widget.export_labeled_video(export)
    with tifffile.TiffFile(export) as tif:
        means = [page.asarray()[1:, 1:, 0].mean() for page in tif.pages]
    assert max(means) - min(means) < 0.05 * np.mean(means)


def test_stimulation_marks_scale_with_the_image(qtbot: QtBot, tmp_path: Path) -> None:
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim import StimType

    size = 512
    segmentation = -np.ones((size, size), dtype=np.int32)
    segmentation[250:262, 250:262] = 0
    path = tmp_path / "large.tif"
    tifffile.imwrite(path, np.full((2, size, size), 100, dtype=np.uint16))
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_cellpose_data(
        CellposeImportResult(
            image_path=tmp_path / "cells.png",
            segmentation_array=segmentation,
            overlay_image=np.zeros((size, size, 3), dtype=np.uint8),
            trace_data=None,
            spike_data=None,
            original_cellpose_ids=np.array([0], dtype=np.int32),
            stimulus_to_cellpose_mapping=None,
        )
    )
    widget.set_stim_arrays(
        {"stim": StimArray("stim", StimType.Excitation, np.ones((100, 1)), "mW", 0.01)}
    )
    widget.set_video_paths([path], 10.0, warn_if_missing_cellpose=False)

    frame = widget._labeled_frame(0)
    red = np.all(frame == (255, 0, 0), axis=-1)
    rows, cols = np.nonzero(red)
    # An "x" about 1/32 of the image wide, not a 3-pixel plus.
    assert cols.max() - cols.min() >= 16
    assert rows.max() - rows.min() >= 16
    assert red[255, 255] or red[256, 256]  # centred on the cell
