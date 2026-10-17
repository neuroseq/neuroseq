"""Showing one channel of a movie with interleaved channels (issue #599)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from pytestqt.qtbot import QtBot

from neuroseq.calcium_video_widget import CalciumVideoWidget


@pytest.fixture
def two_channel_movie(tmp_path: Path) -> Path:
    # Pages alternate channel 1 and channel 2; each page holds its own index.
    path = tmp_path / "two_channels.tif"
    pages = np.stack([np.full((4, 4), page, dtype=np.uint16) for page in range(6)])
    tifffile.imwrite(path, pages)
    return path


def _shown_page(widget: CalciumVideoWidget) -> int:
    assert widget.current_frame_index is not None
    return int(widget._read_frame(widget.current_frame_index)[0, 0])


def test_one_channel_plays_at_the_channel_frame_rate(
    qtbot: QtBot, two_channel_movie: Path
) -> None:
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths(
        [two_channel_movie],
        10.0,
        warn_if_missing_cellpose=False,
        channels=2,
        shown_channel=2,
    )
    assert widget.frame_count() == 3
    assert widget.channel_combo.currentText() == "Channel 2"
    widget.set_time_ms(150)
    assert _shown_page(widget) == 3

    widget.set_channels(2, 1)
    assert _shown_page(widget) == 2  # same time, other channel
    # Every frame: both channels share each 100 ms frame period.
    widget.set_channels(2, 0)
    assert widget.frame_count() == 6
    widget.set_time_ms(150)
    assert _shown_page(widget) == 3
    widget.set_time_ms(250)
    assert _shown_page(widget) == 5


def test_channel_controls_emit_the_choice(
    qtbot: QtBot, two_channel_movie: Path
) -> None:
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_video_paths([two_channel_movie], 10.0, warn_if_missing_cellpose=False)
    assert widget.frame_count() == 6
    assert not widget.channel_combo.isEnabled()

    with qtbot.waitSignal(widget.channels_changed) as blocker:
        widget.channels_spin.setValue(2)
    assert blocker.args == [2, 1]
    assert widget.frame_count() == 3

    widget.channel_combo.setCurrentIndex(widget.channel_combo.findData(2))
    with qtbot.waitSignal(widget.channels_changed) as blocker:
        widget.channel_combo.activated.emit(widget.channel_combo.currentIndex())
    assert blocker.args == [2, 2]
    widget.set_time_ms(0)
    assert _shown_page(widget) == 1


def test_gui_keeps_the_channel_choice_per_video(
    qtbot: QtBot, two_channel_movie: Path, tmp_path: Path
) -> None:
    from neuroseq.project import Project
    from neuroseq.project_workspace import ProjectWorkspace
    from neuroseq.save_load_options import SaveLoadOptions

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    # A TIFF import with two channels shows its functional channel.
    gui.project.add_calcium_video(
        [two_channel_movie], 10.0, channels=2, shown_channel=2
    )
    gui._refresh_calcium_video_widgets()
    video = gui.calcium_video_widget
    assert (video._channels, video._shown_channel) == (2, 2)

    video.channel_combo.setCurrentIndex(video.channel_combo.findData(1))
    video.channel_combo.activated.emit(video.channel_combo.currentIndex())
    assert gui.project.calcium_video_channels == (2, 1)

    archive = tmp_path / "channels.nseq"
    gui.project.save(archive, SaveLoadOptions(True, True, False, False))
    assert Project(archive).calcium_video_channels == (2, 1)
    gui._close_confirmed = True
    gui.close()


def test_video_labels_cells_with_their_neuron_ids(qtbot: QtBot, tmp_path: Path) -> None:
    """Issue #596: the video shows each mapped cell's neuron id."""
    from neuroseq.importers.importer_cellpose import CellposeImportResult

    segmentation = -np.ones((4, 4), dtype=np.int32)
    segmentation[0:2, 0:2] = 0
    segmentation[2:4, 2:4] = 1
    segmentation[0, 3] = 2  # not mapped to a neuron
    cellpose = CellposeImportResult(
        image_path=tmp_path / "cells.png",
        segmentation_array=segmentation,
        overlay_image=np.zeros((4, 4, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1, 2], dtype=np.int32),
        stimulus_to_cellpose_mapping={7: 0, 3: 1},
    )
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, np.zeros((2, 4, 4), dtype=np.uint16))
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_cellpose_data(cellpose)
    widget.set_video_paths([path], 10.0, warn_if_missing_cellpose=False)

    labels = {g.text: tuple(g.offset[:2]) for g in widget._id_label_graphics}
    assert labels == {"7": (0.5, 0.5), "3": (2.5, 2.5)}
    widget.ids_checkbox.setChecked(False)
    assert widget._id_label_graphics == []
    widget.ids_checkbox.setChecked(True)
    assert len(widget._id_label_graphics) == 2
