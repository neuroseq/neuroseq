from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from pytestqt.qtbot import QtBot

from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget


@pytest.mark.parametrize(
    ("world", "expected"),
    [
        # Pixel centres sit at integer world coordinates, so a pointer at
        # x=0.6 is over pixel 1 (truncation picked pixel 0's cell).
        ((0.6, 0.0), 1),
        ((0.4, 0.0), 0),
        ((-0.4, 0.0), 0),
        ((-0.6, 0.0), None),
    ],
)
def test_hover_picks_the_cell_under_the_nearest_pixel(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: tuple[float, float],
    expected: int | None,
) -> None:
    from neuroseq.importers.importer_cellpose import CellposeImportResult

    segmentation = -np.ones((2, 2), dtype=np.int32)
    segmentation[:, 0] = 0
    segmentation[:, 1] = 1
    cellpose = CellposeImportResult(
        image_path=tmp_path / "cells.png",
        segmentation_array=segmentation,
        overlay_image=np.zeros((2, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
    )
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, np.zeros((2, 2, 2), dtype=np.uint16))
    widget = CalciumVideoWidget()
    qtbot.addWidget(widget)
    widget.set_cellpose_data(cellpose)
    widget.set_video_paths([path], 10.0, warn_if_missing_cellpose=False)
    monkeypatch.setattr(
        widget.subplot,
        "map_screen_to_world",
        lambda event: np.array([world[0], world[1], 0.0]),
    )

    widget._on_pointer_move(object())

    assert widget._hovered_cellpose_id == expected
