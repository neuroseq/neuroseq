from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from neuroseq.__main__ import main


class FakeGui:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def show(self) -> None:
        pass


class FakeQApplication:
    def __init__(self, *args: Any) -> None:
        pass

    def exec(self) -> None:
        pass


@pytest.fixture()
def cli_runner_fake_gui(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setattr("neuroseq.gui.Gui", FakeGui)
    monkeypatch.setattr("qtpy.QtWidgets.QApplication", FakeQApplication)
    return CliRunner()


@pytest.mark.parametrize("log_level", ["DEBUG", "inFo", "warning", "ERROR", "Critical"])
def test_main_logging_valid(
    log_level: str, tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    log_file = str(tmp_path / "test.log")
    result = cli_runner_fake_gui.invoke(
        main, ["--log-level", log_level, "--log-file", log_file]
    )
    assert log_level.upper() in result.output
    assert log_file in result.output
    assert result.exit_code == 0


@pytest.mark.parametrize("log_level", ["INVALIDLEVEL", "Warn"])
def test_main_logging_invalid(
    log_level: str, tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    result = cli_runner_fake_gui.invoke(main, ["--log-level", log_level])
    assert "Error" in result.output
    assert result.exit_code != 0


def test_generate_cellpose_test_data_cli(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    image_path = tmp_path / "cellpose_grid.tiff"
    segmentation_path = tmp_path / "cellpose_grid_masks.npy"

    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-segmentation-output",
            str(segmentation_path),
            "--cellpose-grid-rows",
            "10",
            "--cellpose-grid-cols",
            "10",
            "--cellpose-radius",
            "10",
            "--cellpose-gap",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert image_path.exists()
    assert segmentation_path.exists()
    assert str(image_path) in result.output
    assert str(segmentation_path) in result.output

    image_array = np.array(Image.open(image_path))
    segmentation_array = np.load(segmentation_path)

    assert image_array.shape == segmentation_array.shape
    assert segmentation_array.dtype == np.int32
    assert segmentation_array.max() == 100
    assert np.unique(segmentation_array).shape[0] == 101


def test_cellpose_segmentation_output_requires_image_path(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    segmentation_path = tmp_path / "cellpose_grid_masks.npy"
    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--cellpose-segmentation-output",
            str(segmentation_path),
        ],
    )

    assert result.exit_code != 0
    assert "--generate-cellpose-test-data" in result.output
