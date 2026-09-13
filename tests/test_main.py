from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from neuroseq.__main__ import main


class FakeGui:
    last_instance: FakeGui | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.filename = kwargs.get("filename", args[0] if args else None)
        self.shown = False
        FakeGui.last_instance = self

    def show(self) -> None:
        self.shown = True


class FakeQApplication:
    def __init__(self, *args: Any) -> None:
        self.quit_called = False
        self.exec_called = False

    def exec(self) -> None:
        self.exec_called = True

    def quit(self) -> None:
        self.quit_called = True


class FakeGuiSettings:
    def __init__(self) -> None:
        self.theme = "retro"


class FakeStartupDialog:
    """Stands in for the real StartupDialog: accepts immediately with a
    fixed filename chosen, unless a test overrides its class attributes."""

    accepts: bool = True
    chosen_filename: str | None = "/fake/chosen_project.nseq"
    chosen_project: Any | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.chosen_filename = type(self).chosen_filename
        self.chosen_project = type(self).chosen_project

    def exec(self) -> bool:
        return type(self).accepts


@pytest.fixture()
def cli_runner_fake_gui(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    FakeGui.last_instance = None
    FakeStartupDialog.accepts = True
    FakeStartupDialog.chosen_filename = "/fake/chosen_project.nseq"
    FakeStartupDialog.chosen_project = None
    monkeypatch.setattr("neuroseq.gui.Gui", FakeGui)
    monkeypatch.setattr("qtpy.QtWidgets.QApplication", FakeQApplication)
    monkeypatch.setattr("neuroseq.gui_settings.GuiSettings", FakeGuiSettings)
    monkeypatch.setattr("neuroseq.startup_dialog.StartupDialog", FakeStartupDialog)
    monkeypatch.setattr("neuroseq.theme.apply_app_theme", lambda *a, **k: None)
    monkeypatch.setattr(
        "neuroseq.__main__._install_interrupt_handlers", lambda app: None
    )
    return CliRunner()


class FakeTimerSignal:
    def __init__(self) -> None:
        self.callback: Any | None = None

    def connect(self, callback: Any) -> None:
        self.callback = callback


class FakeTimer:
    def __init__(self, parent: Any) -> None:
        self.parent = parent
        self.interval: int | None = None
        self.started = False
        self.timeout = FakeTimerSignal()

    def setInterval(self, interval: int) -> None:
        self.interval = interval

    def start(self) -> None:
        self.started = True


class FakeSignalModule:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}

    def signal(self, signum: int, handler: Any) -> None:
        self.handlers[signum] = handler


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


def test_main_shows_startup_dialog_and_opens_its_choice_when_no_filename_given(
    cli_runner_fake_gui: CliRunner,
) -> None:
    FakeStartupDialog.chosen_filename = "/fake/chosen_project.nseq"
    result = cli_runner_fake_gui.invoke(main)
    assert result.exit_code == 0
    assert FakeGui.last_instance is not None
    assert FakeGui.last_instance.filename == "/fake/chosen_project.nseq"
    assert FakeGui.last_instance.shown is True


def test_main_skips_startup_dialog_when_a_filename_is_given(
    cli_runner_fake_gui: CliRunner,
) -> None:
    result = cli_runner_fake_gui.invoke(main, ["/some/explicit/project.nseq"])
    assert result.exit_code == 0
    assert FakeGui.last_instance is not None
    assert FakeGui.last_instance.filename == "/some/explicit/project.nseq"


def test_main_exits_without_launching_gui_when_startup_dialog_is_dismissed(
    cli_runner_fake_gui: CliRunner,
) -> None:
    FakeStartupDialog.accepts = False
    result = cli_runner_fake_gui.invoke(main)
    assert result.exit_code == 0
    assert FakeGui.last_instance is None


@pytest.mark.parametrize("log_level", ["INVALIDLEVEL", "Warn"])
def test_main_logging_invalid(
    log_level: str, tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    result = cli_runner_fake_gui.invoke(main, ["--log-level", log_level])
    assert "Error" in result.output
    assert result.exit_code != 0


def test_install_interrupt_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuroseq.__main__ import _install_interrupt_handlers

    app = FakeQApplication()
    fake_signals = FakeSignalModule()

    monkeypatch.setattr("neuroseq.__main__.sys.platform", "linux")
    timer = _install_interrupt_handlers(
        app,
        timer_factory=FakeTimer,
        signal_module=fake_signals,
    )

    assert isinstance(timer, FakeTimer)
    assert timer.parent is app
    assert timer.interval == 200
    assert timer.started is True
    assert callable(timer.timeout.callback)
    assert fake_signals.SIGINT in fake_signals.handlers
    assert fake_signals.SIGTERM in fake_signals.handlers

    fake_signals.handlers[fake_signals.SIGINT](fake_signals.SIGINT, None)
    assert app.quit_called is True


def test_install_interrupt_handlers_skips_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.__main__ import _install_interrupt_handlers

    app = FakeQApplication()
    fake_signals = FakeSignalModule()

    monkeypatch.setattr("neuroseq.__main__.sys.platform", "win32")
    timer = _install_interrupt_handlers(
        app,
        timer_factory=FakeTimer,
        signal_module=fake_signals,
    )

    assert timer is None
    assert fake_signals.handlers == {}


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


def test_generate_cellpose_test_data_cli_square_size(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    rows = 8
    cols = 12
    square_size = 1300
    image_path = tmp_path / "cellpose_grid_square.tiff"
    segmentation_path = tmp_path / "cellpose_grid_square_masks.npy"

    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-segmentation-output",
            str(segmentation_path),
            "--cellpose-grid-rows",
            str(rows),
            "--cellpose-grid-cols",
            str(cols),
            "--cellpose-radius",
            "10",
            "--cellpose-gap",
            "10",
            "--cellpose-square-size",
            str(square_size),
        ],
    )

    assert result.exit_code == 0
    assert image_path.exists()
    assert segmentation_path.exists()

    image_array = np.array(Image.open(image_path))
    segmentation_array = np.load(segmentation_path)

    assert image_array.shape == (square_size, square_size)
    assert segmentation_array.shape == (square_size, square_size)
    assert segmentation_array.max() == rows * cols
    assert np.unique(segmentation_array).shape[0] == (rows * cols) + 1


def test_generate_cellpose_test_data_cli_square_size_preserve_radius_gap(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    rows = 10
    cols = 10
    radius = 10
    gap = 10
    square_size = 1300
    image_path = tmp_path / "cellpose_grid_square_fixed_geometry.tiff"
    segmentation_path = tmp_path / "cellpose_grid_square_fixed_geometry_masks.npy"

    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-segmentation-output",
            str(segmentation_path),
            "--cellpose-grid-rows",
            str(rows),
            "--cellpose-grid-cols",
            str(cols),
            "--cellpose-radius",
            str(radius),
            "--cellpose-gap",
            str(gap),
            "--cellpose-square-size",
            str(square_size),
            "--cellpose-preserve-radius-gap",
        ],
    )

    assert result.exit_code == 0
    assert image_path.exists()
    assert segmentation_path.exists()

    image_array = np.array(Image.open(image_path))
    segmentation_array = np.load(segmentation_path)
    y1, x1 = np.where(segmentation_array == 1)
    y2, x2 = np.where(segmentation_array == 2)
    y_down, x_down = np.where(segmentation_array == (cols + 1))
    center_1 = (x1.mean(), y1.mean())
    center_2 = (x2.mean(), y2.mean())
    center_down = (x_down.mean(), y_down.mean())

    assert image_array.shape == (square_size, square_size)
    assert segmentation_array.shape == (square_size, square_size)
    assert segmentation_array.max() == rows * cols
    assert np.unique(segmentation_array).shape[0] == (rows * cols) + 1
    assert np.isclose(center_2[0] - center_1[0], (2 * radius) + gap)
    assert np.isclose(center_down[1] - center_1[1], (2 * radius) + gap)


def test_generate_cellpose_test_data_cli_square_size_preserve_radius_auto_gap(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    rows = 8
    cols = 12
    radius = 10
    square_size = 1300
    expected_gap = min(
        (square_size - (2 * radius * cols)) // (cols - 1),
        (square_size - (2 * radius * rows)) // (rows - 1),
    )
    image_path = tmp_path / "cellpose_grid_square_fixed_radius_auto_gap.tiff"
    segmentation_path = (
        tmp_path / "cellpose_grid_square_fixed_radius_auto_gap_masks.npy"
    )

    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-segmentation-output",
            str(segmentation_path),
            "--cellpose-grid-rows",
            str(rows),
            "--cellpose-grid-cols",
            str(cols),
            "--cellpose-radius",
            str(radius),
            "--cellpose-gap",
            "1",
            "--cellpose-square-size",
            str(square_size),
            "--cellpose-preserve-radius-auto-gap",
        ],
    )

    assert result.exit_code == 0
    assert image_path.exists()
    assert segmentation_path.exists()

    image_array = np.array(Image.open(image_path))
    segmentation_array = np.load(segmentation_path)
    y1, x1 = np.where(segmentation_array == 1)
    y2, x2 = np.where(segmentation_array == 2)
    y_down, x_down = np.where(segmentation_array == (cols + 1))
    center_1 = (x1.mean(), y1.mean())
    center_2 = (x2.mean(), y2.mean())
    center_down = (x_down.mean(), y_down.mean())

    assert image_array.shape == (square_size, square_size)
    assert segmentation_array.shape == (square_size, square_size)
    assert segmentation_array.max() == rows * cols
    assert np.unique(segmentation_array).shape[0] == (rows * cols) + 1
    assert np.isclose(center_2[0] - center_1[0], (2 * radius) + expected_gap)
    assert np.isclose(center_down[1] - center_1[1], (2 * radius) + expected_gap)


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


def test_cellpose_preserve_radius_gap_requires_square_size(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    image_path = tmp_path / "cellpose_grid.tiff"
    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-preserve-radius-gap",
        ],
    )

    assert result.exit_code != 0
    assert "--cellpose-square-size" in result.output


def test_cellpose_preserve_radius_auto_gap_requires_square_size(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    image_path = tmp_path / "cellpose_grid.tiff"
    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-preserve-radius-auto-gap",
        ],
    )

    assert result.exit_code != 0
    assert "--cellpose-square-size" in result.output


def test_cellpose_preserve_modes_are_mutually_exclusive(
    tmp_path: Path, cli_runner_fake_gui: CliRunner
) -> None:
    image_path = tmp_path / "cellpose_grid.tiff"
    result = cli_runner_fake_gui.invoke(
        main,
        [
            "--generate-cellpose-test-data",
            str(image_path),
            "--cellpose-square-size",
            "1300",
            "--cellpose-preserve-radius-gap",
            "--cellpose-preserve-radius-auto-gap",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be used together" in result.output
