from __future__ import annotations

import logging
import pathlib
import signal
import sys
from collections.abc import Callable
from types import FrameType
from typing import Any

import click

from neuroseq.cellpose_test_data import save_cellpose_test_data


def _install_interrupt_handlers(
    app: Any,
    *,
    timer_factory: Callable[[Any], Any] | None = None,
    signal_module: Any = signal,
) -> Any | None:
    """Keep Ctrl-C responsive while the Qt event loop is running."""
    if sys.platform == "win32":
        return None

    if timer_factory is None:
        from qtpy import QtCore

        timer = QtCore.QTimer(app)
    else:
        timer = timer_factory(app)

    def _quit_on_signal(signum: int, _frame: FrameType | None) -> None:
        logging.getLogger(__name__).info(
            "Received signal %s, quitting GUI event loop", signum
        )
        app.quit()

    signal_module.signal(signal_module.SIGINT, _quit_on_signal)
    signal_module.signal(signal_module.SIGTERM, _quit_on_signal)
    timer.setInterval(200)
    timer.timeout.connect(lambda: None)
    timer.start()
    return timer


@click.command()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Log level",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
    help="Log to this file instead of the console",
    show_default=True,
)
@click.option(
    "--generate-cellpose-test-data",
    "test_cellpose_image_path",
    default=None,
    type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
    help=(
        "Generate a synthetic Cellpose test TIFF at this path and matching "
        "segmentation .npy, then exit."
    ),
)
@click.option(
    "--cellpose-segmentation-output",
    "test_cellpose_segmentation_path",
    default=None,
    type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
    help=(
        "Optional output path for generated Cellpose segmentation .npy "
        "(requires --generate-cellpose-test-data)."
    ),
)
@click.option(
    "--cellpose-grid-rows",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of target rows in generated Cellpose test data.",
)
@click.option(
    "--cellpose-grid-cols",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of target columns in generated Cellpose test data.",
)
@click.option(
    "--cellpose-radius",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Circle radius in pixels for generated Cellpose test data.",
)
@click.option(
    "--cellpose-gap",
    default=10,
    type=click.IntRange(min=0),
    show_default=True,
    help="Gap in pixels between adjacent circles in generated Cellpose test data.",
)
@click.option(
    "--cellpose-square-size",
    default=None,
    type=click.IntRange(min=1),
    help=(
        "Optional square output size in pixels for generated Cellpose test data. "
        "When set, radius and gap are auto-fitted while preserving rows/cols."
    ),
)
@click.option(
    "--cellpose-preserve-radius-gap/--cellpose-auto-fit-radius-gap",
    default=False,
    show_default=True,
    help=(
        "When --cellpose-square-size is set, preserve the provided "
        "--cellpose-radius and --cellpose-gap instead of auto-fitting them."
    ),
)
@click.option(
    "--cellpose-preserve-radius-auto-gap",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "When --cellpose-square-size is set, preserve --cellpose-radius and "
        "auto-fit only --cellpose-gap."
    ),
)
@click.argument("filename", required=False)
def main(
    log_level: str,
    log_file: pathlib.Path | None,
    test_cellpose_image_path: pathlib.Path | None,
    test_cellpose_segmentation_path: pathlib.Path | None,
    cellpose_grid_rows: int,
    cellpose_grid_cols: int,
    cellpose_radius: int,
    cellpose_gap: int,
    cellpose_square_size: int | None,
    cellpose_preserve_radius_gap: bool,
    cellpose_preserve_radius_auto_gap: bool,
    filename: str | None,
) -> None:
    if test_cellpose_segmentation_path is not None and test_cellpose_image_path is None:
        raise click.UsageError(
            "--cellpose-segmentation-output requires --generate-cellpose-test-data"
        )
    if cellpose_preserve_radius_gap and cellpose_square_size is None:
        raise click.UsageError(
            "--cellpose-preserve-radius-gap requires --cellpose-square-size"
        )
    if cellpose_preserve_radius_auto_gap and cellpose_square_size is None:
        raise click.UsageError(
            "--cellpose-preserve-radius-auto-gap requires --cellpose-square-size"
        )
    if cellpose_preserve_radius_gap and cellpose_preserve_radius_auto_gap:
        raise click.UsageError(
            "--cellpose-preserve-radius-gap and --cellpose-preserve-radius-auto-gap "
            "cannot be used together"
        )

    if test_cellpose_image_path is not None:
        image_path, segmentation_path = save_cellpose_test_data(
            test_cellpose_image_path,
            segmentation_path=test_cellpose_segmentation_path,
            rows=cellpose_grid_rows,
            cols=cellpose_grid_cols,
            radius=cellpose_radius,
            gap=cellpose_gap,
            square_size=cellpose_square_size,
            preserve_radius_gap=cellpose_preserve_radius_gap,
            preserve_radius_auto_gap=cellpose_preserve_radius_auto_gap,
        )
        click.echo(
            "Generated Cellpose test data:\n"
            f"- image: {image_path}\n"
            f"- segmentation: {segmentation_path}"
        )
        return

    from neuroseq.utils import init_brian2_logging

    init_brian2_logging()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(funcName)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
    )
    from qtpy import API_NAME
    from qtpy import QtWidgets

    from neuroseq.gui import Gui

    msg = f"Launching {API_NAME} GUI with {log_level} logging to {log_file if log_file else 'console'}"
    click.echo(msg)
    logging.getLogger(__name__).info(msg)
    app = QtWidgets.QApplication(sys.argv)
    app._interrupt_timer = _install_interrupt_handlers(app)
    gui = Gui(filename=filename)
    gui.show()
    logging.getLogger(__name__).info("GUI initialized, entering Qt event loop")
    app.exec()


if __name__ == "__main__":
    main()
