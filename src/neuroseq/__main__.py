from __future__ import annotations

import logging
import pathlib
import sys

import click
from qtpy import API_NAME
from qtpy import QtWidgets

from neuroseq.gui import Gui
from neuroseq.profiling import configure_profiling
from neuroseq.profiling import start_profiling
from neuroseq.profiling import stop_profiling
from neuroseq.utils import init_brian2_logging


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
    "--profile",
    is_flag=True,
    default=False,
    help="Enable cProfile and write stats on exit",
    show_default=True,
)
@click.option(
    "--profile-output",
    default=pathlib.Path("neuroseq.prof"),
    type=click.Path(path_type=pathlib.Path, dir_okay=False, writable=True),
    help="Output file for cProfile stats",
    show_default=True,
)
@click.argument("filename", required=False)
def main(
    log_level: str,
    log_file: pathlib.Path,
    profile: bool,
    profile_output: pathlib.Path,
    filename: str | None,
) -> None:
    configure_profiling(profile, profile_output)
    init_brian2_logging()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(funcName)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
    )
    msg = f"Launching {API_NAME} GUI with {log_level} logging to {log_file if log_file else 'console'}"
    click.echo(msg)
    logging.getLogger(__name__).info(msg)
    app = QtWidgets.QApplication(sys.argv)
    gui = Gui(filename=filename)
    gui.show()
    start_profiling()
    try:
        app.exec()
    finally:
        stop_profiling()


if __name__ == "__main__":
    main()
