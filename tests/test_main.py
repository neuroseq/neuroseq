from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

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
    monkeypatch.setattr("neuroseq.__main__.Gui", FakeGui)
    monkeypatch.setattr("neuroseq.__main__.QtWidgets.QApplication", FakeQApplication)
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
