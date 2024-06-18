from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import formozov_project.__main__ as main
import pytest


def test_main_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    assert not log_file.exists()
    main_path = inspect.getfile(main)
    # ugly hack to test logging: start the app in a subprocess and then kill it after a few seconds
    # I couldn't prevent pytest / cliRunner from modifying the logging setup in the test
    # todo: find a better way to test this
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.call(
            [
                sys.executable,
                main_path,
                "--log-level",
                "DEBUG",
                "--log-file",
                f"{log_file}",
            ],
            timeout=5,
        )
    assert log_file.exists()
    with open(log_file) as f:
        log_file_contents = f.read()
    assert "DEBUG" in log_file_contents
    assert str(log_file) in log_file_contents
