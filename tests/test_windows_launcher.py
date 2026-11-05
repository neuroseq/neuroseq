"""Desktop launcher update rules (scripts/windows/launch_neuroseq.ps1, #617).

Runs the real script with PowerShell 7 against throwaway git repositories and
a stand-in for uv; Windows only, as that is the only platform it runs on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).parents[1] / "scripts" / "windows" / "launch_neuroseq.ps1"
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or PWSH is None,
    reason="Windows launcher; needs PowerShell 7 (pwsh)",
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull},
    ).stdout.strip()


def _commit(repo: Path, name: str) -> None:
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name)


@pytest.fixture()
def rig(tmp_path: Path) -> dict[str, Path]:
    """A clone of `origin` that is one commit behind on main."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    launcher = seed / "scripts" / "windows" / "launch_neuroseq.ps1"
    launcher.parent.mkdir(parents=True)
    shutil.copy(LAUNCHER, launcher)
    _git(seed, "add", ".")
    _commit(seed, "pyproject.toml")
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _commit(seed, "newer.txt")
    _git(seed, "push", "-q", str(origin), "main")

    # Records the commit it was started at, then exits with $FAKE_EXIT.
    # %~n strips the quotes the launcher puts around the --project path.
    uv = tmp_path / "uv.cmd"
    uv.write_text(
        '@>"%FAKE_RECORD%" echo %~1 %~2 %~3 %~4\n'
        '@git -C "%FAKE_CLONE%" rev-parse HEAD >> "%FAKE_RECORD%"\n'
        "@echo boom 1>&2\n"
        "@exit /b %FAKE_EXIT%\n"
    )
    return {
        "origin": origin,
        "clone": clone,
        "uv": uv,
        "logs": tmp_path / "logs",
        "record": tmp_path / "record.txt",
    }


def _launch(rig: dict[str, Path], *args: str, exit_code: int = 0):
    assert PWSH is not None
    env = {
        **os.environ,
        "LOCALAPPDATA": str(rig["logs"]),
        "FAKE_RECORD": str(rig["record"]),
        "FAKE_CLONE": str(rig["clone"]),
        "FAKE_EXIT": str(exit_code),
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    script = rig["clone"] / "scripts" / "windows" / "launch_neuroseq.ps1"
    command = [PWSH, "-NoProfile", "-File", str(script), "-Console"]
    command += ["-Uv", str(rig["uv"]), *args]
    # Output goes to files: processes started by the launcher inherit pipes, so
    # one that hangs would also block reading the output after a timeout.
    out, err = rig["logs"].with_name("pwsh.out"), rig["logs"].with_name("pwsh.err")
    with out.open("w") as stdout, err.open("w") as stderr:
        try:
            # Below pytest's 60 s timeout, so a hang fails only this test.
            process = subprocess.run(
                command, stdout=stdout, stderr=stderr, env=env, timeout=30
            )
        except subprocess.TimeoutExpired:
            log = rig["logs"] / "neuroseq" / "launcher.log"
            pytest.fail(
                f"launcher timed out\n{out.read_text()}{err.read_text()}"
                f"{log.read_text() if log.exists() else 'no launcher.log'}"
            )
    return subprocess.CompletedProcess(
        command, process.returncode, out.read_text(), err.read_text()
    )


def _started_at(rig: dict[str, Path]) -> tuple[str, str]:
    args, head = rig["record"].read_text().splitlines()
    return args, head


def test_main_is_fast_forwarded_before_starting(rig) -> None:
    result = _launch(rig)
    assert result.returncode == 0, result.stdout + result.stderr
    args, head = _started_at(rig)
    assert head == _git(rig["origin"], "rev-parse", "main")
    assert args == f"run --project {rig['clone']} neuroseq"
    assert "not updated" not in result.stdout


def test_untracked_files_do_not_block_the_update(rig) -> None:
    (rig["clone"] / "scratch.txt").write_text("x")
    _launch(rig)
    assert _started_at(rig)[1] == _git(rig["origin"], "rev-parse", "main")


@pytest.mark.parametrize(
    ("prepare", "reason"),
    [
        (lambda c: _git(c, "switch", "-qc", "feature"), "on branch 'feature'"),
        (lambda c: (c / "pyproject.toml").write_text("edited"), "local changes"),
        (
            lambda c: _git(c, "remote", "set-url", "origin", str(c / "missing")),
            "GitHub could not be reached",
        ),
        (lambda c: _commit(c, "local.txt"), "not on GitHub"),
    ],
    ids=["other-branch", "local-changes", "offline", "diverged"],
)
def test_skipped_update_still_starts_and_says_why(rig, prepare, reason) -> None:
    prepare(rig["clone"])
    before = _git(rig["clone"], "rev-parse", "HEAD")
    result = _launch(rig)
    assert result.returncode == 0, result.stdout + result.stderr
    assert reason in result.stdout
    assert _started_at(rig)[1] == before


def test_no_update_switch(rig) -> None:
    before = _git(rig["clone"], "rev-parse", "HEAD")
    result = _launch(rig, "-NoUpdate")
    assert result.returncode == 0
    assert _started_at(rig)[1] == before


def test_app_error_is_reported_with_its_output(rig) -> None:
    result = _launch(rig, exit_code=3)
    assert result.returncode == 3
    assert "exit code 3" in result.stdout
    assert "boom" in result.stdout
    assert (
        "exited with code 3" in (rig["logs"] / "neuroseq" / "launcher.log").read_text()
    )


def test_missing_uv_is_reported(rig) -> None:
    rig["uv"].unlink()
    env_path = os.environ["PATH"]
    os.environ["PATH"] = os.pathsep.join(
        p for p in env_path.split(os.pathsep) if not (Path(p) / "uv.exe").exists()
    )
    try:
        result = _launch(rig)
    finally:
        os.environ["PATH"] = env_path
    assert result.returncode == 1
    assert "uv was not found" in result.stdout
