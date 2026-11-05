"""Print native backtraces of processes that crash during a CI job.

`python .github/crash_reports.py setup` before the tests makes crashing
processes leave a core dump (Linux), minidump (Windows) or crash report
(macOS) in $RUNNER_TEMP/crash-dumps. `... report` after a failure prints their
backtraces. Only the standard library is used.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DUMPS = Path(os.environ.get("RUNNER_TEMP", ".")) / "crash-dumps"
SETUP_MARKER = DUMPS / "setup-time"
MACOS_REPORTS = [
    Path.home() / "Library/Logs/DiagnosticReports",
    Path("/Library/Logs/DiagnosticReports"),
]
CDB = (
    Path(os.environ.get("PROGRAMFILES(X86)", ""))
    / "Windows Kits/10/Debuggers/x64/cdb.exe"
)


def setup() -> None:
    DUMPS.mkdir(parents=True, exist_ok=True)
    SETUP_MARKER.touch()
    if sys.platform == "linux":
        subprocess.run(["sudo", "apt-get", "install", "-y", "gdb"], check=True)
        # Unlike a file, a core piped to a program ignores the core size
        # limit, which is 0 by default.
        pattern = f"|/usr/bin/dd of={DUMPS}/core.%p"
        subprocess.run(
            ["sudo", "sysctl", "-w", f"kernel.core_pattern={pattern}"], check=True
        )
    elif sys.platform == "win32":
        import winreg

        with winreg.CreateKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps",
        ) as key:
            winreg.SetValueEx(key, "DumpFolder", 0, winreg.REG_EXPAND_SZ, str(DUMPS))
            winreg.SetValueEx(key, "DumpType", 0, winreg.REG_DWORD, 1)  # minidump
    # macOS always writes crash reports, there is nothing to set up.


def collect_macos_reports() -> None:
    """Copy crash reports written since setup, waiting for ReportCrash."""
    since = SETUP_MARKER.stat().st_mtime
    for _ in range(60):
        for folder in MACOS_REPORTS:
            for report in folder.glob("*.ips"):
                if "python" in report.name.lower() and report.stat().st_mtime > since:
                    (DUMPS / report.name).write_bytes(report.read_bytes())
        if any(DUMPS.glob("*.ips")):
            return
        time.sleep(1)


def print_ips(path: Path) -> None:
    with path.open() as f:
        header = json.loads(f.readline())
        report = json.loads(f.read())
    print(header.get("name"), report.get("exception"), report.get("termination"))
    images = report.get("usedImages", [])
    for index, thread in enumerate(report.get("threads", [])):
        crashed = thread.get("triggered", False)
        print(f"--- thread {index}{' (crashed)' if crashed else ''}")
        for frame in thread.get("frames", [])[: 150 if crashed else 25]:
            image = images[frame["imageIndex"]] if "imageIndex" in frame else {}
            name = image.get("name") or image.get("path", "???").rsplit("/", 1)[-1]
            symbol = frame.get("symbol", hex(frame.get("imageOffset", 0)))
            print(f"  {name:30} {symbol} + {frame.get('symbolLocation', 0)}")


def report() -> None:
    if not SETUP_MARKER.exists():
        print("Crash dumps were not set up")
        return
    if sys.platform == "darwin":
        collect_macos_reports()
    dumps = [p for p in sorted(DUMPS.iterdir()) if p != SETUP_MARKER]
    if not dumps:
        print(f"No crash dumps in {DUMPS}")
    for dump in dumps:
        print(f"=== {dump}", flush=True)
        if dump.suffix == ".ips":
            print_ips(dump)
        elif dump.suffix == ".dmp":
            if CDB.exists():
                cdb_commands = ".symfix; .reload; .ecxr; kn 100; ~*kn 40; q"
                subprocess.run([str(CDB), "-z", str(dump), "-c", cdb_commands])
            else:
                print(f"cdb not found at {CDB}")
        else:
            gdb_commands = ["-ex", "bt 150", "-ex", "thread apply all bt 40"]
            subprocess.run(["gdb", "-batch", *gdb_commands, sys.executable, str(dump)])


if __name__ == "__main__":
    {"setup": setup, "report": report}[sys.argv[1]]()
