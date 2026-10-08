# Testing

A proper development of the code implies the use of tests with the `pytest` framework. Tests are located in the `tests/` folder.

## To run all tests

```bash
pytest
```

## To run a specific subset of tests

```bash
pytest tests/test_gui.py
```

## SysCon2 control

The protocol, connection manager, panel, and routine tests use fake transports and
recordings. On Windows, `tests/test_syscon2_client.py` also opens a real local named
pipe to check timeouts when a server never replies or drains a write. These tests
do not require SysCon2 or a connected stimulation device.

```bash
uv run pytest tests/test_syscon2_client.py tests/test_syscon2_connection_manager.py tests/test_syscon2_panel.py tests/test_rheobase_dialog.py tests/test_stdp_window_dialog.py
```

Real-rig checks are still needed for command execution and acquisition alignment.
The SCI v1.2 manual confirms the `Sequence Manager` instance name for the Holo4D
profile and that Runs = 0 means endless repetition, but does not specify the wire
width of the Runs integer. The client currently uses a 32-bit DWord.

Pipe commands have a 30-second deadline, configurable through
`NamedPipeTransport(timeout=...)`. Disconnect and shutdown cancel pending I/O.
A timeout closes the connection and requires an explicit reconnect; it does not
confirm that a queued Stop command reached the device.

## Linux teardown crash with Conda

If tests report success and then segfault during process exit, inspect the native
backtrace. On the development host, this occurred in Conda's `libstdc++` exit
handlers after graphics libraries unloaded. Running with the system C++ runtime
resolved the crash without changing application code:

```bash
conda run --no-capture-output -n neuroseq env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 QT_QPA_PLATFORM=offscreen QT_API=pyqt6 PYTEST_QT_API=pyqt6 .venv/bin/python -m pytest
```

The library path is specific to Debian/Ubuntu x86-64; check the system library
location before using this workaround elsewhere.

## Pre-commit

pre-commit run --all-files
