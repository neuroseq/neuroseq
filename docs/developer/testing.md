# Testing

Tests use pytest and pytest-qt and live flat in `tests/` as `test_<module>.py`,
with fixtures in `tests/conftest.py` and inputs in `tests/sampledata/`. Every
behaviour change needs a test. With a Conda install, replace `uv run pytest`
with `python -m pytest` in the activated environment.

## Running tests

| Task                             | Command                                            |
| -------------------------------- | -------------------------------------------------- |
| Full suite (up to 8 workers)     | `uv run pytest`                                    |
| One module                       | `uv run pytest tests/test_gui.py`                  |
| Serial (`-s`, `--pdb`, one test) | `uv run pytest -n 0 ...`                           |
| Coverage report                  | `uv run pytest --cov --cov-report=xml`             |
| Diff coverage gate (75%)         | `uv run diff-cover coverage.xml -c pyproject.toml` |
| Update reference screenshots     | `uv run pytest --update-screenshot`                |

`-n auto` (pytest-xdist) is the default. Tests marked `@pytest.mark.screenshot`
are skipped unless `--update-screenshot` is given, which runs only them. Brian2
uses the `numpy` code generation target in tests. About 27 ELISE tests skip when
the private `elise` package is not installed (see `CLAUDE.md`).

## Writing GUI tests

- Wait for a condition (`qtbot.waitUntil`, `waitSignal`, ...) instead of
  `qtbot.wait(ms)`, and pass `timeout=WAIT_TIMEOUT_MS` from `tests/wait_timeout.py`
  (5 s by default; CI passes `--wait-timeout=30`).
- Use `waitExposed`/`waitActive` as context managers: `with qtbot.waitExposed(w): w.show()`.
- Autouse fixtures silence `QMessageBox` and replace dialog `.exec()` with
  `.show()`. Use the `qmessagebox_calls` fixture to assert on message boxes.
- `tests/test_import_layering.py` enforces the package import rules.

## Scientific validation tests

Passing unit tests do not show that a result is scientifically right. When a
change affects simulation, analysis or export output, add a test that compares
the scientific outcome with an independent reference: an analytic solution, a
published value, or a trusted dataset. State the reference and tolerance in the
test. Examples:

- `tests/test_elise_opsin_kinetics.py`: integrated opsin state matches the
  analytic formula.
- `tests/test_membrane_parameters_analysis.py`: the fit recovers the parameters
  of a synthetic membrane.

## CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`:

| Job         | Setup                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------- |
| conda-forge | `environment.yml` on Linux, macOS, Windows; `pytest --wait-timeout=30 --reruns 1`              |
| pypi        | uv, Python 3.12 and 3.14 on all three OSes; same pytest flags with coverage, then `diff-cover` |
| pre-commit  | `uvx prek run -a` (ruff, mypy, prettier, nbstripout, workflow checks)                          |

For Qt parity with CI, set `QT_API=pyqt6` and `PYTEST_QT_API=pyqt6`.

## Hardware integration

SysCon2 tests use fake transports and need no rig:

```bash
uv run pytest tests/test_syscon2_client.py tests/test_syscon2_connection_manager.py tests/test_syscon2_panel.py tests/test_rheobase_dialog.py tests/test_stdp_window_dialog.py
```

On Windows, `test_syscon2_client.py` also opens a real named pipe to test
timeouts. Pipe commands time out after 30 s (`NamedPipeTransport(timeout=...)`);
a timeout closes the connection and does not confirm that a queued Stop arrived.

### Real-rig checklist

Run NeuroSeq and SysCon2 on the same Windows PC with the Holo4D profile
(its `.seq` folder is `C:\ProgramData\ROE\SysCon 2\2.7.2\profiles\Holo4D\Sequence Manager\Experiments\NeuroSeq`)
and a verified Cellpose to stim ID mapping. Check:

1. **Connect** opens the pipe only; **Get Status** finds the light sources (940 nm preferred).
2. **Export with settings…**: target power is in mW; LUT percentages are divided
   by 100 in `.seq` files (verify on the rig).
3. **Check Input Sanity** resolves every violation; the timestep is exactly
   520.8125 µs. Inspect the overview PNG and debug CSV.
4. After import, the sequence appears in SysCon2 and red target crosses appear
   (they mark the acknowledged preview, not live illumination).
5. **Runs = 1**, trigger settings, **Upload**, **Run**: progress passes through
   uploading, ready, starting, running; Stop during upload sends no Run. Measure
   timing, pattern and delivered power with the bench-test procedure.
6. **Export current array to stimulation setup** reuses the last dialog settings
   and writes to a temporary directory.
7. Rheobase/STDP, automatic blocks: ScanImage connected, TTL output
   (`TargetConnectorPort`) wired to the acquisition trigger. Each block's `.seq`,
   overview and sidecar are kept in the run folder, each recording `.json` names
   the right `.seq`, recordings start with the block, and the logged ΔF/F or spike
   count matches the visible response.
8. Manual blocks: each block imported into a new, uniquely named experiment
   before clicking **Calcium data imported**; stale recordings are rejected.
9. **Routine Stop** cancels only its own block; **Holo4D Stop** stops the
   hardware. Disconnect cancels communication, not stimulation.

Record the script, debug CSV, sequence, Runs, trigger, wavelength, power, frame
rate, threshold and per-probe response. Still unconfirmed: the wire width of the
Runs integer (sent as a 32-bit DWord).

## Segfault at exit

If tests pass and then segfault on exit inside Conda's `libstdc++`, preload the
system runtime (Debian/Ubuntu x86-64 path):

```bash
conda run --no-capture-output -n neuroseq env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 QT_QPA_PLATFORM=offscreen QT_API=pyqt6 PYTEST_QT_API=pyqt6 .venv/bin/python -m pytest
```
