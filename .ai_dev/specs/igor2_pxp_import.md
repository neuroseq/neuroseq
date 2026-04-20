# Spec: Igor2 Packed Experiment (.pxp) Import

**Date:** 2026-04-15
**Branch context:** `import_igor2`

---

## Background & Motivation

NeuroSeq already imports single-wave Igor Binary Wave (`.ibw`) files and a simple single-column `.csv` format. The pipeline is:

```
load_igor_wave_file() → IgorWave
    → prepare_igor_wave_for_neuroseq() → PreparedIgorWave
        → create_stim_array_from_igor_wave()  [new stim array]
        → import_igor_wave_into_stim_array()  [into existing]
```

Igor Pro users frequently export data as **Packed Experiment** (`.pxp`) files, which bundle many waves in a folder hierarchy. A representative test file `260316_0.pxp` (13 MB) is available at `/mnt/live3/svintila/code_scraps/igor2_files/test_files/`. The goal of this spec is to extend the import pipeline to support `.pxp` files.

---

## Current State

### Implemented

| File                                             | Role                                                |
| ------------------------------------------------ | --------------------------------------------------- |
| `src/neuroseq/importer_igor2.py`                 | Core loading, resampling, stim-array creation logic |
| `src/neuroseq/igor_import_dialog.py`             | Qt dialog; calls `load_igor_wave_file()` on browse  |
| `src/neuroseq/exporter_igor2.py`                 | CSV export back to Igor patch-clamp format          |
| `src/neuroseq/igor2_patchclamp_export_dialog.py` | Export dialog                                       |
| `tests/test_importer_igor2.py`                   | Unit tests (all use synthetic data or mocked igor2) |
| `tests/test_igor_import_dialog.py`               | Dialog integration tests                            |
| `tests/test_exporter_igor2.py`                   | Export tests                                        |

### Gaps

- No `.pxp` loader exists anywhere in the codebase.
- `igor_import_dialog.py` file filter only shows `*.ibw *.csv`.
- No wave-selection UI for multi-wave containers.
- No tests against any of the real test `.ibw` / `.pxp` files.

---

## igor2 Library Capabilities (available in `.venv`)

```python
from igor2.packed import load, walk
from igor2.record.wave import WaveRecord

records, filesystem = load("260316_0.pxp")
# filesystem is a nested dict: { b'folder': { b'wave_name': WaveRecord, ... }, ... }
# WaveRecord.wave is the full IBW dict (same structure as igor2.binarywave.load())

def callback(dirpath, key, value):
    if isinstance(value, WaveRecord):
        # key  = bytes wave name, e.g. b'HodgkinHuxley'
        # dirpath = list of bytes folder names, e.g. [b'root', b'stims']
        pass

walk(filesystem, callback)
```

`WaveRecord.wave` has the same dict structure as the output of `igor2.binarywave.load()`, so the existing `parse_igor_binary_wave_data()` function can parse it directly.

---

## Proposed Design

### 1. New function in `importer_igor2.py`

```python
def load_igor_pxp_file(path: str | Path) -> dict[str, IgorWave]:
    """Load all 1-D waves from an Igor Packed Experiment file.

    Returns a dict mapping slash-joined paths to IgorWave objects,
    e.g. {"root/stims/HodgkinHuxley": IgorWave(...), ...}.

    Only 1-D waves are included; multi-dimensional waves are skipped
    with a debug log message.
    """
```

- Imports `igor2.packed` lazily (same pattern as `load_igor_ibw_file` uses `importlib.import_module`).
- Uses `igor2.packed.walk()` to enumerate all `WaveRecord` leaves.
- For each wave, calls `parse_igor_binary_wave_data(wave_record.wave, fallback_name=...)` — reusing the existing parser.
- Skips non-1-D waves silently (logs at DEBUG level).
- Raises `ImportError` if `igor2` is not installed.
- Raises `ValueError` if the file contains no importable 1-D waves.
- Returns `{}` for an empty PXP (should not happen in practice, but guard).

Path key format: folder segments joined with `/`, converted from bytes with `decode("utf-8", errors="ignore")`. Root folder key omitted if there's only one level.

### 2. Extend `load_igor_wave_file()` dispatcher

Add `.pxp` to the known extensions. However, since `.pxp` returns many waves (not one), the dispatcher **should not** be modified to return a single `IgorWave` — that would force an arbitrary choice. Instead, callers handle `.pxp` explicitly.

Add a small helper:

```python
def is_pxp_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pxp"
```

### 3. Dialog changes in `igor_import_dialog.py`

#### Browse filter update

Add `*.pxp` to the file open dialog filter string.

#### PXP flow

When the user selects a `.pxp` file, the dialog:

1. Calls `load_igor_pxp_file(filename)` → `dict[str, IgorWave]`.
2. Stores the dict as `self.loaded_pxp_waves: dict[str, IgorWave]`.
3. Shows a **wave-picker group** (`QGroupBox` + `QListWidget`) listing the wave paths.
4. When the user selects a wave from the list, sets `self.loaded_wave` to the corresponding `IgorWave` — from this point the existing preview / settings / import flow is unchanged.
5. If the PXP contains exactly one wave, auto-selects it.
6. Wave picker is hidden/disabled for `.ibw` / `.csv` files.

#### UI layout addition

Insert a `QGroupBox("Wave Selection")` containing a `QListWidget` (single-selection) between the file-selection group and the preview group. It is shown only when a `.pxp` file is loaded.

#### State fields to add

```python
self.loaded_pxp_waves: dict[str, IgorWave] = {}
self.pxp_wave_list: QtWidgets.QListWidget  # added in _setup_ui
self.wave_picker_group: QtWidgets.QGroupBox
```

### 4. Tests

#### `tests/test_importer_igor2.py` additions

- `test_load_igor_pxp_file_returns_all_1d_waves` — monkeypatches `igor2.packed`, verifies dict keys and `IgorWave` fields.
- `test_load_igor_pxp_file_skips_multidimensional_waves` — verifies 2-D waves are excluded.
- `test_load_igor_pxp_file_raises_if_no_waves` — verifies `ValueError` for empty PXP.
- `test_load_igor_pxp_file_raises_if_igor2_not_installed` — verifies `ImportError`.

#### `tests/test_igor_import_dialog.py` additions

- `test_igor_import_dialog_loads_pxp_and_shows_wave_picker` — loads a mocked PXP, verifies the wave-picker is visible and populated.
- `test_igor_import_dialog_pxp_wave_selection_sets_loaded_wave` — selects a wave from the picker, verifies `loaded_wave` is set and preview updates.
- `test_igor_import_dialog_pxp_single_wave_auto_selects` — single-wave PXP auto-selects.

---

## Out of Scope

- Importing multiple waves from a PXP in a single action (import one at a time, user re-opens dialog for each).
- Exporting to `.pxp`.
- Importing directly into `Experiment` (patchclamp traces) from a `.pxp`; only `StimArray` target for now.
- Any changes to `exporter_igor2.py` or `igor2_patchclamp_export_dialog.py`.

---

## Acceptance Criteria

1. Opening a `.pxp` file in the Import Igor2 Wave dialog shows a populated wave-picker list.
2. Selecting a wave from the picker enables the full existing import flow (preview, source dt, new/existing stim target).
3. Single-wave PXPs skip the picker interaction (auto-selected).
4. `.ibw` and `.csv` import flow is completely unchanged.
5. All new tests pass. No existing tests broken.
6. `mypy` and `ruff` pass cleanly.
7. Test coverage on `importer_igor2.py` and `igor_import_dialog.py` remains ≥ 75%.
