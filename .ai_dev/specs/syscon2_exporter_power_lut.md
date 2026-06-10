# SysCon2 Exporter — Power Control via Project-Bundled LUTs

## Overview

NeuroSeq can now drive the delivered laser power in mW for SysCon2 exports.
SysCon reads `TimelineInfo0_Intensity_0` on both the Shape and the laser
TTLPulse and applies its hardware-configured percent→mW lookup table; previously
NeuroSeq wrote `"0"` on both fields. The exporter now accepts a target power
(mW) and a `PowerLut`, inverts the LUT to compute the percent, and writes that
percent into both fields. A bundled default LUT for 1100 nm ships with the
package so the feature works out of the box; users override it per project via
a "Load LUT…" button in the stim-generator dialog.

Applies to:

- `src/neuroseq/power_lut.py` (new) — parser, inverse, bundled-default
  registry, resolver
- `src/neuroseq/resources/syscon_luts/default_1100nm.lut` (new) — bundled
  reference LUT from `20260526_1100-LUT_RAPP.lut`
- `src/neuroseq/project.py` — `Project.power_luts: dict[int, PowerLut]` field
  - JSON persistence inside `project_metadata.json`
- `src/neuroseq/exporter_syscon2.py` — `target_power_mw` / `power_lut` kwargs
  on both `generate_syscon2_sequence_text` and `check_syscon2_input_sanity`
- `src/neuroseq/stimulation_generator_dialog.py` — "Target power" spinbox,
  "Load LUT…" button, status label, saturation warning
- `pyproject.toml` — `[tool.hatch.build.targets.wheel]` `force-include` for the
  new resources dir

---

## LUT data model and inverse lookup

```python
@dataclass(frozen=True)
class PowerLut:
    description: str
    tuples: tuple[tuple[float, float], ...]   # ((percent, mW), …), sorted by percent

def parse_power_lut_text(text: str) -> PowerLut: ...
def power_lut_mw_to_percent(lut: PowerLut, target_mw: float) -> tuple[float, bool]:
    """Returns (percent, saturated). target_mw <= 0 → (0.0, False).
    Linear interpolation between bracketing tuples; saturation clamps to the
    SMALLEST percent reaching max_mw (most efficient operating point)."""
```

A `.lut` file's hash line, `Enabled` flag, and `TuplesCount` are accepted but
ignored — only `Description`, `Unit`, and the `Tuple_n_` rows matter. Unit
**must** be `mW`; anything else raises `ValueError`.

## Default LUT registry

```python
DEFAULT_POWER_LUTS: dict[int, PowerLut] = load_bundled_default_luts()

def resolve_power_lut(
    project_luts: dict[int, PowerLut],
    wavelength_nm: int,
) -> tuple[PowerLut | None, bool]: ...
```

`load_bundled_default_luts()` enumerates `default_<wavelength>nm.lut` files in
the `neuroseq.resources.syscon_luts` package via `importlib.resources`. Only
1100 nm currently has a bundled default. `resolve_power_lut(...)` returns the
project's override when present, falling back to the bundled default; the
boolean flag (`is_override`) lets the UI tag the status as "(default)".

## Project persistence

`Project.power_luts` is stored in `project_metadata.json` as a `dict[str, dict]`
keyed by wavelength (string for JSON compatibility):

```json
{
  "power_luts": {
    "1100": {"description": "1100nm 1 spot", "tuples": [[0, 0], [5, 0], …]}
  }
}
```

`Project.__init__` re-hydrates `PowerLut` objects via `PowerLut.from_serializable`.
`Project.save` only writes the field when non-empty — projects that haven't
overridden anything stay minimal, and updates to the bundled default propagate
to old projects automatically. `Project.import_project` doesn't touch the LUTs
(LUTs aren't a per-component import).

## Exporter wiring

Both `generate_syscon2_sequence_text` and `check_syscon2_input_sanity` gained:

```python
target_power_mw: float = 0.0,
power_lut: PowerLut | None = None,
```

When `target_power_mw > 0` and `power_lut` is truthy, the exporter:

- **Shape `Intensity_0`** = `power_lut_mw_to_percent(target_power_mw × window.intensity)`
  via a new `_shape_intensity_for_window` helper. With "Force binary" checked,
  `window.intensity == 1.0` so every Shape gets the full target percent;
  unchecked, the per-window 0–1 amplitude scales the target.
- **Laser TTLPulse `Intensity_0`** = inverse-LUT of `target_power_mw × max(window.intensity for overlapping window)`.
  The TTL pulse spans potentially multiple shapes after merge, so we take the
  per-pulse effective intensity as the maximum across overlapping shape
  windows.
- **TTL Out 1 / TTL Out 2** stay at `Intensity_0 = "0"` with no lightsource
  attached (these are digital triggers, not laser pulses).

When `target_power_mw == 0` or `power_lut is None`, both `Intensity_0` fields
write `"0"` — preserving the previous behaviour for callers/tests that don't
opt in.

`SysconExportReport` grew three fields:

```python
power_saturation_warnings: list[str] = field(default_factory=list)
target_power_mw: float = 0.0
power_lut_description: str = ""
```

`check_syscon2_input_sanity` de-dupes saturation messages into its own
`warnings` list. The overrun-fix path inside the sanity check stays tied to the
unshifted pulse geometry (carried over from the spiralling patch).

## Dialog UI

The stim-generator dialog gained, between the laser-timeline row and the
HOLO4D config text edit:

- `QDoubleSpinBox` `target_power_spin` — range 0.0–1000.0 mW, suffix " mW",
  step 0.5, default 0.0 (= power control off).
- `QPushButton` `load_lut_button` — opens a `QFileDialog` filtered to
  `*.lut`; the result is parsed via `parse_power_lut_text` and stored in
  `project.power_luts[selected_wavelength]`. Parse errors surface as a
  `QMessageBox.warning` and leave the project state untouched.
- `QLabel` `lut_status_label` — `"LUT: {description} (max {max_mw:.1f} mW)"`,
  with " (default)" appended when the LUT comes from `DEFAULT_POWER_LUTS`,
  or `"No LUT loaded for {wavelength} nm"` when none is available.

The label refreshes on every `laser_combo` change. After a successful export,
any `export_report.power_saturation_warnings` are shown in a single
`QMessageBox.warning` (truncated to 20 lines + "…" overflow).

`_selected_power_lut(self) -> PowerLut | None` is a thin convenience over
`resolve_power_lut(self.project.power_luts, self._selected_laser_wavelength_nm())`.

## Tests

In `tests/test_power_lut.py` (new, 9 tests):

| Test                                                                 | Asserts                                                                            |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `test_parse_power_lut_text_parses_reference_example`                 | The reference 1100 nm `.lut` parses into 9 tuples with description "1100nm 1 spot" |
| `test_power_lut_mw_to_percent_interpolates_linearly`                 | Linear LUT (0,0)→(50,25): 12.5 mW → 25% percent                                    |
| `test_power_lut_mw_to_percent_saturates_at_first_max_percent`        | Reference LUT, 60 mW → (80%, saturated=True)                                       |
| `test_power_lut_mw_to_percent_zero_target_returns_zero`              | target_mw = 0 → (0.0, False)                                                       |
| `test_power_lut_serializable_roundtrip`                              | `PowerLut.from_serializable(lut.to_serializable()) == lut`                         |
| `test_default_power_luts_includes_1100nm`                            | Bundled LUT loaded for 1100 nm with correct tuples                                 |
| `test_resolve_power_lut_prefers_project_override_over_default`       | Project override wins, `is_override=True`                                          |
| `test_resolve_power_lut_falls_back_to_default_when_project_has_none` | Empty project_luts → bundled default with `is_override=False`                      |
| `test_resolve_power_lut_returns_none_for_unknown_wavelength`         | 940 nm (no bundled default) → `(None, False)`                                      |

In `tests/test_exporter_syscon2.py` (three new tests):

| Test                                                                  | Asserts                                                                           |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `test_export_seq_writes_lut_inverted_intensity_when_power_requested`  | target=25 mW on a (0,0)→(100,50) linear LUT → Shape and laser TTL both write "50" |
| `test_export_seq_clamps_and_warns_when_target_power_exceeds_lut_max`  | target=200 mW on the same LUT → both fields clamped to "100" + warning emitted    |
| `test_export_seq_uses_window_intensity_as_multiplier_when_non_binary` | force-binary off + window.intensity=0.4 + target=50 mW → 40% percent on both      |

In `tests/test_stimulation_generator_dialog.py` (two new dialog tests):

| Test                                                                        | Asserts                                                                                                                      |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `test_stimulation_generator_dialog_load_lut_assigns_to_selected_wavelength` | After `_on_load_lut_clicked` with the reference `.lut`, `project.power_luts[1100]` is set and status label drops "(default)" |
| `test_stimulation_generator_dialog_passes_target_power_and_lut_to_exporter` | `target_power_mw` and the synthetic `power_lut` are captured in the kwargs passed to `generate_syscon2_sequence_text`        |

In `tests/test_project.py`:

| Test                                                | Asserts                                                               |
| --------------------------------------------------- | --------------------------------------------------------------------- |
| `test_project_power_luts_persist_through_save_load` | A `Project` with `power_luts[1100]` round-trips across save / re-open |

Full suite: **856 passed, 2 skipped** (no regressions).

---

## Manual verification recipe

1. Open the stim-generator dialog on a fresh project with 1100 nm selected.
   Status label should read `"LUT: 1100nm 1 spot (max 49.0 mW) (default)"` —
   the bundled default is picked up with no user action.
2. Set target = 20 mW, force-binary ON, export. Both Shape and laser TTLPulse
   `TimelineInfo0_Intensity_0` should be ≈ 36.6 (linear interpolation
   between (40,21.9) and (20,8.4) at 20 mW).
3. Set target = 60 mW, export. Both fields clamp to "80" and the saturation
   `QMessageBox.warning` appears.
4. Click "Load LUT…", pick a different `.lut` file → status label drops the
   "(default)" suffix. Save the project as `.nseq`, re-open → the user-loaded
   LUT is still in effect.
5. Switch the laser combo to 940 nm. Status label reads "No LUT loaded for
   940 nm"; setting a target with no LUT keeps `Intensity_0` at "0".
