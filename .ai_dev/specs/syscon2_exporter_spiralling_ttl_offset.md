# SysCon2 Exporter — Spiralling TTL Offset

## Overview

SysCon v3 in spiralling mode requires the laser TTLPulse to start **one tick
before** the SLM completes its 16.666 ms (= 32-tick) load, so the laser edge
aligns with the spiraling hardware's ~1-tick warm-up. NeuroSeq's exporter now
emits the laser TTL (and the mirrored Holo4D TTL Out 1) at this shifted position
by default. A new **Spiralling** checkbox in the stim-generator dialog (default
ON) lets the user opt out for legacy non-spiralling hardware, in which case the
TTL fires exactly at SLM-ready.

The shift is intentionally **uniform**: pulse_start and pulse_end both move by
-1 tick (preserving duration). Multi-burst-per-shape behaviour is unchanged —
this is Option B from the design discussion (rather than the more invasive
Option A "one TTL per shape, anchored absolutely to shape_start + 31").

Applies to:

- `src/neuroseq/exporter_syscon2.py` — `generate_syscon2_sequence_text`,
  `check_syscon2_input_sanity`
- `src/neuroseq/stimulation_generator_dialog.py` — new spiralling checkbox,
  plumbed into both sanity-check and export call sites

---

## What the shift does

| `spiralling`     | Laser TTL `starttime`                     | Laser TTL duration |
| ---------------- | ----------------------------------------- | ------------------ |
| `True` (default) | `state.pulse_intervals[i].start − 1` tick | unchanged          |
| `False`          | `state.pulse_intervals[i].start`          | unchanged          |

The same shift applies to the Holo4D TTL Out 1 mirror (it iterates the same
`pulse_intervals` list). TTL Out 2 stays at intensity 0 / no lightsource and is
unaffected. Shape blocks are unaffected.

---

## Constants and signatures

In `exporter_syscon2.py`:

```python
_SPIRALLING_TTL_LEAD_TICKS = 1   # TTL fires 1 tick before SLM load completes

def generate_syscon2_sequence_text(
    ...,
    active_laser_wavelength_nm: int = _DEFAULT_ACTIVE_LASER_WAVELENGTH_NM,
    spiralling: bool = True,
) -> tuple[str, SysconExportReport]: ...

def check_syscon2_input_sanity(
    ...,
    active_laser_wavelength_nm: int = _DEFAULT_ACTIVE_LASER_WAVELENGTH_NM,
    spiralling: bool = True,
) -> SysconInputSanityReport: ...
```

The shift is applied as a fresh list (not in-place on `state.pulse_intervals`),
so the `_PulseWindowState` object remains valid for callers that need the
unshifted geometry — specifically `check_syscon2_input_sanity`'s overrun-fix
path (`_find_pulse_overrun_violations` at exporter_syscon2.py:2013) keeps using
the unshifted intervals because the fix-region math is tied to the original
window geometry.

No defensive clamping (`max(0, start - 1)`): pulses always have `start >= 32`
from upstream preblank insertion, so the shifted `start >= 31` is always
positive. A negative value would mean the preblank logic upstream broke and
should surface loudly via the validator.

---

## Dialog

`StimulationGeneratorDialog` now exposes a checkbox (default checked) between
the binary-mode checkbox and the laser combo:

```python
self.spiralling_checkbox = QtWidgets.QCheckBox(
    "Spiralling (TTL fires 1 tick before SLM load completes)"
)
self.spiralling_checkbox.setChecked(True)
```

Both call sites pass the checkbox state through:

- `_run_sanity_check` → `check_syscon2_input_sanity(..., spiralling=self.spiralling_checkbox.isChecked())`
- `_generate_holo4d_script` → `generate_syscon2_sequence_text(..., spiralling=self.spiralling_checkbox.isChecked())`

---

## Tests

In `tests/test_exporter_syscon2.py`:

| Test                                                                   | Asserts                                                                                                                                                                               |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_export_seq_spiralling_shifts_laser_pulse_one_tick_earlier` (new) | With identical input, `spiralling=False` laser pulse starts exactly `_TTL_QUANTUM_US` later than `spiralling=True`; repeats (duration) unchanged; TTL Out 1 mirror shifts identically |
| All other exporter tests (updated)                                     | Each `generate_syscon2_sequence_text` call explicitly passes `spiralling=False` to preserve pre-existing start-tick assertions                                                        |

In `tests/test_stimulation_generator_dialog.py`:

| Test                                                                                      | Asserts                                                                                                                  |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `test_stimulation_generator_dialog_binary_mode_checkbox_defaults_to_enabled` (updated)    | `dialog.spiralling_checkbox.isChecked()` (default ON)                                                                    |
| `test_stimulation_generator_dialog_sanity_check_uses_selected_laser_wavelength` (updated) | Captures the `spiralling` kwarg passed to `check_syscon2_input_sanity` (True by default)                                 |
| `test_stimulation_generator_dialog_generate_passes_spiralling_checkbox` (new)             | After unchecking the spiralling checkbox, the captured `spiralling` kwarg to `generate_syscon2_sequence_text` is `False` |

All 78 tests in the two affected test files pass.

---

## Manual verification recipe

Export a one-neuron / one-pulse stim against the reference template with:

- **Spiralling ON** → laser TTL `TimelineInfo0_starttime` matches the reference
  `.seq` exactly (e.g. 49477.1875 µs = tick 95 for the 1100 nm reference, when
  the drawn stim sits in the second SLM window).
- **Spiralling OFF** → same TTL appears one tick later (= 49998.0 µs in the
  same example), and TTL Out 1 mirrors it identically.
