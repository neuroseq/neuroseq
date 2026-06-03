# SysCon2 Exporter — Fixed Absolute Timeline Layout

## Overview

The SysCon2 `.seq` exporter previously placed laser TTLPulse and TTL-out objects
on **dynamic** timeline rows computed as `n_entity_lanes + offset`. SysCon v3
actually reserves timeline rows **1..5** as fixed hardware lanes, so any stim
needing more than one shape lane caused the laser pulses to spill onto the
TTL-out rows or a different wavelength's row. The wavelength↔lightsource_id
mapping also had 940 nm and 1030 nm swapped, and the exporter labelled the
1100 nm laser as "1140 nm" — inconsistent with SysCon's UI and the Rapp LUT
file naming.

The exporter now emits the fixed SysCon v3 layout established by three reference
`.seq` files exported from the up-to-date SysCon software:

```
~/Downloads/laser_1100nm_TL2.seq
~/Downloads/laser_940nm_TL3.seq
~/Downloads/laser_1030nm_TL4.seq
```

(Filename "TL*n*" is the 1-based SysCon UI row; the on-disk
`TimelineInfo0_TimelineIndex` is `n - 1`.)

Applies to:

- `src/neuroseq/exporter_syscon2.py`
- `src/neuroseq/stimulation_generator_dialog.py` — laser combo + fallback
- `tests/test_exporter_syscon2.py` — golden indices rewritten
- `tests/test_stimulation_generator_dialog.py` — golden report indices

---

## SysCon v3 timeline layout (now hard-coded)

| TimelineIndex | Purpose             | LightsourceID | Port               |
| ------------- | ------------------- | ------------- | ------------------ |
| 0             | Shape (SLM)         | `dummy_2_0`   | — (Intensity_0=0)  |
| 1             | Laser TTL @ 1100 nm | `dummy_2_0`   | `dummy_2_0 RMI In` |
| 2             | Laser TTL @ 940 nm  | `dummy_1_0`   | `dummy_1_0 RMI In` |
| 3             | Laser TTL @ 1030 nm | `dummy_0_0`   | `dummy_0_0 RMI In` |
| 4             | TTL Out 1 (mirror)  | —             | `Holo4D TTL Out 1` |
| 5             | TTL Out 2 (gate)    | —             | `Holo4D TTL Out 2` |

The row↔lightsource binding is fixed in SysCon's hardware config; the exporter
must match exactly. TTL Out 1 mirrors the laser TTL (same `starttime`, same
duration) and acts as the "spiraling trigger" downstream.

---

## Constants reorganised in `exporter_syscon2.py`

```python
_SUPPORTED_LASER_WAVELENGTHS_NM = (940, 1030, 1100)
_LEGACY_LASER_WAVELENGTH_ALIASES_NM = {1040: 1030, 1140: 1100}
_DEFAULT_ACTIVE_LASER_WAVELENGTH_NM = 1100

# Absolute SysCon v3 indices — NOT offsets.
_LASER_TIMELINE_INDEX_BY_WAVELENGTH_NM = {1100: 1, 940: 2, 1030: 3}
_TTL_OUT_1_TIMELINE_INDEX = 4
_TTL_OUT_2_TIMELINE_INDEX = 5

_LIGHTSOURCE_ID_BY_WAVELENGTH_NM = {
    940:  "dummy_1_0",   # swapped vs. previous code
    1030: "dummy_0_0",   # swapped vs. previous code
    1100: "dummy_2_0",   # renamed from 1140
}
_SHAPE_LIGHTSOURCE_ID = "dummy_2_0"   # SLM driver, always emitted on Shape
```

`_LASER_TIMELINE_OFFSETS` and `_infer_holo4d_timeline_base` are removed — the
PNG overview labels are now read directly from the fixed constants.

---

## Shape-lane → timeline-index mapping

Shape lane 0 stays on TimelineIndex 0 (the SLM row every reference file uses);
additional simultaneous lanes are pushed past the reserved hardware rows:

```python
def _shape_lane_to_timeline_index(lane: int) -> int:
    # Rows 1..5 are reserved for lasers (1..3) and TTL outs (4..5).
    return 0 if lane == 0 else lane + 5
```

Used in both the Shape-block writer in `generate_syscon2_sequence_text` and the
debug CSV in `_build_debug_csv_text`.

---

## Shape block now emits the SLM driver, intensity 0

Every reference file has `Intensity_0 = 0` and `LightsourceID_0 = dummy_2_0` on
the Shape regardless of which laser is active — the laser TTLPulse carries the
actual power signal. The exporter follows suit:

```python
("TimelineInfo0_LightsourceID_0", _SHAPE_LIGHTSOURCE_ID),
("TimelineInfo0_Intensity_0",     "0"),
```

`force_binary_intensity=False` still affects sanitisation upstream (controlling
how ON samples are normalised before binning); it no longer changes the Shape's
emitted intensity field. The dialog tooltip was updated to reflect this.

**Downstream consequence:** the roundtrip path
`generate_syscon2_sequence_text → import_sequence_into_stim_array(include_non_shape=False)`
no longer recovers ON samples from Shape blocks alone — callers that need the
roundtrip must use the laser TTLPulse path (`include_non_shape=True`). Two
existing tests that exercised the Shape-only roundtrip were updated to assert
the new (empty) behaviour.

---

## Wavelength relabel: 1140 → 1100

SysCon and the Rapp LUT (`20260526_1100-LUT_RAPP.lut`) both label the
`dummy_2_0` lightsource as "1100 nm". NeuroSeq now uses 1100 as the canonical
value with 1140 kept as a legacy alias (same pattern as `1040 → 1030`).

- `StimulationGeneratorDialog.laser_combo` shows "1100 nm digital", default
  selection 1100.
- `_selected_laser_wavelength_nm` fallback returns 1100.
- Any external caller passing 1140 still works via the alias dict and is
  reported in `SysconExportReport.active_laser_wavelength_nm` as 1100.

---

## TTL Out 2 — explicitly out of scope

The reference files all emit a TTL Out 2 envelope that pre/post-pads the active
stim window, but the timing (start tick, duration) varies between files and
isn't deterministic from the user's description. TTL Out 2 emission is **not**
added in this patch. `SysconExportReport.ttl_out2_timeline_index` still reports
the correct reserved index (5).

---

## Tests

In `tests/test_exporter_syscon2.py`:

| Test                                                                                        | Asserts                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `test_export_seq_uses_fixed_laser_and_ttl_out_timeline_layout` (updated)                    | At 1100 nm: laser=1, TTL1=4, TTL2=5; two shape lanes appear at TimelineIndex 0 and 6                                                     |
| `test_export_seq_routes_ttl_pulses_to_selected_1030_laser_timeline` (upd.)                  | 1030 nm → laser_timeline_index 3; rows 1 and 2 are empty; TTL Out 1 still mirrors the laser                                              |
| `test_export_seq_accepts_legacy_1040_alias_as_1030_timeline` (updated)                      | 1040 alias → wavelength 1030, TimelineIndex 3                                                                                            |
| `test_export_seq_accepts_legacy_1140_alias_as_1100_timeline` (new)                          | 1140 alias → wavelength 1100, TimelineIndex 1                                                                                            |
| `test_export_seq_lightsource_id_matches_syscon_v3_template` (new)                           | Parametrized across (1100,1,`dummy_2_0`) / (940,2,`dummy_1_0`) / (1030,3,`dummy_0_0`); also asserts Shape=`dummy_2_0` with Intensity_0=0 |
| `test_export_seq_coalesces_nearby_segments_into_one_sanitized_window`                       | Shape-only roundtrip now empty (Shape Intensity_0 forced to 0)                                                                           |
| `test_export_seq_roundtrip_preserves_burst_timing_at_exact_tick_positions`                  | Same — `segment_counts == [0]*6` for shape-only path                                                                                     |
| `test_export_seq_shape_intensity_remains_zero_when_non_binary_input_is_requested` (renamed) | Shape intensity is 0 even with `force_binary_intensity=False`                                                                            |

In `tests/test_stimulation_generator_dialog.py`:

| Test                                                                         | Updated values                                                     |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `_build_export_report` helper                                                | default `active_laser_wavelength_nm=1100`; laser=1, TTL1=4, TTL2=5 |
| `test_stimulation_generator_dialog_binary_mode_checkbox_defaults_to_enabled` | `_selected_laser_wavelength_nm() == 1100`                          |

All 76 tests in `tests/test_exporter_syscon2.py` + `tests/test_stimulation_generator_dialog.py` pass.

---

## Manual verification recipe

Export a one-neuron / one-pulse stim at each of the three wavelengths and diff
against the matching reference file. The structural fields must match:

- **Shape block**: `TimelineInfo0_TimelineIndex == 0`,
  `TimelineInfo0_LightsourceID_0 == dummy_2_0`,
  `TimelineInfo0_Intensity_0 == 0`.
- **Laser TTLPulse**: `TimelineInfo0_TimelineIndex` matches the reserved row for
  the active wavelength (1100→1, 940→2, 1030→3); `LightsourceID_0` and `port`
  match the row↔device binding.

Start times and durations will differ — the references use hand-tuned positions.
