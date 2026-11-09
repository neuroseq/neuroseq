# Patch-clamp data

Igor Pro (WaveMetrics) is the default supported software: its `.ibw` and
`.pxp` files are read directly, with units, sample interval and wave notes.
This covers SutterPatch, which runs inside Igor Pro. Data from other
acquisition software can be imported as `.csv` or `.txt`.

Patch-clamp waveforms come in through two menu entries:

| Menu                                          | Imports                                                         | Into                        |
| --------------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| **File → Import data → Input → Patch Clamp**  | Command waveforms (stimuli)                                     | Input arrays                |
| **File → Import data → Output → Patch Clamp** | Recordings (voltage in current clamp, current in voltage clamp) | Trace data (and spike data) |

## File formats

- `.ibw`: Igor Binary Wave. 1-D waves, or 2-D samples × sweeps waves, whose
  sweeps are joined end to end (inter-sweep gaps are not kept).
- `.pxp`: Igor Packed Experiment. Importable waves are listed by folder path
  (e.g. `root/stims/HodgkinHuxley`); one is selected automatically if it is
  the only one. Malformed records are skipped with a warning and logged.
- `.csv` / `.txt`: comma, tab or whitespace separated; set **Source sampling
  rate** (Hz).
  - Input: one value per row, optional header with the waveform name.
  - Output: one trace per column, optional header row with unique trace
    names. All columns are signals (remove any time column); rows must be
    complete and numeric. Columns get consecutive neuron IDs from the selected
    one.

```csv
cell_a,cell_b
-65,-62
-64,-61
```

The preview shows name, format, sample count, sample interval, units, value
range and Igor note. Data keep their source sample interval: no resampling on
import, so different rates coexist and align by time.

## Import targets

**Input** dialog:

- **Create new input array**: name, type (Excitatory, Inhibitory, External),
  units; defaults come from the wave.
- **Import into existing input array**: pick array and **Target neuron
  index**. **Extend target input array to fit imported duration** (else
  overflow is cut); **Replace target neuron values instead of adding to them**
  (else values are added). Other neurons are untouched.

**Output** dialog:

- **Create new trace data** (experiment, name, neuron ID, units) or add to
  existing trace data (**Target neuron ID**, extend, replace an existing row).
- **Detect spikes** (current-clamp voltage only) stores detected spikes as
  spike data in the same experiment.
- Packed experiments: import as many waves as needed, then click **Close**.

If the target uses a different sample interval, NeuroSeq asks:

- **Resample wave to target Δt** (default): the target is unchanged.
- **Remap target to wave Δt**: the whole target moves to the wave's grid.
  Timing can shift by one sample, and a fast recording can make the target
  very large.
- **Cancel**.

Each import is one undoable edit.

## Limitations

- One wave per import, also for `.pxp`.
- Sutter routine structure in packed experiments is not reconstructed, only the
  waves.

To export input arrays for SutterPatch, see
[Patch clamp export](../input_design/patch_clamp_export.md).
