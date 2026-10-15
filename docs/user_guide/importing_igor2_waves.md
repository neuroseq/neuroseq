# Importing Igor2 Waves

NeuroSeq can import stimulation waveforms from Igor Pro / Sutter workflows into stimulation arrays.

Open the dialog from `File → Import data → Import stimulus → Import patch clamp`. The pre-dialog asks for the source software — choose `Igor2` — and the file picker accepts `.ibw`, `.pxp`, and `.csv` files.

## Supported input formats

CSV recordings can also be imported through `File → Input data → Output → Patch Clamp`
by choosing `CSV / Text (.csv, .txt)`. Both extensions use the same importer.
Each column is a separate trace and each row is a sample.
Comma, tab, and whitespace separators are detected automatically.
An optional header row names the traces; without it, names use the filename and
column number. Set the shared **Source sampling rate** in Hz, experiment, and
recorded signal units. Tables import all columns as new traces with consecutive
neuron IDs starting at the selected neuron ID. All columns are treated as signals,
so remove any time column before import. Column names must be non-empty and unique;
all rows must contain the same number of finite numeric samples.

For example, this imports traces named `cell_a` and `cell_b`:

```csv
cell_a,cell_b
-65,-62
-64,-61
-63,-60
```

For CSV patch-clamp input, choose `File → Input data → Input → Patch Clamp`,
then `CSV / Text (.csv, .txt)`. Both extensions accept a single column with one numeric sample per row; an
optional first row can contain the waveform name. Without a header, the file
name supplies the waveform name. Set **Source sampling rate** in Hz and choose
the input units before importing. The preview and imported duration use this
sampling rate. By default, the waveform is resampled onto the 1 ms input grid;
enable **Adjust NeuroSeq input grid to source/file grid** to preserve its timestep.

- `.ibw`: a single Igor Binary Wave
- `.pxp`: an Igor Packed Experiment containing one or more waves
- `.csv`: NeuroSeq's simple Igor-style CSV import format

Only one-dimensional waves can be imported into NeuroSeq stimulation arrays. Multi-dimensional waves inside a packed experiment are skipped.

## Packed experiments (`.pxp`)

When you open a `.pxp` file, NeuroSeq scans the packed experiment and lists all importable 1-D waves in a `Wave Selection` panel.

- If the file contains exactly one importable wave, it is selected automatically.
- If the file contains multiple waves, choose one from the list before importing.
- Wave names are shown using their packed-experiment folder path, for example `root/stims/HodgkinHuxley`.

Once a wave is selected, the rest of the import workflow is the same as for `.ibw` and `.csv`.

### When some records cannot be parsed

Real-world `.pxp` files occasionally contain wave records whose stored shape does not match the data bytes on disk (for example, dependency-formula waves written without a full data block). NeuroSeq loads such files tolerantly: the malformed record is skipped and logged at WARNING level, and the remaining waves are still imported.

If any records were skipped, a warning dialog lists the count after the file is opened. The wave picker still populates with every wave that loaded successfully, and the application log records each skipped record's index, type, and reason.

## Preview and timing

After selecting a wave, the preview shows:

- wave name
- source format
- source sample count
- detected source sample interval (`dt`) when present
- imported sample count on the currently selected NeuroSeq import grid
- units and axis units
- value range
- optional Igor note text

For CSV files, set the source sampling rate in Hz before import. For Igor binary
files, the source sample interval is read from the file.

By default, NeuroSeq imports stimulation data onto its 1 ms timeline and uses zero-order-hold resampling when the source `dt` differs from `0.001 s`.

If you enable `Adjust NeuroSeq stim grid to source/file grid`, NeuroSeq instead uses the selected source sample interval as the imported stimulation grid:

- new stimulation arrays are created with that `dt`
- existing stimulation arrays are remapped to that `dt` before import, after confirmation
- when source and target `dt` match, the Igor wave is copied without timeline resampling

## Import targets

The dialog offers two modes:

- **Create new stim array** — creates a new single-neuron `StimArray` from the imported wave. Suggested name and units default to the Igor wave name and units.
- **Import into existing stim array** — writes the imported wave into one selected neuron column of an existing stimulation array. Options:
  - _Target neuron index_ selects which column to write.
  - _Extend target stim array to fit imported duration_ grows the target in time if the imported wave is longer. Otherwise overflow samples are truncated.
  - _Replace target neuron values instead of adding to them_ clears the destination column before writing. When unchecked, imported values are added sample-by-sample to the existing column.
  - Other neuron columns are always preserved.

If the target stim array uses a different `dt` than the selected import grid, NeuroSeq asks for confirmation and remaps the target to the import grid before writing the wave.

For packed experiments, the suggested stim name and units update automatically when you choose a different wave.

## Current limitations

- Import is still one wave at a time, even for `.pxp` files.
- Packed-experiment metadata is not reconstructed into higher-level Sutter routine structures; NeuroSeq only imports the contained waves.
