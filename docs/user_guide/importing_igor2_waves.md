# Importing Igor2 Waves

NeuroSeq can import stimulation waveforms from Igor Pro / Sutter workflows into stimulation arrays.

Open the dialog from `File -> Import Igor2 Wave (.ibw/.csv)`. The current menu label still mentions `.ibw/.csv`, but the file picker now also accepts packed experiment files (`.pxp`).

## Supported input formats

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

## Preview and timing

After selecting a wave, the preview shows:

- wave name
- source format
- source sample count
- detected source sample interval (`dt`) when present
- imported sample count on NeuroSeq's fixed 1 ms grid
- units and axis units
- value range
- optional Igor note text

The source sample interval can always be edited before import. NeuroSeq imports stimulation data onto its 1 ms timeline and uses zero-order-hold resampling when the source `dt` differs from `0.001 s`.

## Import targets

The dialog supports two modes:

- `Create new stim array`: creates a new single-neuron `StimArray` using the imported wave
- `Import into existing stim array`: writes the imported wave into one selected neuron column of an existing stimulation array

For packed experiments, the suggested stim name and units update automatically when you choose a different wave.

## Current limitations

- Import is still one wave at a time, even for `.pxp` files.
- Import into an existing stim array requires the target stimulation array to already use a 1 ms `dt`.
- Packed-experiment metadata is not reconstructed into higher-level Sutter routine structures; NeuroSeq only imports the contained waves.
