# Experiments

## Experiment workflow

1. Create experiment definition.
2. Attach model and stimulus arrays.
3. Run and record outputs.
4. Export and compare results.

## Managing project data

**File → Manage Data** opens the data manager dialog,
which lists project data grouped by simulation run or recording and lets you
inspect or delete selected groups (see
[Connecting simulations and experiments](simulations_and_experiments.md#manage-data)
for details).

## Generating synthetic data (GUI)

**File → Generate Synthetic Data…** (`Ctrl+G`) opens a dialog that produces
synthetic patch-clamp traces or imaging traces directly inside the project,
useful for prototyping analyses without a real recording.

The dialog has two groups:

### Patch clamp

- **Select traces** — which template trace shapes to draw from.
- **Indexes to import** — which neurons receive synthesized traces.
- **Noise (mV)** — gaussian noise standard deviation added on top.

### Imaging

Choose **Indicator response** to control the fluorescence response to each spike:

- **Exponential decay** (default) preserves the instantaneous rise followed by
  exponential decay.
- **Rise and decay** uses a difference of exponentials with a gradual rise.
  **Rise time constant (s)** defaults to 0.010 s and must be shorter than
  **Decay time (s)**, which defaults to 0.050 s. These settings give a peak
  about 20.1 ms after an isolated spike. The rise constant is not the time to peak.

For both models, **dF/F** sets the mean peak fluorescence change of an isolated
spike before sampling and noise. Overlapping spike responses add together.
The response is calculated at 0.1 ms resolution before sampling at the selected
imaging rate, so a sampled trace may miss the exact peak. Both **Export as a
file** and **Add to the project** use the selected response model.

- **Select spikes** — source spike pattern that drives the synthetic
  fluorescence.
- **Recording duration / Number of neurons** — output shape.
- **F0 (photoelectrons)**, **dF/F**, **dF/F variability**, **Decay time
  (s)**, **Sampling rate (Hz)** — calcium-indicator parameters.
- **Lognormal mean / sigma (F0 distribution)** — distribution from which
  per-neuron baseline brightness is drawn.
- **Photoelectron noise factor** — shot-noise scaling.

A **Data Preview** panel updates as you change parameters so you can see
what will be added before committing. The generated traces are appended
to the project as a new experiment.

## Cellpose test data (CLI)

To generate synthetic Cellpose test data (TIFF image + matching `.npy`
segmentation) outside the GUI:

```bash
neuroseq --generate-cellpose-test-data ./cellpose_test_grid.tif
```

See [Multiphoton Holographic Stimulation → Generating test data](multiphoton_holographic_stimulation.md#generating-test-data-cli)
for the full set of options.
