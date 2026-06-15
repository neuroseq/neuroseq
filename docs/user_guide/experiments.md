# Experiments

## Experiment workflow

1. Create experiment definition.
2. Attach model and stimulus arrays.
3. Run and record outputs.
4. Export and compare results.

## Managing experiment sets

**Experiment → Manage Experiments** opens the experiment manager dialog,
which lists every experiment in the current project and lets you rename,
delete, or mark the active experiment (see
[Connecting simulations and experiments](simulations_and_experiments.md#manage-experiments)
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
