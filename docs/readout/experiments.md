# Experiments

Each simulation run or imported recording becomes a data group (an
`Experiment`) holding trace, spike and synapse data. Groups appear in the
**Neuron data**, **Spike data**, **Synapse data** and **All data** selectors of
the Data toolbar.

## Manage Data

**File → Manage Data** lists every data group with its duration, data sets and
details (neurons, spikes, matrix size). **Delete Selected** removes groups and
all their data.

## Generate synthetic data

**File → Generate Synthetic Data...** (`Ctrl+G`) creates test traces for
prototyping analyses. Each group has **Export as a file** and **Add to the
project**; **Data Preview** updates as you change parameters.

**Patch clamp**: pick a template in **Select traces**, the neurons in
**Indexes to import**, **Noise (mV)** (default 0.5) and **Output sampling rate
(Hz)** (default 1000).

**Imaging**: fluorescence driven by a spike pattern (**Select spikes**).

| Parameter                  | Default           | Notes                                                                                                  |
| -------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------ |
| Recording duration (s)     | 10                |                                                                                                        |
| Number of neurons          | 10                |                                                                                                        |
| F0 (photoelectrons)        | 1000              | Baseline brightness, scaled per neuron by a lognormal (mean 0, sigma 0.25).                            |
| dF/F                       | 0.4               | Mean peak change for an isolated spike.                                                                |
| dF/F variability           | 0.04              |                                                                                                        |
| Indicator response         | Exponential decay | Or **Rise and decay** (difference of exponentials).                                                    |
| Rise time constant (s)     | 0.010             | Rise and decay only; must be shorter than the decay. Not the time to peak (about 20 ms with defaults). |
| Decay time (s)             | 0.050             |                                                                                                        |
| Sampling rate (Hz)         | 30                |                                                                                                        |
| Photoelectron noise factor | 1.0               | Shot-noise scaling.                                                                                    |

Overlapping spike responses add up. Responses are computed at 0.1 ms and then
sampled, so a sampled trace can miss the exact peak.

## Segmentation test data

To make a synthetic Cellpose image and matching segmentation from the command
line:

```bash
neuroseq --generate-cellpose-test-data ./cellpose_test_grid.tif
```

Options are in
[Holographic stimulation](../networks/holographic_stimulation.md).
