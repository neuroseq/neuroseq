from pathlib import Path

import numpy as np
import pytest
import tifffile

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_scanimage_traces import DFF_UNITS
from neuroseq.importers.importer_scanimage_traces import baseline_frame_count
from neuroseq.importers.importer_scanimage_traces import delta_f_over_f
from neuroseq.importers.importer_scanimage_traces import extract_recording_dff
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    calcium_rise_fraction,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType

FPS = 30.0


def _cellpose(mapping: dict[int, int] | None = None) -> CellposeImportResult:
    segmentation = np.full((6, 8), -1, dtype=np.int32)
    segmentation[:3, :4] = 0
    segmentation[3:, 4:] = 1
    return CellposeImportResult(
        image_path=Path("fov.tif"),
        segmentation_array=segmentation,
        overlay_image=np.zeros((6, 8, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1]),
        stimulus_to_cellpose_mapping=mapping,
        mapping_strategy="Custom" if mapping else "Linear",
    )


def _block(onset_s: float, seconds: float, dt: float = 0.001) -> StimArray:
    values = np.zeros((int(seconds / dt), 2))
    values[int(onset_s / dt) : int((onset_s + 0.02) / dt), 1] = 1.0
    return StimArray("Block", StimType.External, values, "", dt)


def _synthetic_recording(path: Path, onset_s: float, frames: int) -> None:
    """Two interleaved channels, as ScanImage saves them. ROI 1 (cellpose id 1) has a
    jGCaMP-like transient after ``onset_s``; channel 2 is constant."""
    t = np.arange(frames) / FPS
    response = np.where(t >= onset_s, np.exp(-(t - onset_s) / 0.5), 0.0)
    pages = []
    for frame in range(frames):
        green = np.full((6, 8), 100.0)
        green[3:, 4:] = 100.0 * (1.0 + 0.8 * response[frame])
        pages += [green.astype(np.int16), np.full((6, 8), 7, dtype=np.int16)]
    tifffile.imwrite(path, np.stack(pages))


def test_delta_f_over_f_uses_the_frames_before_the_stimulus() -> None:
    traces = np.array([[100.0, 100.0, 150.0, 200.0]])
    np.testing.assert_allclose(delta_f_over_f(traces, 2), [[0.0, 0.0, 0.5, 1.0]])
    np.testing.assert_allclose(
        delta_f_over_f(np.array([[10.0, 10.0, 10.0, 10.0, 20.0]]), 0)[0, -1], 1.0
    )


def test_baseline_frame_count_stops_at_the_first_stimulus() -> None:
    assert baseline_frame_count(_block(0.5, 1.0), FPS) == 15
    assert baseline_frame_count(None, FPS) == 0


def test_extracts_dff_of_each_mapped_roi_from_a_recording(tmp_path) -> None:
    path = tmp_path / "2026-10-22_00001.tif"
    onset = 0.5
    _synthetic_recording(path, onset, frames=60)
    trace = extract_recording_dff(
        [path],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
    )
    assert trace.units == DFF_UNITS
    assert trace.dt == pytest.approx(1 / FPS)
    assert trace.traces.shape == (2, 60)
    np.testing.assert_array_equal(trace.raw_cellpose_ids, [0, 1])
    np.testing.assert_array_equal(trace.indices, [3, 7])
    # The responding cell's ΔF/F peaks near 0.8; the frame lit by the
    # stimulation laser is interpolated, the other cell stays flat.
    assert np.max(trace.traces[1]) == pytest.approx(0.8, abs=0.1)
    np.testing.assert_allclose(trace.traces[0], 0.0, atol=1e-6)
    # What the rheobase routine measures for neuron 7 after the pulse.
    rise = calcium_rise_fraction(trace, 7, onset, 250.0, 300.0)
    assert rise > 0.5
    assert calcium_rise_fraction(trace, 3, onset, 250.0, 300.0) == pytest.approx(0)


def test_extraction_needs_masks_and_the_functional_channel(tmp_path) -> None:
    path = tmp_path / "rec.tif"
    _synthetic_recording(path, 0.5, frames=4)
    with pytest.raises(ValueError, match="Cellpose"):
        extract_recording_dff(
            [path],
            fps=FPS,
            saved_channels=[1],
            functional_channel=1,
            cellpose_data=None,
        )
    with pytest.raises(ValueError, match="Channel 3 was not saved"):
        extract_recording_dff(
            [path],
            fps=FPS,
            saved_channels=[1, 2],
            functional_channel=3,
            cellpose_data=_cellpose(),
        )


def test_raw_fluorescence_rise_is_relative_to_its_baseline() -> None:
    from neuroseq.data import TraceData

    trace = TraceData(
        "F", np.array([[100.0] * 10 + [130.0] * 5]), np.array([0]), "fluorescence", 0.1
    )
    assert calcium_rise_fraction(trace, 0, 1.0, 500.0, 300.0) == pytest.approx(0.3)


def _transients(spike_frames: dict[int, list[int]], n_frames: int = 150) -> np.ndarray:
    """ΔF/F of three cells: 0.4 per spike, decaying with jGCaMP8m's 0.5 s, plus noise."""
    t = np.arange(n_frames) / FPS
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.02, (3, n_frames))
    for row, frames in spike_frames.items():
        for frame in frames:
            onset = frame / FPS
            values[row] += np.where(t >= onset, 0.4 * np.exp(-(t - onset) / 0.5), 0.0)
    return values.astype(np.float32)


def test_deconvolution_finds_the_spikes_of_each_cell() -> None:
    from neuroseq.data import TraceData
    from neuroseq.importers.importer_scanimage_traces import deconvolve_dff

    trace = TraceData(
        "dF/F",
        _transients({1: [30, 75], 2: [100]}),
        np.array([5, 7, 9]),
        DFF_UNITS,
        1 / FPS,
        np.array([0, 1, 2]),
    )
    spikes = deconvolve_dff(trace)
    np.testing.assert_array_equal(spikes.spike_indices, [7, 7, 9])
    np.testing.assert_allclose(spikes.spike_times, np.array([30, 75, 100]) / FPS)
    np.testing.assert_array_equal(spikes.raw_cellpose_ids, [1, 1, 2])
    # Noise alone (cell 5) never reaches the threshold; a higher one hides all.
    assert deconvolve_dff(trace, threshold=1.0).spike_indices.size == 0


def test_recorded_probe_is_counted_from_deconvolved_spikes(tmp_path) -> None:
    from neuroseq.importers.importer_scanimage_traces import deconvolve_dff
    from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
        measure_calcium_spike_count,
    )

    path = tmp_path / "rec.tif"
    onset = 0.5
    _synthetic_recording(path, onset, frames=60)
    trace = extract_recording_dff(
        [path],
        fps=FPS,
        saved_channels=[1, 2],
        functional_channel=1,
        cellpose_data=_cellpose({7: 1, 3: 0}),
        stim_block=_block(onset, 2.0),
    )
    spikes = deconvolve_dff(trace)
    assert set(spikes.spike_indices) == {7}
    assert measure_calcium_spike_count(spikes, 7, onset, 300.0, 1 / FPS) >= 1
    assert measure_calcium_spike_count(spikes, 3, onset, 300.0, 1 / FPS) == 0


def test_spike_in_the_onset_frame_counts_only_with_frame_tolerance() -> None:
    from neuroseq.data import SpikeData
    from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
        measure_calcium_spike_count,
    )

    frame = 1 / FPS
    # Timed at the start of frame 15, which contains an onset at 0.51 s.
    spikes = SpikeData("s", np.array([0, 0]), np.array([14 * frame, 15 * frame]))
    assert measure_calcium_spike_count(spikes, 0, 0.51, 100.0) == 0
    assert measure_calcium_spike_count(spikes, 0, 0.51, 100.0, frame) == 1
