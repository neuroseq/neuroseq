"""Calcium traces of a ScanImage recording, extracted with the project's Cellpose masks.

Used by the closed-loop routines (rheobase, STDP window) to turn each recorded block into a
ΔF/F trace, and its OASIS-deconvolved spikes, without the TIFF import dialog: every block
reuses the same masks, so neuron IDs stay those of the stimulation targets. Qt-free.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping
from neuroseq.importers.importer_suite2p import remap_raw_cellpose_ids
from neuroseq.importers.importer_tif import DEFAULT_TAU
from neuroseq.importers.importer_tif import ProgressCallback
from neuroseq.importers.importer_tif import TifPipelineParams
from neuroseq.importers.importer_tif import cellpose_result_to_masks
from neuroseq.importers.importer_tif import impute_contaminated_frames
from neuroseq.importers.importer_tif import run_roi_average_pipeline
from neuroseq.importers.importer_tif import run_suite2p_pipeline
from neuroseq.importers.importer_tif import stimulation_contaminated_frames

if TYPE_CHECKING:
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.stimulation.stim import StimArray

#: How a recorded block's fluorescence is extracted: the mean of each Cellpose ROI's
#: pixels, or Suite2p's extraction with neuropil subtraction (slower).
ExtractionMethod = Literal["roi_average", "suite2p"]
EXTRACTION_METHODS: dict[str, str] = {
    "suite2p": "Suite2p (motion-corrected, neuropil-corrected)",
    "roi_average": "ROI average (fast)",
}

DFF_TRACE_NAME = "dF/F"
DFF_UNITS = "dF/F"
SPIKES_NAME = "Deconvolved spikes"
#: OASIS event size (in ΔF/F) counted as a spike by default; a single action
#: potential of jGCaMP8m gives a few tenths.
DEFAULT_SPIKE_THRESHOLD = 0.1


def baseline_frame_count(stim_block: StimArray | None, fs: float) -> int:
    """Frames recorded before the block's first stimulus (its ΔF/F baseline)."""
    if stim_block is None:
        return 0
    active = np.flatnonzero(np.any(np.asarray(stim_block.values) != 0, axis=1))
    if active.size == 0:
        return 0
    return int(np.floor(float(active[0]) * float(stim_block.dt) * fs))


def delta_f_over_f(traces: np.ndarray, baseline_frames: int) -> np.ndarray:
    """ΔF/F per row: ``(F - F0) / |F0|``. F0 is the mean of the first
    ``baseline_frames`` frames, or the 20th percentile of the whole trace when there
    are none. Only the divisor is made positive (and non-zero), so a negative
    baseline (offset-corrected detectors) still subtracts correctly."""
    traces = np.asarray(traces, dtype=np.float64)
    if baseline_frames >= 1:
        f0 = traces[:, : min(baseline_frames, traces.shape[1])].mean(axis=1)
    else:
        f0 = np.percentile(traces, 20, axis=1)
    scale = np.maximum(np.abs(f0), 1e-9)
    return ((traces - f0[:, np.newaxis]) / scale[:, np.newaxis]).astype(np.float32)


def extract_recording_dff(
    paths: Sequence[str | Path],
    *,
    fps: float,
    saved_channels: Sequence[int],
    functional_channel: int,
    cellpose_data: CellposeImportResult | None,
    stim_block: StimArray | None = None,
    remove_artefact: bool = True,
    method: ExtractionMethod = "roi_average",
    progress: ProgressCallback | None = None,
    on_registered: Callable[[dict], None] | None = None,
) -> TraceData:
    """ΔF/F of every Cellpose ROI in a ScanImage recording.

    ``saved_channels`` are the ScanImage channel numbers interleaved in the TIFFs and
    ``functional_channel`` the one carrying the calcium indicator. ``stim_block`` is the
    stimulus delivered during the recording, starting at its first frame: frames lit by
    the stimulation laser are interpolated (unless ``remove_artefact`` is False), and the
    frames before its first stimulus are the ΔF/F baseline.

    ``method`` "roi_average" averages each ROI's pixels; "suite2p" motion-corrects the
    recording with Suite2p, aligned to the image the Cellpose masks were drawn on (see
    :func:`fov_reference_image`), runs Suite2p's extraction with those masks and uses
    F - neuropil coefficient x Fneu; ``on_registered`` then gets the registration
    summary (``TifPipelineResult.extra["registration"]``). Both methods are then turned
    into ΔF/F the same way. ``progress`` is passed to the pipeline, which
    calls it between its stages (raise ``PipelineCancelled`` there to stop it).
    """
    if cellpose_data is None:
        raise ValueError(
            "Load a Cellpose segmentation of this field of view first: its masks "
            "define the neurons whose calcium traces are measured."
        )
    channels = [int(channel) for channel in saved_channels]
    if functional_channel not in channels:
        raise ValueError(
            f"Channel {functional_channel} was not saved in the recording "
            f"(saved: {', '.join(map(str, channels)) or 'none'})."
        )
    if fps <= 0:
        raise ValueError("The recording has no valid frame rate.")
    masks = cellpose_result_to_masks(cellpose_data)
    params = TifPipelineParams(
        fs=fps,
        nchannels=len(channels),
        functional_chan=channels.index(functional_channel) + 1,
    )
    tif_paths = [Path(path) for path in paths]
    if method == "suite2p":
        result = run_suite2p_pipeline(
            tif_paths,
            masks,
            params,
            progress=progress,
            register=True,
            reference_image=fov_reference_image(cellpose_data),
        )
        if on_registered is not None and "registration" in result.extra:
            on_registered(result.extra["registration"])
        fluorescence = np.asarray(result.F, dtype=np.float64) - float(
            params.neuropil_coefficient
        ) * np.asarray(result.Fneu, dtype=np.float64)
        if remove_artefact and stim_block is not None:
            impute_contaminated_frames(
                fluorescence,
                stimulation_contaminated_frames(stim_block, fps, fluorescence.shape[1]),
            )
        # Suite2p's ROIs are the mask labels in increasing order.
        labels = np.unique(masks)
        labels = labels[labels != 0]
    elif method == "roi_average":
        result = run_roi_average_pipeline(
            tif_paths,
            masks,
            params,
            progress=progress,
            stim_array=stim_block if remove_artefact else None,
        )
        fluorescence = result.F
        labels = result.extra["roi_labels"]
    else:
        raise ValueError(f"Unknown trace extraction method '{method}'.")
    cellpose_ids = np.asarray(labels, dtype=np.uint32) - 1
    return TraceData(
        DFF_TRACE_NAME,
        delta_f_over_f(fluorescence, baseline_frame_count(stim_block, fps)),
        remap_raw_cellpose_ids(
            cellpose_ids, resolve_stimulus_to_cellpose_mapping(cellpose_data)
        ),
        DFF_UNITS,
        1.0 / fps,
        cellpose_ids,
    )


def fov_reference_image(cellpose_data: CellposeImportResult) -> np.ndarray | None:
    """The image the Cellpose masks were drawn on (e.g. a "Grab FOV" mean TIFF), as
    a 2D float array to register recordings to; None if it is not a readable TIFF."""
    path = Path(cellpose_data.image_path)
    if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
        return None
    try:
        import tifffile

        image = np.asarray(tifffile.imread(path), dtype=np.float64)
    except Exception:
        return None
    while image.ndim > 2:
        image = image.mean(axis=0)
    return image if image.ndim == 2 else None


def deconvolve_dff(
    trace: TraceData,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    tau: float = DEFAULT_TAU,
) -> SpikeData:
    """Spikes of a ΔF/F trace, deconvolved with OASIS (as Suite2p does).

    The trace is already baseline-corrected, so Suite2p's sliding-window baseline,
    meant for long recordings, is skipped. OASIS estimates events in ΔF/F units;
    every frame whose event reaches ``threshold`` is a spike at that frame's start,
    so the threshold is absolute and a silent cell's noise is not scaled up into
    spikes.
    """
    from suite2p.extraction import dcnv

    events = dcnv.oasis(
        F=np.ascontiguousarray(trace.traces, dtype=np.float32),
        batch_size=200,
        tau=float(tau),
        fs=1.0 / float(trace.dt),
    )
    rows, frames = np.nonzero(np.asarray(events) >= threshold)
    order = np.argsort(frames, kind="stable")
    rows, frames = rows[order], frames[order]
    raw_ids = (
        trace.raw_cellpose_ids[rows] if trace.raw_cellpose_ids is not None else None
    )
    return SpikeData(
        SPIKES_NAME,
        trace.indices[rows],
        frames.astype(np.float64) * float(trace.dt),
        raw_ids,
    )


def functional_channel_mean(
    paths: Sequence[str | Path],
    *,
    saved_channels: Sequence[int],
    functional_channel: int,
) -> np.ndarray:
    """Mean image of ``functional_channel`` over all frames of a recording.

    ScanImage interleaves the saved channels page by page (page ``i`` holds
    channel ``i mod len(saved_channels)``); the mean of a short recording is the
    field-of-view image cell segmentation (Cellpose) runs on, in the TIFFs' pixel
    coordinates.
    """
    import tifffile

    channels = [int(channel) for channel in saved_channels]
    if functional_channel not in channels:
        raise ValueError(
            f"Channel {functional_channel} was not saved in the recording "
            f"(saved: {', '.join(map(str, channels)) or 'none'})."
        )
    position = channels.index(functional_channel)
    total: np.ndarray | None = None
    count = 0
    index = 0  # frame number across files, for the channel interleave
    for path in paths:
        # Page by page: a long grab never has to fit in memory at once. ScanImage
        # writes one frame per page; other writers may put several in one page.
        with tifffile.TiffFile(Path(path)) as tif:
            for page in tif.pages:
                data = np.asarray(page.asarray())
                for frame in data if data.ndim == 3 else (data,):
                    if index % len(channels) == position:
                        frame = frame.astype(np.float64)
                        total = frame if total is None else total + frame
                        count += 1
                    index += 1
    if total is None or count == 0:
        raise ValueError("The recording holds no frames of the functional channel.")
    return (total / count).astype(np.float32)
