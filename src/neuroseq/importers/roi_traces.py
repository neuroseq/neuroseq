"""Streaming ROI and neuropil traces from TIFF stacks, and their ΔF/F. Qt-free.

Used by the "Simply average from ROI" TIFF pipeline
(:func:`neuroseq.importers.importer_tif.run_roi_average_pipeline`). Frames are
read in chunks rather than whole files, so a 15-minute 512×512 recording (~14 GB)
does not have to fit in memory. Every ROI (and neuropil annulus) is a row of one
sparse averaging matrix, so all of a chunk's traces come from one product.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: Frames read and averaged at a time (64 frames of 512×512 float32 are 64 MB).
DEFAULT_CHUNK_FRAMES = 64
#: Pixels between a cell and its neuropil annulus, as Suite2p's
#: ``inner_neuropil_radius``.
NEUROPIL_INNER_RADIUS_PX = 2.0
#: Pixels in each neuropil annulus (fewer only if the image has no more), as
#: Suite2p's ``min_neuropil_pixels``.
NEUROPIL_MIN_PIXELS = 350
#: How far from its cell an annulus may reach.
NEUROPIL_MAX_RADIUS_PX = 100


def averaging_matrix(pixel_sets: list[npt.NDArray[np.intp]], n_pixels: int) -> Any:
    """Sparse (len(pixel_sets), n_pixels) matrix whose row ``i`` averages the
    flat pixel indices ``pixel_sets[i]`` (an empty set gives a row of zeros)."""
    from scipy import sparse

    rows = np.concatenate(
        [np.full(len(pixels), i, dtype=np.intp) for i, pixels in enumerate(pixel_sets)]
        or [np.array([], dtype=np.intp)]
    )
    columns = np.concatenate(
        [np.asarray(p, dtype=np.intp) for p in pixel_sets]
        or [np.array([], dtype=np.intp)]
    )
    weights = np.concatenate(
        [
            np.full(len(pixels), 1.0 / len(pixels), dtype=np.float64)
            for pixels in pixel_sets
            if len(pixels)
        ]
        or [np.array([], dtype=np.float64)]
    )
    return sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(pixel_sets), n_pixels)
    )


def roi_pixels(
    masks: npt.NDArray[Any], labels: npt.NDArray[Any]
) -> list[npt.NDArray[np.intp]]:
    """Flat pixel indices of each label in ``masks``."""
    flat = np.asarray(masks).ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.searchsorted(sorted_labels, labels, side="left")
    ends = np.searchsorted(sorted_labels, labels, side="right")
    return [order[s:e] for s, e in zip(starts, ends, strict=True)]


def neuropil_pixels(
    masks: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    *,
    inner_radius_px: float = NEUROPIL_INNER_RADIUS_PX,
    min_pixels: int = NEUROPIL_MIN_PIXELS,
    max_radius_px: int = NEUROPIL_MAX_RADIUS_PX,
) -> list[npt.NDArray[np.intp]]:
    """Flat pixel indices of a neuropil annulus around each label.

    As Suite2p's: the closest ``min_pixels`` pixels to the cell that are more than
    ``inner_radius_px`` from every cell (its own included), so no cell's light
    leaks into any annulus.
    """
    from scipy import ndimage

    masks = np.asarray(masks)
    height, width = masks.shape
    cells = masks > 0
    # Pixels near any cell (cells included) belong to no annulus.
    allowed = ndimage.distance_transform_edt(~cells) > inner_radius_px
    objects = ndimage.find_objects(np.where(cells, masks, 0).astype(np.int64))
    annuli: list[npt.NDArray[np.intp]] = []
    for label in labels:
        index = int(label) - 1
        box = objects[index] if 0 <= index < len(objects) else None
        if box is None:
            annuli.append(np.array([], dtype=np.intp))
            continue
        y0 = max(0, box[0].start - max_radius_px)
        y1 = min(height, box[0].stop + max_radius_px)
        x0 = max(0, box[1].start - max_radius_px)
        x1 = min(width, box[1].stop + max_radius_px)
        window = masks[y0:y1, x0:x1] == label
        distance = ndimage.distance_transform_edt(~window)
        candidates = allowed[y0:y1, x0:x1] & (distance <= max_radius_px)
        ys, xs = np.nonzero(candidates)
        nearest = np.argsort(distance[ys, xs], kind="stable")[:min_pixels]
        flat = (ys[nearest] + y0) * width + (xs[nearest] + x0)
        annuli.append(np.sort(flat).astype(np.intp))
    return annuli


@dataclass
class TiffLayout:
    """Where a TIFF's functional-channel frames are."""

    #: Image planes in the file (frames × channels).
    n_planes: int
    #: Frames of the functional channel.
    n_frames: int
    frame_shape: tuple[int, int]


def tiff_layout(
    series_shape: tuple[int, ...], nchannels: int, path: Path
) -> TiffLayout:
    """Check a TIFF series' shape against ``nchannels``: ``(frame, y, x)`` stacks
    interleave the channels, ``(time, channel, y, x)`` stacks hold them on axis 1."""
    shape = tuple(int(n) for n in series_shape)
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) not in (3, 4):
        raise ValueError(
            f"TIFF {path} must have shape (frame, y, x) or "
            f"(time, channel, y, x), got {shape}"
        )
    if len(shape) == 4:
        if shape[1] != nchannels:
            raise ValueError(
                f"TIFF {path} contains {shape[1]} channel(s), but the "
                f"pipeline is configured for {nchannels}"
            )
        n_planes = shape[0] * shape[1]
    else:
        n_planes = shape[0]
        if n_planes % nchannels != 0:
            raise ValueError(
                f"TIFF {path} has {n_planes} frame(s), which is not divisible by "
                f"nchannels={nchannels} for interleaved acquisition"
            )
    return TiffLayout(n_planes, n_planes // nchannels, (shape[-2], shape[-1]))


def iter_functional_chunks(
    path: Path,
    nchannels: int,
    functional_chan: int,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    on_layout: Callable[[TiffLayout], None] | None = None,
) -> Iterator[npt.NDArray[Any]]:
    """Yield the functional channel's frames of ``path`` as ``(k, y, x)`` chunks.

    Files with a page per plane are read page by page, and contiguous files
    described by one page (ImageJ hyperstacks over 4 GB) memory-mapped, so only
    one chunk is in memory at a time; anything else is read whole. ``on_layout`` gets the
    file's layout before the first chunk.
    """
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        # The shape the file was written with (tifffile drops length-1 axes from
        # series.shape), as tifffile.imread returns it.
        shaped = tif.shaped_metadata[0] if tif.shaped_metadata else None
        shape = tuple(shaped["shape"]) if shaped and "shape" in shaped else series.shape
        layout = tiff_layout(shape, nchannels, path)
        if on_layout is not None:
            on_layout(layout)
        planes = np.arange(layout.n_frames) * nchannels + (functional_chan - 1)
        height, width = layout.frame_shape
        read: Callable[[npt.NDArray[np.intp]], npt.NDArray[Any]]
        if len(series.pages) == layout.n_planes:
            read = lambda idx: np.asarray(  # noqa: E731
                tif.asarray(key=[int(i) for i in idx], series=0)
            ).reshape(len(idx), height, width)
        elif series.dataoffset is not None:
            # e.g. an ImageJ hyperstack over 4 GB: one page describes them all.
            mapped = tifffile.memmap(path, series=0, mode="r").reshape(
                -1, height, width
            )
            read = lambda idx: np.asarray(mapped[idx])  # noqa: E731
        else:
            whole = np.asarray(series.asarray()).reshape(-1, height, width)
            read = lambda idx: whole[idx]  # noqa: E731
        for start in range(0, layout.n_frames, max(1, int(chunk_frames))):
            yield read(planes[start : start + chunk_frames])


def sliding_baseline(
    traces: npt.NDArray[Any],
    fs: float,
    *,
    method: str = "maximin",
    win_baseline_s: float = 60.0,
    sig_baseline_frames: float = 10.0,
    prctile_baseline: float = 8.0,
) -> npt.NDArray[np.float64]:
    """Per-frame baseline of each row, as Suite2p's ``preprocess``: "maximin" (a
    Gaussian-smoothed trace's running minimum, then running maximum, over
    ``win_baseline_s``), "constant" (the smoothed trace's minimum) or
    "constant_prctile" (the ``prctile_baseline`` percentile)."""
    from scipy import ndimage

    traces = np.asarray(traces, dtype=np.float64)
    if method == "constant_prctile":
        level = np.percentile(traces, prctile_baseline, axis=1, keepdims=True)
        return np.broadcast_to(level, traces.shape).copy()
    smooth = ndimage.gaussian_filter1d(traces, sig_baseline_frames, axis=1)
    if method == "constant":
        return np.broadcast_to(smooth.min(axis=1, keepdims=True), traces.shape).copy()
    if method != "maximin":
        raise ValueError(f"Unknown baseline method '{method}'.")
    window = max(1, int(round(win_baseline_s * fs)))
    low = ndimage.minimum_filter1d(smooth, window, axis=1)
    return ndimage.maximum_filter1d(low, window, axis=1)


def roi_delta_f_over_f(
    F: npt.NDArray[Any],
    Fneu: npt.NDArray[Any] | None,
    fs: float,
    *,
    neuropil_coefficient: float = 0.7,
    **baseline: Any,
) -> npt.NDArray[np.float32]:
    """ΔF/F of ROI traces: ``(Fc - B(Fc)) / |B(F)|`` with ``Fc = F - c·Fneu``
    (``F`` itself without ``Fneu``) and ``B`` :func:`sliding_baseline`.

    Dividing by the raw F's baseline keeps a dim cell, whose neuropil-corrected
    baseline can be near zero or negative, from blowing up or flipping sign.
    """
    raw = np.asarray(F, dtype=np.float64)
    corrected = (
        raw
        if Fneu is None
        else raw - float(neuropil_coefficient) * np.asarray(Fneu, dtype=np.float64)
    )
    divisor = np.maximum(np.abs(sliding_baseline(raw, fs, **baseline)), 1e-9)
    return ((corrected - sliding_baseline(corrected, fs, **baseline)) / divisor).astype(
        np.float32
    )
