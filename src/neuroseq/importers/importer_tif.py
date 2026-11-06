"""Suite2p + Cellpose pipeline for raw 2-photon TIFF stacks.

Defaults target jGCaMP8m at 30 Hz with Cellpose-derived ROIs (no Suite2p detection).
The pipeline writes Suite2p's binary cache into a caller-supplied work dir (a
tempdir, by design) and returns arrays in memory — nothing is left on disk.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt

if TYPE_CHECKING:
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.stimulation.stim import StimArray

logger = logging.getLogger(__name__)


# Defaults taken from /mnt/live3/svintila/random/mannheim_data/run_suite2p_FOV6.py
# (jGCaMP8m, 30 Hz, 2-channel Chrimson stimulation recordings).
DEFAULT_FS: float = 30.0
DEFAULT_TAU: float = 0.5
DEFAULT_NEUROPIL_COEFFICIENT: float = 0.7
DEFAULT_BATCH_SIZE: int = 200
DEFAULT_BASELINE: str = "maximin"
DEFAULT_WIN_BASELINE: float = 30.0
DEFAULT_SIG_BASELINE: float = 10.0
DEFAULT_PRCTILE_BASELINE: float = 8.0


@dataclass
class TifPipelineParams:
    fs: float = DEFAULT_FS
    tau: float = DEFAULT_TAU
    nchannels: int = 2
    functional_chan: int = 1  # 1-indexed, matching Suite2p convention
    neuropil_coefficient: float = DEFAULT_NEUROPIL_COEFFICIENT
    diameter: float = 12.0
    batch_size: int = DEFAULT_BATCH_SIZE
    baseline: str = DEFAULT_BASELINE
    win_baseline: float = DEFAULT_WIN_BASELINE
    sig_baseline: float = DEFAULT_SIG_BASELINE
    prctile_baseline: float = DEFAULT_PRCTILE_BASELINE


@dataclass
class TifPipelineResult:
    F: npt.NDArray[Any]
    Fneu: npt.NDArray[Any]
    spks: npt.NDArray[Any]
    stat: npt.NDArray[Any]
    iscell: npt.NDArray[Any]
    ops: dict[str, Any]
    mean_image: npt.NDArray[Any]
    masks: npt.NDArray[Any]
    extra: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[str, float], None]


class PipelineCancelled(Exception):
    """Raised from a progress callback to abort a running pipeline cleanly."""


def cellpose_result_to_masks(result: CellposeImportResult) -> npt.NDArray[Any]:
    """Convert stored masks back to Cellpose's raw label layout.

    Masks share the TIFFs' pixel coordinates; only Cellpose's 1-based labels
    and zero-valued background need restoring.
    """
    if result.segmentation_array is None:
        raise ValueError("Cellpose segmentation is missing; re-run Cellpose import.")
    seg = np.asarray(result.segmentation_array, dtype=np.int32)
    masks = seg.copy()
    # In CellposeImportResult: background = -1, cells = 0..N-1
    # Suite2p / Cellpose raw layout: background = 0, cells = 1..N
    masks[masks >= 0] += 1
    masks[masks < 0] = 0
    return masks


def _noop_progress(_message: str, _fraction: float) -> None:
    return None


def run_roi_average_pipeline(
    tif_paths: list[Path],
    masks: npt.NDArray[Any],
    params: TifPipelineParams,
    *,
    progress: ProgressCallback | None = None,
    stim_array: StimArray | None = None,
) -> TifPipelineResult:
    """Average ROI pixels from the selected channel across ordered TIFFs.

    Both ``(time, channel, y, x)`` stacks and ``(frame, y, x)`` stacks with
    interleaved channels are supported. Channel numbering is 1-based, matching
    the Suite2p acquisition convention used by the dialog.
    """
    if not tif_paths:
        raise ValueError("tif_paths must contain at least one TIFF file")
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError(f"masks must be 2D, got shape {masks.shape}")
    if params.nchannels < 1:
        raise ValueError("nchannels must be at least 1")
    if not 1 <= params.functional_chan <= params.nchannels:
        raise ValueError("functional_chan must be between 1 and nchannels (inclusive)")

    import tifffile

    progress = progress or _noop_progress
    labels = np.unique(masks)
    labels = labels[labels != 0]
    if labels.size == 0:
        raise ValueError("masks must contain at least one labelled ROI")

    trace_parts: list[npt.NDArray[Any]] = []
    first_stack: npt.NDArray[Any] | None = None
    for index, path in enumerate(tif_paths):
        progress(f"Reading {path.name}", index / len(tif_paths))
        stack = np.asarray(tifffile.imread(path))
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        if stack.ndim not in (3, 4):
            raise ValueError(
                f"TIFF {path} must have shape (frame, y, x) or "
                f"(time, channel, y, x), got {stack.shape}"
            )
        if stack.shape[-2:] != masks.shape:
            raise ValueError(
                f"masks shape {masks.shape} does not match TIFF frame shape "
                f"{stack.shape[-2:]}"
            )
        if stack.ndim == 4:
            file_channels = stack.shape[1]
            if file_channels != params.nchannels:
                raise ValueError(
                    f"TIFF {path} contains {file_channels} channel(s), but the "
                    f"pipeline is configured for {params.nchannels}"
                )
            stack = stack[:, params.functional_chan - 1, :, :]
        else:
            if stack.shape[0] % params.nchannels != 0:
                raise ValueError(
                    f"TIFF {path} has {stack.shape[0]} frame(s), which is not divisible by "
                    f"nchannels={params.nchannels} for interleaved acquisition"
                )
            stack = stack[params.functional_chan - 1 :: params.nchannels]
        if stack.shape[0] == 0:
            raise ValueError(
                f"TIFF {path} contains no frames for functional channel "
                f"{params.functional_chan}"
            )
        if first_stack is None:
            first_stack = stack
        rows = []
        for label_index, label in enumerate(labels):
            # Reported per ROI so a cancel request is honoured promptly even
            # for large stacks.
            progress(
                f"Averaging ROIs in {path.name}",
                (index + label_index / len(labels)) / len(tif_paths),
            )
            rows.append(stack[:, masks == label].mean(axis=1))
        trace_parts.append(np.vstack(rows).astype(np.float32))

    F = np.concatenate(trace_parts, axis=1)
    contaminated_frames = np.array([], dtype=np.int64)
    if stim_array is not None:
        contaminated_frames = stimulation_contaminated_frames(
            stim_array, params.fs, F.shape[1]
        )
        impute_contaminated_frames(F, contaminated_frames)
    assert first_stack is not None
    progress("Done", 1.0)
    return TifPipelineResult(
        F=F,
        Fneu=np.zeros_like(F),
        spks=np.zeros_like(F),
        stat=np.array([], dtype=object),
        iscell=np.ones((len(labels), 2), dtype=np.float32),
        ops={},
        mean_image=first_stack.mean(axis=0).astype(np.float32),
        masks=masks,
        extra={
            "pipeline": "roi_average",
            "roi_labels": np.asarray(labels),
            "nchannels": params.nchannels,
            "functional_chan": params.functional_chan,
            "stimulation_artefact_suppression": stim_array is not None,
            "contaminated_frames": contaminated_frames,
        },
    )


def stimulation_contaminated_frames(
    stim_array: StimArray, fs: float, n_frames: int
) -> npt.NDArray[np.int64]:
    """Return camera frames whose exposure overlaps non-zero stimulation."""
    if fs <= 0:
        raise ValueError("fs must be greater than zero")
    if stim_array.dt <= 0:
        raise ValueError("stim_array.dt must be greater than zero")
    values = np.asarray(stim_array.values)
    if values.ndim != 2:
        raise ValueError("stim_array.values must be a 2D array")
    active_samples = np.flatnonzero(np.any(values != 0, axis=1))
    contaminated = np.zeros(n_frames, dtype=bool)
    for sample in active_samples:
        start = float(sample) * float(stim_array.dt)
        end = float(sample + 1) * float(stim_array.dt)
        first_frame = max(0, int(np.floor(start * fs)))
        # nextafter prevents an interval ending exactly on a frame boundary
        # from contaminating the following frame.
        last_frame = min(n_frames - 1, int(np.floor(np.nextafter(end * fs, -np.inf))))
        if first_frame <= last_frame:
            contaminated[first_frame : last_frame + 1] = True
    return np.flatnonzero(contaminated).astype(np.int64)


def impute_contaminated_frames(
    traces: npt.NDArray[Any], contaminated_frames: npt.NDArray[np.int64]
) -> None:
    """Replace each contaminated run with its neighboring clean-value mean.

    Runs at the beginning or end of a recording use their single available
    clean neighbor. A fully contaminated trace cannot be inferred and is left
    unchanged.
    """
    if contaminated_frames.size == 0:
        return
    run_starts = np.r_[0, np.flatnonzero(np.diff(contaminated_frames) > 1) + 1]
    run_ends = np.r_[run_starts[1:] - 1, contaminated_frames.size - 1]
    for start_index, end_index in zip(run_starts, run_ends, strict=True):
        first = int(contaminated_frames[start_index])
        last = int(contaminated_frames[end_index])
        neighbors: list[npt.NDArray[Any]] = []
        if first > 0:
            neighbors.append(traces[:, first - 1])
        if last + 1 < traces.shape[1]:
            neighbors.append(traces[:, last + 1])
        if neighbors:
            traces[:, first : last + 1] = np.mean(neighbors, axis=0)[:, np.newaxis]


def select_torch_device() -> tuple[Any, str]:
    """Pick the best available torch device and a short human label.

    Preference order: CUDA GPU > Apple-Silicon MPS GPU > CPU. Set the
    ``NEUROSEQ_FORCE_CPU`` environment variable to force CPU (an escape hatch if
    a GPU backend misbehaves). Returns ``(torch.device, label)``.
    """
    import torch

    if os.environ.get("NEUROSEQ_FORCE_CPU"):
        return torch.device("cpu"), "CPU (forced)"
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU (CUDA)"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return torch.device("mps"), "GPU (Apple MPS)"
    return torch.device("cpu"), "CPU"


def _run_with_cpu_fallback(
    fn: Callable[[Any], Any], device: Any, device_label: str
) -> tuple[Any, Any, str]:
    """Run ``fn(device)``; on a GPU runtime error, retry once on CPU.

    Returns ``(result, device, device_label)`` reflecting the device actually
    used, so a GPU that is present but unusable (e.g. an op missing on MPS)
    degrades gracefully to CPU instead of failing the whole import.
    """
    try:
        return fn(device), device, device_label
    except Exception as exc:
        if device.type == "cpu":
            raise
        # Only reached when a GPU attempt failed — torch is guaranteed importable
        # here (the caller already used it to build the GPU device).
        import torch

        logger.warning(
            "GPU compute on %s failed (%s: %s); falling back to CPU.",
            device_label,
            type(exc).__name__,
            exc,
        )
        cpu = torch.device("cpu")
        return fn(cpu), cpu, "CPU (GPU fallback)"


def _masks_to_stat(masks: npt.NDArray[Any], Ly: int, Lx: int) -> npt.NDArray[Any]:
    """Convert a Cellpose label image into a Suite2p `stat` array."""
    from suite2p.detection import assign_overlaps
    from suite2p.detection import roi_stats

    unique_ids = np.unique(masks)
    # masks=0 is background; iterate the rest in order
    cell_ids = unique_ids[unique_ids != 0]
    stat: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        ypix, xpix = np.nonzero(masks == cell_id)
        npix = int(len(xpix))
        stat.append(
            {
                "ypix": ypix,
                "xpix": xpix,
                "npix": npix,
                "lam": np.ones(npix, np.float32),
                "med": [float(np.mean(ypix)), float(np.mean(xpix))],
            }
        )
    stat_arr = np.array(stat)
    stat_arr = roi_stats(stat_arr, Ly, Lx)
    stat_arr = assign_overlaps(stat_arr, Ly, Lx)
    return stat_arr


def _stat_to_masks(stat: npt.NDArray[Any], Ly: int, Lx: int) -> npt.NDArray[Any]:
    """Rebuild a Cellpose-style label image from a Suite2p `stat` array."""
    mask_img = np.zeros((Ly, Lx), dtype=np.int32)
    for idx, entry in enumerate(stat):
        mask_img[entry["ypix"], entry["xpix"]] = idx + 1
    return mask_img


def run_suite2p_pipeline(
    tif_paths: list[Path],
    masks: npt.NDArray[Any] | None,
    params: TifPipelineParams,
    *,
    work_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    register: bool = False,
    reference_image: npt.NDArray[Any] | None = None,
) -> TifPipelineResult:
    """Run TIFF → Suite2p binary → trace extraction → OASIS deconvolution.

    With ``register``, the binary is first motion-corrected in place by Suite2p's
    rigid + non-rigid registration, aligned to ``reference_image`` (e.g. the image
    the masks were drawn on) when it has the movie's shape, else to a reference
    Suite2p computes from the movie. ``extra["registration"]`` then reports which
    reference was used and the largest rigid shift, in pixels.

    `masks` is an optional 2D integer label image (Cellpose output,
    0 = background). If `masks` is None, ROIs are detected automatically using
    Suite2p's own `sparsery` algorithm; `params.diameter` sets the expected
    cell size in pixels. A temporary directory is created under `work_dir`
    (or system tempdir) for Suite2p's binary cache and is removed before
    returning.
    """
    if not tif_paths:
        raise ValueError("tif_paths must contain at least one TIFF file")
    if masks is not None and masks.ndim != 2:
        raise ValueError(f"masks must be 2D, got shape {masks.shape}")

    import suite2p
    import suite2p.classification as s2cls
    import suite2p.detection as s2detect
    import suite2p.extraction as s2ext
    import suite2p.io as s2io

    progress = progress or _noop_progress
    progress("Preparing work directory", 0.0)

    # Deduplicate parent directories while preserving order — Suite2p scans
    # each directory in data_path for matching TIFFs.
    seen: set[str] = set()
    data_paths: list[str] = []
    for p in tif_paths:
        parent = str(p.parent)
        if parent not in seen:
            seen.add(parent)
            data_paths.append(parent)

    tmp_root = Path(tempfile.mkdtemp(prefix="neuroseq_suite2p_", dir=work_dir))
    try:
        db = {
            **suite2p.default_db(),
            "data_path": data_paths,
            "save_path0": str(tmp_root),
            "fast_disk": str(tmp_root),
            "nchannels": int(params.nchannels),
            "functional_chan": int(params.functional_chan),
            "look_one_level_down": False,
        }
        # get_file_list returns the file list plus a numpy `first_files` mask
        # that downstream Suite2p code calls .sum() on — building those by hand
        # breaks tiff_to_binary.
        file_list, first_files = s2io.get_file_list(db)
        # Suite2p scans the full directory; restrict to only the user-selected
        # files so that other TIFFs in the same folder (e.g. different-FOV
        # acquisitions with mismatched dimensions) are not included.
        selected = {str(p.resolve()) for p in tif_paths}
        keep = np.array(
            [str(Path(f).resolve()) in selected for f in file_list], dtype=bool
        )
        if keep.any():
            file_list = [f for f, k in zip(file_list, keep, strict=True) if k]
            first_files = first_files[keep]
            # Ensure the first kept file in each folder is still marked True so
            # frames_per_folder is sized correctly inside tiff_to_binary.
            seen_folders: set[str] = set()
            for i, f in enumerate(file_list):
                folder = str(Path(f).parent)
                if folder not in seen_folders:
                    first_files[i] = True
                    seen_folders.add(folder)
        db["file_list"] = file_list
        db["first_files"] = first_files

        defaults = suite2p.default_settings()
        settings = {
            **defaults,
            "fs": float(params.fs),
            "tau": float(params.tau),
            "extraction": {
                **defaults["extraction"],
                "neuropil_coefficient": float(params.neuropil_coefficient),
                "batch_size": int(params.batch_size),
            },
            "dcnv_preprocess": {
                "baseline": params.baseline,
                "win_baseline": float(params.win_baseline),
                "sig_baseline": float(params.sig_baseline),
                "prctile_baseline": float(params.prctile_baseline),
            },
        }

        device, device_label = select_torch_device()
        logger.info("Suite2p pipeline using %s (device=%s)", device_label, device)
        progress(f"Running suite2p on {device_label}", 0.05)
        if getattr(device, "type", None) == "cpu":
            from neuroseq.gpu import torch_gpu_status

            _, gpu_msg = torch_gpu_status()
            logger.warning("Suite2p on CPU. %s", gpu_msg.replace("\n", " "))

        progress("Initialising Suite2p output dirs", 0.05)
        dbs = s2io.init_dbs(db)
        plane_db = dbs[0]

        progress("Converting TIFF to binary", 0.1)
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(plane_db["reg_file"], "wb"))]
            files_chan2 = []
            if params.nchannels >= 2:
                files_chan2 = [
                    stack.enter_context(open(plane_db["reg_file_chan2"], "wb"))
                ]
            dbs = s2io.tiff_to_binary(dbs, settings, files, files_chan2)

        plane_db = dbs[0]
        Ly, Lx, nframes = (
            int(plane_db["Ly"]),
            int(plane_db["Lx"]),
            int(plane_db["nframes"]),
        )
        logger.info("Suite2p binary written: %d frames %dx%d", nframes, Ly, Lx)

        if masks is not None and masks.shape != (Ly, Lx):
            raise ValueError(
                f"masks shape {masks.shape} does not match TIFF shape ({Ly}, {Lx})"
            )

        # Registration motion-corrects the binary in place, so it must be writable.
        writable: dict[str, Any] = (
            {"n_frames": nframes, "write": True} if register else {}
        )
        f_bin = s2io.BinaryFile(Ly=Ly, Lx=Lx, filename=plane_db["reg_file"], **writable)
        f_bin_chan2 = None
        if params.nchannels >= 2:
            f_bin_chan2 = s2io.BinaryFile(
                Ly=Ly, Lx=Lx, filename=plane_db["reg_file_chan2"], **writable
            )

        registration: dict[str, Any] | None = None
        if register:
            import suite2p.registration as s2reg

            ref = None
            if reference_image is not None:
                if np.shape(reference_image) == (Ly, Lx):
                    ref = np.clip(
                        np.rint(np.asarray(reference_image, dtype=np.float64)),
                        np.iinfo(np.int16).min,
                        np.iinfo(np.int16).max,
                    ).astype(np.int16)
                else:
                    logger.warning(
                        "Reference image %s does not match the movie (%d, %d); "
                        "registering to the movie's own reference",
                        np.shape(reference_image),
                        Ly,
                        Lx,
                    )
            reg_settings = dict(settings["registration"])
            # Non-rigid registration needs more than one block across the frame
            # (Suite2p fails on a single one); small frames are registered rigidly.
            if min(Ly, Lx) <= max(reg_settings.get("block_size", (128, 128))):
                reg_settings["nonrigid"] = False
            progress(f"Motion-correcting with Suite2p ({device_label})", 0.2)
            reg_outputs, device, device_label = _run_with_cpu_fallback(
                lambda dev: s2reg.registration_wrapper(
                    f_bin,
                    f_reg_chan2=f_bin_chan2,
                    refImg=ref,
                    settings=reg_settings,
                    device=dev,
                ),
                device,
                device_label,
            )
            shifts = np.hypot(
                np.asarray(reg_outputs["yoff"], dtype=np.float64),
                np.asarray(reg_outputs["xoff"], dtype=np.float64),
            )
            registration = {
                "reference": "provided" if ref is not None else "movie",
                "max_shift_px": float(shifts.max()) if shifts.size else 0.0,
            }

        progress("Computing mean projection", 0.3)
        mean_image = f_bin.data.mean(axis=0).astype(np.float32)

        if masks is not None:
            progress("Building Suite2p stat from Cellpose masks", 0.4)
            stat = _masks_to_stat(masks, Ly, Lx)
            masks_out = np.asarray(masks)
        else:
            progress(f"Detecting ROIs with Suite2p ({device_label})", 0.4)
            _new_settings, stat, _redcell = s2detect.detection_wrapper(
                f_bin,
                diameter=[float(params.diameter), float(params.diameter)],
                tau=float(params.tau),
                fs=float(params.fs),
                yrange=[0, Ly],
                xrange=[0, Lx],
                settings=defaults["detection"],
                classifier_path=s2cls.builtin_classfile,
                device=device,
            )
            masks_out = _stat_to_masks(stat, Ly, Lx)

        progress(f"Extracting fluorescence traces ({device_label})", 0.5)
        (F, Fneu, F_chan2, Fneu_chan2), device, device_label = _run_with_cpu_fallback(
            lambda dev: s2ext.extraction_wrapper(
                stat,
                f_bin,
                f_reg_chan2=f_bin_chan2,
                settings=settings["extraction"],
                device=dev,
            ),
            device,
            device_label,
        )

        progress("Computing per-cell SNR/skew/std", 0.7)
        try:
            from scipy.stats import skew as scipy_skew
        except ImportError:  # scipy is a hard dep; defensive only
            scipy_skew = None

        dF = F - settings["extraction"]["neuropil_coefficient"] * Fneu
        diff_var = np.diff(dF, axis=1).var(axis=1)
        dF_var = dF.var(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 1 - 0.5 * np.where(dF_var > 0, diff_var / dF_var, 0.0)
        sk = scipy_skew(dF, axis=1) if scipy_skew is not None else np.zeros(len(stat))
        sd = dF.std(axis=1)
        for k, s in enumerate(stat):
            s["snr"] = float(snr[k])
            s["skew"] = float(sk[k])
            s["std"] = float(sd[k])

        progress(f"Baseline correction + OASIS deconvolution ({device_label})", 0.8)
        dF_preproc, device, device_label = _run_with_cpu_fallback(
            lambda dev: s2ext.preprocess(
                F=dF.copy(),
                fs=settings["fs"],
                batch_size=settings["extraction"]["batch_size"],
                device=dev,
                **settings["dcnv_preprocess"],
            ),
            device,
            device_label,
        )
        spks = s2ext.oasis(
            F=dF_preproc,
            batch_size=settings["extraction"]["batch_size"],
            tau=settings["tau"],
            fs=settings["fs"],
        )

        progress("Classifying ROIs", 0.95)
        iscell = s2cls.classify(stat=stat, classfile=s2cls.builtin_classfile)
        if masks is not None:
            # All ROIs come from Cellpose — accept them all.
            iscell[:, 0] = 1

        extra: dict[str, Any] = {"compute_device": device_label}
        if registration is not None:
            extra["registration"] = registration
        if masks is None:
            extra["diameter"] = params.diameter
        if params.nchannels >= 2:
            extra["F_chan2"] = F_chan2
            extra["Fneu_chan2"] = Fneu_chan2

        progress("Done", 1.0)

        return TifPipelineResult(
            F=np.asarray(F),
            Fneu=np.asarray(Fneu),
            spks=np.asarray(spks),
            stat=stat,
            iscell=np.asarray(iscell),
            ops={**settings, **plane_db},
            mean_image=mean_image,
            masks=masks_out,
            extra=extra,
        )
    finally:
        # Suite2p caches can be tens of GB — never leave them behind.
        with contextlib.suppress(OSError):
            shutil.rmtree(tmp_root, ignore_errors=True)
        # Defensive: ensure tempdir is gone even on Windows-style file locks.
        if os.path.exists(tmp_root):
            logger.warning("Suite2p tempdir %s could not be removed", tmp_root)
