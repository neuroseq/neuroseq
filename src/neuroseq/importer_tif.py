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
from typing import Any

import brian2.numpy_ as np
import numpy.typing as npt

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


def _noop_progress(_message: str, _fraction: float) -> None:
    return None


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


def run_suite2p_pipeline(
    tif_paths: list[Path],
    masks: npt.NDArray[Any],
    params: TifPipelineParams,
    *,
    work_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> TifPipelineResult:
    """Run TIFF → Suite2p binary → trace extraction → OASIS deconvolution.

    `masks` is a 2D integer label image (Cellpose output, 0 = background).
    A temporary directory is created under `work_dir` (or system tempdir) for
    Suite2p's binary cache and is removed before returning.
    """
    if not tif_paths:
        raise ValueError("tif_paths must contain at least one TIFF file")
    if masks.ndim != 2:
        raise ValueError(f"masks must be 2D, got shape {masks.shape}")

    import suite2p
    import suite2p.classification as s2cls
    import suite2p.extraction as s2ext
    import suite2p.io as s2io
    import torch

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Suite2p pipeline using device=%s", device)

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

        progress("Computing mean projection", 0.3)
        if masks.shape != (Ly, Lx):
            raise ValueError(
                f"masks shape {masks.shape} does not match TIFF shape ({Ly}, {Lx})"
            )

        f_bin = s2io.BinaryFile(Ly=Ly, Lx=Lx, filename=plane_db["reg_file"])
        f_bin_chan2 = None
        if params.nchannels >= 2:
            f_bin_chan2 = s2io.BinaryFile(
                Ly=Ly, Lx=Lx, filename=plane_db["reg_file_chan2"]
            )

        mean_image = f_bin.data.mean(axis=0).astype(np.float32)

        progress("Building Suite2p stat from Cellpose masks", 0.4)
        stat = _masks_to_stat(masks, Ly, Lx)

        progress("Extracting fluorescence traces", 0.5)
        F, Fneu, F_chan2, Fneu_chan2 = s2ext.extraction_wrapper(
            stat,
            f_bin,
            f_reg_chan2=f_bin_chan2,
            settings=settings["extraction"],
            device=device,
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

        progress("Baseline correction + OASIS deconvolution", 0.8)
        dF_preproc = s2ext.preprocess(
            F=dF.copy(),
            fs=settings["fs"],
            batch_size=settings["extraction"]["batch_size"],
            device=device,
            **settings["dcnv_preprocess"],
        )
        spks = s2ext.oasis(
            F=dF_preproc,
            batch_size=settings["extraction"]["batch_size"],
            tau=settings["tau"],
            fs=settings["fs"],
        )

        progress("Classifying ROIs", 0.95)
        iscell = s2cls.classify(stat=stat, classfile=s2cls.builtin_classfile)
        # All ROIs come from Cellpose — accept them all.
        iscell[:, 0] = 1

        extra: dict[str, Any] = {}
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
            masks=np.asarray(masks),
            extra=extra,
        )
    finally:
        # Suite2p caches can be tens of GB — never leave them behind.
        with contextlib.suppress(OSError):
            shutil.rmtree(tmp_root, ignore_errors=True)
        # Defensive: ensure tempdir is gone even on Windows-style file locks.
        if os.path.exists(tmp_root):
            logger.warning("Suite2p tempdir %s could not be removed", tmp_root)
