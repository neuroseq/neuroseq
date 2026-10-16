"""Replay-quality metrics shared by analysis plugins.

Implements the two performance measures from Kriener et al. (2024),
Sec. 5.6, Eq. 22–26: per-cycle MSE and Pearson correlation between an
observed signal and a target pattern, with a time-shift search over one
full pattern length so cycle-by-cycle drift in the replay is corrected
for. The shift search uses an FFT-based cross-correlation.

Two entry points:

* :func:`compute_performance_against_target` — generic. Compares each
  cycle of ``traces`` to an externally supplied ``target`` of shape
  ``(n_signals, T)``. This is the paper-faithful version: ``target`` is
  the teaching pattern ``y``.
* :func:`compute_cycle_performance` — convenience wrapper that uses
  ``traces[:, :T]`` (cycle 0 of the recording) as the target. Useful
  when no separate teaching pattern is available.

The last whole cycle of the recording is always dropped because it has
no shift slack — see Eq. 25–26 in the paper.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import numpy as np


def compute_performance_against_target(
    traces: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cycle MSE and Pearson correlation of ``traces`` vs ``target``.

    For every cycle ``k = 0 .. n_cycles_full - 2`` of ``traces`` (the
    last cycle is dropped — no shift slack), find the alignment shift
    ``s ∈ [0, T]`` that minimises the across-signal mean MSE between the
    shifted segment ``traces[:, k·T + s : (k+1)·T + s]`` and ``target``.
    MSE and Pearson correlation are reported at that same shift.

    Parameters
    ----------
    traces : np.ndarray, shape ``(n_signals, n_samples)``
        Observed signals. ``n_signals`` must match ``target``.
    target : np.ndarray, shape ``(n_signals, T)``
        Reference pattern (one cycle). ``T`` is the cycle length.

    Returns
    -------
    mses, corrs, shifts : np.ndarray, each of shape ``(n_cycles - 1,)``
        ``mses[k]`` and ``corrs[k]`` are the across-signal averages at
        the best-fit shift ``shifts[k]`` (in samples).
    """
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (n_signals, n_samples)")
    if target.ndim != 2:
        raise ValueError("target must be 2-D (n_signals, T)")
    if traces.shape[0] != target.shape[0]:
        raise ValueError(
            f"traces and target must have the same number of signals; "
            f"got {traces.shape[0]} and {target.shape[0]}"
        )
    T = int(target.shape[1])
    if T < 2:
        raise ValueError("target cycle length must be at least 2 samples")
    n_signals, n_total = traces.shape
    n_cycles_full = n_total // T
    if n_cycles_full < 2:
        raise ValueError(
            "traces must contain at least two whole cycles "
            f"(needed {2 * T} samples; have {n_total})"
        )

    reference = target.astype(np.float64)
    ref_mean = reference.mean(axis=1)
    ref_centered = reference - ref_mean[:, None]
    ref_var = (ref_centered**2).mean(axis=1)
    ref_sq_sum = (reference**2).sum(axis=1)

    n_fft = 1
    while n_fft < 3 * T - 1:
        n_fft <<= 1
    ref_freq_conj = np.fft.rfft(reference, n=n_fft, axis=1).conj()

    n_eval = n_cycles_full - 1
    mses = np.zeros(n_eval, dtype=np.float64)
    corrs = np.zeros(n_eval, dtype=np.float64)
    shifts = np.zeros(n_eval, dtype=np.int64)

    for k in range(n_eval):
        window = traces[:, k * T : (k + 2) * T].astype(np.float64)

        win_freq = np.fft.rfft(window, n=n_fft, axis=1)
        cross_full = np.fft.irfft(win_freq * ref_freq_conj, n=n_fft, axis=1)
        cross = cross_full[:, : T + 1]

        cumsum_sq = np.cumsum(window**2, axis=1)
        seg_sq = np.empty((n_signals, T + 1))
        seg_sq[:, 0] = cumsum_sq[:, T - 1]
        seg_sq[:, 1:] = cumsum_sq[:, T : 2 * T] - cumsum_sq[:, :T]

        cumsum_lin = np.cumsum(window, axis=1)
        seg_sum = np.empty((n_signals, T + 1))
        seg_sum[:, 0] = cumsum_lin[:, T - 1]
        seg_sum[:, 1:] = cumsum_lin[:, T : 2 * T] - cumsum_lin[:, :T]

        mse_per_signal = (seg_sq - 2.0 * cross + ref_sq_sum[:, None]) / T
        np.maximum(mse_per_signal, 0.0, out=mse_per_signal)
        mse_avg = mse_per_signal.mean(axis=0)

        best_s = int(np.argmin(mse_avg))
        mses[k] = float(mse_avg[best_s])
        shifts[k] = best_s

        seg_mean = seg_sum[:, best_s] / T
        seg_var = seg_sq[:, best_s] / T - seg_mean**2
        np.maximum(seg_var, 0.0, out=seg_var)
        cov_seg_ref = cross[:, best_s] / T - seg_mean * ref_mean

        denom = np.sqrt(seg_var * ref_var)
        valid = denom > 1e-12
        per_signal_corr = np.where(
            valid, cov_seg_ref / np.where(valid, denom, 1.0), np.nan
        )
        if np.any(valid):
            corrs[k] = float(np.nanmean(per_signal_corr))
        else:
            corrs[k] = np.nan

    return mses, corrs, shifts


def compute_cycle_performance(
    traces: np.ndarray, n_cycle_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cycle MSE and Pearson correlation vs ``traces[:, :T]`` as target.

    Convenience wrapper around :func:`compute_performance_against_target`
    that uses cycle 0 of the recording as the reference. The first
    returned entry is therefore ``(0, 1)`` by construction.
    """
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (n_signals, n_samples)")
    if n_cycle_samples < 2:
        raise ValueError("n_cycle_samples must be at least 2")
    T = int(n_cycle_samples)
    if traces.shape[1] < 2 * T:
        raise ValueError(
            "selection must contain at least two whole cycles "
            f"(needed {2 * T} samples; have {traces.shape[1]})"
        )
    return compute_performance_against_target(traces, traces[:, :T])


def save_replay_metrics_csv(
    path: str | Path,
    mses: np.ndarray,
    corrs: np.ndarray,
    shifts: np.ndarray,
    *,
    cycle_dt_ms: float,
    cycle_length_samples: int,
    n_output_neurons: int,
    source: str | None = None,
    analysis: str = "ELiSe target MSE (Kriener et al. 2024, Eq. 22-26)",
) -> Path:
    """Write per-cycle replay metrics to a CSV file with a metadata header.

    Header lines start with ``#`` and carry the run context (source,
    timestamp, cycle length, number of output neurons, mean MSE and
    correlation). The data table has one row per evaluated cycle.
    """
    out_path = Path(path)
    cycle_ms = cycle_length_samples * cycle_dt_ms
    n_cycles = int(mses.size)
    mean_mse = float(np.mean(mses)) if n_cycles else float("nan")
    mean_corr = float(np.nanmean(corrs)) if n_cycles else float("nan")

    with out_path.open("w", newline="") as fh:
        fh.write(f"# Analysis: {analysis}\n")
        if source is not None:
            fh.write(f"# Source: {source}\n")
        fh.write(
            f"# Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        )
        fh.write(f"# Pattern cycle: {cycle_length_samples} samples ({cycle_ms:g} ms)\n")
        fh.write(f"# Trace dt: {cycle_dt_ms:g} ms\n")
        fh.write(f"# Output neurons: {n_output_neurons}\n")
        fh.write(f"# Mean MSE: {mean_mse:.6g}\n")
        fh.write(f"# Mean correlation: {mean_corr:.6g}\n")

        writer = csv.writer(fh)
        writer.writerow(
            [
                "cycle_index",
                "start_time_ms",
                "mse",
                "corr_coeff",
                "shift_samples",
                "shift_ms",
            ]
        )
        for k in range(n_cycles):
            writer.writerow(
                [
                    k,
                    f"{k * cycle_ms:.6g}",
                    f"{float(mses[k]):.8g}",
                    f"{float(corrs[k]):.8g}",
                    int(shifts[k]),
                    f"{int(shifts[k]) * cycle_dt_ms:.6g}",
                ]
            )
    return out_path


def save_learning_curves_csv(
    path: str | Path,
    curves: dict,
    *,
    source: str | None = None,
) -> Path:
    """Write per-epoch train + replay learning curves to a CSV.

    ``curves`` is the dict returned by
    :meth:`neuroseq.example_models.elise.Elise.run_protocol`. The output has a
    ``#``-commented header with the run config and one data row per epoch,
    tagged by phase (``training`` = teacher-off validation during training,
    ``replay`` = free replay after training).
    """
    out_path = Path(path)
    cfg = curves.get("config", {}) or {}
    val_mse = np.asarray(curves.get("val_mse", []), dtype=np.float64)
    val_corr = np.asarray(curves.get("val_corr", []), dtype=np.float64)
    rep_mse = np.asarray(curves.get("rep_mse", []), dtype=np.float64)
    rep_corr = np.asarray(curves.get("rep_corr", []), dtype=np.float64)

    with out_path.open("w", newline="") as fh:
        fh.write("# Analysis: ELiSe learning curves (Kriener et al. 2024, Fig. 2b/c)\n")
        if source is not None:
            fh.write(f"# Source: {source}\n")
        fh.write(
            f"# Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        )
        for key in (
            "teacher_mode",
            "training_epochs",
            "training_cycles",
            "validation_cycles",
            "replay_epochs",
            "replay_cycles",
            "replay_learning",
        ):
            if key in cfg:
                fh.write(f"# {key}: {cfg[key]}\n")
        if rep_mse.size:
            fh.write(f"# Mean replay MSE: {float(np.mean(rep_mse)):.6g}\n")
            fh.write(f"# Mean replay corr: {float(np.nanmean(rep_corr)):.6g}\n")

        writer = csv.writer(fh)
        writer.writerow(["phase", "epoch", "mse", "corr_coeff"])
        for k in range(val_mse.size):
            writer.writerow(
                ["training", k, f"{float(val_mse[k]):.8g}", f"{float(val_corr[k]):.8g}"]
            )
        for k in range(rep_mse.size):
            writer.writerow(
                ["replay", k, f"{float(rep_mse[k]):.8g}", f"{float(rep_corr[k]):.8g}"]
            )
    return out_path


def compute_cycle_mse(traces: np.ndarray, n_cycle_samples: int) -> np.ndarray:
    """Per-cycle MSE of ``traces`` against cycle 0, *without* shift correction.

    Kept for backward compatibility with the original
    ``replay_mse.compute_cycle_mse`` API: returns one entry per whole
    cycle in the selection (including the last), no time-shift search.
    """
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (n_neurons, n_timepoints)")
    if n_cycle_samples < 2:
        raise ValueError("n_cycle_samples must be at least 2")
    if traces.shape[1] < 2 * n_cycle_samples:
        raise ValueError(
            "selection must contain at least two whole cycles "
            f"(needed {2 * n_cycle_samples} samples; have {traces.shape[1]})"
        )

    n_cycles = traces.shape[1] // n_cycle_samples
    reference = traces[:, :n_cycle_samples]
    out = np.empty(n_cycles, dtype=np.float64)
    for k in range(n_cycles):
        segment = traces[:, k * n_cycle_samples : (k + 1) * n_cycle_samples]
        out[k] = float(np.mean((segment - reference) ** 2))
    return out
