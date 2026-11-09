"""ELiSe target MSE — replay error against the binary teaching pattern.

Implements Kriener et al. (2024), Sec. 5.6 + Eq. 22–26, evaluated against
the paper's target ``y`` (Eq. 6 applied to the binary teaching pattern,
Sec. 5.3.3) rather than against an arbitrary cycle of the recording.

This plugin is meant to be triggered from the dedicated "Analyse ELiSe
replay" toolbar action — the action computes the target firing rate from
the project's opsin stim and attaches it to the analysis widget before
rendering. Triggering this plugin directly from the right-click menu (no
target attached) shows a helpful message instead of failing silently.
"""

from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.elise_support.plugin_keys import TARGET_ATTR
from neuroseq.elise_support.plugin_keys import TARGET_DT_ATTR
from neuroseq.elise_support.plugin_keys import TARGET_MSE_PLUGIN_ID
from neuroseq.elise_support.replay_metrics import compute_performance_against_target

PLUGIN_ID = TARGET_MSE_PLUGIN_ID
PLUGIN_NAME = "ELiSe target MSE (vs teaching pattern)"
PLUGIN_ICON = "mdi6.chart-bell-curve-cumulative"

_CONTROLS_ATTR = "_elise_target_mse_controls"


def _make_controls(widget) -> QtWidgets.QWidget:
    bar = QtWidgets.QWidget(widget)
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(4, 2, 4, 2)
    btn = QtWidgets.QPushButton("Refresh")
    btn.setFixedSize(btn.sizeHint())
    btn.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Fixed,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    layout.addWidget(btn)
    layout.addStretch()
    bar._refresh_btn = btn
    return bar


def _inject_controls(widget, on_refresh) -> QtWidgets.QWidget:
    # The widget removes the bar again before the next analysis renders.
    bar = _make_controls(widget)
    widget.add_plugin_controls(bar)
    bar._refresh_btn.clicked.connect(on_refresh)
    setattr(widget, _CONTROLS_ATTR, bar)
    return bar


def _show_message(widget, message: str) -> None:
    ax = widget.subplot
    fig = ax.figure
    for other_ax in list(fig.axes):
        if other_ax is not ax:
            other_ax.remove()
    ax.clear()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
    )
    widget._redraw()


def _plot(widget, data) -> None:
    if not isinstance(data, TraceData):
        _show_message(widget, "ELiSe target MSE only supports trace data.")
        return

    target = getattr(widget, TARGET_ATTR, None)
    if target is None:
        _show_message(
            widget,
            "No teaching pattern attached. Trigger this analysis from "
            "the 'Analyse ELiSe replay' toolbar button so the target "
            "is built automatically from the project's opsin stim.",
        )
        return

    traces = np.asarray(widget.traces) if widget.traces is not None else None
    if traces is None or traces.size == 0:
        _show_message(widget, "No trace samples in selected region.")
        return

    target_arr = np.asarray(target, dtype=np.float64)
    if target_arr.ndim != 2:
        _show_message(widget, "Target pattern must be 2-D (n_output, T).")
        return
    if target_arr.shape[0] != traces.shape[0]:
        _show_message(
            widget,
            f"Target / trace shape mismatch: "
            f"target has {target_arr.shape[0]} output neurons but "
            f"selection covers {traces.shape[0]} neurons. The toolbar "
            "action restricts to output neurons automatically; if you "
            "got here some other way, narrow the neuron selection to "
            "the first num_vis neurons.",
        )
        return

    T = int(target_arr.shape[1])
    if traces.shape[1] < 2 * T:
        _show_message(
            widget,
            f"Need at least 2 full cycles in the selection "
            f"(≥ {2 * T} samples; got {traces.shape[1]}).",
        )
        return

    target_dt_s = float(getattr(widget, TARGET_DT_ATTR, 0.0))
    trace_dt_s = float(widget.dt_ms) / 1000.0
    if target_dt_s > 0 and trace_dt_s > 0 and abs(target_dt_s - trace_dt_s) > 1e-9:
        _show_message(
            widget,
            f"Target dt ({target_dt_s * 1000:.4g} ms) differs from "
            f"trace dt ({trace_dt_s * 1000:.4g} ms). The toolbar action "
            "resamples them to match — refresh through the button.",
        )
        return

    mses, corrs, _shifts = compute_performance_against_target(traces, target_arr)
    n_cycles = int(mses.size)
    mean_mse = float(np.mean(mses)) if n_cycles else 0.0
    mean_corr = float(np.nanmean(corrs)) if n_cycles else 1.0
    cycle_ms = T * (trace_dt_s * 1000.0)

    ax = widget.subplot
    fig = ax.figure
    for other_ax in list(fig.axes):
        if other_ax is not ax:
            other_ax.remove()
    ax.clear()

    bars_x = np.arange(n_cycles)
    ax.bar(bars_x, mses, width=1.0, color="tab:purple", label="MSE")
    ax.set_xlabel("Replay cycle index")
    ax.set_ylabel("MSE [rate²]")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    # Adaptive tick stride: aim for at most ~20 labelled ticks so the
    # axis is readable for runs with many hundreds of cycles.
    tick_step = max(1, int(np.ceil(n_cycles / 20)))
    ax.set_xticks(bars_x[::tick_step])
    ax.set_xlim(-0.5, n_cycles - 0.5)

    ax_corr = ax.twinx()
    ax_corr.plot(
        bars_x,
        corrs,
        color="tab:orange",
        marker="o" if n_cycles <= 50 else None,
        linewidth=1.5,
        label="Corr. coeff.",
    )
    ax_corr.axhline(1.0, color="tab:orange", linestyle=":", linewidth=0.6, alpha=0.5)
    ax_corr.set_ylabel("Pearson correlation coefficient")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title(
        f"ELiSe replay vs target — cycle {cycle_ms:g} ms\n"
        f"{n_cycles} cycles · mean MSE {mean_mse:.4g} · mean corr {mean_corr:.3g}",
        fontsize=10,
    )
    widget._redraw()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    _inject_controls(widget, lambda: _plot(widget, data))
    _plot(widget, data)
