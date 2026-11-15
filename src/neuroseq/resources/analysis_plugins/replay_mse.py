"""Replay MSE — measure how well visible neurons replay a learned cycle.

Use this on a TraceData slice that starts on the **last training cycle**
(used as the reference) and extends through one or more ring-out cycles.
The plugin splits the selection into cycles of the configured length and
reports, for each cycle, the mean squared error and the Pearson
correlation coefficient against cycle 0 (the reference), following the
two performance measures defined in Kriener et al. (2024), Sec. 5.6,
Eq. 22–26.

Because a free-running ELiSe network's replay drifts in time relative to
the trained pattern, performance is evaluated at the time-shift
``s ∈ [0, T]`` (one pattern length) that minimises MSE for each cycle
(Eq. 25–26). The shift search uses an FFT-based cross-correlation. The
last cycle of the selection cannot be evaluated because it has no shift
slack — see the paper for details.
"""

from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.elise_support.replay_metrics import compute_cycle_mse
from neuroseq.elise_support.replay_metrics import compute_cycle_performance

__all__ = ["compute_cycle_mse", "compute_cycle_performance"]

PLUGIN_ID = "replay_mse"
PLUGIN_NAME = "Replay MSE (cycle vs reference)"
PLUGIN_ICON = "mdi6.delta"

_CONTROLS_ATTR = "_replay_mse_controls"
_DEFAULT_CYCLE_MS = 1920.0  # Für Elise intro phrase length.


def _make_controls(widget) -> QtWidgets.QWidget:
    bar = QtWidgets.QWidget(widget)
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(4, 2, 4, 2)

    layout.addWidget(QtWidgets.QLabel("Cycle length (ms):"))
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(1.0, 1_000_000.0)
    spin.setDecimals(1)
    spin.setSingleStep(10.0)
    spin.setValue(_DEFAULT_CYCLE_MS)
    spin.setFixedSize(110, spin.sizeHint().height())
    spin.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Fixed,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    layout.addWidget(spin)

    btn = QtWidgets.QPushButton("Update")
    btn.setFixedSize(btn.sizeHint())
    btn.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Fixed,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    layout.addWidget(btn)
    layout.addStretch()

    bar._spin_cycle = spin
    bar._update_btn = btn
    return bar


def _inject_controls(widget, on_update) -> QtWidgets.QWidget:
    # The widget removes the bar again before the next analysis renders.
    bar = _make_controls(widget)
    widget.add_plugin_controls(bar)
    bar._update_btn.clicked.connect(lambda: on_update(bar._spin_cycle.value()))
    setattr(widget, _CONTROLS_ATTR, bar)
    return bar


def _show_message(widget, message: str) -> None:
    ax = widget.subplot
    for other_ax in list(ax.figure.axes):
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


def _replot(widget, data, cycle_ms: float = _DEFAULT_CYCLE_MS) -> None:
    if not isinstance(data, TraceData):
        _show_message(widget, "Replay MSE only supports trace data.")
        return

    traces = np.asarray(widget.traces) if widget.traces is not None else None
    if traces is None or traces.size == 0:
        _show_message(widget, "No trace samples in selected region.")
        return

    dt_ms = float(widget.dt_ms)
    if dt_ms <= 0:
        _show_message(
            widget, "Trace dt is non-positive — cannot determine cycle length."
        )
        return

    n_cycle_samples = int(round(cycle_ms / dt_ms))
    n_total = int(traces.shape[1])
    if n_cycle_samples < 2 or n_total < 2 * n_cycle_samples:
        _show_message(
            widget,
            f"Need at least 2 full cycles in the selection "
            f"(≥ {max(4, 2 * n_cycle_samples)} samples; got {n_total}). "
            "Widen the time selection or shrink the cycle length.",
        )
        return

    mses, corrs, _ = compute_cycle_performance(traces, n_cycle_samples)
    n_cycles = int(mses.size)
    total_mse = float(np.mean(mses[1:])) if n_cycles > 1 else 0.0
    total_corr = float(np.nanmean(corrs[1:])) if n_cycles > 1 else 1.0

    ax = widget.subplot
    fig = ax.figure
    for other_ax in list(fig.axes):
        if other_ax is not ax:
            other_ax.remove()
    ax.clear()

    bars_x = np.arange(n_cycles)
    ax.bar(bars_x, mses, color="tab:purple", label="MSE")
    ax.set_xlabel("Cycle index (0 = reference)")
    ax.set_ylabel(f"MSE [({data.units})²]" if data.units else "MSE")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_xticks(bars_x)

    ax_corr = ax.twinx()
    ax_corr.plot(
        bars_x,
        corrs,
        color="tab:orange",
        marker="o",
        linewidth=1.5,
        label="Corr. coeff.",
    )
    ax_corr.axhline(1.0, color="tab:orange", linestyle=":", linewidth=0.6, alpha=0.5)
    ax_corr.set_ylabel("Pearson correlation coefficient")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title(
        f"Replay performance vs cycle 0  ·  cycle = {cycle_ms:g} ms  ·  "
        f"mean MSE (1..{n_cycles - 1}) = {total_mse:.4g}  ·  "
        f"mean corr = {total_corr:.3g}"
    )
    widget._redraw()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    def _on_update(cycle_ms: float) -> None:
        _replot(widget, data, cycle_ms=cycle_ms)

    _inject_controls(widget, _on_update)
    _replot(widget, data, cycle_ms=_DEFAULT_CYCLE_MS)
