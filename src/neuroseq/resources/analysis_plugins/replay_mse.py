"""Replay MSE — measure how well visible neurons replay a learned cycle.

Use this on a TraceData slice that starts on the **last training cycle**
(used as the reference) and extends through one or more ring-out cycles.
The plugin splits the selection into cycles of the configured length and
reports the mean squared error of every cycle against cycle 0 (the
reference). Cycle 0 is therefore always exactly zero by construction;
cycles 1..N show how well the free-running network reproduces the
trained pattern.
"""

from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.data import TraceData

PLUGIN_ID = "replay_mse"
PLUGIN_NAME = "Replay MSE (cycle vs reference)"
PLUGIN_ICON = "mdi6.delta"

_CONTROLS_ATTR = "_replay_mse_controls"
_DEFAULT_CYCLE_MS = 1920.0  # Für Elise intro phrase length.


def compute_cycle_mse(traces: np.ndarray, n_cycle_samples: int) -> np.ndarray:
    """Return per-cycle MSE of ``traces`` against the first cycle.

    Cycle 0 is the reference, so ``result[0] == 0`` by construction. The
    last partial cycle (if any) is dropped — cycles must be whole.
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
    layout = widget.layout()
    if hasattr(widget, _CONTROLS_ATTR):
        old_bar = getattr(widget, _CONTROLS_ATTR)
        if old_bar is not None:
            if layout is not None:
                layout.removeWidget(old_bar)
            old_bar.deleteLater()

    bar = _make_controls(widget)
    if layout is not None:
        layout.insertWidget(0, bar)
    bar._update_btn.clicked.connect(lambda: on_update(bar._spin_cycle.value()))
    setattr(widget, _CONTROLS_ATTR, bar)
    return bar


def _show_message(widget, message: str) -> None:
    ax = widget.subplot
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

    cycle_mses = compute_cycle_mse(traces, n_cycle_samples)
    n_cycles = int(cycle_mses.size)
    total_mse = float(np.mean(cycle_mses[1:])) if n_cycles > 1 else 0.0

    ax = widget.subplot
    ax.clear()
    bars_x = np.arange(n_cycles)
    ax.bar(bars_x, cycle_mses, color="tab:purple")
    ax.set_xlabel("Cycle index (0 = reference)")
    ax.set_ylabel(f"MSE [({data.units})²]" if data.units else "MSE")
    ax.set_title(
        f"Replay MSE vs cycle 0  ·  cycle = {cycle_ms:g} ms  ·  "
        f"mean MSE (cycles 1..{n_cycles - 1}) = {total_mse:.4g}"
    )
    ax.set_xticks(bars_x)
    widget._redraw()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    def _on_update(cycle_ms: float) -> None:
        _replot(widget, data, cycle_ms=cycle_ms)

    _inject_controls(widget, _on_update)
    _replot(widget, data, cycle_ms=_DEFAULT_CYCLE_MS)
