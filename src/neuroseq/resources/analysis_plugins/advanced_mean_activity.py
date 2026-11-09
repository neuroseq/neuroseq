from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData

PLUGIN_ID = "advanced_mean_activity"
PLUGIN_NAME = "Advanced: Mean activity"
PLUGIN_ICON = "mdi6.chart-timeline-variant-shimmer"

_CONTROLS_ATTR = "_advanced_mean_activity_controls"


def _make_controls(widget, data, neuron_slice, time_slice) -> QtWidgets.QWidget | None:
    """Build a controls bar for SpikeData only."""
    if not isinstance(data, SpikeData):
        return None

    bar = QtWidgets.QWidget(widget)
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(4, 2, 4, 2)

    spin = None
    if isinstance(data, SpikeData):
        layout.addWidget(QtWidgets.QLabel("Histogram bin size (ms):"))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.1, 10000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(1.0)
        spin.setFixedSize(90, spin.sizeHint().height())
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

    def _update():
        _replot(
            widget,
            data,
            neuron_slice,
            time_slice,
            dt_ms=spin.value() if spin is not None else 1.0,
        )

    btn.clicked.connect(_update)

    bar._spin_dt = spin
    return bar


def _inject_controls(widget, data, neuron_slice, time_slice) -> None:
    """Show the controls bar above the plot; the widget removes it again
    before the next analysis renders."""
    bar = _make_controls(widget, data, neuron_slice, time_slice)
    if bar is not None:
        widget.add_plugin_controls(bar)
    setattr(widget, _CONTROLS_ATTR, bar)


def _replot(widget, data, neuron_slice, time_slice, dt_ms: float = 1.0) -> None:
    ax = widget.subplot
    ax.clear()

    if isinstance(data, TraceData):
        if widget.traces is None or np.asarray(widget.traces).size == 0:
            ax.text(
                0.5,
                0.5,
                "No trace samples in selected region",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            trace_values = np.asarray(widget.traces)
            mean_trace = np.mean(trace_values, axis=0)
            times_ms = np.arange(mean_trace.shape[0], dtype=float) * widget.dt_ms
            if widget.time_slice is not None and widget.time_slice.start is not None:
                times_ms += float(widget.time_slice.start) * widget.dt_ms
            ax.plot(times_ms, mean_trace, color="tab:orange", linewidth=2.0)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(f"Mean {data.units}")

    elif isinstance(data, SpikeData):
        spike_times_ms = (
            np.asarray(widget.spike_times_ms)
            if widget.spike_times_ms is not None
            else np.asarray([], dtype=float)
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Spike count")
        if spike_times_ms.size == 0:
            ax.text(
                0.5,
                0.5,
                "No spikes in selected region",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            t_max = float(np.max(spike_times_ms))
            t_min = float(np.min(spike_times_ms))
            n_bins = max(1, int(np.ceil((t_max - t_min) / dt_ms)))
            ax.hist(spike_times_ms, bins=n_bins)

    elif isinstance(data, SynapseData):
        ax.text(
            0.5,
            0.5,
            "This plugin supports only trace and spike data.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    widget._redraw()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    _inject_controls(widget, data, neuron_slice, time_slice)
    _replot(widget, data, neuron_slice, time_slice, dt_ms=1.0)
