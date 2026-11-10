from __future__ import annotations

import brian2.numpy_ as np

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData

PLUGIN_ID = "mean_activity"
PLUGIN_NAME = "Mean activity"
PLUGIN_ICON = "mdi6.chart-timeline-variant"


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    """Example analysis plugin that draws mean activity for selected data."""
    ax = widget.subplot
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
            return
        trace_values = np.asarray(widget.traces)
        mean_trace = np.mean(trace_values, axis=0)
        times_ms = np.arange(mean_trace.shape[0], dtype=float) * widget.dt_ms
        if widget.time_slice is not None and widget.time_slice.start is not None:
            times_ms += float(widget.time_slice.start) * widget.dt_ms
        ax.plot(times_ms, mean_trace, color="tab:orange", linewidth=2.0)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Mean {data.units}")
        return

    if isinstance(data, SpikeData):
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
            return

        # 1 ms bins over the data range (at least one, e.g. for a lone spike).
        t_min = float(np.min(spike_times_ms))
        bins = max(1, int(np.ceil(np.max(spike_times_ms) - t_min)))
        ax.hist(spike_times_ms, bins=bins, range=(t_min, t_min + bins))
        return

    if isinstance(data, SynapseData):
        ax.text(
            0.5,
            0.5,
            "This example plugin supports only trace and spike data.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
