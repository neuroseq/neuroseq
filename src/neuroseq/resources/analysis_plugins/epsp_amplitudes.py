"""Detect EPSPs in membrane-voltage traces and plot their amplitudes."""

from __future__ import annotations

import re
from dataclasses import dataclass

import brian2.numpy_ as np
from qtpy import QtWidgets
from scipy.signal import find_peaks

from neuroseq.data import SynapseData
from neuroseq.data import TraceData

PLUGIN_ID = "epsp_amplitudes"
PLUGIN_NAME = "EPSP amplitudes"
PLUGIN_ICON = "mdi6.chart-scatter-plot"

_CONTROLS_ATTR = "_epsp_amplitude_controls"
_NEURON_MARKERS = ("+", "x", "o", "s", "^", "v", "D", "P", "*", "<", ">")


@dataclass(frozen=True)
class EPSPDetection:
    times_ms: np.ndarray
    amplitudes_mv: np.ndarray


def _to_mv(values: np.ndarray, units: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    unit = str(units).strip().lower().replace("µ", "u").replace("μ", "u")
    # Unit labels can originate in Brian2, Igor, or user-authored imports. Strip
    # harmless display punctuation and accept both symbols and full names.
    unit = re.sub(r"[\[\](){}\s._-]", "", unit)
    factors = {
        "v": 1000.0,
        "volt": 1000.0,
        "volts": 1000.0,
        "mv": 1.0,
        "mvolt": 1.0,
        "millivolt": 1.0,
        "millivolts": 1.0,
        "uv": 0.001,
        "uvolt": 0.001,
        "microvolt": 0.001,
        "microvolts": 0.001,
    }
    if not unit:
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            return values
        # Imported traces sometimes omit unit metadata. Membrane voltages are
        # commonly stored either in SI volts (~0.07) or millivolts (~70).
        factor = 1000.0 if float(np.percentile(finite, 95)) < 2.0 else 1.0
        return values * factor
    if unit not in factors:
        raise ValueError(
            "EPSP analysis requires a voltage trace "
            f"(V, mV, or µV); this trace is labelled {units!r}."
        )
    return values * factors[unit]


def detect_epsps(trace_mv: np.ndarray, dt_ms: float) -> EPSPDetection:
    """Return EPSP peak times and local-baseline amplitudes for one trace.

    The adaptive prominence is based on the median absolute first difference.
    Peaks reaching action-potential voltages, or rising >=20 mV above their
    local baseline, are discarded.
    """
    trace = np.asarray(trace_mv, dtype=float)
    if trace.size < 3 or dt_ms <= 0 or not np.all(np.isfinite(trace)):
        return EPSPDetection(np.asarray([], dtype=float), np.asarray([], dtype=float))

    noise = float(np.median(np.abs(np.diff(trace)))) / 0.6745
    prominence_mv = max(0.1, 3.0 * noise)
    min_distance = max(1, int(round(2.0 / dt_ms)))
    peak_indices, _ = find_peaks(trace, prominence=prominence_mv, distance=min_distance)

    baseline_window = max(2, int(round(10.0 / dt_ms)))
    baseline_gap = max(1, int(round(1.0 / dt_ms)))
    times: list[float] = []
    amplitudes: list[float] = []
    for peak in peak_indices:
        baseline_end = peak - baseline_gap
        baseline_start = max(0, baseline_end - baseline_window)
        if baseline_end <= baseline_start:
            continue
        baseline = float(np.median(trace[baseline_start:baseline_end]))
        amplitude = float(trace[peak] - baseline)
        # APs typically cross -20 mV; the amplitude guard also handles traces
        # whose absolute voltage offset has been removed.
        if amplitude < prominence_mv or trace[peak] >= -20.0 or amplitude >= 20.0:
            continue
        times.append(float(peak) * dt_ms)
        amplitudes.append(amplitude)
    return EPSPDetection(np.asarray(times), np.asarray(amplitudes))


def _remove_controls(widget) -> None:
    old = getattr(widget, _CONTROLS_ATTR, None)
    if old is not None and widget.layout() is not None:
        widget.layout().removeWidget(old)
        old.deleteLater()
    setattr(widget, _CONTROLS_ATTR, None)


def _replot(
    widget,
    data: TraceData,
    normalize: bool,
    normalization_count: int,
    input_units: str = "Auto",
) -> None:
    ax = widget.subplot
    ax.clear()
    traces = np.asarray(widget.traces)
    time_offset = 0.0
    if widget.time_slice is not None and widget.time_slice.start is not None:
        time_offset = float(widget.time_slice.start) * widget.dt_ms

    detected = 0
    units = data.units if input_units == "Auto" else input_units
    for row, trace in enumerate(_to_mv(traces, units)):
        result = detect_epsps(trace, widget.dt_ms)
        if result.amplitudes_mv.size == 0:
            continue
        values = result.amplitudes_mv
        if normalize:
            count = min(normalization_count, values.size)
            reference = float(np.mean(values[:count]))
            if reference > 0:
                values = values / reference * 100.0
        source_row = row + (widget.neuron_slice.start or 0)
        label = None
        if traces.shape[0] > 1:
            label = f"Neuron {source_row}"
        ax.scatter(
            result.times_ms + time_offset,
            values,
            marker=_NEURON_MARKERS[source_row % len(_NEURON_MARKERS)],
            s=48,
            linewidths=1.5,
            color="black",
            label=label,
        )
        detected += values.size

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("EPSP amplitude (%)" if normalize else "EPSP amplitude (mV)")
    if detected == 0:
        ax.text(
            0.5,
            0.5,
            "No EPSPs detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    elif traces.shape[0] > 1:
        ax.legend()
    widget._redraw()


def _inject_controls(widget, data: TraceData, neuron_slice, time_slice) -> None:
    _remove_controls(widget)
    layout = widget.layout()
    if layout is None:
        return
    bar = QtWidgets.QWidget(widget)
    controls = QtWidgets.QHBoxLayout(bar)
    controls.setContentsMargins(4, 2, 4, 2)
    controls.addWidget(QtWidgets.QLabel("Input units:"))
    units = QtWidgets.QComboBox()
    units.addItems(["Auto", "V", "mV", "µV"])
    units.setToolTip(
        "Auto uses the trace's unit label; for an unlabeled trace it infers "
        "V or mV from the value range."
    )
    controls.addWidget(units)
    percent = QtWidgets.QCheckBox("Express amplitude as %")
    controls.addWidget(percent)
    controls.addWidget(QtWidgets.QLabel("First EPSPs used as 100% per neuron:"))
    count = QtWidgets.QSpinBox()
    count.setRange(1, 1000)
    count.setValue(3)
    count.setEnabled(False)
    count.setToolTip(
        "Each neuron is normalized independently to the mean of its own first N EPSPs."
    )
    controls.addWidget(count)
    controls.addStretch()

    def update() -> None:
        _replot(
            widget,
            data,
            percent.isChecked(),
            count.value(),
            input_units=units.currentText(),
        )

    units.currentTextChanged.connect(update)
    percent.toggled.connect(count.setEnabled)
    percent.toggled.connect(update)
    count.valueChanged.connect(update)
    layout.insertWidget(0, bar)
    setattr(widget, _CONTROLS_ATTR, bar)


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    _remove_controls(widget)
    if isinstance(data, SynapseData) or not isinstance(data, TraceData):
        widget.subplot.text(
            0.5,
            0.5,
            "EPSP analysis supports voltage traces only.",
            ha="center",
            va="center",
            transform=widget.subplot.transAxes,
        )
        return
    try:
        _to_mv(np.asarray([0.0]), data.units)
    except ValueError as exc:
        widget.subplot.text(
            0.5,
            0.5,
            str(exc),
            ha="center",
            va="center",
            transform=widget.subplot.transAxes,
        )
        return
    _inject_controls(widget, data, neuron_slice, time_slice)
    _replot(
        widget,
        data,
        normalize=False,
        normalization_count=3,
        input_units="Auto",
    )
