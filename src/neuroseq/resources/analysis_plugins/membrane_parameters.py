"""Estimate passive membrane parameters from voltage and injected current."""

from __future__ import annotations

import re
from dataclasses import dataclass

import brian2.numpy_ as np
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.stim import StimArray

PLUGIN_ID = "membrane_parameters"
PLUGIN_NAME = "Estimate membrane parameters"
PLUGIN_ICON = "mdi6.chart-bell-curve"


@dataclass(frozen=True)
class MembraneParameterEstimate:
    neuron: int
    r_m: float
    v_rest: float
    tau_m: float


def _normalized_unit(units: str) -> str:
    unit = str(units).strip().lower().replace("µ", "u").replace("μ", "u")
    return re.sub(r"[\[\](){}\s._-]", "", unit)


def is_current_unit(units: str) -> bool:
    """Return whether a display unit denotes electrical current."""
    return _normalized_unit(units) in {
        "a",
        "amp",
        "amps",
        "ampere",
        "amperes",
        "ma",
        "mamp",
        "milliamp",
        "ua",
        "uamp",
        "microamp",
        "na",
        "namp",
        "nanoamp",
        "pa",
        "pamp",
        "picoamp",
        "fa",
        "famp",
        "femtoamp",
    }


def _unit_factor(units: str, factors: dict[str, float], quantity: str) -> float:
    unit = _normalized_unit(units)
    if unit not in factors:
        raise ValueError(f"Unsupported {quantity} unit: {units!r}.")
    return factors[unit]


def voltage_factor(units: str) -> float:
    """Return the multiplier that converts a supported voltage unit to volts."""
    return _unit_factor(
        units,
        {
            "v": 1.0,
            "volt": 1.0,
            "volts": 1.0,
            "mv": 1e-3,
            "millivolt": 1e-3,
            "millivolts": 1e-3,
            "uv": 1e-6,
            "microvolt": 1e-6,
            "microvolts": 1e-6,
        },
        "voltage",
    )


def current_factor(units: str) -> float:
    """Return the multiplier that converts a supported current unit to amperes."""
    return _unit_factor(
        units,
        {
            "a": 1.0,
            "amp": 1.0,
            "amps": 1.0,
            "ampere": 1.0,
            "amperes": 1.0,
            "ma": 1e-3,
            "mamp": 1e-3,
            "milliamp": 1e-3,
            "ua": 1e-6,
            "uamp": 1e-6,
            "microamp": 1e-6,
            "na": 1e-9,
            "namp": 1e-9,
            "nanoamp": 1e-9,
            "pa": 1e-12,
            "pamp": 1e-12,
            "picoamp": 1e-12,
            "fa": 1e-15,
            "famp": 1e-15,
            "femtoamp": 1e-15,
        },
        "current",
    )


def fit_a_b_c(dv_dt_trace, v, i_stim) -> tuple[float, float, float]:
    """Fit ``dv/dt = (-(v - b) + a*i_stim) / c`` by least squares."""
    dv_dt_trace = np.asarray(dv_dt_trace, dtype=float)
    v = np.asarray(v, dtype=float)
    i_stim = np.asarray(i_stim, dtype=float)
    if not (dv_dt_trace.shape == v.shape == i_stim.shape):
        raise ValueError("dv_dt_trace, v, and i_stim must have the same shape.")
    if dv_dt_trace.size < 3:
        raise ValueError("At least three trace samples are required.")
    finite = np.isfinite(dv_dt_trace) & np.isfinite(v) & np.isfinite(i_stim)
    if np.count_nonzero(finite) < 3:
        raise ValueError("At least three finite trace samples are required.")
    x = np.column_stack((v[finite], i_stim[finite], np.ones(np.count_nonzero(finite))))
    p_hat, q_hat, r_hat = np.linalg.lstsq(x, dv_dt_trace[finite], rcond=None)[0]
    if np.isclose(p_hat, 0.0):
        raise ValueError(
            "Could not estimate tau_m because the fitted voltage coefficient is near zero."
        )
    c_hat = -1.0 / p_hat
    return float(q_hat * c_hat), float(r_hat * c_hat), float(c_hat)


def estimate_parameters(
    traces,
    trace_dt: float,
    stimulus: StimArray,
    voltage_units: str,
    *,
    first_neuron: int = 0,
    first_sample: int = 0,
) -> list[MembraneParameterEstimate]:
    """Estimate parameters per trace row, aligning stimulus samples by time."""
    # TraceData stores voltage samples in SI volts. Its unit string specifies
    # the preferred display unit, as it does in TraceWidget.
    voltage_factor(voltage_units)  # validate that this is a voltage trace
    voltage = np.asarray(traces, dtype=float)
    stimulus_values = np.asarray(stimulus.values, dtype=float) * current_factor(
        stimulus.units
    )
    if voltage.ndim != 2 or voltage.shape[1] < 4:
        raise ValueError("The selection must contain at least four samples.")
    if trace_dt <= 0 or stimulus.dt <= 0:
        raise ValueError("Trace and input-array time steps must be positive.")
    trace_times = (first_sample + np.arange(voltage.shape[1] - 1)) * trace_dt
    stim_indices = np.floor(trace_times / stimulus.dt + 1e-12).astype(int)
    if stim_indices.size == 0 or stim_indices[-1] >= stimulus.values.shape[0]:
        raise ValueError("The selected trace extends beyond the input array.")
    estimates: list[MembraneParameterEstimate] = []
    for row, trace in enumerate(voltage):
        neuron = first_neuron + row
        if neuron >= stimulus.values.shape[1]:
            raise ValueError(f"Input array has no column for neuron {neuron}.")
        dv_dt = np.diff(trace) / trace_dt
        r_m, v_rest, tau_m = fit_a_b_c(
            dv_dt, trace[:-1], stimulus_values[stim_indices, neuron]
        )
        estimates.append(MembraneParameterEstimate(neuron, r_m, v_rest, tau_m))
    return estimates


class MembraneParametersDialog(QtWidgets.QDialog):
    """Compact input-array selector and textual parameter report."""

    def __init__(
        self,
        data: TraceData,
        current_arrays: list[StimArray],
        neuron_slice: slice | None,
        time_slice: slice | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(PLUGIN_NAME)
        self.resize(480, 320)
        self._data = data
        self._arrays = {array.name: array for array in current_arrays}
        self._neuron_slice = neuron_slice or slice(None)
        self._time_slice = time_slice or slice(None)
        layout = QtWidgets.QVBoxLayout(self)
        selector = QtWidgets.QHBoxLayout()
        selector.addWidget(QtWidgets.QLabel("Input current array:"))
        self.input_array_combo = QtWidgets.QComboBox()
        self.input_array_combo.addItems(self._arrays)
        selector.addWidget(self.input_array_combo, 1)
        layout.addLayout(selector)
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.results, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.input_array_combo.currentTextChanged.connect(self._update_results)
        self._update_results()

    def _update_results(self) -> None:
        stimulus = self._arrays[self.input_array_combo.currentText()]
        first_neuron = self._neuron_slice.start or 0
        first_sample = self._time_slice.start or 0
        traces = np.asarray(self._data.traces)[self._neuron_slice, self._time_slice]
        try:
            estimates = estimate_parameters(
                traces,
                self._data.dt,
                stimulus,
                self._data.units,
                first_neuron=first_neuron,
                first_sample=first_sample,
            )
        except ValueError as exc:
            self.results.setPlainText(f"Could not estimate parameters:\n\n{exc}")
            return
        lines: list[str] = []
        for result in estimates:
            if lines:
                lines.append("")
            lines.extend(
                (
                    f"Neuron {result.neuron}",
                    f"r_m:    {_format_resistance(result.r_m)}",
                    f"v_rest: {result.v_rest / voltage_factor(self._data.units):.6g} "
                    f"{self._data.units}",
                    f"tau_m:  {result.tau_m * 1000.0:.6g} ms",
                )
            )
        self.results.setPlainText("\n".join(lines))


def _format_resistance(value_ohm: float) -> str:
    absolute = abs(value_ohm)
    if absolute >= 1e9:
        return f"{value_ohm / 1e9:.6g} GΩ"
    if absolute >= 1e6:
        return f"{value_ohm / 1e6:.6g} MΩ"
    if absolute >= 1e3:
        return f"{value_ohm / 1e3:.6g} kΩ"
    return f"{value_ohm:.6g} Ω"


def open_analysis_dialog(parent, project, data, neuron_slice, time_slice) -> None:
    """Open one dialog containing both input selection and fit results."""
    if not isinstance(data, TraceData):
        QtWidgets.QMessageBox.warning(
            parent, PLUGIN_NAME, "Membrane parameter fitting supports trace data only."
        )
        return
    arrays = [
        array for array in project.stim_arrays.values() if is_current_unit(array.units)
    ]
    if not arrays:
        QtWidgets.QMessageBox.warning(
            parent,
            PLUGIN_NAME,
            "No input arrays with current units are available. Supported examples "
            "include A, mA, µA, nA, and pA.",
        )
        return
    MembraneParametersDialog(data, arrays, neuron_slice, time_slice, parent).exec()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    """Fallback required by the plugin interface; normal use opens the dialog."""
    widget.subplot.text(
        0.5,
        0.5,
        "Use the membrane-parameter dialog from the trace analysis menu.",
        ha="center",
        va="center",
        transform=widget.subplot.transAxes,
    )
