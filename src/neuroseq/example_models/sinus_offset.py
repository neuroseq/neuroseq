from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.example_models.simulator import ParamValue
from neuroseq.example_models.simulator import Simulator

FlatParams = dict[str, ParamValue]
N_UNITS = 5

DEFAULT_PARAMS: dict[str, dict[str, ParamValue | str]] = {
    "dt_ms": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "Simulation",
        "description": "Integration time step in milliseconds.",
    },
    "baseline_mv": {
        "value": -60.0,
        "default": -60.0,
        "type": "float",
        "group": "Wave",
        "description": "Baseline membrane potential added to each sinusoid.",
    },
    "amplitude_mv": {
        "value": 8.0,
        "default": 8.0,
        "type": "float",
        "group": "Wave",
        "description": "Amplitude of the unit sinusoids in mV.",
    },
    "frequency_hz": {
        "value": 3.0,
        "default": 3.0,
        "type": "float",
        "group": "Wave",
        "description": "Common sinusoid frequency in Hz.",
    },
    "phase_step_rad": {
        "value": 0.4,
        "default": 0.4,
        "type": "float",
        "group": "Wave",
        "description": "Additional phase shift per unit in radians.",
    },
    "stim_gain": {
        "value": 1.0,
        "default": 1.0,
        "type": "float",
        "group": "Stimulus",
        "description": "Multiplier applied to summed stimulation offsets.",
    },
}


def _float_param(params: Mapping[str, ParamValue], key: str) -> float:
    value = params[key]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Parameter '{key}' must be float-compatible, got {type(value)}")


def _stim_as_unit_time_series(
    stim_values: np.ndarray,
    sample_indices: np.ndarray,
    n_units: int,
) -> np.ndarray:
    values = np.asarray(stim_values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "Each stim array must be 1D or 2D with non-zero time dimension; "
            f"got shape {values.shape}."
        )

    sampled = values[sample_indices]
    n_channels = sampled.shape[1]
    if n_channels == n_units:
        return sampled
    if n_channels == 1:
        return np.repeat(sampled, n_units, axis=1)
    if n_channels > n_units:
        return sampled[:, :n_units]

    padded = np.zeros((sampled.shape[0], n_units), dtype=np.float32)
    padded[:, :n_channels] = sampled
    return padded


class SinusOffset(Simulator):
    """Five-unit sinusoidal simulator with additive stimulus offsets.

    The simulator keeps cumulative simulation time so repeated calls continue
    the waveform phase instead of restarting from t=0.
    """

    def __init__(self) -> None:
        super().__init__(DEFAULT_PARAMS)

    def reset(self) -> None:
        self._simulated_time_s = 0.0

    def simulate(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, ParamValue] | None = None,
    ) -> Experiment:
        self.update_params(params)

        dt_s = _float_param(self._params, "dt_ms") / 1000.0
        if dt_s <= 0:
            raise ValueError("dt_ms must be > 0")

        n_steps = int(np.round(simulation_time / dt_s))
        if n_steps <= 0:
            raise ValueError("simulation_time too short for current dt_ms")

        t_local = np.arange(n_steps, dtype=np.float64) * dt_s
        t_abs = self._simulated_time_s + t_local

        baseline = _float_param(self._params, "baseline_mv")
        amplitude = _float_param(self._params, "amplitude_mv")
        frequency_hz = _float_param(self._params, "frequency_hz")
        phase_step_rad = _float_param(self._params, "phase_step_rad")
        stim_gain = _float_param(self._params, "stim_gain")

        unit_phase = phase_step_rad * np.arange(N_UNITS, dtype=np.float64)
        base_wave = baseline + amplitude * np.sin(
            2.0 * np.pi * frequency_hz * t_abs[:, None] + unit_phase[None, :]
        )

        stim_offset = np.zeros((n_steps, N_UNITS), dtype=np.float32)
        for stim_info in stims_dict.values():
            stim_dt_s = float(stim_info["dt"])
            if stim_dt_s <= 0:
                raise ValueError("Stim dt must be > 0")

            stim_values = np.asarray(stim_info["array"], dtype=np.float32)
            stim_indices = np.minimum(
                (t_abs / stim_dt_s).astype(np.int64),
                stim_values.shape[0] - 1,
            )
            stim_offset += _stim_as_unit_time_series(stim_values, stim_indices, N_UNITS)

        membrane = base_wave + stim_gain * stim_offset
        self._simulated_time_s += simulation_time

        indices = np.arange(N_UNITS, dtype=np.uint32)
        data: list[Data] = [
            TraceData(
                name="sinusoid_membrane_potential",
                traces=membrane.T.astype(np.float32, copy=False),
                indices=indices,
                units="mV",
                dt=dt_s,
            ),
            TraceData(
                name="stimulus_offset",
                traces=stim_offset.T.astype(np.float32, copy=False),
                indices=indices,
                units="mV",
                dt=dt_s,
            ),
        ]
        return Experiment(
            name="5-unit sinusoid with stimulation offset",
            duration=simulation_time,
            timepoints=[],
            data=data,
        )


def simulate(
    stims_dict: dict,
    simulation_time: float,
    params: Mapping[str, ParamValue] | None = None,
) -> Experiment:
    return _SIMULATOR.simulate(stims_dict, simulation_time, params=params)


def get_simulator() -> SinusOffset:
    return _SIMULATOR


_SIMULATOR = SinusOffset()
