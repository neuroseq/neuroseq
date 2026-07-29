from __future__ import annotations

import importlib
import json
import logging
import math
import pathlib
import pickle
import runpy
import zipfile
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Protocol
from typing import overload

import numpy as np

from neuroseq.b2_model import B2Model
from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.io import append_to_zip
from neuroseq.io import load_from_zip
from neuroseq.power_lut import PowerLut
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_timing import ensure_2d_stim_values
from neuroseq.stim_timing import remap_stim_values_to_dt
from neuroseq.stim_timing import stim_dt_matches
from neuroseq.stim_timing import validate_stim_dt_seconds
from neuroseq.utils import append_data
from neuroseq.utils import find_by_name_and_type
from neuroseq.utils import save_store_name


class ExternalSimulateFn(Protocol):
    def __call__(
        self,
        stims_dict: dict,
        simulation_time: float,
        params: Mapping[str, object] | None = None,
    ) -> Experiment: ...


def _to_seconds(value: float | int) -> float:
    return float(value)


def _matching_potential_to_opsins(
    u_inp: np.ndarray,
    E_exc: float,
    E_inh: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    denom = E_inh - E_exc
    if denom == 0:
        return None
    g_exc_vals = ((E_inh - u_inp) / denom).astype(np.float32)
    g_inh_vals = ((u_inp - E_exc) / denom).astype(np.float32)
    np.clip(g_exc_vals, 0.0, None, out=g_exc_vals)
    np.clip(g_inh_vals, 0.0, None, out=g_inh_vals)
    return g_exc_vals, g_inh_vals


def _trace_data_neuron_count(data: TraceData) -> int:
    if data.indices.size > 0:
        return int(np.max(data.indices)) + 1
    if data.traces.ndim >= 1:
        return int(data.traces.shape[0])
    return 0


def _spike_data_neuron_count(data: SpikeData) -> int:
    if data.spike_indices.size == 0:
        return 0
    return int(np.max(data.spike_indices)) + 1


def _synapse_data_neuron_count(data: SynapseData) -> int:
    if data.values.ndim < 2:
        return 0
    return int(max(data.values.shape[0], data.values.shape[1]))


def _experiment_data_neuron_count(
    data: TraceData | SpikeData | SynapseData,
) -> int:
    if isinstance(data, TraceData):
        return _trace_data_neuron_count(data)
    if isinstance(data, SpikeData):
        return _spike_data_neuron_count(data)
    return _synapse_data_neuron_count(data)


def _cellpose_neuron_count(cellpose_data: CellposeImportResult | None) -> int:
    if cellpose_data is None:
        return 0
    if cellpose_data.original_cellpose_ids is not None:
        return int(np.asarray(cellpose_data.original_cellpose_ids).size)
    if cellpose_data.segmentation_array is None:
        return 0
    unique_ids = np.unique(cellpose_data.segmentation_array)
    return int(np.sum(unique_ids >= 0))


class Project:
    def __init__(
        self,
        filepath: str | pathlib.Path | None = None,
        report: Callable[[int], None] | None = None,
    ) -> None:
        self.filepath = None
        self.b2_model: B2Model = B2Model()
        self.project_name: str | None = None
        self.project_section: str | None = None
        self.experiments: dict[str, Experiment] = {}
        self.stim_arrays: dict[str, StimArray] = {}
        self.cellpose_data: CellposeImportResult | None = None
        self.syscon2_export_template_text: str | None = None
        self.power_luts: dict[int, PowerLut] = {}

        self.max_ram_synapse_plot_matrix = 256
        self.max_time_secs = 0.0
        self.max_neurons = 0
        self.dt: float = 0.001

        # External-simulation function state. `external_simulate_path` is either
        # a filesystem path to a `.py` file or a `neuroseq:<module_name>`
        # sentinel resolving to `neuroseq.example_models.<module_name>.simulate`.
        # `external_simulate_function` is transient and re-resolved from the
        # path on load (see `load_NN`).
        self.external_simulate_path: str | None = None
        self.external_simulate_params: dict[str, dict] = {}
        self.external_simulator: object | None = None
        self.external_simulate_function: ExternalSimulateFn | None = None
        # Schema version of the most recently loaded archive (1 == pre-kinetic).
        # New archives are written at version 2.
        self._loaded_schema_version: int = 1

        self.logger = logging.getLogger(__name__)
        if filepath:
            self.filepath = pathlib.Path(filepath)
            self.logger.info(f"Loading project from '{filepath}'")
            _dt_from_metadata = False
            with zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED) as zf:
                b2_model_data = load_from_zip("b2_model", zf)
                if "b2_stored_state.pkl" in zf.namelist():
                    b2_stored_state = pickle.loads(zf.read("b2_stored_state.pkl"))
                else:
                    b2_stored_state = {}
                self.b2_model = B2Model.from_dict(
                    b2_model_data, b2_stored_state, report
                )
                self.experiments = load_from_zip("experimental_data", zf)
                self.stim_arrays = load_from_zip("stim_arrays", zf)
                self.cellpose_data = load_from_zip("cellpose_data", zf).get(
                    "cellpose_import"
                )
                if "project_metadata.json" in zf.namelist():
                    _meta = json.loads(zf.read("project_metadata.json"))
                    self.dt = float(_meta.get("dt", 0.001))
                    _dt_from_metadata = True
                    self.external_simulate_path = _meta.get("external_simulate_path")
                    self.external_simulate_params = _meta.get(
                        "external_simulate_params", {}
                    )
                    self.power_luts = {
                        int(wavelength_str): PowerLut.from_serializable(payload)
                        for wavelength_str, payload in _meta.get(
                            "power_luts", {}
                        ).items()
                    }
                    self._loaded_schema_version = int(_meta.get("schema_version", 1))
            if not _dt_from_metadata:
                self.dt = self._derive_dt_from_data()
            if self.external_simulate_path is not None:
                try:
                    self.load_NN()
                    self._migrate_legacy_elise_stim()
                    self._migrate_legacy_elise_kinetics_units()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to resolve external_simulate_path "
                        f"{self.external_simulate_path!r}: {e}"
                    )
                    self.external_simulate_function = None
            for experiment in self.experiments.values():
                self.max_time_secs = max(self.max_time_secs, experiment.duration)
            for stim_array in self.stim_arrays.values():
                self.max_time_secs = max(
                    self.max_time_secs, stim_array.dt * stim_array.values.shape[0]
                )

        self.update_limits()

    def import_project(
        self,
        filepath: str | pathlib.Path,
        options: SaveLoadOptions,
        report: Callable[[int], None] | None = None,
    ) -> None:
        filepath = pathlib.Path(filepath)
        self.logger.info(
            f"Importing components from '{filepath}' into existing project"
        )
        with zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED) as zf:
            if options.include_model:
                self.logger.info("  - importing b2_model")
                self.b2_model = B2Model()
                b2_model_data = load_from_zip("b2_model", zf)
                if (
                    options.include_simulation_data
                    and "b2_stored_state.pkl" in zf.namelist()
                ):
                    self.logger.info("  - importing simulation data")
                    b2_stored_state = pickle.loads(zf.read("b2_stored_state.pkl"))
                else:
                    b2_stored_state = {}
                self.b2_model = B2Model.from_dict(
                    b2_model_data, b2_stored_state, report
                )
            if options.include_experimental_data:
                self.logger.info("  - importing experimental data")
                self.experiments = load_from_zip("experimental_data", zf)
            if options.include_stim_arrays:
                self.logger.info("  - importing stim arrays")
                self.stim_arrays = load_from_zip("stim_arrays", zf)
        self.update_limits()

    def get_data_names(
        self, data_type: type[TraceData | SpikeData | SynapseData]
    ) -> list[str]:
        """
        Get names of all data of the given type

        Parameters
        ----------
        data_type : type[TraceData | SpikeData | SynapseData]
            Type of data to get names for

        Returns
        -------
        list[str]
            List of data names that can be passed to get_data()
        """
        data_names = []
        if data_type is TraceData:
            data_names.extend(self.b2_model.get_simulation_trace_names())
        elif data_type is SpikeData:
            data_names.extend(self.b2_model.get_simulation_spike_names())
        elif data_type is SynapseData:
            data_names.extend(self.b2_model.get_simulation_synapse_names())
        for experiment in self.experiments.values():
            for datum in experiment.data:
                if isinstance(datum, data_type):
                    data_names.append(f"{experiment.name} :: {datum.name}")
        return data_names

    @overload
    def get_data(self, name: str, data_type: type[TraceData]) -> TraceData | None: ...
    @overload
    def get_data(self, name: str, data_type: type[SpikeData]) -> SpikeData | None: ...
    @overload
    def get_data(
        self, name: str, data_type: type[SynapseData]
    ) -> SynapseData | None: ...
    def get_data(
        self,
        name: str,
        data_type: type[Data],
    ) -> Data | None:
        """
        Get experimental data from the project

        Parameters
        ----------
        name : str
            Name of the data to get, as provided by get_data_names()
        data_type : type[TraceData | SpikeData | SynapseData]
            Type of data to get

        Returns
        -------
        TraceData | SpikeData | SynapseData | None
            The requested data, or None if not found
        """
        if name == "":
            return None
        self.logger.info(f"Getting '{name}' of type {data_type}")
        experiment_name, data_name = name.split(" :: ")
        if experiment_name == "Simulation":
            if data_type is TraceData:
                return self.b2_model.get_simulation_trace_data(name)
            if data_type is SpikeData:
                return self.b2_model.get_simulation_spike_data(name)
            if data_type is SynapseData:
                return self.b2_model.get_simulation_synapse_data(
                    name, self.max_ram_synapse_plot_matrix
                )
        if experiment_name not in self.experiments:
            self.logger.info(f"  - experiment '{experiment_name}' not found")
            return None
        for data in self.experiments[experiment_name].data:
            if data.name == data_name and isinstance(data, data_type):
                return data
        self.logger.info(f"  - data '{data_name}' not found")
        return None

    def add_stim_array(self, stim_array: StimArray):
        stim_dt = validate_stim_dt_seconds(stim_array.dt)
        stim_array.values = ensure_2d_stim_values(stim_array.values)
        stim_array.dt = stim_dt
        self.stim_arrays[stim_array.name] = stim_array
        self.update_limits()

    def delete_stim_array(self, name: str):
        if name in self.stim_arrays:
            del self.stim_arrays[name]
        self.update_limits()

    def update_stim_array_properties(
        self, name: str, n_neurons: int, stim_type: StimType, units: str, dt: float
    ):
        dt = validate_stim_dt_seconds(dt)
        stim_array = self.stim_arrays[name]
        old_dt = validate_stim_dt_seconds(stim_array.dt)
        stim_array.stim_type = stim_type
        stim_array.units = units
        if not stim_dt_matches(old_dt, dt):
            stim_array.values = remap_stim_values_to_dt(
                stim_array.values,
                old_dt_seconds=old_dt,
                new_dt_seconds=dt,
            )
        stim_array.values = ensure_2d_stim_values(stim_array.values)
        stim_array.dt = dt
        current_n_neurons = int(stim_array.values.shape[1])
        if n_neurons < current_n_neurons:
            self.logger.info(
                f"Decreasing neurons in StimArray {name} from {current_n_neurons} to {n_neurons}"
            )
            stim_array.values = stim_array.values[:, :n_neurons]
        elif n_neurons > current_n_neurons:
            self.logger.info(
                f"Increasing neurons in StimArray {name} from {current_n_neurons} to {n_neurons}"
            )
            stim_array.values = np.pad(
                stim_array.values,
                ((0, 0), (0, n_neurons - current_n_neurons)),
                mode="constant",
                constant_values=0,
            )
        self.update_limits()

    def rename_stim_array(self, name: str, new_name: str):
        stim_array = self.stim_arrays.pop(name)
        stim_array.name = new_name
        self.stim_arrays[stim_array.name] = stim_array

    def update_limits(self):
        self.max_time_secs = _to_seconds(self.b2_model.get_sim_time_secs())
        self.max_neurons = int(self.b2_model.get_max_neurons())
        for experiment in self.experiments.values():
            self.max_time_secs = max(
                self.max_time_secs, _to_seconds(experiment.duration)
            )
            for datum in experiment.data:
                self.max_neurons = max(
                    self.max_neurons,
                    _experiment_data_neuron_count(datum),
                )
        for stim_array in self.stim_arrays.values():
            stim_array.values = ensure_2d_stim_values(stim_array.values)
            stim_dt = validate_stim_dt_seconds(stim_array.dt)
            stim_array.dt = stim_dt
            self.max_time_secs = max(
                self.max_time_secs,
                _to_seconds(stim_dt) * float(stim_array.values.shape[0]),
            )
            n_stim_neurons = int(stim_array.values.shape[1])
            self.max_neurons = max(self.max_neurons, n_stim_neurons)
        self.max_neurons = max(
            self.max_neurons,
            _cellpose_neuron_count(self.cellpose_data),
        )

    def _derive_dt_from_data(self) -> float:
        """Infer canonical dt from loaded stim/trace data; fall back to 1 ms."""
        for stim_array in self.stim_arrays.values():
            return float(stim_array.dt)
        for experiment in self.experiments.values():
            for data in experiment.data:
                if isinstance(data, TraceData):
                    return float(data.dt)
        return 0.001

    def remap_all_data_to_dt(self, new_dt: float) -> None:
        """ZOH-remap every StimArray and every TraceData in the project to new_dt."""
        from neuroseq.importer_igor2 import (  # noqa: PLC0415
            estimate_resampled_sample_count,
        )
        from neuroseq.importer_igor2 import (  # noqa: PLC0415
            resample_wave_zero_order_hold,
        )

        new_dt = float(new_dt)
        for stim_array in self.stim_arrays.values():
            if not stim_dt_matches(float(stim_array.dt), new_dt):
                stim_array.values = remap_stim_values_to_dt(
                    stim_array.values,
                    old_dt_seconds=float(stim_array.dt),
                    new_dt_seconds=new_dt,
                )
                stim_array.dt = new_dt

        for experiment in self.experiments.values():
            for data in experiment.data:
                if not isinstance(data, TraceData):
                    continue
                if stim_dt_matches(float(data.dt), new_dt):
                    continue
                n_rows, n_cols = data.traces.shape
                if n_cols == 0:
                    data.dt = new_dt
                    continue
                if n_rows == 0:
                    new_cols = estimate_resampled_sample_count(
                        n_cols, float(data.dt), target_dt=new_dt
                    )
                    data.traces = np.zeros((0, new_cols), dtype=np.float32)
                else:
                    new_rows = [
                        resample_wave_zero_order_hold(
                            data.traces[i],
                            source_dt=float(data.dt),
                            target_dt=new_dt,
                        )
                        for i in range(n_rows)
                    ]
                    data.traces = np.stack(new_rows, axis=0).astype(
                        np.float32, copy=False
                    )
                data.dt = new_dt

        self.dt = new_dt
        self.update_limits()

    def remappable_data_description(self, new_dt: float) -> str:
        """Human-readable summary of stims/traces that would be remapped to new_dt."""
        stim_count = sum(
            1
            for sa in self.stim_arrays.values()
            if not stim_dt_matches(float(sa.dt), new_dt)
        )
        trace_count = sum(
            1
            for exp in self.experiments.values()
            for data in exp.data
            if isinstance(data, TraceData)
            and not stim_dt_matches(float(data.dt), new_dt)
        )
        parts: list[str] = []
        if stim_count:
            parts.append(f"{stim_count} stim array(s)")
        if trace_count:
            parts.append(f"{trace_count} trace dataset(s)")
        return " and ".join(parts)

    def issues_with_appending_experimental_data(
        self,
        experiment_name: str,
        new_data: Sequence[TraceData | SpikeData | SynapseData],
    ) -> str:
        """
        Check for any issues with appending additional experimental data to an existing experiment

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to append to
        new_data : Sequence[TraceData | SpikeData | SynapseData]
            List of data objects to be appended to the experiment

        Returns
        -------
        str
            Description of any issues found, or an empty string if no issues
        """
        if experiment_name not in self.experiments:
            return f"Experiment '{experiment_name}' does not exist"
        existing_experiment = self.experiments[experiment_name]

        for datum in new_data:
            if isinstance(datum, TraceData):
                existing_trace = find_by_name_and_type(
                    existing_experiment.data,
                    datum.name,
                    TraceData,
                )
                if existing_trace is None:
                    return (
                        f"Cannot append data '{datum.name}': data with this name and type "
                        f"does not exist in experiment '{experiment_name}'"
                    )
                if existing_trace.dt != datum.dt:
                    return (
                        f"Cannot append trace data '{datum.name}': dt does not match "
                        f"({existing_trace.dt} != {datum.dt})"
                    )
            elif isinstance(datum, SpikeData):
                existing_spike = find_by_name_and_type(
                    existing_experiment.data,
                    datum.name,
                    SpikeData,
                )
                if existing_spike is None:
                    return (
                        f"Cannot append data '{datum.name}': data with this name and type "
                        f"does not exist in experiment '{experiment_name}'"
                    )
            elif isinstance(datum, SynapseData):
                existing_synapse = find_by_name_and_type(
                    existing_experiment.data,
                    datum.name,
                    SynapseData,
                )
                if existing_synapse is None:
                    return (
                        f"Cannot append data '{datum.name}': data with this name and type "
                        f"does not exist in experiment '{experiment_name}'"
                    )
                if (
                    existing_synapse.values.shape[0] != datum.values.shape[0]
                    or existing_synapse.values.shape[1] != datum.values.shape[1]
                ):
                    return (
                        f"Cannot append synapse data '{datum.name}': number of neurons does not match "
                        f"({existing_synapse.values.shape[0]}x{existing_synapse.values.shape[1]} != "
                        f"{datum.values.shape[0]}x{datum.values.shape[1]})"
                    )
        return ""

    def append_experimental_data(
        self,
        experiment_name: str,
        duration_seconds: float,
        new_data: Sequence[TraceData | SpikeData | SynapseData],
        time_offset_seconds: float,
    ):
        """
        Append new experimental data to an existing experiment

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to append to
        duration_seconds: float
            Duration in seconds of the new data to append
        new_data : Sequence[TraceData | SpikeData | SynapseData]
            List of data objects to append to the experiment
        time_offset_seconds : float
            Time in seconds at which to insert the new data.
        """
        if experiment_name not in self.experiments:
            self.logger.warning(
                f"Append requested but experiment '{experiment_name}' does not exist"
            )
            raise ValueError(f"Experiment '{experiment_name}' does not exist")
        self.logger.info(
            f"Appending experimental data to experiment '{experiment_name}'"
        )
        existing_experiment = self.experiments[experiment_name]
        if time_offset_seconds is None:
            time_offset_seconds = existing_experiment.duration
        # match data by type and name and append
        for datum in new_data:
            existing_datum = next(
                (
                    d
                    for d in existing_experiment.data
                    if d.name == datum.name and isinstance(d, type(datum))
                ),
                None,
            )
            if existing_datum is not None:
                self.logger.info(
                    f"  - appending '{datum.name}' to '{datum.name}' at {time_offset_seconds}s"
                )
                append_data(
                    existing_datum, datum, duration_seconds, time_offset_seconds
                )
            else:
                self.logger.warning(
                    f"'{datum.name}' not found when trying to append - adding new data instead of appending."
                )
                self.experiments[experiment_name].data.append(datum)
        for t in [time_offset_seconds, time_offset_seconds + duration_seconds]:
            if t not in existing_experiment.timepoints:
                existing_experiment.timepoints.append(t)
        existing_experiment.timepoints.sort()
        existing_experiment.duration = max(
            existing_experiment.duration, time_offset_seconds + duration_seconds
        )
        self.update_limits()

    def add_experiment(
        self,
        experiment: Experiment,
    ):
        """
        Add a new experiment to the model

        Parameters
        ----------
        experiment : Experiment
            Experiment object to add

        Raises
        ------
        ValueError
            If an experiment with the same name already exists
        """
        self.logger.info(f"Adding experimental data for experiment '{experiment.name}'")
        if experiment.name in self.experiments:
            self.logger.warning(
                f"Add requested but experiment '{experiment.name}' already exists"
            )
            raise ValueError(f"Experiment '{experiment.name}' already exists")
        self.logger.info(f"Adding experimental data for experiment '{experiment.name}'")
        for t in [0.0, experiment.duration]:
            if t not in experiment.timepoints:
                experiment.timepoints.append(t)
        self.experiments[experiment.name] = experiment
        self.update_limits()

    def add_experimental_data(
        self,
        experiment_name: str,
        data_list: list[TraceData | SpikeData | SynapseData],
    ):
        """
        Add experimental data as a new experiment to the project.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment to create
        data_list : list[TraceData | SpikeData | SynapseData]
            List of data objects to add to the experiment

        Raises
        ------
        ValueError
            If an experiment with the same name already exists
        """
        if not data_list:
            raise ValueError("Cannot create experiment with empty data list")

        # Calculate duration from the data
        # All data should have the same duration, so we use the first one
        from neuroseq.data import TraceData

        duration = (
            len(data_list[0].traces) * data_list[0].dt
            if isinstance(data_list[0], TraceData)
            else 0.0
        )

        # Create and add the experiment
        experiment = Experiment(
            name=experiment_name,
            duration=duration,
            timepoints=[],
            data=data_list,
        )
        self.add_experiment(experiment)

    def add_data_to_experiment(
        self,
        experiment_name: str,
        data: TraceData | SpikeData | SynapseData,
    ) -> None:
        """Append data to an existing experiment and keep timing metadata coherent."""
        experiment = self.experiments.get(experiment_name)
        if experiment is None:
            raise KeyError(f"Experiment '{experiment_name}' does not exist")

        experiment.data.append(data)
        if isinstance(data, TraceData):
            new_duration = float(data.traces.shape[1] * data.dt)
            if new_duration > experiment.duration:
                experiment.duration = new_duration
                if new_duration not in experiment.timepoints:
                    experiment.timepoints.append(new_duration)
                experiment.timepoints.sort()
        self.update_limits()

    def pad_stim_arrays(self, total_time_seconds: float):
        """
        Pad all StimArrays in the project to a length of total_time_seconds
        """
        for name, stim_array in self.stim_arrays.items():
            n_timestamps = stim_array.values.shape[0]
            array_time_secs = stim_array.dt * n_timestamps
            padding_required_secs = total_time_seconds - array_time_secs
            if padding_required_secs > 0:
                n_pad = math.ceil(padding_required_secs / stim_array.dt)
                self.logger.info(
                    f"Adding {padding_required_secs}s / {n_pad} timestamps of padding to {name}: {stim_array.values.shape}"
                )
                self.stim_arrays[name].values = np.pad(
                    stim_array.values, pad_width=((0, n_pad), (0, 0))
                )
        self.update_limits()

    def pad_stim_arrays_to_neuron_count(self, n_neurons: int) -> None:
        """
        Widen every StimArray along the neuron axis so it has at least
        ``n_neurons`` columns. New columns are filled with zeros (no
        stimulation). Arrays that are already wide enough are left untouched;
        arrays are never shrunk (so no user data is lost).

        Parameters
        ----------
        n_neurons : int
            Target minimum number of neuron columns.
        """
        if n_neurons <= 0:
            return
        for name, stim_array in self.stim_arrays.items():
            current_n_neurons = int(stim_array.values.shape[1])
            if current_n_neurons >= n_neurons:
                continue
            self.logger.info(
                f"Widening StimArray {name} from {current_n_neurons} to "
                f"{n_neurons} neurons: {stim_array.values.shape}"
            )
            stim_array.values = np.pad(
                stim_array.values,
                ((0, 0), (0, n_neurons - current_n_neurons)),
                mode="constant",
                constant_values=0,
            )
        self.update_limits()

    def trim_stim_arrays(self, total_time_seconds: float):
        """
        Trim all StimArrays in the project to a length of total_time_seconds
        """
        for name, stim_array in self.stim_arrays.items():
            n_timestamps = stim_array.values.shape[0]
            new_n_timestamps = int(total_time_seconds / stim_array.dt)
            if new_n_timestamps < n_timestamps:
                self.logger.info(
                    f"Reducing {name} number of timestamps: {n_timestamps} -> {new_n_timestamps}"
                )
                self.stim_arrays[name].values = stim_array.values[:new_n_timestamps, :]
        self.update_limits()

    def loop_stim_arrays(self, n_loops: int) -> None:
        """Tile every stim array's values along the time axis."""
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        if n_loops == 1:
            return
        for stim_array in self.stim_arrays.values():
            stim_array.values = np.tile(stim_array.values, (n_loops, 1))
        self.update_limits()

    def suggested_simulation_time_seconds(self) -> float:
        """Return the longest stim-array duration across the project.

        The full stim length is used as the suggested simulation time so
        that any pre-Run modification (Pad, Loop, Trim, manual editing)
        is faithfully reflected in the Run dialog's default. In particular,
        padding a stim with trailing zeros to create a "ring-out" period
        produces the same end state as the dialog would: the simulator
        reads the entire stim, including the trailing zeros, and produces
        no teaching current during that interval.
        """
        if not self.stim_arrays:
            return float(self.max_time_secs) if self.max_time_secs > 0 else 1.0
        candidates: list[float] = []
        for stim_array in self.stim_arrays.values():
            n_rows = int(stim_array.values.shape[0])
            if n_rows == 0:
                continue
            candidates.append(n_rows * stim_array.dt)
        if not candidates:
            return 1.0
        return max(candidates)

    def tile_stim_arrays_to_length(self, target_seconds: float) -> None:
        """Extend every stim array via tiling so each covers `target_seconds`."""
        if target_seconds < 0:
            raise ValueError(f"target_seconds must be >= 0, got {target_seconds}")
        for stim_array in self.stim_arrays.values():
            current_n = int(stim_array.values.shape[0])
            if current_n == 0:
                continue
            n_target = int(math.ceil(target_seconds / stim_array.dt))
            if n_target <= current_n:
                continue
            full_tiles = (n_target + current_n - 1) // current_n
            tiled = np.tile(stim_array.values, (full_tiles, 1))
            stim_array.values = tiled[:n_target]
        self.update_limits()

    def trim_stim_array_would_delete_data(self, total_time_seconds: float) -> bool:
        """
        Check whether trimming all StimArrays in the project to a length of total_time_seconds would delete data
        """
        for stim_array in self.stim_arrays.values():
            new_n_timestamps = int(total_time_seconds / stim_array.dt)
            if (
                new_n_timestamps < stim_array.values.shape[0]
                and np.sum(stim_array.values[new_n_timestamps:, :]) > 0
            ):
                return True
        return False

    def load_NN(self) -> None:
        """Resolve `external_simulate_path` into `external_simulate_function`.

        `external_simulate_path` is either a filesystem path to a `.py` file
        (loaded via `runpy.run_path`) or a `neuroseq:<module_name>` sentinel
        resolving to `neuroseq.example_models.<module_name>.simulate`.
        """
        path = self.external_simulate_path
        if path is None:
            self.external_simulator = None
            self.external_simulate_function = None
            return
        if path.startswith("neuroseq:"):
            module_name = path[len("neuroseq:") :]
            if module_name == "elise_simulate":
                module_name = "elise"
            module = importlib.import_module(f"neuroseq.example_models.{module_name}")

            simulator = None
            if hasattr(module, "get_simulator"):
                simulator = module.get_simulator()
            elif hasattr(module, "NN"):
                simulator = module.NN()
            self.external_simulator = simulator

            if simulator is not None and hasattr(simulator, "simulate"):
                self.external_simulate_function = simulator.simulate
            elif hasattr(module, "simulate"):
                self.external_simulate_function = module.simulate
            else:
                raise AttributeError(
                    f"Module neuroseq.example_models.{module_name} has no simulator entry point"
                )
        else:
            self.external_simulator = None
            self.external_simulate_function = runpy.run_path(path)["simulate"]

    def _migrate_legacy_elise_stim(self) -> None:
        """Convert a single `u_inp` ELISE stim into an opsin-activation pair."""
        if self.external_simulate_path != "neuroseq:elise":
            return
        if "u_inp" not in self.stim_arrays:
            return
        if "opsin_exc" in self.stim_arrays or "opsin_inh" in self.stim_arrays:
            return

        params = self.external_simulate_params or {}
        E_exc = float(params.get("E_exc", {}).get("value", 0.0))
        E_inh = float(params.get("E_inh", {}).get("value", -75.0))
        old = self.stim_arrays["u_inp"]
        u_inp = np.asarray(old.values, dtype=np.float64)
        converted = _matching_potential_to_opsins(u_inp, E_exc, E_inh)
        if converted is None:
            self.logger.warning("Cannot migrate legacy u_inp stim: E_inh == E_exc")
            return
        g_exc_vals, g_inh_vals = converted
        del self.stim_arrays["u_inp"]
        self.stim_arrays["opsin_exc"] = StimArray(
            name="opsin_exc",
            stim_type=StimType.Excitation,
            values=g_exc_vals,
            units="1",
            dt=float(old.dt),
        )
        self.stim_arrays["opsin_inh"] = StimArray(
            name="opsin_inh",
            stim_type=StimType.Inhibition,
            values=g_inh_vals,
            units="1",
            dt=float(old.dt),
        )
        self.logger.info(
            "Migrated legacy ELISE 'u_inp' stim to "
            "'opsin_exc'/'opsin_inh' opsin-activation pair."
        )

    def _migrate_legacy_elise_kinetics_units(self) -> None:
        """Convert pre-kinetic ELISE opsin stims (values in [0,1]) to photon flux.

        Pre-kinetic archives store opsin activations as fractional
        conductances in [0, 1]. After the 3-state kinetic layer
        landed, the stim arrays are interpreted as photon flux density
        (mm^-2 s^-1) instead. Without migration, a 1.0 stim value
        would be ~17 orders of magnitude below the Hill K_M and the
        opsin would never open.

        Detection: external_simulate_path is ELISE, both opsin stims
        exist, the loaded archive predates schema_version 2, and the
        stim values are within [-2, 2] (allows small float noise above
        1.0). Migration uses the kinetic defaults from
        ``Elise.DEFAULT_PARAMS`` to invert the Hill — if a user had
        already retuned k_a_max etc., they'd have saved at
        schema_version >= 2 and this migration wouldn't fire.
        """
        if self.external_simulate_path != "neuroseq:elise":
            return
        if self._loaded_schema_version >= 2:
            return
        if "opsin_exc" not in self.stim_arrays or "opsin_inh" not in self.stim_arrays:
            return

        exc = self.stim_arrays["opsin_exc"]
        inh = self.stim_arrays["opsin_inh"]
        exc_max = float(np.max(np.abs(exc.values))) if exc.values.size else 0.0
        inh_max = float(np.max(np.abs(inh.values))) if inh.values.size else 0.0
        if exc_max > 2.0 or inh_max > 2.0:
            # Values already look like photon flux; leave alone.
            return

        from neuroseq.example_models.elise import DEFAULT_PARAMS
        from neuroseq.example_projects import _calibrate_phi_for_target_open_fraction

        def _side_kin(side: str) -> dict:
            return dict(
                k_a_max=float(DEFAULT_PARAMS[f"k_a_max_{side}"]["default"]),
                phi_m=float(DEFAULT_PARAMS[f"phi_m_{side}"]["default"]),
                k_d=float(DEFAULT_PARAMS[f"k_d_{side}"]["default"]),
                k_r=float(DEFAULT_PARAMS[f"k_r_{side}"]["default"]),
            )

        def _convert(values: np.ndarray, side: str) -> np.ndarray:
            kin = _side_kin(side)
            g0 = float(DEFAULT_PARAMS[f"g0_{side}"]["default"])
            # Old fractional conductance g_old corresponds to an open
            # fraction O_target = g_old / g0; invert the Hill to get φ.
            # The largest O reachable at finite flux is the saturating
            # steady-state O_sat = k_a_max·k_r / (k_a_max·k_r + k_a_max·k_d + k_d·k_r).
            # Legacy g values above g0·O_sat saturate: clamp O_target so
            # the inverse Hill stays finite. Migration is lossy here —
            # this is the documented trade-off for non-default archives.
            O_sat = (kin["k_a_max"] * kin["k_r"]) / (
                kin["k_a_max"] * kin["k_r"]
                + kin["k_a_max"] * kin["k_d"]
                + kin["k_d"] * kin["k_r"]
            )
            O_max = O_sat * (1.0 - 1e-3)  # 0.1% below saturation
            out = np.empty_like(values, dtype=np.float32)
            flat_in = values.ravel()
            flat_out = out.ravel()
            for i, g_old in enumerate(flat_in):
                # noqa naming: 'O' is the open-state fraction (per the 3-state spec).
                O_target = max(0.0, min(float(g_old) / g0, O_max))
                flat_out[i] = _calibrate_phi_for_target_open_fraction(O_target, **kin)
            return out

        exc.values = _convert(exc.values, "exc")
        inh.values = _convert(inh.values, "inh")
        exc.units = "mm**-2 / second"
        inh.units = "mm**-2 / second"
        self.logger.info(
            "Migrated pre-kinetic ELISE opsin stims (values in [0,1]) "
            "to photon-flux density (mm^-2 s^-1)."
        )

    def load_external_simulator(self) -> None:
        """Compatibility wrapper for loading external simulation entry points.

        This delegates to `load_NN`, which remains the canonical implementation.
        """
        self.load_NN()

    def external_simulate_params_flat(self) -> dict[str, object]:
        """Flatten `external_simulate_params` to `{name: value}` for the call site."""
        return {k: v["value"] for k, v in self.external_simulate_params.items()}

    def run_simulation(
        self,
        time_seconds: float,
        report: Callable | None = None,
        stim_array_padding_seconds: float = 0.0,
    ) -> None:
        self.pad_stim_arrays(
            self.b2_model.get_sim_time_secs()
            + time_seconds
            + stim_array_padding_seconds
        )
        # Stim arrays index the neuron axis (e.g. ``stimulus(t, i)``), so they
        # must be at least as wide as the model's neuron count. Widen with
        # trailing zero columns (= no stimulation) so increasing the neuron
        # count in the model editor doesn't crash with an out-of-bounds index.
        self.pad_stim_arrays_to_neuron_count(self.b2_model.get_defined_neuron_count())
        self.b2_model.run_simulation(time_seconds, self.stim_arrays, report)
        self.update_limits()

    def request_cancel_simulation(self) -> None:
        """Request that an in-progress simulation stop early."""
        self.b2_model.request_cancel()

    def save(
        self,
        filepath: str | pathlib.Path,
        options: SaveLoadOptions,
    ) -> None:
        """
        Saves the project to a zipfile which contains
            - and optional stim_arrays.json file representing the StimArrays
            - an optional b2_model.json file representing the B2Model
            - an optional pickle file for the stored_state containing the B2Model simulation checkpoints
            - an optional experimental_data.json file representing the experimental data
            - npy files for each numpy array in any TimedArray, StimArray or TraceData/SpikeData/SynapseData objects

        Parameters
        ----------
        filepath : str | pathlib.Path
            path to the file where the current project should be stored
        options: SaveLoadOptions
            options for what to include in the saved project
        """

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata: dict = {"dt": self.dt, "schema_version": 2}
            if self.external_simulate_path is not None:
                metadata["external_simulate_path"] = self.external_simulate_path
            if self.external_simulate_params:
                metadata["external_simulate_params"] = self.external_simulate_params
            if self.power_luts:
                metadata["power_luts"] = {
                    str(wavelength_nm): lut.to_serializable()
                    for wavelength_nm, lut in self.power_luts.items()
                }
            zf.writestr("project_metadata.json", json.dumps(metadata))
            include_simulation_data = (
                options.include_simulation_data and self.b2_model.has_simulation_data()
            )
            if options.include_stim_arrays:
                append_to_zip(self.stim_arrays, "stim_arrays", zf)
            if options.include_model:
                append_to_zip(
                    self.b2_model.to_dict(include_simulation_data), "b2_model", zf
                )
            if include_simulation_data:
                # store the current simulation data state to a temporary checkpoint to be able to restore it on load
                self.b2_model.network.store(save_store_name)
                zf.writestr(
                    "b2_stored_state.pkl",
                    pickle.dumps(self.b2_model.network._stored_state, protocol=4),
                )
            elif options.include_simulation_data:
                self.logger.info("Skipping simulation data save: no simulation data")
            # Todo: We could consider another option for cellpose data. I originally leaned towards that, but after merging  #388 and #390 this cellpose data may be in experimental data anyway, and seems inextricably related to it.
            if options.include_experimental_data:
                append_to_zip(self.experiments, "experimental_data", zf)
                if self.cellpose_data is not None:
                    append_to_zip(
                        {"cellpose_import": self.cellpose_data}, "cellpose_data", zf
                    )
        self.filepath = pathlib.Path(filepath)
