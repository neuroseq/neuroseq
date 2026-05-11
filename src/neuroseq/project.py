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
            if not _dt_from_metadata:
                self.dt = self._derive_dt_from_data()
            if self.external_simulate_path is not None:
                try:
                    self.load_NN()
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
        self.b2_model.run_simulation(time_seconds, self.stim_arrays, report)
        self.update_limits()

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
            metadata: dict = {"dt": self.dt}
            if self.external_simulate_path is not None:
                metadata["external_simulate_path"] = self.external_simulate_path
            if self.external_simulate_params:
                metadata["external_simulate_params"] = self.external_simulate_params
            zf.writestr("project_metadata.json", json.dumps(metadata))
            if options.include_stim_arrays:
                append_to_zip(self.stim_arrays, "stim_arrays", zf)
            if options.include_model:
                append_to_zip(self.b2_model.to_dict(), "b2_model", zf)
            if options.include_simulation_data:
                # store the current simulation data state to a temporary checkpoint to be able to restore it on load
                self.b2_model.store(save_store_name)
                zf.writestr(
                    "b2_stored_state.pkl",
                    pickle.dumps(self.b2_model.network._stored_state, protocol=4),
                )
            # Todo: We could consider another option for cellpose data. I originally leaned towards that, but after merging  #388 and #390 this cellpose data may be in experimental data anyway, and seems inextricably related to it.
            if options.include_experimental_data:
                append_to_zip(self.experiments, "experimental_data", zf)
                if self.cellpose_data is not None:
                    append_to_zip(
                        {"cellpose_import": self.cellpose_data}, "cellpose_data", zf
                    )
        self.filepath = pathlib.Path(filepath)
