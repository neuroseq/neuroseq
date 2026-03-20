from __future__ import annotations

import logging
import math
import pathlib
import pickle
import zipfile
from collections.abc import Callable
from collections.abc import Sequence
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
from neuroseq.utils import append_data
from neuroseq.utils import find_by_name_and_type
from neuroseq.utils import save_store_name


class Project:
    def __init__(
        self,
        filepath: str | pathlib.Path | None = None,
        report: Callable[[int], None] | None = None,
    ) -> None:
        self.filepath = None
        self.b2_model: B2Model = B2Model()
        self.experiments: dict[str, Experiment] = {}
        self.stim_arrays: dict[str, StimArray] = {}
        self.cellpose_data: CellposeImportResult | None = None
        self.syscon2_export_template_text: str | None = None

        self.max_ram_synapse_plot_matrix = 256
        self.max_time_secs = 0.0
        self.max_neurons = 0

        self.logger = logging.getLogger(__name__)
        if filepath:
            self.filepath = pathlib.Path(filepath)
            self.logger.info(f"Loading project from '{filepath}'")
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
        self.stim_arrays[stim_array.name] = stim_array
        self.update_limits()

    def delete_stim_array(self, name: str):
        if name in self.stim_arrays:
            del self.stim_arrays[name]
        self.update_limits()

    def update_stim_array_properties(
        self, name: str, n_neurons: int, stim_type: StimType, units: str, dt: float
    ):
        stim_array = self.stim_arrays[name]
        stim_array.stim_type = stim_type
        stim_array.units = units
        stim_array.dt = dt
        if n_neurons < stim_array.values.shape[1]:
            self.logger.info(
                f"Decreasing neurons in StimArray {name} from {stim_array.values.shape[1]} to {n_neurons}"
            )
            stim_array.values = stim_array.values[:, :n_neurons]
        elif n_neurons > stim_array.values.shape[1]:
            self.logger.info(
                f"Increasing neurons in StimArray {name} from {stim_array.values.shape[1]} to {n_neurons}"
            )
            stim_array.values = np.pad(
                stim_array.values,
                ((0, 0), (0, n_neurons - stim_array.values.shape[1])),
                mode="constant",
                constant_values=0,
            )
        self.update_limits()

    def rename_stim_array(self, name: str, new_name: str):
        stim_array = self.stim_arrays.pop(name)
        stim_array.name = new_name
        self.stim_arrays[stim_array.name] = stim_array

    def update_limits(self):
        self.max_time_secs = self.b2_model.get_sim_time_secs()
        self.max_neurons = self.b2_model.get_max_neurons()
        for experiment in self.experiments.values():
            self.max_time_secs = max(self.max_time_secs, experiment.duration)
        for stim_array in self.stim_arrays.values():
            self.max_time_secs = max(
                self.max_time_secs, stim_array.dt * stim_array.values.shape[0]
            )
            self.max_neurons = max(self.max_neurons, int(stim_array.values.shape[1]))

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

    def pad_stim_arrays(self, total_time_seconds: float):
        """
        Pad all StimArrays in the project to a length of total_time_seconds
        """
        for name, stim_array in self.stim_arrays.items():
            if stim_array.values.ndim > 1:
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
            if stim_array.values.ndim > 1:
                n_timestamps = stim_array.values.shape[0]
                new_n_timestamps = int(total_time_seconds / stim_array.dt)
                if new_n_timestamps < n_timestamps:
                    self.logger.info(
                        f"Reducing {name} number of timestamps: {n_timestamps} -> {new_n_timestamps}"
                    )
                    self.stim_arrays[name].values = stim_array.values[
                        :new_n_timestamps, :
                    ]
        self.update_limits()

    def trim_stim_array_would_delete_data(self, total_time_seconds: float) -> bool:
        """
        Check whether trimming all StimArrays in the project to a length of total_time_seconds would delete data
        """
        for stim_array in self.stim_arrays.values():
            if stim_array.values.ndim > 1:
                new_n_timestamps = int(total_time_seconds / stim_array.dt)
                if (
                    new_n_timestamps < stim_array.values.shape[0]
                    and np.sum(stim_array.values[new_n_timestamps:, :]) > 0
                ):
                    return True
        return False

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
