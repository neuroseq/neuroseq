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
from neuroseq.utils import save_store_name


class Model:
    def __init__(
        self,
        filepath: str | pathlib.Path | None = None,
        report: Callable[[int], None] | None = None,
    ) -> None:
        self.filepath = None
        self.b2_model: B2Model = B2Model()
        self.experiments: dict[str, list[Data]] = {}
        self.stim_arrays: dict[str, StimArray] = {}
        self.cellpose_data: CellposeImportResult | None = None

        self.max_ram_synapse_plot_matrix = 256
        self.max_time_secs = 0.0
        self.max_neurons = 0

        self.logger = logging.getLogger(__name__)
        if filepath:
            self.filepath = pathlib.Path(filepath)
            self.logger.info(f"Loading model from '{filepath}'")
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
                cellpose_payload = load_from_zip("cellpose_data", zf)
                # todo: Check load_from_zip after merge and see if we need the isinstance check or if laod_from_zip handles a presence check, or can..
                # For now we will onyl keep it as a dict anyway, probably going forward also even if we support multiple cellpose imports at once.
                if isinstance(cellpose_payload, dict):
                    self.cellpose_data = cellpose_payload.get("cellpose_import")

        self.update_limits()

    def import_model(
        self,
        filepath: str | pathlib.Path,
        options: SaveLoadOptions,
        report: Callable[[int], None] | None = None,
    ) -> None:
        filepath = pathlib.Path(filepath)
        self.logger.info(f"Importing components from '{filepath}' into existing model")
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
        for experiment_name, data_list in self.experiments.items():
            for datum in data_list:
                if isinstance(datum, data_type):
                    data_names.append(f"{experiment_name} :: {datum.name}")
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
        Get experimental data from the model

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
        for data in self.experiments[experiment_name]:
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
            for data in experiment:
                if isinstance(data, TraceData):
                    self.max_time_secs = max(
                        self.max_time_secs, data.dt * data.traces.shape[1]
                    )
                    self.max_neurons = max(
                        self.max_neurons, int(np.max(data.indices) + 1)
                    )
        for stim_array in self.stim_arrays.values():
            self.max_time_secs = max(
                self.max_time_secs, stim_array.dt * stim_array.values.shape[0]
            )
            self.max_neurons = max(self.max_neurons, int(stim_array.values.shape[1]))

    def add_experimental_data(
        self,
        experiment_name: str,
        new_data: Sequence[TraceData | SpikeData | SynapseData],
        append: bool = False,
    ):
        """
        Add experimental data to the model

        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        new_data : Sequence[TraceData | SpikeData | SynapseData]
            List of data objects to add
        append : bool, optional
            Whether to append to existing data for the experiment, by default False
        """
        if not append:
            self.logger.info(
                f"Adding experimental data for experiment '{experiment_name}'"
            )
            self.experiments[experiment_name] = list(new_data)
            self.update_limits()
            return
        if experiment_name not in self.experiments:
            self.logger.warning(
                f"Append requested but experiment '{experiment_name}' does not exist yet - creating"
            )
            self.experiments[experiment_name] = []
        self.logger.info(
            f"Appending experimental data to experiment '{experiment_name}'"
        )
        # (for now) automatically get time offset when appending from existing trace data for this experiment
        existing_trace_data = next(
            (d for d in self.experiments[experiment_name] if isinstance(d, TraceData)),
            None,
        )
        if existing_trace_data is not None:
            time_offset_seconds = (
                existing_trace_data.dt * existing_trace_data.traces.shape[1]
            )
            self.logger.info(
                f"Setting time offset to {time_offset_seconds}s from existing trace data '{existing_trace_data.name}'"
            )
        else:
            self.logger.warning(
                f"No existing trace data found for experiment '{experiment_name}', setting time offset to 0s"
            )
            time_offset_seconds = 0.0
        # match data by type and name and append
        for data in new_data:
            existing_datum = next(
                (
                    d
                    for d in self.experiments[experiment_name]
                    if d.name == data.name and isinstance(d, type(data))
                ),
                None,
            )
            if existing_datum is not None:
                self.logger.info(f"  - appending experimental data to '{data.name}'")
                append_data(existing_datum, data, time_offset_seconds)
            else:
                self.logger.warning(
                    f"'{data.name}' not found when trying to append - adding new data instead of appending."
                )
                self.experiments[experiment_name].append(data)
        self.update_limits()

    def pad_stim_arrays(self, total_time_seconds: float):
        """
        Pad all StimArrays in the model to a length of total_time_seconds
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
        Trim all StimArrays in the model to a length of total_time_seconds
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
        Check whether trimming all StimArrays in the model to a length of total_time_seconds would delete data
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
        Saves the model to a zipfile which contains
            - and optional stim_arrays.json file representing the StimArrays
            - an optional b2_model.json file representing the B2Model
            - an optional pickle file for the stored_state containing the B2Model simulation checkpoints
            - an optional experimental_data.json file representing the experimental data
            - npy files for each numpy array in any TimedArray, StimArray or TraceData/SpikeData/SynapseData objects

        Parameters
        ----------
        filepath : str | pathlib.Path
            path to the file where the current model should be stored
        options: SaveLoadOptions
            options for what to include in the saved model
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
