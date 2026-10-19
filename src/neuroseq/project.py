from __future__ import annotations

import importlib
import json
import logging
import math
import pathlib
import pickle
import runpy
import uuid
import zipfile
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import contextmanager
from copy import copy
from typing import Protocol
from typing import overload

import numpy as np

from neuroseq.b2_model import B2Model
from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.export.power_lut import PowerLut
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import resolve_stimulus_to_cellpose_mapping
from neuroseq.importers.importer_scanimage import compact_index
from neuroseq.importers.importer_suite2p import remap_raw_cellpose_ids
from neuroseq.io import append_to_zip
from neuroseq.io import load_from_zip
from neuroseq.project_history import ProjectCommand
from neuroseq.project_history import ProjectHistory
from neuroseq.project_history import StaleCommandError
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_timing import ensure_2d_stim_values
from neuroseq.stimulation.stim_timing import remap_stim_values_to_dt
from neuroseq.stimulation.stim_timing import seconds_to_samples_nearest
from neuroseq.stimulation.stim_timing import stim_dt_matches
from neuroseq.stimulation.stim_timing import validate_stim_dt_seconds
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


# Tolerance (seconds) when deciding whether a simulated sample falls within the
# stim timeline, so float representation error at the boundary does not drop a
# sample that lands exactly on the trimmed length.
_TIMELINE_CLIP_EPS_SECONDS = 1e-9


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
    """Neuron/stimulus-index-space count implied by ``cellpose_data`` alone.

    Before a stimulus-to-Cellpose mapping is chosen, every segmented ROI is
    assumed to become its own neuron (1:1), so the full segmentation count
    applies. Once a mapping picks a subset of ROIs to actually be neurons
    (e.g. "Custom with areas of interest"), the neuron/stimulus index space
    is the mapping's size, not the total number of segmented ROIs - the rest
    of the segmentation is only ever addressed by Cellpose id, never by
    neuron index, so counting it here would inflate every neuron-indexed
    plot (trace/stim/spike) with empty rows the mapping deliberately dropped.
    """
    if cellpose_data is None:
        return 0
    if cellpose_data.stimulus_to_cellpose_mapping is not None:
        return len(cellpose_data.stimulus_to_cellpose_mapping)
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
        # Whether this project has changes not yet written to filepath - used
        # to skip the "save before closing/discarding" prompts when there is
        # nothing to lose. Set by the mutating methods below and by callers
        # that change project state directly instead of through one of them
        # (see mark_dirty()'s docstring); cleared by save() and, defensively,
        # at the very end of this constructor regardless of how it got here.
        self.dirty = False
        self.history = ProjectHistory()
        self.history.subscribe(self._history_changed)
        self.filepath = None
        self.b2_model: B2Model = B2Model()
        self.project_name: str | None = None
        self.project_section: str | None = None
        self.experiments: dict[str, Experiment] = {}
        self.stim_arrays: dict[str, StimArray] = {}
        self.cellpose_data: CellposeImportResult | None = None
        # Each video: {"id", "name", "paths", "fps"} plus "run_id" for a
        # ScanImage recording, whose index lives in imaging_acquisitions.
        self.calcium_videos: list[dict] = []
        self.selected_calcium_video_id: str | None = None
        self.imaging_acquisitions: list[dict] = []
        self.syscon2_export_template_text: str | None = None
        self.power_luts: dict[int, PowerLut] = {}

        self.max_ram_synapse_plot_matrix = 256
        self.max_ram_trace_plot_matrix = 256
        self.max_time_secs = 0.0
        self.max_neurons = 0

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
                # "cellpose_data_v2" holds results in their raw orientation
                # (#601); older NeuroSeq versions do not read it, so they see no
                # segmentation instead of a mirrored one. Older archives keep
                # "cellpose_data", which io.py migrates on load.
                self.cellpose_data = (
                    load_from_zip("cellpose_data_v2", zf)
                    or load_from_zip("cellpose_data", zf)
                ).get("cellpose_import")
                if "project_metadata.json" in zf.namelist():
                    _meta = json.loads(zf.read("project_metadata.json"))
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
                    self.imaging_acquisitions = _meta.get("imaging_acquisitions", [])
                    for acquisition in self.imaging_acquisitions:
                        if "recording" in acquisition:
                            acquisition["recording"] = compact_index(
                                acquisition["recording"]
                            )
                        if acquisition.get("status") in {
                            "requested",
                            "starting",
                            "recording",
                            "waiting_for_first_frame",
                        }:
                            acquisition["status"] = "interrupted"
                    self._load_calcium_videos(_meta)
                    self._loaded_schema_version = int(_meta.get("schema_version", 1))
            if self.external_simulate_path is not None:
                try:
                    self.load_NN()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to resolve external_simulate_path "
                        f"{self.external_simulate_path!r}: {e}"
                    )
                    self.external_simulate_function = None
                else:
                    migrate_project = getattr(
                        self.external_simulator, "migrate_project", None
                    )
                    if callable(migrate_project):
                        # A failing migration leaves the archive data as
                        # stored but must not unload a working backend.
                        try:
                            migrate_project(self)
                        except Exception:
                            self.logger.exception(
                                f"External simulation backend migration failed "
                                f"for {self.external_simulate_path!r}"
                            )
            for experiment in self.experiments.values():
                self.max_time_secs = max(self.max_time_secs, experiment.duration)
            for stim_array in self.stim_arrays.values():
                self.max_time_secs = max(
                    self.max_time_secs, stim_array.dt * stim_array.values.shape[0]
                )

        self.update_limits()
        # Constructing a Project (fresh or loaded from filepath) always
        # yields a clean baseline, regardless of anything above that
        # happened to route through a mutating method.
        self.dirty = False

    def mark_dirty(self) -> None:
        """Flag this project as having changes not yet saved to filepath.

        Call this directly wherever project state is changed without going
        through one of the mutating methods below (e.g. gui.py assigning
        ``project.b2_model`` or ``project.cellpose_data`` after a dialog
        accepts new state).
        """
        self.dirty = True

    def _history_changed(self, event: str) -> None:
        if event != "clear":
            self.mark_dirty()

    def _experiment_restorer(
        self, names: Sequence[str]
    ) -> Callable[[Mapping[str, Experiment | None] | None], None]:
        """Retain affected data versions without duplicating large NumPy buffers.

        The mutators using this helper replace array buffers (append_data does
        this too). They must never write into an existing buffer in place.
        Preserve object identity for callers holding an experiment or dataset.

        ``restore(expected)`` refuses (raising :class:`StaleCommandError` before
        changing anything) when a group's current object is not the one the
        command left there, e.g. after a same-named group was added outside
        the history.
        """
        order = list(self.experiments)
        states = {}
        for name in names:
            if name not in self.experiments:
                continue
            experiment = self.experiments[name]
            saved = copy(experiment)
            saved.timepoints = list(experiment.timepoints)
            saved.data = list(experiment.data)
            states[name] = (
                experiment,
                saved,
                [(datum, vars(datum).copy()) for datum in experiment.data],
            )

        def restore(expected: Mapping[str, Experiment | None] | None = None) -> None:
            if expected is not None:
                for name in names:
                    if self.experiments.get(name) is not expected.get(name):
                        raise StaleCommandError(
                            f"Data group '{name}' was changed outside the undo history"
                        )
            for name in names:
                self.experiments.pop(name, None)
            for name, (experiment, saved, data_states) in states.items():
                experiment.name = saved.name
                experiment.duration = saved.duration
                experiment.timepoints = list(saved.timepoints)
                experiment.data = list(saved.data)
                for datum, attributes in data_states:
                    vars(datum).update(attributes)
                self.experiments[name] = experiment
            # Restore display order, retaining any unrelated entries.
            ordered_names = [name for name in order if name in self.experiments]
            ordered_names.extend(name for name in self.experiments if name not in order)
            self.experiments = {name: self.experiments[name] for name in ordered_names}
            self.update_limits()

        return restore

    @contextmanager
    def edit_experiments(self, names: Sequence[str], label: str) -> Iterator[None]:
        """Record the edits made to the named data groups as one undoable command.

        Edits inside the block must replace array buffers instead of writing
        into them (see :meth:`_experiment_restorer`). An exception rolls the
        groups back and records nothing.
        """
        names = list(dict.fromkeys(names))
        undo = self._experiment_restorer(names)
        before = {name: self.experiments.get(name) for name in names}
        was_dirty = self.dirty
        try:
            yield
        except Exception:
            undo(None)
            self.dirty = was_dirty
            raise
        else:
            redo = self._experiment_restorer(names)
            after = {name: self.experiments.get(name) for name in names}
            self.history.record_applied(
                ProjectCommand(label, lambda: redo(before), lambda: undo(after))
            )

    def remove_experiments(self, names: Sequence[str]) -> None:
        """Delete a selection of data groups as one undoable edit."""
        names = [name for name in dict.fromkeys(names) if name in self.experiments]
        if not names:
            return
        with self.edit_experiments(names, "Delete data groups"):
            for name in names:
                del self.experiments[name]
            self.update_limits()
            self.mark_dirty()

    @property
    def selected_calcium_video(self) -> dict | None:
        return next(
            (
                video
                for video in self.calcium_videos
                if video["id"] == self.selected_calcium_video_id
            ),
            None,
        )

    @property
    def calcium_video_paths(self) -> list[pathlib.Path]:
        video = self.selected_calcium_video
        return [pathlib.Path(p) for p in video["paths"]] if video else []

    @property
    def calcium_video_fps(self) -> float | None:
        video = self.selected_calcium_video
        return video["fps"] if video else None

    @property
    def calcium_video_channels(self) -> tuple[int, int]:
        """Interleaved channels of the selected video and the one shown
        (counted from 1, 0 for every frame)."""
        video = self.selected_calcium_video
        if video is None:
            return 1, 1
        return int(video.get("channels", 1)), int(video.get("shown_channel", 1))

    def set_calcium_video_channels(self, channels: int, shown_channel: int) -> None:
        video = self.selected_calcium_video
        if video is None:
            return
        if (video.get("channels", 1), video.get("shown_channel", 1)) == (
            channels,
            shown_channel,
        ):
            return
        video["channels"] = int(channels)
        video["shown_channel"] = int(shown_channel)
        self.mark_dirty()

    @property
    def calcium_scanimage_index(self) -> dict | None:
        video = self.selected_calcium_video
        if video is None:
            return None
        if "index" in video:
            return video["index"]
        return self.scanimage_recording(video.get("run_id"))

    def scanimage_recording(self, run_id: str | None) -> dict | None:
        if run_id is None:
            return None
        return next(
            (
                record["recording"]
                for record in self.imaging_acquisitions
                if record.get("run_id") == run_id and "recording" in record
            ),
            None,
        )

    def add_calcium_video(
        self,
        paths: list[pathlib.Path],
        fps: float | None,
        name: str | None = None,
        run_id: str | None = None,
        channels: int = 1,
        shown_channel: int = 1,
    ) -> dict:
        """Add a video and select it; returns the new entry."""
        base = name or (pathlib.Path(paths[0]).stem if paths else "Imaging")
        names = {video["name"] for video in self.calcium_videos}
        unique, suffix = base, 2
        while unique in names:
            unique, suffix = f"{base} ({suffix})", suffix + 1
        video: dict = {
            "id": uuid.uuid4().hex,
            "name": unique,
            "paths": [str(path) for path in paths],
            "fps": float(fps) if fps is not None else None,
        }
        if channels > 1:
            video["channels"] = int(channels)
            video["shown_channel"] = int(shown_channel)
        if run_id is not None:
            video["run_id"] = run_id
        self.calcium_videos.append(video)
        self.selected_calcium_video_id = video["id"]
        self.mark_dirty()
        return video

    def select_calcium_video(self, video_id: str | None) -> None:
        if video_id != self.selected_calcium_video_id:
            self.selected_calcium_video_id = video_id
            self.mark_dirty()

    def remove_calcium_video(self, video_id: str | None = None) -> None:
        """Remove a video (the selected one by default); select the last left."""
        video_id = video_id or self.selected_calcium_video_id
        self.calcium_videos = [v for v in self.calcium_videos if v["id"] != video_id]
        if self.selected_calcium_video_id == video_id:
            self.selected_calcium_video_id = (
                self.calcium_videos[-1]["id"] if self.calcium_videos else None
            )
        self.mark_dirty()

    def _load_calcium_videos(self, meta: dict) -> None:
        if "calcium_videos" in meta:
            self.calcium_videos = meta["calcium_videos"]
            for video in self.calcium_videos:
                if "index" in video:
                    video["index"] = compact_index(video["index"])
            self.selected_calcium_video_id = meta.get("selected_calcium_video_id")
            return
        # Projects saved before multiple videos stored a single selected one.
        index = meta.get("calcium_scanimage_index")
        recording = self.scanimage_recording((index or {}).get("run_id"))
        if recording is None and index is not None and "frames" in index:
            recording = compact_index(index)
        paths = meta.get("calcium_video_paths") or (recording or {}).get("paths", [])
        if not paths:
            return
        fps = meta.get("calcium_video_fps", (recording or {}).get("fps"))
        video = self.add_calcium_video(
            [pathlib.Path(p) for p in paths], float(fps) if fps is not None else None
        )
        if recording is not None:
            if self.scanimage_recording(recording.get("run_id")) is recording:
                video["run_id"] = recording["run_id"]
            else:
                video["index"] = recording

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
            # Replacing model/components is a new baseline, not a data command.
            self.history.clear()
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
        self.mark_dirty()

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
            # The stim array is the authoritative timeline: simulated data past
            # the longest stim array is windowed out so trimming a stim also
            # trims its traces/spikes/synapses. This is non-destructive (the
            # Brian2 monitors keep the full recording), so undoing the trim
            # restores the simulated data along with the stim length.
            if data_type is TraceData:
                return self._clip_trace_to_timeline(
                    self.b2_model.get_simulation_trace_data(
                        name, self.max_ram_trace_plot_matrix
                    )
                )
            if data_type is SpikeData:
                return self._clip_spike_to_timeline(
                    self.b2_model.get_simulation_spike_data(name)
                )
            if data_type is SynapseData:
                return self._clip_synapse_to_timeline(
                    self.b2_model.get_simulation_synapse_data(
                        name, self.max_ram_synapse_plot_matrix
                    )
                )
        if experiment_name not in self.experiments:
            self.logger.info(f"  - experiment '{experiment_name}' not found")
            return None
        for data in self.experiments[experiment_name].data:
            if data.name == data_name and isinstance(data, data_type):
                return data
        self.logger.info(f"  - data '{data_name}' not found")
        return None

    def simulation_timeline_seconds(self) -> float:
        """Duration of the authoritative stim timeline in seconds.

        The longest stim array defines the project's timeline; simulated data
        is windowed to this length (see :meth:`get_data`) so trimming a stim
        array also trims the simulated traces/spikes/synapses that extend past
        it. Returns ``0.0`` when there are no stim arrays, in which case
        simulated data is returned unclipped (there is no timeline to clip to).
        """
        longest = 0.0
        for stim_array in self.stim_arrays.values():
            longest = max(
                longest, float(stim_array.dt) * float(stim_array.values.shape[0])
            )
        return longest

    def _clip_trace_to_timeline(self, data: TraceData | None) -> TraceData | None:
        """Window a simulated trace to the stim timeline (non-destructive)."""
        if data is None:
            return None
        timeline = self.simulation_timeline_seconds()
        if timeline <= 0.0 or data.dt <= 0.0:
            return data
        n_time = int(data.traces.shape[1])
        n_keep = seconds_to_samples_nearest(timeline, data.dt)
        if 1 <= n_keep < n_time:
            data.traces = data.traces[:, :n_keep]
        return data

    def _clip_spike_to_timeline(self, data: SpikeData | None) -> SpikeData | None:
        """Window simulated spikes to the stim timeline (non-destructive)."""
        if data is None:
            return None
        timeline = self.simulation_timeline_seconds()
        if timeline <= 0.0:
            return data
        keep = data.spike_times <= timeline + _TIMELINE_CLIP_EPS_SECONDS
        if not bool(np.all(keep)):
            data.spike_times = data.spike_times[keep]
            data.spike_indices = data.spike_indices[keep]
        return data

    def _clip_synapse_to_timeline(self, data: SynapseData | None) -> SynapseData | None:
        """Window simulated synapse data to the stim timeline (non-destructive)."""
        if data is None:
            return None
        timeline = self.simulation_timeline_seconds()
        if timeline <= 0.0:
            return data
        keep = data.timestamps <= timeline + _TIMELINE_CLIP_EPS_SECONDS
        if not bool(np.all(keep)):
            data.timestamps = data.timestamps[keep]
            data.values = data.values[:, :, keep]
        return data

    def add_stim_array(self, stim_array: StimArray):
        stim_dt = validate_stim_dt_seconds(stim_array.dt)
        stim_array.values = ensure_2d_stim_values(stim_array.values)
        stim_array.dt = stim_dt
        self.history.clear()
        self.stim_arrays[stim_array.name] = stim_array
        self.update_limits()
        self.mark_dirty()

    def delete_stim_array(self, name: str):
        if name in self.stim_arrays:
            self.history.clear()
            del self.stim_arrays[name]
        self.update_limits()
        self.mark_dirty()

    def update_stim_array_properties(
        self, name: str, n_neurons: int, stim_type: StimType, units: str, dt: float
    ):
        dt = validate_stim_dt_seconds(dt)
        stim_array = self.stim_arrays[name]
        old_dt = validate_stim_dt_seconds(stim_array.dt)
        if not stim_dt_matches(old_dt, dt) or n_neurons != stim_array.values.shape[1]:
            self.history.clear()
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
                f"Decreasing neurons in StimArray {name} from {current_n_neurons} to "
                f"{n_neurons} - trimming the rest of the project to match "
                "(same behavior as Trim to Mapped Neurons)"
            )
            self._trim_to_neuron_ids(list(range(n_neurons)))
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
        self.mark_dirty()

    def replace_stim_values(
        self,
        name: str,
        values: np.ndarray,
        dt: float | None = None,
        label: str | None = None,
    ) -> None:
        """Replace a stim array's whole sample grid as one undoable edit.

        Use this instead of assigning ``values``/``dt`` directly: slice-based
        stimulation commands restore samples by index, so an unrecorded grid
        change would make them write into the wrong samples. Swapping the whole
        buffer (never writing into it) keeps every earlier command valid, because
        undo restores exactly the grid those commands were recorded on.
        """
        stim_array = self.stim_arrays[name]
        new_values = ensure_2d_stim_values(values)
        old_values = stim_array.values
        if new_values is old_values:
            new_values = new_values.copy()
        old_dt = validate_stim_dt_seconds(stim_array.dt)
        new_dt = old_dt if dt is None else validate_stim_dt_seconds(dt)

        def apply(applied_values: np.ndarray, applied_dt: float) -> None:
            stim_array.values = applied_values
            stim_array.dt = applied_dt
            self.update_limits()

        apply(new_values, new_dt)
        self.history.record_applied(
            ProjectCommand(
                label or f"Replace input {name}",
                lambda: apply(new_values, new_dt),
                lambda: apply(old_values, old_dt),
            )
        )
        self.mark_dirty()

    def rename_stim_array(self, name: str, new_name: str):
        self.history.clear()
        stim_array = self.stim_arrays.pop(name)
        stim_array.name = new_name
        self.stim_arrays[stim_array.name] = stim_array
        self.mark_dirty()

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
        with self.edit_experiments([experiment_name], "Append experimental data"):
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
            self.mark_dirty()

    def remap_cellpose_indices(self) -> list[str]:
        """
        Re-resolve ``TraceData.indices``/``SpikeData.spike_indices`` for every
        Suite2p-imported experiment against the current ``cellpose_data`` mapping.

        Trace/spike data carries its raw, pre-mapping Cellpose ROI id
        (``raw_cellpose_ids``) alongside the already-resolved ``indices``/
        ``spike_indices``. Whenever ``cellpose_data``'s mapping changes, call this to
        keep them in sync instead of leaving the old resolution in place.

        Returns
        -------
        list[str]
            Names of experiments whose data could not be remapped (e.g. the current
            mapping collides for their raw ids) - those are left unchanged rather than
            raising, so the caller can surface a warning.
        """
        self.history.clear()
        mapping = resolve_stimulus_to_cellpose_mapping(self.cellpose_data)
        failed: list[str] = []
        for name, experiment in self.experiments.items():
            for datum in experiment.data:
                if not isinstance(datum, TraceData | SpikeData):
                    continue
                if datum.raw_cellpose_ids is None:
                    continue
                try:
                    new_ids = remap_raw_cellpose_ids(datum.raw_cellpose_ids, mapping)
                except ValueError:
                    self.logger.warning(
                        f"Could not remap trace/spike indices for experiment "
                        f"'{name}' under the current Cellpose mapping",
                        exc_info=True,
                    )
                    failed.append(name)
                    continue
                if isinstance(datum, TraceData):
                    datum.indices = new_ids
                else:
                    datum.spike_indices = new_ids
        self.mark_dirty()
        return failed

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
        with self.edit_experiments([experiment.name], "Add data group"):
            for t in [0.0, experiment.duration]:
                if t not in experiment.timepoints:
                    experiment.timepoints.append(t)
            self.experiments[experiment.name] = experiment
            self.update_limits()
            self.mark_dirty()

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

        with self.edit_experiments([experiment_name], "Add dataset"):
            experiment.data.append(data)
            if isinstance(data, TraceData):
                new_duration = float(data.traces.shape[1] * data.dt)
                if new_duration > experiment.duration:
                    experiment.duration = new_duration
                    if new_duration not in experiment.timepoints:
                        experiment.timepoints.append(new_duration)
                    experiment.timepoints.sort()
            self.update_limits()
            self.mark_dirty()

    def pad_stim_arrays(
        self, total_time_seconds: float, only_stim_name: str | None = None
    ):
        """
        Pad all StimArrays in the project to a length of total_time_seconds.

        A selected array is padded to the nearest sample, for duration editing.
        """
        for name, stim_array in self.stim_arrays.items():
            if only_stim_name is not None and name != only_stim_name:
                continue
            n_timestamps = stim_array.values.shape[0]
            array_time_secs = stim_array.dt * n_timestamps
            padding_required_secs = total_time_seconds - array_time_secs
            if padding_required_secs > 0:
                n_pad = (
                    max(
                        0,
                        seconds_to_samples_nearest(total_time_seconds, stim_array.dt)
                        - n_timestamps,
                    )
                    if only_stim_name is not None
                    else math.ceil(padding_required_secs / stim_array.dt)
                )
                self.logger.info(
                    f"Adding {padding_required_secs}s / {n_pad} timestamps of padding to {name}: {stim_array.values.shape}"
                )
                self.stim_arrays[name].values = np.pad(
                    stim_array.values, pad_width=((0, n_pad), (0, 0))
                )
        self.update_limits()
        self.mark_dirty()

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
        if any(arr.values.shape[1] < n_neurons for arr in self.stim_arrays.values()):
            self.history.clear()
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
        self.mark_dirty()

    def resize_stim_arrays_to_neuron_count(self, n_neurons: int) -> None:
        """
        Resize every StimArray along the neuron axis to *exactly* ``n_neurons``
        columns: grow with trailing zero columns or trim surplus columns. Used
        when the model's neuron count is reduced so the stim arrays follow the
        network size back down (``pad_stim_arrays_to_neuron_count`` only grows).

        Parameters
        ----------
        n_neurons : int
            Exact number of neuron columns to resize to.
        """
        if n_neurons <= 0:
            return
        if any(arr.values.shape[1] != n_neurons for arr in self.stim_arrays.values()):
            self.history.clear()
        for name, stim_array in self.stim_arrays.items():
            current_n_neurons = int(stim_array.values.shape[1])
            if current_n_neurons == n_neurons:
                continue
            self.logger.info(
                f"Resizing StimArray {name} from {current_n_neurons} to "
                f"{n_neurons} neurons: {stim_array.values.shape}"
            )
            if current_n_neurons < n_neurons:
                stim_array.values = np.pad(
                    stim_array.values,
                    ((0, 0), (0, n_neurons - current_n_neurons)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                stim_array.values = stim_array.values[:, :n_neurons]
        self.update_limits()
        self.mark_dirty()

    def _trim_to_neuron_ids(self, kept_ids: list[int]) -> None:
        """Keep only the given neuron/stim ids across every StimArray and
        every experiment's TraceData/SpikeData/SynapseData in the project,
        re-indexing the kept ones to a contiguous ``0..n_kept-1`` numbering
        (in the order given). Also drops/renumbers any
        ``cellpose_data.stimulus_to_cellpose_mapping`` entry the same way,
        so a neuron dropped here can't leave a dangling mapping entry
        behind. Does not call `update_limits()` - callers do that once
        they're done (e.g. after also renaming/resizing the array that
        triggered the trim).

        Shared core behind `trim_to_mapped_neurons()` (kept_ids = the
        Cellpose-mapped stim ids) and `update_stim_array_properties()`'s
        neuron-count decrease (kept_ids = ``range(new_n_neurons)``) - both
        need the same project-wide consistency, not just a resize of the
        one array/dataset that triggered it.
        """
        self.history.clear()
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
        kept_ids_set = set(kept_ids)

        for name, stim_array in self.stim_arrays.items():
            n_neurons = int(stim_array.values.shape[1])
            valid_ids = [i for i in kept_ids if i < n_neurons]
            if len(valid_ids) == n_neurons:
                continue
            self.logger.info(
                f"Trimming StimArray {name} from {n_neurons} to "
                f"{len(valid_ids)} neurons"
            )
            stim_array.values = stim_array.values[:, valid_ids]

        for exp_name, experiment in self.experiments.items():
            for datum in experiment.data:
                if isinstance(datum, TraceData):
                    keep_mask = np.isin(datum.indices, kept_ids)
                    if keep_mask.all():
                        continue
                    self.logger.info(
                        f"Trimming TraceData '{exp_name} :: {datum.name}' from "
                        f"{datum.traces.shape[0]} to {int(keep_mask.sum())} neurons"
                    )
                    datum.traces = datum.traces[keep_mask]
                    datum.indices = np.asarray(
                        [old_to_new[int(i)] for i in datum.indices[keep_mask]],
                        dtype=np.uint32,
                    )
                    if datum.raw_cellpose_ids is not None:
                        datum.raw_cellpose_ids = datum.raw_cellpose_ids[keep_mask]
                elif isinstance(datum, SpikeData):
                    keep_mask = np.isin(datum.spike_indices, kept_ids)
                    if keep_mask.all():
                        continue
                    self.logger.info(
                        f"Trimming SpikeData '{exp_name} :: {datum.name}': dropping "
                        f"{int((~keep_mask).sum())} spikes from removed neurons"
                    )
                    datum.spike_times = datum.spike_times[keep_mask]
                    datum.spike_indices = np.asarray(
                        [old_to_new[int(i)] for i in datum.spike_indices[keep_mask]],
                        dtype=np.uint32,
                    )
                    if datum.raw_cellpose_ids is not None:
                        datum.raw_cellpose_ids = datum.raw_cellpose_ids[keep_mask]
                elif isinstance(datum, SynapseData):
                    if datum.values.ndim < 2:
                        continue
                    n_pre, n_post = datum.values.shape[0], datum.values.shape[1]
                    if n_pre != n_post:
                        self.logger.warning(
                            f"Skipping SynapseData '{exp_name} :: {datum.name}': "
                            f"non-square neuron axes ({n_pre} x {n_post})"
                        )
                        continue
                    valid_ids = [i for i in kept_ids if i < n_pre]
                    if len(valid_ids) == n_pre:
                        continue
                    self.logger.info(
                        f"Trimming SynapseData '{exp_name} :: {datum.name}' from "
                        f"{n_pre} to {len(valid_ids)} neurons"
                    )
                    n_timesteps = datum.values.shape[2]
                    valid_ids_arr = np.asarray(valid_ids, dtype=np.int64)
                    datum.values = datum.values[
                        np.ix_(valid_ids_arr, valid_ids_arr, np.arange(n_timesteps))
                    ]

        if (
            self.cellpose_data is not None
            and self.cellpose_data.stimulus_to_cellpose_mapping
        ):
            self.cellpose_data.stimulus_to_cellpose_mapping = {
                old_to_new[old_stim_id]: cellpose_id
                for old_stim_id, cellpose_id in (
                    self.cellpose_data.stimulus_to_cellpose_mapping.items()
                )
                if old_stim_id in kept_ids_set
            }

    def trim_to_mapped_neurons(self) -> None:
        """
        Permanently keep only the neurons that have a Cellpose cell mapped to
        them (per ``cellpose_data.stimulus_to_cellpose_mapping``), discarding
        every other neuron from every StimArray and every experiment's
        TraceData/SpikeData/SynapseData. The kept neurons are re-indexed to a
        contiguous ``0..n_kept-1`` numbering (in mapping order), and
        ``stimulus_to_cellpose_mapping`` is rewritten to match.

        This is a one-way, in-memory change - the caller is responsible for
        having the result saved under a new file name rather than silently
        overwriting anything the trim discarded data from.

        Raises
        ------
        ValueError
            If there is no Cellpose data, or its stimulus mapping is empty
            (nothing to trim to).
        """
        if (
            self.cellpose_data is None
            or not self.cellpose_data.stimulus_to_cellpose_mapping
        ):
            raise ValueError(
                "No Cellpose stimulus mapping to trim to - import Cellpose "
                "data with a mapping first."
            )
        kept_stim_ids = sorted(self.cellpose_data.stimulus_to_cellpose_mapping.keys())
        self._trim_to_neuron_ids(kept_stim_ids)
        self.update_limits()
        self.mark_dirty()

    def trim_stim_arrays(self, total_time_seconds: float):
        """
        Trim all StimArrays in the project to a length of total_time_seconds
        """
        for name, stim_array in self.stim_arrays.items():
            n_timestamps = stim_array.values.shape[0]
            new_n_timestamps = seconds_to_samples_nearest(
                total_time_seconds, stim_array.dt
            )
            if new_n_timestamps < n_timestamps:
                self.logger.info(
                    f"Reducing {name} number of timestamps: {n_timestamps} -> {new_n_timestamps}"
                )
                self.stim_arrays[name].values = stim_array.values[:new_n_timestamps, :]
        self.update_limits()
        self.mark_dirty()

    def loop_stim_arrays(self, n_loops: int) -> None:
        """Tile every stim array's values along the time axis."""
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        if n_loops == 1:
            return
        self.history.clear()
        for stim_array in self.stim_arrays.values():
            stim_array.values = np.tile(stim_array.values, (n_loops, 1))
        self.update_limits()
        self.mark_dirty()

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
        self.mark_dirty()

    def trim_stim_array_would_delete_data(self, total_time_seconds: float) -> bool:
        """
        Check whether trimming all StimArrays in the project to a length of total_time_seconds would delete data
        """
        for stim_array in self.stim_arrays.values():
            new_n_timestamps = seconds_to_samples_nearest(
                total_time_seconds, stim_array.dt
            )
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
        self.mark_dirty()

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
            metadata: dict = {"schema_version": 2}
            if self.external_simulate_path is not None:
                metadata["external_simulate_path"] = self.external_simulate_path
            if self.external_simulate_params:
                metadata["external_simulate_params"] = self.external_simulate_params
            if self.power_luts:
                metadata["power_luts"] = {
                    str(wavelength_nm): lut.to_serializable()
                    for wavelength_nm, lut in self.power_luts.items()
                }
            if self.calcium_videos:
                metadata["calcium_videos"] = self.calcium_videos
                metadata["selected_calcium_video_id"] = self.selected_calcium_video_id
            if self.calcium_video_paths:
                # Lets earlier NeuroSeq versions still open the selected video.
                metadata["calcium_video_paths"] = [
                    str(path) for path in self.calcium_video_paths
                ]
                metadata["calcium_video_fps"] = self.calcium_video_fps
            if self.imaging_acquisitions:
                metadata["imaging_acquisitions"] = self.imaging_acquisitions
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
                        {"cellpose_import": self.cellpose_data}, "cellpose_data_v2", zf
                    )
        self.filepath = pathlib.Path(filepath)
        self.dirty = False
