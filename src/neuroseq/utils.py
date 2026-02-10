from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from collections.abc import Iterable
from enum import Enum
from typing import Any
from typing import TypeVar

import brian2 as b2
import brian2.numpy_ as np
from cmap import Colormap
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.profiling import profile_section
from neuroseq.stim import StimType

save_store_name = "___current-simulation-state-at-save-time___"


class InteractMode(Enum):
    DRAW = 1
    SELECT = 2


def timed_array_with_new_data(
    timed_array: b2.TimedArray, new_data: np.ndarray
) -> b2.TimedArray:
    return b2.TimedArray(
        b2.Quantity(new_data, timed_array.dim), timed_array.dt * b2.second
    )


def get_downsample_factor(
    n_floats: int, max_ram_mb: int, bits_per_float: int = 64
) -> int:
    """Calculate the downsample factor for a given number of floats and maximum allowed RAM usage in MB."""
    if n_floats < 0 or max_ram_mb <= 0:
        return 1
    mb_in_bits = 1024 * 1024 * 8
    mbits_per_float = float(bits_per_float) / mb_in_bits
    ram_mb = n_floats * mbits_per_float
    return 1 + int(ram_mb // max_ram_mb)


def construct_synapse_matrix(
    monitor: b2.StateMonitor, variable: str, max_ram_mb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs n x n x t timeseries of synapse matrices and timestamps from a Brian2 StateMonitor of synapse variables.

    Where n is the largest index of any source or target neuron + 1, and t is the number of timestamps.

    Automatically downsamples the data if it exceeds the maximum RAM limit.
    """
    with profile_section("synapse.construct_matrix"):
        synapse_group = monitor.source
        values = monitor.variables[variable].get_value().T
        n = int(max(np.max(synapse_group.i), np.max(synapse_group.j)) + 1)
        n_timestamps = int(values.shape[1])
        if n_timestamps == 0:
            return np.array([]), np.array([])
        n_floats = n * n * n_timestamps
        downsample_factor = get_downsample_factor(
            n_floats, max_ram_mb, bits_per_float=32
        )
        if downsample_factor > 1:
            logging.info(
                f"Downsampling {n}x{n}x{n_timestamps} matrix by factor {downsample_factor} due to max RAM limit of {max_ram_mb} MB"
            )
        matrix = np.zeros(
            (n, n, values[0, ::downsample_factor].shape[0]),
            dtype=np.float32,
        )
        for i, j, value in zip(
            synapse_group.i,
            synapse_group.j,
            values[:, ::downsample_factor],
            strict=True,
        ):
            matrix[i, j, :] = value
        timestamps = (monitor.t / b2.second)[::downsample_factor]
        return matrix, timestamps


def neuron_indices_to_colors(indices: np.ndarray) -> np.ndarray:
    """Transform the supplied neuron indices to colors in a fixed but psuedo-random order.
    Such that a given index always maps to the same color, but nearby indices map to different colors.
    """
    colormap = Colormap("matlab:hsv")
    # these values of a and m taken from table 2 of https://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
    a = 23234
    m = 32749
    return colormap(np.mod(np.asarray(indices) * a, m) / m)


def init_brian2_logging(log_level: str = "ERROR") -> None:
    b2.prefs["logging"]["console_log_level"] = log_level
    b2.prefs["logging"]["display_brian_error_message"] = False
    b2.prefs["logging"]["file_log"] = False
    b2.prefs["logging"]["save_script"] = False
    b2.prefs["logging"]["std_redirection"] = False
    b2.BrianLogger.initialize()


def with_wait_cursor(func: Callable) -> Callable:
    """Decorator to show a wait cursor while a function is running."""

    @functools.wraps(func)
    def with_wait_cursor_wrapper(*args: Any, **kwargs: Any):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        return_value = func(*args, **kwargs)
        QtWidgets.QApplication.restoreOverrideCursor()
        return return_value

    return with_wait_cursor_wrapper


def get_primary_available_geometry() -> QtCore.QRect | None:
    screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
    if screen is None:
        screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return None
    return screen.availableGeometry()


def is_vertical_screen(available: QtCore.QRect | None, ratio: float = 0.8) -> bool:
    if available is None or available.height() <= 0:
        return False
    return available.width() < ratio * available.height()


def add_action(
    name: str,
    callback: Callable,
    menu: QtWidgets.QMenu | None = None,
    toolbar: QtWidgets.QToolBar | None = None,
    widget: QtWidgets.QWidget | None = None,
    shortcut: str | None = None,
    icon: QtGui.QIcon | None = None,
    checkable: bool = False,
    is_checked: bool = False,
) -> QtWidgets.QAction:
    parent = menu or toolbar
    action = QtWidgets.QAction(name, parent)
    action.setCheckable(checkable)
    if checkable:
        action.setChecked(is_checked)
    action.triggered.connect(callback)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
    if menu is not None:
        menu.addAction(action)
    if toolbar is not None:
        toolbar.addAction(action)
    if widget is not None:
        widget.addAction(action)
    return action


def add_items_to_combobox(combobox: QtWidgets.QComboBox, items: list[str]) -> None:
    """Add items to a combobox and adjust its width to avoid contents getting truncated on Windows."""
    if len(items) == 0:
        return
    combobox.addItems(items)
    longest_item = max(items, key=len)
    font_metrics = QtGui.QFontMetrics(combobox.font())
    max_width = font_metrics.boundingRect(longest_item).width()
    combobox.view().setMinimumWidth(max_width + 20)


def update_nested_dict(original: dict[Any, Any], updates: Any) -> None:
    """
    update_nested_dict Update nested dictionary with another dictionary. Paths within dictionaries
    are assumed to be unique. Violation of this premise can result in unpredictable behavior

    Parameters
    ----------
    original : dict
        dictionary to be updated
    updates : dict
        dictionary to update with
    """
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            update_nested_dict(original[key], value)
        else:
            original[key] = value


def safe_concat(arrays: list[np.ndarray], axis=0) -> np.ndarray:
    """
    Concatenate a list of NumPy arrays along the given axis, ignoring any arrays that are empty (size == 0).
    """
    return np.concatenate([a for a in arrays if a.size > 0], axis=axis)


def append_data(
    existing_data: TraceData | SpikeData | SynapseData,
    new_data: TraceData | SpikeData | SynapseData,
    data_duration_seconds: float,
    time_offset_seconds: float,
):
    """Append new data in-place to the existing data, overwriting the existing data if the time ranges overlap.

    Parameters
    ----------
    existing_data : TraceData | SpikeData | SynapseData
        Existing data to append to.
    new_data : TraceData | SpikeData | SynapseData
        New trace or spike data to append.
    data_duration_seconds: float
        Duration in seconds of the new data being appended.
    time_offset_seconds: float
        Time offset in seconds to apply to the timestamps in the new data when appending

    Raises
    -------
    ValueError
        If existing and new data are not compatible.
    TypeError
        If existing_data and new_data are not of the same type.
    """
    if type(existing_data) is not type(new_data):
        raise TypeError("Existing data and new data must be of the same type.")
    if isinstance(existing_data, TraceData) and isinstance(new_data, TraceData):
        if existing_data.dt != new_data.dt:
            raise ValueError(
                f"Cannot append TraceData with different dt: {existing_data.dt} != {new_data.dt}"
            )
        existing_ids = np.array(existing_data.indices, copy=False)
        new_ids = np.array(new_data.indices, copy=False)
        existing_id_set = set(map(int, existing_ids))
        new_id_set = set(map(int, new_ids))
        overlap_ids = existing_id_set.intersection(new_id_set)
        disjoint = not overlap_ids

        # when there is no crossover in neuron ids, we only allow a full timeline
        if disjoint:
            # no crossover means we only allow a full timeline match
            existing_duration = existing_data.traces.shape[1] * existing_data.dt
            new_duration = new_data.traces.shape[1] * new_data.dt
            if time_offset_seconds != 0.0 or not np.isclose(
                new_duration, existing_duration
            ):
                raise ValueError(
                    "Cannot merge TraceData with non-overlapping neuron IDs unless the timelines fully overlap (offset 0, equal duration)."
                )
            existing_data.traces = safe_concat(
                [existing_data.traces, new_data.traces], axis=0
            )
            existing_data.indices = safe_concat(
                [existing_data.indices, new_data.indices], axis=0
            )
            return

        if existing_id_set != new_id_set:
            raise ValueError(
                "Cannot append TraceData when neuron IDs partially overlap. Either replace the same set of neurons or import a fully disjoint set on the same timeline."
            )

        if not np.array_equal(existing_ids, new_ids):
            # reorder incoming traces to match existing indices for safe replacement
            index_map = {int(idx): i for i, idx in enumerate(new_ids)}
            new_order = [index_map[int(idx)] for idx in existing_ids]
            new_traces = new_data.traces[new_order, :]
        else:
            new_traces = new_data.traces

        idx0 = int(time_offset_seconds / existing_data.dt)
        idx1 = int((time_offset_seconds + data_duration_seconds) / existing_data.dt)
        existing_data.traces = safe_concat(
            [
                existing_data.traces[:, 0:idx0],
                new_traces,
                existing_data.traces[:, idx1:],
            ],
            axis=1,
        )
    elif isinstance(existing_data, SpikeData) and isinstance(new_data, SpikeData):
        existing_ids = set(map(int, existing_data.spike_indices))
        new_ids = set(map(int, new_data.spike_indices))
        overlap_ids = existing_ids.intersection(new_ids)

        if overlap_ids and existing_ids != new_ids:
            raise ValueError(
                "Cannot append SpikeData when neuron IDs partially overlap. Either replace the same set of neurons or import a fully disjoint set on the same timeline."
            )

        if not existing_ids:
            existing_data.spike_indices = safe_concat(
                [existing_data.spike_indices, new_data.spike_indices], axis=0
            )
            existing_data.spike_times = safe_concat(
                [
                    existing_data.spike_times,
                    new_data.spike_times + time_offset_seconds,
                ],
                axis=0,
            )
            if existing_data.spike_times.size > 1:
                order = np.argsort(existing_data.spike_times)
                existing_data.spike_times = existing_data.spike_times[order]
                existing_data.spike_indices = existing_data.spike_indices[order]
            return

        # find the index of the last spike time in the existing data before the time offset
        idx0 = np.searchsorted(
            existing_data.spike_times, time_offset_seconds, side="left"
        )
        # find the index of the first spike time in the existing data after the time offset + duration
        idx1 = np.searchsorted(
            existing_data.spike_times,
            time_offset_seconds + data_duration_seconds,
            side="left",
        )
        existing_data.spike_indices = safe_concat(
            [
                existing_data.spike_indices[0:idx0],
                new_data.spike_indices,
                existing_data.spike_indices[idx1:],
            ],
            axis=0,
        )
        existing_data.spike_times = safe_concat(
            [
                existing_data.spike_times[0:idx0],
                new_data.spike_times + time_offset_seconds,
                existing_data.spike_times[idx1:],
            ],
            axis=0,
        )
    elif isinstance(existing_data, SynapseData) and isinstance(new_data, SynapseData):
        # find the index of the last timestamp in the existing data before the time offset
        idx0 = np.searchsorted(
            existing_data.timestamps, time_offset_seconds, side="left"
        )
        # find the index of the first timestamp in the existing data after the time offset + duration
        idx1 = np.searchsorted(
            existing_data.timestamps,
            time_offset_seconds + data_duration_seconds,
            side="left",
        )
        existing_data.values = safe_concat(
            [
                existing_data.values[0:idx0],
                new_data.values,
                existing_data.values[idx1:],
            ],
            axis=2,
        )
        existing_data.timestamps = safe_concat(
            [
                existing_data.timestamps[0:idx0],
                new_data.timestamps + time_offset_seconds,
                existing_data.timestamps[idx1:],
            ],
            axis=0,
        )


def remove_neurons_from_data(
    data: TraceData | SpikeData, neuron_indices_to_remove: set[int]
) -> None:
    """Remove specific neurons from TraceData or SpikeData in-place.

    Parameters
    ----------
    data : TraceData | SpikeData
        Data to remove neurons from
    neuron_indices_to_remove : set[int]
        Set of neuron indices to remove
    """
    if isinstance(data, TraceData):
        # Find which positions in the arrays correspond to neurons we want to keep
        indices_to_keep = []
        for i, neuron_idx in enumerate(data.indices):
            if neuron_idx not in neuron_indices_to_remove:
                indices_to_keep.append(i)

        if indices_to_keep:
            # Keep only the specified neuron traces and indices
            data.traces = data.traces[indices_to_keep, :]
            data.indices = data.indices[indices_to_keep]
        else:
            # All neurons removed - create empty arrays with correct shape
            data.traces = np.zeros((0, data.traces.shape[1]), dtype=data.traces.dtype)
            data.indices = np.array([], dtype=data.indices.dtype)

    elif isinstance(data, SpikeData):
        # Find which spike events belong to neurons we want to keep
        spikes_to_keep = []
        for i, neuron_idx in enumerate(data.spike_indices):
            if neuron_idx not in neuron_indices_to_remove:
                spikes_to_keep.append(i)

        if spikes_to_keep:
            # Keep only spikes from specified neurons
            data.spike_indices = data.spike_indices[spikes_to_keep]
            data.spike_times = data.spike_times[spikes_to_keep]
        else:
            # All spikes removed - create empty arrays
            data.spike_indices = np.array([], dtype=data.spike_indices.dtype)
            data.spike_times = np.array([], dtype=data.spike_times.dtype)


def getSimulationTime(parent: QtWidgets.QWidget | None = None) -> float | None:
    simulation_time_str, ok = QtWidgets.QInputDialog.getText(
        parent,
        "Simulate",
        "Simulation time (s)",
        QtWidgets.QLineEdit.EchoMode.Normal,
        "1.0",
    )
    if not ok:
        return None
    try:
        simulation_time = float(simulation_time_str)
    except ValueError:
        QtWidgets.QMessageBox.critical(
            parent,
            "Invalid input",
            f"Could not parse simulation time '{simulation_time_str}' as a number",
        )
        return None
    return simulation_time


def get_stim_type(stim_name: str) -> StimType:
    """Determine the stimulus type based on the stimulus name - to be replaced once we have a way of storing this info properly."""
    stim_type = StimType.Excitation
    if "inh" in stim_name.lower():
        stim_type = StimType.Inhibition
    elif "ext" in stim_name.lower():
        stim_type = StimType.External
    return stim_type


def get_cmap(stim_type: StimType, alpha: float = 1.0) -> Colormap:
    match stim_type:
        case StimType.Excitation:
            return Colormap([(1.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, alpha)])
        case StimType.Inhibition:
            return Colormap([(1.0, 1.0, 1.0, 0.0), (0.1255, 0.6784, 0.9725, alpha)])
        case StimType.External:
            return Colormap(
                [
                    (0.251, 0.0, 0.2941, alpha),
                    (1.0, 1.0, 1.0, 0.0),
                    (0.0, 0.2667, 0.1059, alpha),
                ]
            )
        case _:
            return Colormap([(1.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0, alpha)])


T = TypeVar("T")


def find_by_name_and_type(
    data: Iterable[object],
    name: str,
    cls: type[T],
) -> T | None:
    return next(
        (d for d in data if isinstance(d, cls) and getattr(d, "name", None) == name),
        None,
    )
