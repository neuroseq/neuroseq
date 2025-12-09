from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from cmap import Colormap
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
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
    ram_mb = n_floats * bits_per_float // mb_in_bits
    return 1 + ram_mb // max_ram_mb


def construct_synapse_matrix(
    monitor: b2.StateMonitor, variable: str, max_ram_mb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs N_s x N_t x t timeseries of synapse matrices and timestamps from a Brian2 StateMonitor of synapse variables.

    Where N_s is the number of source neurons, N_t is the number of target neurons, and t is the number of timestamps.

    Automatically downsamples the data if it exceeds the maximum RAM limit.
    """
    synapse_group = monitor.source
    source_neuron_group = synapse_group.source
    target_neuron_group = synapse_group.target
    values = monitor.variables[variable].get_value().T
    n_timestamps = values.shape[1]
    if n_timestamps == 0:
        return np.array([]), np.array([])
    n_floats = source_neuron_group.N * target_neuron_group.N * n_timestamps
    downsample_factor = get_downsample_factor(n_floats, max_ram_mb, bits_per_float=32)
    if downsample_factor > 1:
        logging.info(
            f"Downsampling {source_neuron_group.N}x{target_neuron_group.N}x{n_timestamps} matrix by factor {downsample_factor} due to max RAM limit of {max_ram_mb} MB"
        )
    matrix = np.zeros(
        (
            source_neuron_group.N,
            target_neuron_group.N,
            values[0, ::downsample_factor].shape[0],
        ),
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


'''
I updated to

def neuron_indices_to_colors(indices: np.ndarray | None) -> np.ndarray:
    """Transform neuron indices to RGBA colors in a fixed pseudo-random order.

    A given index always maps to the same color, but nearby indices map to different colors.
    """
    if indices is None or len(indices) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    colormap = Colormap("matlab:hsv")
    # values of a and m from table 2 of
    # https://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
    a = 23234
    m = 32749
    normalized = np.mod(np.asarray(indices) * a, m) / m
    return np.asarray(colormap(normalized), dtype=np.float32)

may be better to undo that, given the way spike/trace data is handled
in the PR

'''


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


def append_data(
    existing_data: TraceData | SpikeData | SynapseData,
    new_data: TraceData | SpikeData | SynapseData,
    time_offset_seconds: float,
):
    """Append new trace or spike data in-place to the existing data

    Parameters
    ----------
    existing_data : TraceData | SpikeData | SynapseData
        Existing trace or spike data to append to.
    new_data : TraceData | SpikeData | SynapseData
        New trace or spike data to append.
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
        raise TypeError("existing_data and new_data must be of the same type.")
    if isinstance(existing_data, TraceData) and isinstance(new_data, TraceData):
        if existing_data.dt != new_data.dt:
            raise ValueError(
                f"Cannot append TraceData with different dt: {existing_data.dt} != {new_data.dt}"
            )

        # Check if we're merging neurons (different indices) vs time-extending same neurons
        existing_indices = set(existing_data.indices)
        new_indices = set(new_data.indices)

        logging.getLogger(__name__).info(
            f"Appending TraceData: existing_indices={existing_indices}, new_indices={new_indices}"
        )

        if existing_indices.intersection(new_indices):
            # Some neurons overlap - this is time extension for existing neurons
            if existing_data.traces.shape[0] != new_data.traces.shape[0]:
                raise ValueError(
                    f"Cannot append TraceData with different number of neurons: {existing_data.traces.shape[0]} != {new_data.traces.shape[0]}"
                )
            if existing_data.traces.shape[1] * existing_data.dt != time_offset_seconds:
                raise ValueError(
                    f"Time offset {time_offset_seconds} does not match the length of the existing trace data {existing_data.traces.shape[1] * existing_data.dt}."
                )
            existing_data.traces = np.concatenate(
                (existing_data.traces, new_data.traces), axis=1
            )
        else:
            # No overlap - this is adding new neurons to the same dataset
            # Merge the traces by expanding the neuron dimension
            existing_n_neurons, existing_n_timepoints = existing_data.traces.shape
            new_n_neurons, new_n_timepoints = new_data.traces.shape

            logging.getLogger(__name__).debug(
                f"Merging neurons: existing shape={existing_data.traces.shape}, new shape={new_data.traces.shape}"
            )

            # Combine traces and indices
            existing_data.traces = np.concatenate(
                (existing_data.traces, new_data.traces), axis=0
            )
            existing_data.indices = np.concatenate(
                (existing_data.indices, new_data.indices), axis=0
            )

            logging.getLogger(__name__).debug(
                f"After merging: shape={existing_data.traces.shape}, indices={existing_data.indices}"
            )
    elif isinstance(existing_data, SpikeData) and isinstance(new_data, SpikeData):
        existing_data.spike_indices = np.concatenate(
            (existing_data.spike_indices, new_data.spike_indices), axis=0
        )
        existing_data.spike_times = np.concatenate(
            (existing_data.spike_times, new_data.spike_times + time_offset_seconds),
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
