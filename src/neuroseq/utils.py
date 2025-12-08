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
    print(n_floats)
    print(bits_per_float)
    print(mb_in_bits)
    mbits_per_float = float(bits_per_float) / mb_in_bits
    ram_mb = n_floats * mbits_per_float
    print("get down")
    print(ram_mb)
    return 1 + int(ram_mb // max_ram_mb)


def construct_synapse_matrix(
    monitor: b2.StateMonitor, variable: str, max_ram_mb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Constructs NxNxt timeseries of synapse matrices and timestamps from a Brian2 StateMonitor of synapse variables.

    Automatically downsamples the data if it exceeds the maximum RAM limit.
    """
    synapse_group = monitor.source
    # neuron_group = synapse_group.source
    values = monitor.variables[variable].get_value().T
    N = max(max(synapse_group.i), max(synapse_group.j)) + 1
    print(N)
    n_timestamps = values.shape[1]
    if n_timestamps == 0:
        return np.array([]), np.array([])
    n_floats = N * N * n_timestamps
    print(n_floats)
    downsample_factor = get_downsample_factor(n_floats, max_ram_mb, bits_per_float=32)
    if downsample_factor > 1:
        logging.info(
            f"Downsampling {N}x{N}x{n_timestamps} matrix by factor {downsample_factor} due to max RAM limit of {max_ram_mb} MB"
        )
    print(downsample_factor)
    matrix = np.zeros(
        (N, N, values[0, ::downsample_factor].shape[0]),
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


def cmap_transform_neuron_indices(
    n_neurons: int, indices: np.ndarray | None = None
) -> np.ndarray:
    """Transform the supplied neuron indices to generate cmap transform indices in a pseudo-random order

    For a given value of n_neurons, this will always return the same transform."""
    m = 415343
    if indices is None:
        indices = np.arange(n_neurons, dtype=np.int32)
    return np.mod(np.asarray(indices) * m, n_neurons)


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
        if existing_data.traces.shape[0] != new_data.traces.shape[0]:
            raise ValueError(
                f"Cannot append TraceData with different number of neurons: {existing_data.traces.shape[0]} != {new_data.traces.shape[0]}"
            )
        if existing_data.dt != new_data.dt:
            raise ValueError(
                f"Cannot append TraceData with different dt: {existing_data.dt} != {new_data.dt}"
            )
        if existing_data.traces.shape[1] * existing_data.dt != time_offset_seconds:
            raise ValueError(
                f"Time offset {time_offset_seconds} does not match the length of the existing trace data {existing_data.traces.shape[1] * existing_data.dt}."
            )
        existing_data.traces = np.concatenate(
            (existing_data.traces, new_data.traces), axis=1
        )
    elif isinstance(existing_data, SpikeData) and isinstance(new_data, SpikeData):
        existing_data.spike_indices = np.concatenate(
            (existing_data.spike_indices, new_data.spike_indices), axis=0
        )
        existing_data.spike_times = np.concatenate(
            (existing_data.spike_times, new_data.spike_times + time_offset_seconds),
            axis=0,
        )


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
