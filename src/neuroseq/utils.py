from __future__ import annotations

import abc
import functools
import logging
import pathlib
import re
from collections.abc import Callable
from enum import Enum
from importlib import resources
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import brian2.units.allunits as b2_units
import pygfx as gfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import neuroseq
from neuroseq.experiment import SpikeData
from neuroseq.experiment import SynapseData
from neuroseq.experiment import TraceData

example_models_path = pathlib.Path(str(resources.files(neuroseq))) / "example_models"

save_store_name = "___current-simulation-state-at-save-time___"


class InteractMode(Enum):
    DRAW = 1
    SELECT = 2


class PrettyPrintEnum(Enum):
    def __str__(self):
        return self.name


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
    """Constructs NxNxt timeseries of synapse matrices and timestamps from a Brian2 StateMonitor of synapse variables.

    Automatically downsamples the data if it exceeds the maximum RAM limit.
    """
    synapse_group = monitor.source
    neuron_group = synapse_group.source
    values = monitor.variables[variable].get_value().T
    n_timestamps = values.shape[1]
    if n_timestamps == 0:
        return np.array([]), np.array([])
    n_floats = neuron_group.N * neuron_group.N * n_timestamps
    downsample_factor = get_downsample_factor(n_floats, max_ram_mb, bits_per_float=32)
    if downsample_factor > 1:
        logging.info(
            f"Downsampling {neuron_group.N}x{neuron_group.N}x{n_timestamps} matrix by factor {downsample_factor} due to max RAM limit of {max_ram_mb} MB"
        )
    matrix = np.zeros(
        (neuron_group.N, neuron_group.N, values[0, ::downsample_factor].shape[0]),
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


def jet_cmap_transform_for_neurons(neuron_indices: np.ndarray) -> np.ndarray:
    """Return a normalized jet colormap transform for the provided neuron indices.

    Uses hash-based mapping to distribute colors evenly across the jet colormap.
    """
    if neuron_indices is None:
        return np.array([], dtype=np.float32)

    safe_indices = np.asarray(neuron_indices, dtype=np.int64)

    # Use a better hash-based distribution across the full [0,1] range
    # This gives much better color spread without patterns
    hash_base = 2654435761  # Large prime for better distribution
    hashed = (safe_indices * hash_base) % (2**32)
    normalized = hashed.astype(np.float32) / (2**32 - 1)

    return normalized


_JET_LUT: np.ndarray | None = None


def jet_lut() -> np.ndarray:
    """Fetch and cache the jet lookup table from pygfx."""
    global _JET_LUT
    if _JET_LUT is None:
        texture_data = gfx.cm.jet.texture.data
        if texture_data is None:
            raise RuntimeError("Jet colormap texture data is unavailable")
        _JET_LUT = np.asarray(texture_data, dtype=np.float32)
    return _JET_LUT


def jet_colors_for_neurons(neuron_indices: np.ndarray | None) -> np.ndarray:
    """Return uint8 RGB colors for the supplied neuron indices using the shared jet mapping."""
    if neuron_indices is None or len(neuron_indices) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    normalized_sorted = jet_cmap_transform_for_neurons(np.asarray(neuron_indices))
    lut = jet_lut()
    indices = (normalized_sorted * (lut.shape[0] - 1)).astype(int)
    indices = np.clip(indices, 0, lut.shape[0] - 1)
    colors = (lut[indices, :3] * 255.0).astype(np.uint8)
    return colors


_JET_LUT: np.ndarray | None = None
_JET_TRANSFORM_TABLE: np.ndarray | None = None


def _jet_transform_table() -> np.ndarray:
    """Precompute normalized transform values for a fixed colormap domain matching the jet LUT length (256)."""
    global _JET_TRANSFORM_TABLE
    if _JET_TRANSFORM_TABLE is None:
        table_size = 256
        base_indices = np.arange(table_size, dtype=np.int32)
        transform = cmap_transform_neuron_indices(table_size, base_indices).astype(
            np.float32
        )
        normalized = transform / float(max(1, table_size - 1))
        _JET_TRANSFORM_TABLE = np.clip(normalized, 0.0, 1.0)
    return _JET_TRANSFORM_TABLE


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


class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):  # type: ignore
    """Metaclass that allows a class to be both a QObject and an Abstract Base Class.

    Note that pyside6 apparently does not fully support abstract methods in a metaclass,
    so we also need to provide a fallback NotImplementedError in the abstract methods.
    """

    pass


def parse_string_to_quantity(quantity_str: str) -> b2.Quantity | str:
    """
    Parse a string into a brian2 quantity or return it if that is not possible. Quantities can be of the form
    number * unit.

    Parameters
    ----------
    quantity_str : str
        string representing a possible quantity

    Returns
    -------
    b2.Quantity | str
        either a brian2 Quantity or the input string

    Raises
    ------
    RuntimeError
        When something goes wrong during evaluation the numerical part of the input quantity string
    RuntimeError
        When something goes wrong during evaluation of the unit part of the input quantity string
    """

    # Regex pattern explanation: (TODO: this is mscopilot derived, possibly needs to be corrected. Needs to be maintained along with code )

    # 1. Character Class `([\(\)\d./\*\-\+]+(?:e[\d\+\-]*)?)`:

    # - `[` and `]` define a character class, which matches any one of the characters inside the brackets.
    # - `\(` and `\)` match literal parentheses `(` and `)`.
    # - `\d` matches any digit (equivalent to `[0-9]`).
    # - `.` matches a literal dot `.`.
    # - `/` matches a literal forward slash `/`.
    # - `\*` matches a literal asterisk `*`.
    # - `\-` matches a literal hyphen `-`.
    # - `\+` matches a literal plus `+`.
    # - The `+` outside the character class means "one or more" of the preceding character class.
    # - `(?:e[\d\+\-]*)?` is a non-capturing group that matches an optional `e` followed by zero or more digits, plus, or minus signs. This ensures that scientific notation like `2e10` is matched correctly.

    # 2. **Whitespace `\s*`**:
    # - `\s` matches any whitespace character (spaces, tabs, newlines).
    # - `*` means "zero or more" of the preceding whitespace characters.

    # 3. **Optional Asterisk `\*?`**:
    # - `\*` matches a literal asterisk `*`.
    # - `?` means "zero or one" of the preceding character (making the asterisk optional).

    # 4. **Whitespace `\s*`**:
    # - Again, `\s*` matches zero or more whitespace characters.

    # 5. **Word Characters `(\w*)`**:
    # - `\w` matches any word character (equivalent to `[a-zA-Z0-9_]`).
    # - `*` means "zero or more" of the preceding word characters.

    pattern = r"([\(\)\d./\*\-\+]+(?:e[\d\+\-]*)?)\s*\*?\s*(\w*)"
    match = re.match(pattern, quantity_str)
    quantity_obj = None
    if match:
        value, unit_str = match.groups()

        v = None
        unit = None

        try:
            v = eval(value)
            # Create the quantity object
        except Exception as e:
            raise RuntimeError(f"Error when interpreting numerical string: {e}") from e

        if unit_str != "":
            # Map the unit string to a Brian2 unit
            try:
                unit = getattr(b2, unit_str)
            except AttributeError:
                try:
                    unit = getattr(b2_units, unit_str)
                except AttributeError as e:
                    message = f"Error when interpreting unit string: {e}"
                    raise RuntimeError(message) from e

            quantity_obj = v * unit
            return quantity_obj
        else:
            return v
    else:
        try:
            quantity = eval(quantity_str)
            return quantity
        except Exception as _:
            logging.debug(f"Failed to parse {quantity_str} as Quantity")
            return quantity_str


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
