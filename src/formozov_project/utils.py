from __future__ import annotations

import abc
import functools
import logging
import pathlib
import re
from collections.abc import Callable
from importlib import resources
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import formozov_project

example_models_path = (
    pathlib.Path(str(resources.files(formozov_project))) / "example_models"
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


def shuffle_index(i: int, n: int) -> tuple[int, int]:
    """Deterministic shuffling of the index i in a list of length n.

    Returns a tuple (j, n) where j is the shuffled index and n is the length of the list.
    """
    assert 0 <= i < n
    m = 415343
    return i * m % n, n


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
) -> QtWidgets.QAction:
    parent = menu or toolbar
    action = QtWidgets.QAction(name, parent)
    action.setCheckable(checkable)
    if checkable:
        action.setChecked(True)
    action.triggered.connect(callback)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(QtGui.QKeySequence(shortcut))
    if menu is not None:
        menu.addAction(action)
    if toolbar is not None:
        toolbar.addAction(action)
    if widget is not None:
        widget.addAction(action)
    return action


class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):  # type: ignore
    """Metaclass that allows a class to be both a QObject and an Abstract Base Class.

    Note that pyside6 apparently does not fully support abstract methods in a metaclass,
    so we also need to provide a fallback NotImplementedError in the abstract methods.
    """

    pass


def parse_string_to_quantity(quantity_str: str | None) -> b2.Quantity | str | None:
    """
    Parse a string into a brian2 quantity or return it if that is not possible. Quantities can be of the from
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
    if quantity_str is None:
        return None

    # extract numerical value via regex. allow for multiple numbers left and right of a '.' and multiple spaces or a '*' between number and unit
    pattern = r"([\(\)\d./\*]+)\s*\*?\s*(\w*)"
    match = re.match(pattern, quantity_str)
    quantity_obj = None
    if match:
        value, unit_str = match.groups()

        v = None
        unit = None
        try:
            v = eval(value)
            # Create the quantity object
            if unit_str != "":
                # Map the unit string to a Brian2 unit
                unit = getattr(b2, unit_str)
                quantity_obj = v * unit
                return quantity_obj
            else:
                return v
        except Exception:
            q = None
            try:
                q = eval(quantity_str)
            except Exception as _:
                q = quantity_str
            return q
    else:
        try:
            quantity = eval(quantity_str)
            return quantity
        except Exception as _:
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
