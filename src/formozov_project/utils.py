from __future__ import annotations

import abc
import functools
import re
from collections.abc import Callable
from typing import Any

import brian2 as b2
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


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
) -> None:
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


class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):  # type: ignore
    """Metaclass that allows a class to be both a QObject and an Abstract Base Class.

    Note that pyside6 apparently does not fully support abstract methods in a metaclass,
    so we also need to provide a fallback NotImplementedError in the abstract methods.
    """

    pass


def parse_string_to_quantity(quantity_str: str) -> tuple[b2.Quantity | str, bool]:
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
    # extract numerical value via regex. allow for multiple numbers left and right of a '.' and multiple spaces or a '*' between number and unit
    pattern = r"([\(\)0-9./\*]+)\s*\*?\s*(\w+)"
    match = re.match(pattern, quantity_str)
    ok = True
    quantity_obj = None
    if match:
        value, unit_str = match.groups()
        # Map the unit string to a Brian2 unit
        try:
            unit = getattr(b2, unit_str)
        except Exception as e:
            message = f"Error when interpreting unit string: {e}"
            ok = False
            raise RuntimeError(message) from e
        if ok:
            try:
                v = eval(value)
                # Create the quantity object
                quantity_obj = v * unit
            except Exception as e:
                ok = False
                raise RuntimeError(
                    f"Error when interpreting numerical string: {e}"
                ) from e

        return (quantity_obj, ok)
    else:
        return (quantity_str, ok)


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
