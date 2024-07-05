from __future__ import annotations

import abc
from typing import Callable

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


def add_action(
    name: str,
    callback: Callable,
    menu: QtWidgets.QMenu | None = None,
    toolbar: QtWidgets.QToolBar | None = None,
    shortcut: str | None = None,
    icon: QtGui.QIcon | None = None,
) -> None:
    parent = menu or toolbar
    action = QtWidgets.QAction(name, parent)
    action.triggered.connect(callback)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(QtGui.QKeySequence(shortcut))
    if menu is not None:
        menu.addAction(action)
    if toolbar is not None:
        toolbar.addAction(action)


class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):  # type: ignore
    """Metaclass that allows a class to be both a QObject and an Abstract Base Class.

    Note that pyside6 apparently does not fully support abstract methods in a metaclass,
    so we also need to provide a fallback NotImplementedError in the abstract methods.
    """

    pass
