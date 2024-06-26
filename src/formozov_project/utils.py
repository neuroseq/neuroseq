from __future__ import annotations

from typing import Callable

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
