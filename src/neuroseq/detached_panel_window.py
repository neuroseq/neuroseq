from __future__ import annotations

from collections.abc import Callable

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class DetachedPanelWindow(QtWidgets.QWidget):
    """Top-level window wrapping one detached panel instance."""

    def __init__(
        self,
        widget: QtWidgets.QWidget,
        title: str,
        *,
        on_close: Callable[[], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.Tool)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self._widget = widget
        self._on_close = on_close

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        layout.addWidget(scroll_area)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._widget.close()
        if self._on_close is not None:
            self._on_close()
        super().closeEvent(event)
