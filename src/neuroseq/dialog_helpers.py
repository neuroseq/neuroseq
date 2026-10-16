"""Shared Qt dialog behavior independent of stimulation routines."""

from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets


class _SelfRaisingDialog(QtWidgets.QDialog):
    """A non-modal QDialog that raises itself above its siblings whenever it
    is activated (clicked into focus).

    Several window managers raise an entire transient-dialog chain as one
    unit on click, rather than reordering siblings within it - so clicking
    one of several sibling popups (all transient for the same parent
    routine dialog: the parameters dialog, its plots window, and its
    stim-pattern heatmap window) does not bring it in front of the others;
    the whole group just moves together, at whatever relative order the
    window manager already had. Explicitly re-raising on ActivationChange
    bypasses that window-manager heuristic and enforces the expected
    last-clicked-is-on-top order.
    """

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.raise_()
