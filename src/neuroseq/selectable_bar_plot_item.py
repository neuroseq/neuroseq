from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from qtpy import QtGui

from neuroseq.selectable_mixin import SelectableMixin


class SelectableBarPlotItem(pg.BarGraphItem, SelectableMixin):
    """A pg.BarGraphItem where a slice of the bars can be selected."""

    def __init__(self, **opts: Any) -> None:
        super().__init__(**opts)
        deselected_alpha = 60
        self.selected_slice: slice = slice(0, -1)
        self.selected_color = self._sharedBrush.color()
        self.deselected_color = self._sharedBrush.color()
        self.deselected_color.setAlpha(deselected_alpha)

    def set_selected_slice(self, selected_slice: slice) -> None:
        self.selected_slice = selected_slice
        self.update()

    def paint(self, p: QtGui.QPainter, *args: Any) -> None:
        if self.selected_slice == slice(0, -1):
            self._sharedBrush.setColor(self.selected_color)
            super().paint(p, *args)
        else:
            self._sharedBrush.setColor(self.deselected_color)
            super().paint(p, *args)
            self._sharedBrush.setColor(self.selected_color)
            p.setBrush(self._sharedBrush)
            rects = self._rectarray.instances()
            for rect in rects[self.selected_slice]:
                p.drawRect(rect)
