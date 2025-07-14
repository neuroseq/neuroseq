from __future__ import annotations

from typing import Any

import pyqtgraph as pg

from neuroseq.selectable_mixin import SelectableMixin


class SelectableLinePlotItem(pg.PlotDataItem, SelectableMixin):
    """A pg.PlotDataItem where a slice of the points can be selected. (slicing not yet implemented)"""

    def __init__(self, *args: Any, **kargs: Any) -> None:
        super().__init__(*args, **kargs)
        self.setDownsampling(auto=True, method="subsample")
        self.setClipToView(True)
        self.setSkipFiniteCheck(True)

    def set_selected_slice(self, selected_slice: slice) -> None:
        pass
