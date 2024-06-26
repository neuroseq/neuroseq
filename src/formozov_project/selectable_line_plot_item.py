from __future__ import annotations

from typing import Any

import pyqtgraph as pg

from formozov_project.selectable_mixin import SelectableMixin


class SelectableLinePlotItem(pg.PlotDataItem, SelectableMixin):
    """A pg.PlotDataItem where a slice of the points can be selected. (slicing not yet implemented)"""

    def __init__(self, *args: Any, **kargs: Any) -> None:
        super().__init__(*args, **kargs)
