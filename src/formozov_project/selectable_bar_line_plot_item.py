from __future__ import annotations

from formozov_project.selectable_bar_plot_item import SelectableBarPlotItem
from formozov_project.selectable_line_plot_item import SelectableLinePlotItem
from formozov_project.selectable_mixin import SelectableMixin


class SelectableBarLinePlotItem(SelectableMixin):
    """A combination of a SelectableBarPlotItem and a SelectableLinePlotItem."""

    def __init__(
        self,
        selectable_bar_plot_item: SelectableBarPlotItem,
        selectable_line_plot_item: SelectableLinePlotItem,
    ) -> None:
        self.bar = selectable_bar_plot_item
        self.line = selectable_line_plot_item

    def set_selected_slice(self, selected_slice: slice) -> None:
        self.bar.set_selected_slice(selected_slice)
        self.line.set_selected_slice(selected_slice)
