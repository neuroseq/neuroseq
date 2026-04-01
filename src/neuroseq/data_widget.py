from __future__ import annotations

import pygfx
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.spike_widget import SpikeWidget
from neuroseq.trace_widget import TraceWidget


class DataWidget(QtWidgets.QWidget):
    """A wrapper widget that displays data in an appropriate widget."""

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stacked_widget = QtWidgets.QStackedWidget()
        layout.addWidget(self.stacked_widget)
        self.trace_widget = TraceWidget(controller=controller, parent=self)
        self.spike_widget = SpikeWidget(controller=controller, parent=self)
        self.heatmap_widget = HeatmapWidget(parent=self)
        self.stacked_widget.addWidget(self.trace_widget)
        self.stacked_widget.addWidget(self.spike_widget)
        self.stacked_widget.addWidget(self.heatmap_widget)
        self.stacked_widget.setCurrentWidget(self.trace_widget)
        self.resize(800, 600)

    def clear_data(self) -> None:
        self.trace_widget.clear_data()
        self.spike_widget.clear_data()
        self.heatmap_widget.clear_data()

    def set_time_ms(self, time_ms: int) -> None:
        self.trace_widget.set_time_ms(time_ms)
        self.spike_widget.set_time_ms(time_ms)
        self.heatmap_widget.set_time_ms(time_ms)

    def set_data(self, data: TraceData | SpikeData | SynapseData | None) -> None:
        self.clear_data()
        if isinstance(data, TraceData):
            self.stacked_widget.setCurrentWidget(self.trace_widget)
            self.trace_widget.set_data(data)
        elif isinstance(data, SpikeData):
            self.stacked_widget.setCurrentWidget(self.spike_widget)
            self.spike_widget.set_data(data)
        elif isinstance(data, SynapseData):
            self.stacked_widget.setCurrentWidget(self.heatmap_widget)
            self.heatmap_widget.set_data(data)

    def closeEvent(self, event):
        # Close child plot widgets explicitly so their renderers shut down before
        # Qt destroys this container.
        self.trace_widget.close()
        self.spike_widget.close()
        self.heatmap_widget.close()
        super().closeEvent(event)
