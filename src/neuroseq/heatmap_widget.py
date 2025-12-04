from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtWidgets

from neuroseq.data import SynapseData
from neuroseq.fpl_widget import FPLWidget


class HeatmapWidget(FPLWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.subplot.title = "Synapses"
        self.data: np.ndarray | None = None
        self.idx: int = 0
        self.ig: fpl.ImageGraphic | None = None
        self.hlut: fpl.HistogramLUTTool | None = None
        self.timestamps: np.ndarray | None = None
        self.logger = logging.getLogger(__name__)

    def clear_data(self) -> None:
        super().clear_data()
        self.data = None
        self.timestamps = None

    def set_data(
        self,
        data: SynapseData | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap."""
        self.clear_data()
        if data is None or data.values.shape[0] == 0:
            self.subplot.title = "Synapses"
            return
        self.subplot.title = f"Synapses: {data.name}"
        self.logger.info(
            f"{data.values.shape} values with {data.timestamps.shape} timestamps"
        )
        self.data = data.values.T.astype(np.float32)
        vmin = np.min(self.data)
        vmax = np.max(self.data)
        if vmin == vmax:
            vmax = vmin + 1.0
        self.idx = 0
        self.timestamps = data.timestamps
        self.ig = self.subplot.add_image(
            self.data[self.idx],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            isolated_buffer=False,
        )
        self.hlut = fpl.HistogramLUTTool(self.data[self.idx], self.ig)
        self.subplot.docks["right"].add_graphic(self.hlut)
        self.subplot.docks["right"].size = 80
        self.subplot.docks["right"].auto_scale(maintain_aspect=False)
        self.subplot.docks["right"].controller.enabled = False
        self.subplot.auto_scale(zoom=1.0, maintain_aspect=True)

    def set_time_ms(self, time_ms: float) -> None:
        """Set the time to display: data from the closest timestamp to the given time is displayed."""
        if (
            self.data is None
            or self.timestamps is None
            or self.ig is None
            or self.hlut is None
        ):
            return
        time_secs = float(time_ms) / 1000.0
        if self.timestamps.size == 0:
            return
        if time_secs <= self.timestamps[0]:
            idx = 0
        elif time_secs >= self.timestamps[-1]:
            idx = self.timestamps.shape[0] - 1
        else:
            idx = int((np.abs(self.timestamps - time_secs)).argmin())
        self.logger.debug(
            f"Nearest timestamp to {time_ms}ms: {self.timestamps[idx]:.3g}s (index {idx})"
        )
        self.idx = idx
        self.ig.data = self.data[self.idx]
        self.hlut.set_data(self.data[self.idx], reset_vmin_vmax=False)

    def get_timestamp_index(self) -> int:
        """Get the current index of the heatmap."""
        return self.idx
