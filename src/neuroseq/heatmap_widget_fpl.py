from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtWidgets

from neuroseq.experiment import SynapseData


class HeatmapWidgetFPL(QtWidgets.QWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        empty_data = np.zeros((1, 1, 1))
        self.image_widget: fpl.ImageWidget = fpl.ImageWidget(
            data=empty_data,
            names=[""],
            cmap="viridis",
            figure_kwargs={"canvas_kwargs": {"parent": self}},
        )
        self.image_widget.figure[0, 0].toolbar = False
        self.timestamps: np.ndarray = np.array([0.0])
        self.suppress_events: bool = False
        self.widget = self.image_widget.show()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        self.setLayout(layout)
        self.playing = False
        self.relative_playback_speed = 1.0
        self.logger = logging.getLogger(__name__)

        self.image_widget.figure.add_animations(self.update_heatmap)

    def clear_data(self) -> None:
        self.suppress_events = True
        self.image_widget.set_data(np.zeros((1, 1, 1)))
        self.timestamps = np.array([0.0])
        self.suppress_events = False

    def set_data(
        self,
        data: SynapseData | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap.
        data should be a 3d numpy array with shape (n_rows, n_columns, n_timesteps).
        """
        self.suppress_events = True
        self.clear_data()
        if data is None or data.values.shape[0] == 0:
            return
        self.logger.info(
            f"{data.values.shape} values with {data.timestamps.shape} timestamps"
        )
        # Data format used by the old HeatmapWidget: (n_rows, n_columns, n_timesteps)
        # rearrange to (n_timesteps, n_rows, n_columns) for fastplotlib
        values = data.values
        values = np.transpose(values, (2, 0, 1))
        self.image_widget.set_data(values)
        self.timestamps = data.timestamps
        self.suppress_events = False

    def connect_timeseries_widget_timeline(
        self, timeseries_widget: QtWidgets.QWidget
    ) -> None:
        """Connect the timeline in this widget to the timeline in the TimeseriesWidget."""

        def _on_index(idx):
            if not self.suppress_events:
                t = self.timestamps[idx["t"]] * 1000.0
                print(f"HeatmapFPL _on_index: {idx}, {t}")
                print(f"HeatmapFPL current index: {self.image_widget.current_index}")
                timeseries_widget.timeline.selection = t

        self.image_widget.add_event_handler(_on_index, "current_index")

    def toggle_pause(self) -> None:
        self.logger.info(f"Playing: {self.playing}")
        self.playing = not self.playing

    def set_current_index(self, idx: int) -> None:
        self.suppress_events = True
        self.image_widget.current_index = {"t": idx}
        self.suppress_events = False

    def update_heatmap(self) -> None:
        if not self.playing:
            return
        current_idx = self.image_widget.current_index["t"]
        next_idx = current_idx + 1
        if next_idx >= self.timestamps.shape[0]:
            next_idx = 0
        self.image_widget.current_index = {"t": next_idx}
