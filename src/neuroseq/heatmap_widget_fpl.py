from __future__ import annotations

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtWidgets


class HeatmapWidgetFPL(QtWidgets.QWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        empty_data = np.zeros((1, 1, 1))
        self.image_widget: fpl.ImageWidget = fpl.ImageWidget(
            data=empty_data, cmap="viridis"
        )
        self.timestamps: np.ndarray = np.array([0.0])
        self.suppress_events: bool = False
        self.widget = self.image_widget.show()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def clear_data(self) -> None:
        self.suppress_events = True
        self.image_widget.set_data(np.zeros((1, 1, 1)))
        self.timestamps = np.array([0.0])
        self.suppress_events = False

    def set_data(
        self,
        data: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap.
        data should be a 3d numpy array with shape (n_rows, n_columns, n_timesteps).
        """
        self.suppress_events = True
        self.clear_data()
        if data is not None:
            # Data format used by the old HeatmapWidget: (n_rows, n_columns, n_timesteps)
            # rearrange to (n_timesteps, n_rows, n_columns) for fastplotlib
            data = np.transpose(data, (2, 0, 1))
            self.image_widget.set_data(data)
            self.timestamps = timestamps
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
