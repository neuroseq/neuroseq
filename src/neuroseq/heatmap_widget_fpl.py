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
        self.widget = self.image_widget.show()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def clear_data(self) -> None:
        self.image_widget.set_data(np.zeros((1, 1, 1)))

    def set_data(
        self,
        data: np.ndarray | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap.
        data should be a 3d numpy array with shape (n_rows, n_columns, n_timesteps).
        """
        self.clear_data()
        if data is not None:
            # Data format used by the old HeatmapWidget: (n_rows, n_columns, n_timesteps)
            # rearrange to (n_timesteps, n_rows, n_columns) for fastplotlib
            data = np.transpose(data, (2, 0, 1))
            self.image_widget.set_data(data)
