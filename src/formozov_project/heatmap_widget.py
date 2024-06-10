from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets


class HeatmapWidget(pg.ImageView):
    """A widget that displays a 3d array as a timeline of 2d heatmaps.

    see https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        xvals: Optional[np.ndarray] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent, discreteTimeLine=True)
        self.setPredefinedGradient("viridis")
        if data is not None:
            self.setImage(data, xvals=xvals)
