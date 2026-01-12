from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SynapseData
from neuroseq.fpl_widget import FPLWidget


class HeatmapNormDialog(QtWidgets.QDialog):
    def __init__(
        self,
        heatmap_widget: HeatmapWidget,
    ) -> None:
        super().__init__(heatmap_widget)
        self.setWindowTitle("Set Heatmap Normalization")
        self.heatmap_widget = heatmap_widget

        self.subtract = (
            heatmap_widget.subtract_idx is not None
            and self.heatmap_widget.data is not None
        )

        self.layout = QtWidgets.QVBoxLayout(self)

        self.chk_subtract = QtWidgets.QCheckBox("Subtract values from timestamp")
        self.chk_subtract.setChecked(self.subtract)
        self.chk_subtract.clicked.connect(self.chk_subtract_clicked)
        if self.heatmap_widget.data is None:
            self.chk_subtract.setEnabled(False)
        self.layout.addWidget(self.chk_subtract)

        self.lbl_subtract_idx = QtWidgets.QLabel("")
        self.lbl_subtract_idx.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.lbl_subtract_idx)
        self._update_label(self.heatmap_widget.subtract_idx)

        self.slider_idx = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        if self.heatmap_widget.timestamps is not None:
            self.slider_idx.setRange(0, self.heatmap_widget.timestamps.shape[0] - 1)
        if self.subtract:
            self.slider_idx.setValue(self.heatmap_widget.subtract_idx)
        else:
            self.slider_idx.setVisible(False)
            self.slider_idx.setEnabled(False)
        self.slider_idx.valueChanged.connect(self.update_idx)
        self.layout.addWidget(self.slider_idx)

        self.btn_reset_defaults = QtWidgets.QPushButton("Reset to defaults", self)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        self.layout.addWidget(self.btn_reset_defaults)

        self.btn_close = QtWidgets.QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        self.layout.addWidget(self.btn_close)

    def reset_defaults(self) -> None:
        self.subtract = False
        self.chk_subtract.setChecked(False)
        self.update_idx(None)

    def chk_subtract_clicked(self, checked: bool) -> None:
        self.subtract = checked
        self.update_idx(self.slider_idx.value() if checked else None)

    def _update_label(self, idx: int | None) -> None:
        self.lbl_subtract_idx.setVisible(idx is not None)
        if idx is not None and self.heatmap_widget.timestamps is not None:
            time_ms = self.heatmap_widget.timestamps[idx] * 1000
            self.lbl_subtract_idx.setText(f"{time_ms:.1f} ms")

    def update_idx(self, value: int | None) -> None:
        self.slider_idx.setEnabled(value is not None)
        self.slider_idx.setVisible(value is not None)
        self._update_label(value)
        self.heatmap_widget.set_subtract_idx(value)


class HeatmapWidget(FPLWidget):
    """A widget that displays a 3d array as a timeline of 2d heatmaps."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.subplot.title = "Synapses"
        self.data: np.ndarray | None = None
        self.idx: int = 0
        self.subtract_idx: int | None = None
        self.ig: fpl.ImageGraphic | None = None
        self.hlut: fpl.HistogramLUTTool | None = None
        self.timestamps: np.ndarray | None = None
        self.logger = logging.getLogger(__name__)

    def heatmap_norm_dialog(self):
        HeatmapNormDialog(self).exec()

    def clear_data(self) -> None:
        super().clear_data()
        self.data = None
        self.idx = 0
        self.subtract_idx = None
        self.ig = None
        self.hlut = None
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
        vmin = float(np.min(self.data))
        vmax = float(np.max(self.data))
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

    def set_time_ms(self, time_ms: int) -> None:
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
        self._set_idx(idx)

    def _set_idx(self, idx: int) -> None:
        if self.data is None or self.ig is None or self.hlut is None:
            return
        self.idx = idx
        data = self.data[self.idx]
        if self.subtract_idx is not None:
            data = data - self.data[self.subtract_idx]
        self.ig.data = data
        self.hlut.set_data(data, reset_vmin_vmax=False)

    def set_subtract_idx(self, idx: int | None) -> None:
        """Set the index of the timestamp to subtract values from when displaying the heatmap."""
        if (
            self.data is None
            or self.timestamps is None
            or self.ig is None
            or self.hlut is None
        ):
            return
        if idx is not None and (idx < 0 or idx >= self.data.shape[0]):
            self.logger.warning(
                f"Index to subtract from {idx} is out of bounds, ignoring"
            )
            return
        self.subtract_idx = idx
        if idx is not None:
            vmin = float(np.min(self.data - self.data[self.subtract_idx]))
            vmax = float(np.max(self.data - self.data[self.subtract_idx]))
            if vmin == vmax:
                vmax = vmin + 1.0
            self.hlut.vmin = vmin
            self.hlut.vmax = vmax
        self._set_idx(self.idx)
