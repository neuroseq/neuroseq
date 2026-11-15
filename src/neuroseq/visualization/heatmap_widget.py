from __future__ import annotations

import logging

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SynapseData
from neuroseq.visualization.fpl_widget import FPLWidget


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
        self.data: np.ndarray | None = None
        self.idx: int = 0
        self.subtract_idx: int | None = None
        self.ig: fpl.ImageGraphic | None = None
        self.hlut: fpl.HistogramLUTTool | None = None
        self.hover_text: fpl.TextGraphic | None = None
        self.timestamps: np.ndarray | None = None
        self._base_camera_width: float | None = None
        self._base_camera_height: float | None = None
        self._last_camera_width: float | None = None
        self._last_camera_height: float | None = None
        self._axis_update_timer = QtCore.QTimer(self)
        self._axis_update_timer.setInterval(100)
        self._axis_update_timer.timeout.connect(self._update_index_axes_from_zoom)
        # Started by showEvent and stopped by hide/close, so a hidden or closed
        # heatmap does not poll its camera forever.
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.logger = logging.getLogger(__name__)
        self.context_menu = QtWidgets.QMenu("Synapses", self)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._axis_update_timer.start()

    def hideEvent(self, event) -> None:
        self._axis_update_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._axis_update_timer.stop()
        super().closeEvent(event)

    def heatmap_norm_dialog(self):
        HeatmapNormDialog(self).exec()

    def _rebuild_context_menu(self) -> None:
        self.context_menu.clear()
        norm_action = self.context_menu.addAction("Normalization…")
        norm_action.triggered.connect(self.heatmap_norm_dialog)

    def contextMenuEvent(self, event):
        self._rebuild_context_menu()
        self.context_menu.exec(event.globalPos())

    def clear_data(self) -> None:
        super().clear_data()
        self.data = None
        self.idx = 0
        self.subtract_idx = None
        self.ig = None
        self.hlut = None
        self.hover_text = None
        self.timestamps = None
        self._base_camera_width = None
        self._base_camera_height = None
        self._last_camera_width = None
        self._last_camera_height = None

    @staticmethod
    def _hover_text_for_position(
        x: float, y: float, n_i: int, n_j: int, value: float | None = None
    ) -> tuple[int, int, str] | None:
        i = int(np.floor(x + 0.5))
        j = int(np.floor(y + 0.5))
        if i < 0 or j < 0 or i >= n_i or j >= n_j:
            return None
        text = f"pre i={i}, post j={j}"
        if value is not None:
            text += f", value={value:.4g}"
        return i, j, text

    @staticmethod
    def _tick_step_from_scale(scale_factor: float) -> int:
        if scale_factor >= 10.0:
            return 1
        if scale_factor >= 5.0:
            return 10
        return 100

    @staticmethod
    def _tick_positions(size: int, step: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=int)
        ticks = np.arange(0, size, max(1, step), dtype=int)
        if ticks.size == 0 or ticks[-1] != size - 1:
            ticks = np.append(ticks, size - 1)
        return ticks

    def _set_index_axes(self, n_x: int, n_y: int, step_x: int, step_y: int) -> None:
        self.subplot.axes.x.ticks = self._tick_positions(n_x, step_x)
        self.subplot.axes.y.ticks = self._tick_positions(n_y, step_y)
        self.subplot.axes.x.tick_format = lambda x, x_min, x_max: f"{x:.0f}"
        self.subplot.axes.y.tick_format = lambda y, y_min, y_max: f"{y:.0f}"
        self.subplot.axes.x.min_tick_distance = 25
        self.subplot.axes.y.min_tick_distance = 25
        self.subplot.axes.x.line.visible = True
        self.subplot.axes.y.line.visible = True

    def _update_index_axes_from_zoom(self) -> None:
        if self.data is None:
            return
        camera_state = self.subplot.camera.get_state()
        width = float(camera_state.get("width", 0.0))
        height = float(camera_state.get("height", 0.0))
        if width <= 0.0 or height <= 0.0:
            return
        if self._base_camera_width is None or self._base_camera_height is None:
            self._base_camera_width = width
            self._base_camera_height = height
        if (
            self._last_camera_width == width
            and self._last_camera_height == height
            and self.subplot.axes.x.ticks is not None
            and self.subplot.axes.y.ticks is not None
        ):
            return
        scale_x = self._base_camera_width / width
        scale_y = self._base_camera_height / height
        step_x = self._tick_step_from_scale(scale_x)
        step_y = self._tick_step_from_scale(scale_y)
        self._set_index_axes(
            n_x=self.data.shape[2],
            n_y=self.data.shape[1],
            step_x=step_x,
            step_y=step_y,
        )
        self._last_camera_width = width
        self._last_camera_height = height

    def set_data(
        self,
        data: SynapseData | None = None,
    ) -> None:
        """Set the data to be displayed in the heatmap."""
        self.clear_data()
        if data is None or data.values.shape[0] == 0:
            return
        self.logger.info(
            f"{data.values.shape} values with {data.timestamps.shape} timestamps"
        )
        self.data = data.values.T
        vmin = float(np.min(self.data))
        vmax = float(np.max(self.data))
        if vmin == vmax:
            vmax = vmin + 1.0
        self.idx = 0
        self.timestamps = data.timestamps
        # the view's slices are non-contiguous; copy the small 2D slice we display
        display_slice = np.ascontiguousarray(self.data[self.idx])
        self.ig = self.subplot.add_image(
            display_slice,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        self.hlut = fpl.HistogramLUTTool(display_slice, self.ig)
        self.subplot.docks["right"].add_graphic(self.hlut)
        self.subplot.docks["right"].size = 80
        self.subplot.docks["right"].auto_scale(maintain_aspect=False)
        self.subplot.docks["right"].controller.enabled = False
        self.hover_text = self.subplot.add_text(
            "",
            font_size=14,
            offset=(8.0, 4.0, 200.0),
            face_color=(0.0, 0.0, 0.0),
            outline_color=(1.0, 1.0, 1.0),
            outline_thickness=2.0,
            center=False,
            anchor="bottom-right",
            screen_space=True,
        )
        self.hover_text.visible = False
        self.subplot.auto_scale(zoom=1.0, maintain_aspect=True)
        camera_state = self.subplot.camera.get_state()
        self._base_camera_width = float(camera_state.get("width", 1.0))
        self._base_camera_height = float(camera_state.get("height", 1.0))
        self._last_camera_width = None
        self._last_camera_height = None
        self._update_index_axes_from_zoom()

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        if self.data is None or self.hover_text is None:
            return
        mouse_pos = self.subplot.map_screen_to_world(ev)
        if mouse_pos is None:
            self.hover_text.visible = False
            return
        hover_text = self._hover_text_for_position(
            x=float(mouse_pos[0]),
            y=float(mouse_pos[1]),
            n_i=self.data.shape[2],
            n_j=self.data.shape[1],
        )
        if hover_text is None:
            self.hover_text.visible = False
            return
        i, j, _ = hover_text
        value = float(self.data[self.idx, j, i])
        if self.subtract_idx is not None:
            value -= float(self.data[self.subtract_idx, j, i])
        text = f"pre i={i}, post j={j}, value={value:.4g}"
        self.hover_text.offset = (float(i), float(j), 200.0)
        self.hover_text.text = text
        self.hover_text.visible = True

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
        # copy the small displayed slice out of the non-contiguous source view
        data = np.ascontiguousarray(self.data[self.idx])
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
            sub = self.data[self.subtract_idx]
            # we need min/max of (data - sub): reduce data over time first
            per_cell_min = self.data.min(axis=0)
            per_cell_max = self.data.max(axis=0)
            vmin = float((per_cell_min - sub).min())
            vmax = float((per_cell_max - sub).max())
            if vmin == vmax:
                vmax = vmin + 1.0
            self.hlut.vmin = vmin
            self.hlut.vmax = vmax
        self._set_idx(self.idx)
