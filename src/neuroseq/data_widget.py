from __future__ import annotations

import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
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
        self.default_controller = controller
        self.controller = ConstrainedPanZoomController()
        self.default_x_scroll_bar: QtWidgets.QScrollBar | None = None
        self.default_y_scroll_bar: QtWidgets.QScrollBar | None = None
        self._sync_in_progress = False

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stacked_widget = QtWidgets.QStackedWidget()
        layout.addWidget(self.stacked_widget, 0, 0, 4, 1)
        self.trace_widget = TraceWidget(controller=self.controller, parent=self)
        self.spike_widget = SpikeWidget(controller=self.controller, parent=self)
        self.heatmap_widget = HeatmapWidget(parent=self)
        self.stacked_widget.addWidget(self.trace_widget)
        self.stacked_widget.addWidget(self.spike_widget)
        self.stacked_widget.addWidget(self.heatmap_widget)
        self.stacked_widget.setCurrentWidget(self.trace_widget)

        # y-axis controls for the all-data widget
        self.y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.controller.connect_y_scroll_bar(self.y_scrollbar)
        self.y_scrollbar.setFixedWidth(12)
        layout.addWidget(self.y_scrollbar, 0, 1)

        self.y_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.y_btn_zoom_in.setFixedSize(14, 24)
        self.y_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_in.setToolTip("Zoom in (Shift + Mouse Wheel Up)")
        self.y_btn_zoom_in.clicked.connect(lambda: self.controller.zoom((0.0, 0.5), ()))
        layout.addWidget(self.y_btn_zoom_in, 1, 1)

        self.y_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.y_btn_zoom_out.setFixedSize(14, 24)
        self.y_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_out.setToolTip("Zoom out (Shift + Mouse Wheel Down)")
        self.y_btn_zoom_out.clicked.connect(
            lambda: self.controller.zoom((0.0, -0.5), ())
        )
        layout.addWidget(self.y_btn_zoom_out, 2, 1)

        self.lock_y_checkbox = self._create_lock_button("Y")
        self.lock_y_checkbox.setToolTip("Lock all-data Y pan/zoom to the default view")
        self.lock_y_checkbox.toggled.connect(self._on_lock_y_toggled)
        layout.addWidget(self.lock_y_checkbox, 3, 1)

        # x-axis controls for the all-data widget
        self.x_controls_widget = QtWidgets.QWidget(self)
        self.x_controls_widget.setFixedHeight(14)
        self.x_controls_layout = QtWidgets.QHBoxLayout(self.x_controls_widget)
        self.x_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.x_controls_layout.setSpacing(0)

        self.x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.controller.connect_x_scroll_bar(self.x_scrollbar)
        self.x_scrollbar.setFixedHeight(12)
        self.x_controls_layout.addWidget(self.x_scrollbar, 1)

        self.x_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.x_btn_zoom_out.setFixedSize(24, 14)
        self.x_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_out.setToolTip("Zoom out (Ctrl + Mouse Wheel Down)")
        self.x_btn_zoom_out.clicked.connect(
            lambda: self.controller.zoom((-0.5, 0.0), ())
        )
        self.x_controls_layout.addWidget(self.x_btn_zoom_out)

        self.x_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.x_btn_zoom_in.setFixedSize(24, 14)
        self.x_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_in.setToolTip("Zoom in (Ctrl + Mouse Wheel Up)")
        self.x_btn_zoom_in.clicked.connect(lambda: self.controller.zoom((0.5, 0.0), ()))
        self.x_controls_layout.addWidget(self.x_btn_zoom_in)

        self.lock_x_checkbox = self._create_lock_button("X")
        self.lock_x_checkbox.setToolTip("Lock all-data X pan/zoom to the default view")
        self.lock_x_checkbox.toggled.connect(self._on_lock_x_toggled)
        self.x_controls_layout.addWidget(self.lock_x_checkbox)

        layout.addWidget(self.x_controls_widget, 4, 0, 1, 2)

        self.x_scrollbar.valueChanged.connect(self._on_local_x_scroll_changed)
        self.x_scrollbar.rangeChanged.connect(self._on_local_x_scroll_changed)
        self.y_scrollbar.valueChanged.connect(self._on_local_y_scroll_changed)
        self.y_scrollbar.rangeChanged.connect(self._on_local_y_scroll_changed)

        self.resize(800, 600)

    def _create_lock_button(self, axis: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setCheckable(True)
        button.setChecked(True)
        button.setFixedSize(14, 14)
        button.setAutoRaise(True)
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        button.setStyleSheet(
            "QToolButton { margin: 0px; padding: 0px; font-size: 12px; }"
        )
        button.toggled.connect(
            lambda checked, btn=button, axis_name=axis: self._update_lock_button(
                btn, axis_name, checked
            )
        )
        self._update_lock_button(button, axis, True)
        return button

    def _make_lock_icon(self, locked: bool) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pen = QtGui.QPen(self.palette().buttonText().color())
        pen.setWidthF(1.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        shackle = QtGui.QPainterPath()
        if locked:
            shackle.moveTo(3.2, 5.8)
            shackle.arcTo(QtCore.QRectF(3.2, 1.0, 5.6, 6.4), 180, 180)
        else:
            shackle.moveTo(8.8, 5.8)
            shackle.arcTo(QtCore.QRectF(3.2, -1.4, 5.6, 6.4), 0, 180)
        painter.drawPath(shackle)
        painter.drawRoundedRect(QtCore.QRectF(2.2, 5.8, 7.6, 5.2), 1.2, 1.2)

        painter.end()
        return QtGui.QIcon(pixmap)

    def _update_lock_button(
        self,
        button: QtWidgets.QToolButton,
        axis: str,
        checked: bool,
    ) -> None:
        button.setText("")
        button.setIcon(self._make_lock_icon(checked))
        button.setIconSize(QtCore.QSize(12, 12))
        state = "linked" if checked else "independent"
        button.setToolTip(f"{axis} axis {state}")
        button.setStatusTip(f"{axis} axis {state}")

    @property
    def x_locked(self) -> bool:
        return self.lock_x_checkbox.isChecked()

    @property
    def y_locked(self) -> bool:
        return self.lock_y_checkbox.isChecked()

    def set_default_scroll_bars(
        self,
        x_scroll_bar: QtWidgets.QScrollBar,
        y_scroll_bar: QtWidgets.QScrollBar,
    ) -> None:
        self.default_x_scroll_bar = x_scroll_bar
        self.default_y_scroll_bar = y_scroll_bar
        self.default_x_scroll_bar.valueChanged.connect(
            self._on_default_x_scroll_changed
        )
        self.default_x_scroll_bar.rangeChanged.connect(
            self._on_default_x_scroll_changed
        )
        self.default_y_scroll_bar.valueChanged.connect(
            self._on_default_y_scroll_changed
        )
        self.default_y_scroll_bar.rangeChanged.connect(
            self._on_default_y_scroll_changed
        )
        self._sync_from_default_axes()

    def set_limits(self, max_time_secs: float, max_neurons: int) -> None:
        self.controller.set_limits(max_time_secs, max_neurons)
        self._sync_from_default_axes()

    def _on_lock_x_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_x_from_default()

    def _on_lock_y_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_y_from_default()

    def _on_default_x_scroll_changed(self, *args) -> None:
        if self.x_locked:
            self._sync_x_from_default()

    def _on_default_y_scroll_changed(self, *args) -> None:
        if self.y_locked:
            self._sync_y_from_default()

    def _on_local_x_scroll_changed(self, *args) -> None:
        if self.x_locked:
            self._sync_x_to_default()

    def _on_local_y_scroll_changed(self, *args) -> None:
        if self.y_locked:
            self._sync_y_to_default()

    def _sync_from_default_axes(self) -> None:
        if self.x_locked:
            self._sync_x_from_default()
        if self.y_locked:
            self._sync_y_from_default()

    def _sync_x_from_default(self) -> None:
        self._sync_axis(
            source_controller=self.default_controller,
            target_controller=self.controller,
            axis="x",
        )

    def _sync_y_from_default(self) -> None:
        self._sync_axis(
            source_controller=self.default_controller,
            target_controller=self.controller,
            axis="y",
        )

    def _sync_x_to_default(self) -> None:
        self._sync_axis(
            source_controller=self.controller,
            target_controller=self.default_controller,
            axis="x",
        )

    def _sync_y_to_default(self) -> None:
        self._sync_axis(
            source_controller=self.controller,
            target_controller=self.default_controller,
            axis="y",
        )

    def _sync_axis(
        self,
        source_controller: pygfx.Controller,
        target_controller: pygfx.Controller,
        axis: str,
    ) -> None:
        if self._sync_in_progress:
            return
        self._sync_in_progress = True
        try:
            try:
                source_state = source_controller._get_camera_state()
                target_state = target_controller._get_camera_state()
            except ValueError:
                return
            if axis == "x":
                new_state = {
                    "width": float(source_state["width"]),
                    "position": (
                        float(source_state["position"][0]),
                        float(target_state["position"][1]),
                        float(target_state["position"][2]),
                    ),
                }
            else:
                new_state = {
                    "height": float(source_state["height"]),
                    "position": (
                        float(target_state["position"][0]),
                        float(source_state["position"][1]),
                        float(target_state["position"][2]),
                    ),
                }
            # _set_camera_state() already pushes the new state to the
            # camera itself - an extra _update_cameras() here would just be
            # a redundant, wasted re-render on every sync.
            target_controller._set_camera_state(new_state)
        finally:
            self._sync_in_progress = False

    def clear_data(self) -> None:
        self.trace_widget.clear_data()
        self.spike_widget.clear_data()
        self.heatmap_widget.clear_data()

    def apply_theme(self, theme: str) -> None:
        self.trace_widget.apply_theme(theme)
        self.spike_widget.apply_theme(theme)
        self.heatmap_widget.apply_theme(theme)

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
