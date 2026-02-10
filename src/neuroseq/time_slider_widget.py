import logging

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.data_widget import DataWidget
from neuroseq.heatmap_widget import HeatmapWidget
from neuroseq.timeseries_widget import TimeseriesWidget


class HandleTooltipSlider(QtWidgets.QSlider):
    """A QSlider that shows a tooltip above the handle with the current value in milliseconds."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setPageStep(1000)  # 1 second in milliseconds
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        self.show_handle_tooltip()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        self.show_handle_tooltip()

    def show_handle_tooltip(self) -> None:
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        handle_rect = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            opt,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )
        handle_center = self.mapToGlobal(handle_rect.center())
        offset = QtCore.QPoint(0, -handle_rect.height() - 35)
        tooltip_pos = handle_center + offset
        QtWidgets.QToolTip.showText(tooltip_pos, f"{self.value()} ms", self)


class TimeSliderWidget(QtWidgets.QWidget):
    """A widget that provides a time slider to control playback of timeseries data."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.relative_playback_speed = 1.0
        self.t_ms = 0
        self.t_max_ms = 0
        self.t_playback_start_ms = 0
        self.widgets: list[TimeseriesWidget | HeatmapWidget | DataWidget] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 9, 0, 9)
        self.slider = HandleTooltipSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.slider.valueChanged.connect(self._value_changed)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)  # 25 fps / 40ms tick interval
        self.timer.timeout.connect(self._tick)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.setMaximumHeight(32)

    def connect_widget(
        self, widget: TimeseriesWidget | HeatmapWidget | DataWidget
    ) -> None:
        self.widgets.append(widget)

    def set_max_time_secs(self, max_time_secs: float) -> None:
        self.t_max_ms = int(max_time_secs * 1000)
        self.slider.setMaximum(self.t_max_ms)

    def set_relative_playback_speed(self, relative_playback_speed: float) -> None:
        self.relative_playback_speed = relative_playback_speed

    def stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self._set_time(0)

    def skip_to_start(self) -> None:
        self._set_time(0)

    def skip_to_end(self) -> None:
        self._set_time(self.t_max_ms)

    def toggle_pause(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.elapsed_timer.restart()
            self.t_playback_start_ms = self.t_ms
            self.timer.start()

    def _set_time(self, time_ms: float) -> None:
        self.t_ms = int(time_ms)
        self.elapsed_timer.restart()
        self.t_playback_start_ms = self.t_ms
        self.slider.setValue(self.t_ms)

    def _tick(self) -> None:
        if not self.timer.isActive():
            return

        elapsed_ms = self.elapsed_timer.elapsed()
        new_t_ms = self.t_playback_start_ms + int(
            elapsed_ms * self.relative_playback_speed
        )

        if new_t_ms >= self.t_max_ms:
            self.timer.stop()
            new_t_ms = self.t_max_ms

        self.t_ms = new_t_ms
        self.slider.setValue(self.t_ms)

    def _value_changed(self, value: int) -> None:
        self.t_ms = value
        for widget in self.widgets:
            widget.set_time_ms(value)
