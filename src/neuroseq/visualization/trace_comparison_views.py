"""Matplotlib popups for comparing trace arrays (#662).

``ComparisonPopup`` is the shared frame (controls row, note line, toolbar
and a scrollable figure with one row per neuron). On top of it:

- ``LinkedArrayWindow`` shows one trace array's chosen neurons stacked; one
  window is opened per array and their time and/or value axes are kept in
  step through an :class:`AxisLinkGroup` while locked.
- ``ConcatenatedTraceWindow`` shows the arrays joined end to end, one long
  timeline per neuron with boundary lines and array names per segment.

The overlay / side-by-side window is ``trace_comparison_window``. All of
them only read the data; adding a joined array to the project is done by
``open_trace_comparison`` through ``Project``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.visualization_support.time_format import format_time_ms
from neuroseq.visualization_support.trace_comparison import TIME_AXIS
from neuroseq.visualization_support.trace_comparison import VALUE_AXIS
from neuroseq.visualization_support.trace_comparison import AxisLinkGroup
from neuroseq.visualization_support.trace_comparison import TraceComparison
from neuroseq.visualization_support.trace_comparison import TraceConcatenation
from neuroseq.visualization_support.trace_comparison import format_neuron_ids
from neuroseq.visualization_support.trace_comparison import union_range

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Height of one neuron row in the figure, in inches.
ROW_HEIGHT_IN = 1.6
# Extra figure height for the legend and the time axis, in inches.
FIGURE_MARGIN_IN = 1.2


def time_formatter(ax: Axes) -> FuncFormatter:
    """Tick labels in ms or s, precision following the visible range."""

    def format_tick(value: float, _position: int | None) -> str:
        low, high = ax.get_xlim()
        return format_time_ms(value, high - low, strip_zeros=True)

    return FuncFormatter(format_tick)


def style_axes(ax: Axes, face: str, text: str) -> None:
    ax.set_facecolor(face)
    ax.tick_params(colors=text, labelsize="small")
    for spine in ax.spines.values():
        spine.set_color(text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)


def neuron_ylabel(neuron_id: int, units: str) -> str:
    return f"Neuron {neuron_id}\n({units})" if units else f"Neuron {neuron_id}"


class ComparisonPopup(QtWidgets.QWidget):
    """Top-level, non-modal Matplotlib window with one row per neuron."""

    def __init__(
        self,
        note: str,
        theme: str = theme_module.LIGHT,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.resize(1000, 700)
        self._theme = theme_module.normalize_theme(theme)

        layout = QtWidgets.QVBoxLayout(self)
        self.controls = QtWidgets.QHBoxLayout()
        layout.addLayout(self.controls)

        self.note_label = QtWidgets.QLabel(note, self)
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(
            f"color: {theme_module.warning_color(self._theme)};"
        )
        self.note_label.setVisible(bool(note))
        layout.addWidget(self.note_label)

        face, _ = theme_module.mpl_colors(self._theme)
        self.figure = Figure(layout="constrained", facecolor=face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area, stretch=1)

    def add_note(self, text: str) -> None:
        """Append a line to the note above the plot."""
        current = self.note_label.text()
        self.note_label.setText(f"{current}\n{text}" if current else text)
        self.note_label.setVisible(True)

    def _fit_rows(self, n_rows: int) -> None:
        self.canvas.setMinimumHeight(
            int((ROW_HEIGHT_IN * max(1, n_rows) + FIGURE_MARGIN_IN) * self.figure.dpi)
        )

    def _show_message(self, message: str) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        ax = self.figure.add_subplot(111, facecolor=face)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=text,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _finish_time_axes(self, axes, max_time_ms: float) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        for ax in axes[-1]:
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(time_formatter(ax))
        if max_time_ms > 0:
            axes[0][0].set_xlim(0.0, max_time_ms)
        for ax in axes.flat:
            style_axes(ax, face, text)

    def closeEvent(self, event) -> None:
        self.figure.clear()
        super().closeEvent(event)


class LinkedArrayWindow(ComparisonPopup):
    """One trace array's chosen neurons, one row each, in its own window.

    Several of these (one per compared array) share an
    :class:`AxisLinkGroup`: while **Lock time axis** is ticked, zooming or
    panning time here moves every other window that has it ticked;
    **Lock value axis** does the same for each neuron row's value range.
    Rows are the same neurons in every window, so row *i* always pairs with
    row *i*. Closing a window just drops it from the group.
    """

    def __init__(
        self,
        comparison: TraceComparison,
        label: str,
        link_group: AxisLinkGroup[LinkedArrayWindow],
        *,
        lock_time: bool = True,
        lock_value: bool = False,
        theme: str = theme_module.LIGHT,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        missing = comparison.missing.get(label, [])
        note = f"Not in this array: {format_neuron_ids(missing)}" if missing else ""
        super().__init__(note, theme=theme, parent=parent)
        self.setWindowTitle(f"Trace array - {label}")
        self.comparison = comparison
        self.label = label
        self.link_group = link_group
        self.axes: list[Axes] = []
        # Rows showing a "not in this array" note: their value range is no data.
        self._empty_rows: set[int] = set()
        self._closed = False

        self.lock_time_checkbox = QtWidgets.QCheckBox("Lock time axis", self)
        self.lock_time_checkbox.setToolTip(
            "Zoom and pan time together with the other locked array windows"
        )
        self.lock_time_checkbox.setChecked(lock_time)
        self.lock_value_checkbox = QtWidgets.QCheckBox("Lock value axis", self)
        self.lock_value_checkbox.setToolTip(
            "Keep each neuron's value range in step with the other locked array windows"
        )
        self.lock_value_checkbox.setChecked(lock_value)
        self.controls.addWidget(self.lock_time_checkbox)
        self.controls.addWidget(self.lock_value_checkbox)
        self.controls.addStretch()

        link_group.add(self, lock_time=lock_time, lock_value=lock_value)
        self.destroyed.connect(partial(link_group.remove, self))
        self.lock_time_checkbox.toggled.connect(partial(self._set_locked, TIME_AXIS))
        self.lock_value_checkbox.toggled.connect(partial(self._set_locked, VALUE_AXIS))
        self._draw()

    def _draw(self) -> None:
        rows = self.comparison.rows
        self._fit_rows(len(rows))
        if not rows:
            self._show_message("None of the chosen neurons is in the chosen arrays.")
            return
        face, text = theme_module.mpl_colors(self._theme)
        color = dict(self.comparison.legend).get(self.label, text)
        axes = self.figure.subplots(len(rows), 1, sharex=True, squeeze=False)
        units = self.comparison.units
        for row_index, (ax, row) in enumerate(zip(axes[:, 0], rows, strict=True)):
            series = next((s for s in row.series if s.label == self.label), None)
            if series is None:
                self._empty_rows.add(row_index)
                ax.text(
                    0.5,
                    0.5,
                    "not in this array",
                    ha="center",
                    va="center",
                    color=theme_module.muted_color(self._theme),
                    transform=ax.transAxes,
                )
            else:
                ax.plot(series.time_ms, series.values, color=color, linewidth=0.8)
            ax.set_ylabel(neuron_ylabel(row.neuron_id, units))
        axes[0][0].set_title(self.label, color=color, fontsize="medium")
        self._finish_time_axes(axes, self.comparison.max_time_ms)
        self.axes = list(axes[:, 0])
        self.axes[0].callbacks.connect("xlim_changed", self._time_range_changed)
        for row_index, ax in enumerate(self.axes):
            ax.callbacks.connect(
                "ylim_changed", partial(self._value_range_changed, row_index)
            )
        self.canvas.draw_idle()

    # -- linking ------------------------------------------------------------

    @property
    def time_range(self) -> tuple[float, float] | None:
        return tuple(self.axes[0].get_xlim()) if self.axes else None  # type: ignore[return-value]

    def value_range(self, row_index: int) -> tuple[float, float] | None:
        if row_index >= len(self.axes):
            return None
        return tuple(self.axes[row_index].get_ylim())  # type: ignore[return-value]

    def apply_time_range(self, limits: tuple[float, float]) -> None:
        if self._closed or not self.axes:
            return
        self.axes[0].set_xlim(limits)
        self.canvas.draw_idle()

    def apply_value_range(self, row_index: int, limits: tuple[float, float]) -> None:
        if self._closed or row_index >= len(self.axes):
            return
        self.axes[row_index].set_ylim(limits)
        self.canvas.draw_idle()

    def _time_range_changed(self, ax: Axes) -> None:
        if self._closed:
            return
        limits = tuple(ax.get_xlim())
        self.link_group.propagate(
            self, TIME_AXIS, lambda target: target.apply_time_range(limits)
        )

    def _value_range_changed(self, row_index: int, ax: Axes) -> None:
        if self._closed:
            return
        limits = tuple(ax.get_ylim())
        self.link_group.propagate(
            self,
            VALUE_AXIS,
            lambda target: target.apply_value_range(row_index, limits),
        )

    def _set_locked(self, axis: str, locked: bool) -> None:
        self.link_group.set_locked(self, axis, locked)
        if not locked:
            return
        # Re-locking snaps this window to the group's current range.
        reference = self.link_group.reference(self, axis)
        if not isinstance(reference, LinkedArrayWindow):
            return
        if axis == TIME_AXIS:
            limits = reference.time_range
            if limits is not None:
                self.apply_time_range(limits)
        else:
            for row_index in range(len(self.axes)):
                row_limits = reference.value_range(row_index)
                if row_limits is not None:
                    self.apply_value_range(row_index, row_limits)

    @staticmethod
    def sync_locked_ranges(link_group: AxisLinkGroup[LinkedArrayWindow]) -> None:
        """Give the group's windows that lock an axis one shared range on it:
        the union of their own (autoscaled) ranges, per row for the value axis.

        Call once every window of the group is built; until then each
        window has only scaled itself to its own data.
        """
        timed = link_group.locked_members(TIME_AXIS)
        limits = union_range(window.time_range for window in timed)
        if limits is not None:
            for window in timed:
                window.apply_time_range(limits)
        valued = link_group.locked_members(VALUE_AXIS)
        rows = max((len(window.axes) for window in valued), default=0)
        for row_index in range(rows):
            row_limits = union_range(
                window.value_range(row_index)
                for window in valued
                if row_index not in window._empty_rows
            )
            if row_limits is not None:
                for window in valued:
                    window.apply_value_range(row_index, row_limits)

    def closeEvent(self, event) -> None:
        self._closed = True
        self.link_group.remove(self)
        self.axes = []
        super().closeEvent(event)


class ConcatenatedTraceWindow(ComparisonPopup):
    """The chosen arrays joined end to end: one long timeline per neuron.

    Each segment is drawn on its own array's time axis, offset by the
    segments before it (plus the gap), in that array's colour; dashed lines
    mark where one array hands over to the next and the array names label
    the segments along the top. A neuron missing from an array leaves that
    segment empty.
    """

    def __init__(
        self,
        concatenation: TraceConcatenation,
        theme: str = theme_module.LIGHT,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(concatenation.missing_note(), theme=theme, parent=parent)
        self.setWindowTitle("Concatenated traces")
        self.concatenation = concatenation
        self._draw()

    def _draw(self) -> None:
        concatenation = self.concatenation
        rows = concatenation.rows
        self._fit_rows(len(rows))
        if not rows:
            self._show_message("None of the chosen neurons is in the chosen arrays.")
            return
        muted = theme_module.muted_color(self._theme)
        axes = self.figure.subplots(len(rows), 1, sharex=True, squeeze=False)
        for ax, row in zip(axes[:, 0], rows, strict=True):
            for segment, values in zip(concatenation.segments, row.values, strict=True):
                ax.plot(segment.time_ms, values, color=segment.color, linewidth=0.8)
            for boundary in concatenation.boundaries_ms:
                ax.axvline(boundary, color=muted, linestyle="--", linewidth=0.8)
            ax.set_ylabel(neuron_ylabel(row.neuron_id, concatenation.units))
        top = axes[0][0]
        transform = blended_transform_factory(top.transData, top.transAxes)
        for segment in concatenation.segments:
            top.text(
                (segment.start_ms + segment.end_ms) / 2,
                0.97,
                segment.label,
                ha="center",
                va="top",
                color=segment.color,
                fontsize="small",
                transform=transform,
                clip_on=True,
            )
        self._finish_time_axes(axes, concatenation.total_ms)
        self.canvas.draw_idle()
