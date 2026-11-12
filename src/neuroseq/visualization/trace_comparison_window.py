"""Popup comparing chosen neurons' traces across several trace arrays.

Opened from the trace view's context menu (**Compare traces…**). The window
is read-only: it copies nothing back into the project and records no undo
step. It draws with Matplotlib rather than fastplotlib: a figure with one
subplot per neuron, a shared (linked) time axis, a legend and a navigation
toolbar is a few lines there, whereas fastplotlib (0.6) has no legends and
every popup would need its own GPU canvas per subplot. The trace view's
vector preview and the custom analysis popups use Matplotlib for the same
reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import TraceData
from neuroseq.visualization.trace_comparison_dialog import TraceComparisonDialog
from neuroseq.visualization_support.time_format import format_time_ms
from neuroseq.visualization_support.trace_comparison import TraceComparison
from neuroseq.visualization_support.trace_comparison import build_trace_comparison

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from neuroseq.project import Project

# Height of one neuron row in the figure, in inches.
ROW_HEIGHT_IN = 1.6
# Extra figure height for the legend and the time axis, in inches.
FIGURE_MARGIN_IN = 1.2

# Parentless comparison windows, kept alive until they are closed.
_open_windows: list[QtWidgets.QWidget] = []


class TraceComparisonWindow(QtWidgets.QWidget):
    """Top-level, non-modal window with one row per neuron.

    Each row overlays that neuron's trace from every compared array in the
    array's legend colour; **Side by side** instead gives every array its own
    column. All rows share one linked time axis; each array is drawn on its
    own time axis, so arrays with different ``dt`` or length stay aligned.
    """

    def __init__(
        self,
        comparison: TraceComparison,
        theme: str = theme_module.LIGHT,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Trace comparison")
        self.resize(1000, 700)
        self.comparison = comparison
        self._theme = theme_module.normalize_theme(theme)

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.side_by_side_checkbox = QtWidgets.QCheckBox("Side by side", self)
        self.side_by_side_checkbox.setToolTip(
            "Give each trace array its own column instead of overlaying them"
        )
        self.side_by_side_checkbox.toggled.connect(self._redraw)
        controls.addWidget(self.side_by_side_checkbox)
        controls.addStretch()
        layout.addLayout(controls)

        self.note_label = QtWidgets.QLabel(comparison.missing_note(), self)
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(
            f"color: {theme_module.warning_color(self._theme)};"
        )
        self.note_label.setVisible(bool(self.note_label.text()))
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
        self._redraw()

    @property
    def side_by_side(self) -> bool:
        return self.side_by_side_checkbox.isChecked()

    def _redraw(self, *_args) -> None:
        figure = self.figure
        figure.clear()
        rows = self.comparison.rows
        legend = self.comparison.legend
        face, text = theme_module.mpl_colors(self._theme)
        n_rows = max(1, len(rows))
        self.canvas.setMinimumHeight(
            int((ROW_HEIGHT_IN * n_rows + FIGURE_MARGIN_IN) * figure.dpi)
        )
        if not rows:
            ax = figure.add_subplot(111, facecolor=face)
            ax.text(
                0.5,
                0.5,
                "None of the chosen neurons is in the chosen arrays.",
                ha="center",
                va="center",
                color=text,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        n_cols = len(legend) if self.side_by_side else 1
        axes = figure.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey="row" if self.side_by_side else False,
            squeeze=False,
        )
        column_of = {label: column for column, (label, _) in enumerate(legend)}
        units = self.comparison.units
        for row_index, row in enumerate(rows):
            for series in row.series:
                column = column_of[series.label] if self.side_by_side else 0
                axes[row_index][column].plot(
                    series.time_ms,
                    series.values,
                    color=series.color,
                    linewidth=0.8,
                    label=series.label,
                )
            if self.side_by_side:
                present = {series.label for series in row.series}
                for label, column in column_of.items():
                    if label not in present:
                        axes[row_index][column].text(
                            0.5,
                            0.5,
                            "not in this array",
                            ha="center",
                            va="center",
                            color=theme_module.muted_color(self._theme),
                            transform=axes[row_index][column].transAxes,
                        )
            ylabel = f"Neuron {row.neuron_id}"
            if units:
                ylabel += f"\n({units})"
            axes[row_index][0].set_ylabel(ylabel)
        if self.side_by_side:
            for (label, color), ax in zip(legend, axes[0], strict=True):
                ax.set_title(label, color=color, fontsize="medium")
        for ax in axes[-1]:
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(_time_formatter(ax))
        max_time = self.comparison.max_time_ms
        if max_time > 0:
            axes[0][0].set_xlim(0.0, max_time)
        for ax in axes.flat:
            _style_axes(ax, face, text)
        handles = [
            Line2D([], [], color=color, linewidth=2, label=label)
            for label, color in legend
        ]
        figure_legend = figure.legend(
            handles=handles,
            loc="outside upper center",
            ncols=min(len(handles), 6),
            frameon=False,
        )
        for legend_text in figure_legend.get_texts():
            legend_text.set_color(text)
        self.canvas.draw_idle()

    def closeEvent(self, event) -> None:
        self.figure.clear()
        super().closeEvent(event)


def _time_formatter(ax: Axes) -> FuncFormatter:
    """Tick labels in ms or s, precision following the visible range."""

    def format_tick(value: float, _position: int | None) -> str:
        low, high = ax.get_xlim()
        return format_time_ms(value, high - low, strip_zeros=True)

    return FuncFormatter(format_tick)


def _style_axes(ax: Axes, face: str, text: str) -> None:
    ax.set_facecolor(face)
    ax.tick_params(colors=text, labelsize="small")
    for spine in ax.spines.values():
        spine.set_color(text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)


def _current_array_name(
    project: Project, names: Sequence[str], data: TraceData | None
) -> str | None:
    """The project data name under which ``data`` is listed, if any."""
    if data is None:
        return None
    for name in names:
        # Simulated traces are rebuilt on every get_data() and carry their
        # full project name; imported arrays are returned as stored.
        if name == data.name:
            return name
    for name in names:
        if (
            name.split(" :: ", 1)[0] != "Simulation"
            and project.get_data(name, TraceData) is data
        ):
            return name
    return None


def open_trace_comparison(
    project: Project,
    data: TraceData | None,
    preselected_neuron_ids: Sequence[int] = (),
    *,
    theme: str = theme_module.LIGHT,
    parent: QtWidgets.QWidget | None = None,
    title_suffix: str = "",
) -> TraceComparisonDialog:
    """Ask which neurons and arrays to compare, then open the comparison popup.

    The neuron list offers the ids of ``data`` (the array the trace view
    shows), ticking ``preselected_neuron_ids``; every trace array of the
    project is listed with the current one ticked. Accepting the dialog
    opens a new :class:`TraceComparisonWindow` (stored on the dialog as
    ``comparison_window``); several may be open at once.
    """
    names = project.get_data_names(TraceData)
    current = _current_array_name(project, names, data)
    neuron_ids = sorted({int(i) for i in data.indices}) if data is not None else []
    dialog = TraceComparisonDialog(
        neuron_ids,
        names,
        checked_neuron_ids=preselected_neuron_ids,
        checked_array_names=[current] if current is not None else [],
        parent=parent,
    )
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.comparison_window = None  # type: ignore[attr-defined]

    def _open_window() -> None:
        arrays: list[tuple[str, TraceData]] = []
        for name in dialog.selected_array_names:
            # Imported arrays are read as stored; nothing is written back.
            if name == current and data is not None:
                arrays.append((name, data))
                continue
            loaded = project.get_data(name, TraceData)
            if loaded is not None:
                arrays.append((name, loaded))
        comparison = build_trace_comparison(arrays, dialog.selected_neuron_ids)
        window = TraceComparisonWindow(comparison, theme=theme, parent=parent)
        title = "Trace comparison"
        if title_suffix:
            title = f"{title} - {title_suffix}"
        window.setWindowTitle(title)
        if parent is None:
            _open_windows.append(window)
            window.destroyed.connect(
                lambda *_: _open_windows.remove(window)
                if window in _open_windows
                else None
            )
        window.show()
        dialog.comparison_window = window  # type: ignore[attr-defined]

    dialog.accepted.connect(_open_window)
    dialog.open()
    return dialog
