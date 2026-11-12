"""Popups comparing chosen neurons' traces across several trace arrays.

Opened from the trace view's context menu (**Compare traces…**). The dialog's
**View** picks one of four presentations (#662):

- overlay per neuron and side by side per neuron (:class:`TraceComparisonWindow`),
- one window per array with linked axes (``LinkedArrayWindow``),
- the arrays joined end to end (``ConcatenatedTraceWindow``).

The windows are read-only. Only the concatenated view can, when asked, store
the joined traces as a new trace array, through ``Project.add_experiment`` so
the step is undoable. Drawing uses Matplotlib rather than fastplotlib: a
figure with one subplot per neuron, a shared (linked) time axis, a legend and
a navigation toolbar is a few lines there, whereas fastplotlib (0.6) has no
legends and every popup would need its own GPU canvas per subplot. The trace
view's vector preview and the custom analysis popups use Matplotlib for the
same reason.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

from matplotlib.lines import Line2D
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme as theme_module
from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.visualization.trace_comparison_dialog import TraceComparisonDialog
from neuroseq.visualization.trace_comparison_views import ComparisonPopup
from neuroseq.visualization.trace_comparison_views import ConcatenatedTraceWindow
from neuroseq.visualization.trace_comparison_views import LinkedArrayWindow
from neuroseq.visualization.trace_comparison_views import neuron_ylabel
from neuroseq.visualization_support.trace_comparison import AxisLinkGroup
from neuroseq.visualization_support.trace_comparison import ComparisonView
from neuroseq.visualization_support.trace_comparison import TraceComparison
from neuroseq.visualization_support.trace_comparison import array_labels
from neuroseq.visualization_support.trace_comparison import build_trace_comparison
from neuroseq.visualization_support.trace_comparison import build_trace_concatenation
from neuroseq.visualization_support.trace_comparison import cascade_offsets
from neuroseq.visualization_support.trace_comparison import concatenation_data_name

if TYPE_CHECKING:
    from neuroseq.project import Project

# Parentless comparison windows, kept alive until they are closed.
_open_windows: list[QtWidgets.QWidget] = []


class TraceComparisonWindow(ComparisonPopup):
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
        *,
        side_by_side: bool = False,
    ) -> None:
        super().__init__(comparison.missing_note(), theme=theme, parent=parent)
        self.setWindowTitle("Trace comparison")
        self.comparison = comparison

        self.side_by_side_checkbox = QtWidgets.QCheckBox("Side by side", self)
        self.side_by_side_checkbox.setToolTip(
            "Give each trace array its own column instead of overlaying them"
        )
        self.side_by_side_checkbox.setChecked(side_by_side)
        self.side_by_side_checkbox.toggled.connect(self._redraw)
        self.controls.addWidget(self.side_by_side_checkbox)
        self.controls.addStretch()
        self._redraw()

    @property
    def side_by_side(self) -> bool:
        return self.side_by_side_checkbox.isChecked()

    def _redraw(self, *_args) -> None:
        figure = self.figure
        figure.clear()
        rows = self.comparison.rows
        legend = self.comparison.legend
        _, text = theme_module.mpl_colors(self._theme)
        self._fit_rows(len(rows))
        if not rows:
            self._show_message("None of the chosen neurons is in the chosen arrays.")
            return

        n_cols = len(legend) if self.side_by_side else 1
        axes = figure.subplots(
            len(rows),
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
            axes[row_index][0].set_ylabel(neuron_ylabel(row.neuron_id, units))
        if self.side_by_side:
            for (label, color), ax in zip(legend, axes[0], strict=True):
                ax.set_title(label, color=color, fontsize="medium")
        self._finish_time_axes(axes, self.comparison.max_time_ms)
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


def _keep_alive(window: QtWidgets.QWidget, parent: QtWidgets.QWidget | None) -> None:
    """Hold a parentless popup until it is closed (a parent owns the others)."""
    if parent is not None:
        return
    _open_windows.append(window)
    window.destroyed.connect(
        lambda *_: _open_windows.remove(window) if window in _open_windows else None
    )


def _titled(title: str, suffix: str) -> str:
    return f"{title} - {suffix}" if suffix else title


def open_trace_comparison(
    project: Project,
    data: TraceData | None,
    preselected_neuron_ids: Sequence[int] = (),
    *,
    theme: str = theme_module.LIGHT,
    parent: QtWidgets.QWidget | None = None,
    title_suffix: str = "",
    on_data_added: Callable[[str], None] | None = None,
) -> TraceComparisonDialog:
    """Ask which neurons and arrays to compare and how, then open the popup(s).

    The neuron list offers the ids of ``data`` (the array the trace view
    shows), ticking ``preselected_neuron_ids``; every trace array of the
    project is listed with the current one ticked. Accepting the dialog opens
    the chosen view; the opened windows are stored on the dialog as
    ``comparison_windows`` (``comparison_window`` is the first). Several may
    be open at once. When the concatenated view also adds the joined traces
    to the project, ``on_data_added`` is called with the new group's name.
    """
    names = project.get_data_names(TraceData)
    current = _current_array_name(project, names, data)
    neuron_ids = sorted({int(i) for i in data.indices}) if data is not None else []

    def _load(name: str) -> TraceData | None:
        # Imported arrays are read as stored; nothing is written back.
        if name == current and data is not None:
            return data
        return project.get_data(name, TraceData)

    def _dt(name: str) -> float | None:
        loaded = _load(name)
        return float(loaded.dt) if loaded is not None else None

    dialog = TraceComparisonDialog(
        neuron_ids,
        names,
        checked_neuron_ids=preselected_neuron_ids,
        checked_array_names=[current] if current is not None else [],
        parent=parent,
        array_dt=_dt,
        existing_group_names=list(project.experiments),
    )
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.comparison_window = None  # type: ignore[attr-defined]
    dialog.comparison_windows = []  # type: ignore[attr-defined]

    def _open_windows_for_choice() -> None:
        arrays: list[tuple[str, TraceData]] = []
        for name in dialog.selected_array_names:
            loaded = _load(name)
            if loaded is not None:
                arrays.append((name, loaded))
        neuron_choice = dialog.selected_neuron_ids
        view = dialog.view
        windows: list[ComparisonPopup] = []
        if view is ComparisonView.CONCATENATED:
            windows.append(_open_concatenation(arrays, neuron_choice))
        elif view is ComparisonView.SEPARATE_WINDOWS:
            comparison = build_trace_comparison(arrays, neuron_choice)
            group = AxisLinkGroup[LinkedArrayWindow]()
            for label, _ in comparison.legend:
                window = LinkedArrayWindow(
                    comparison,
                    label,
                    group,
                    lock_time=dialog.lock_time,
                    lock_value=dialog.lock_value,
                    theme=theme,
                    parent=parent,
                )
                window.setWindowTitle(_titled(f"Trace array {label}", title_suffix))
                windows.append(window)
        else:
            comparison = build_trace_comparison(arrays, neuron_choice)
            window = TraceComparisonWindow(
                comparison,
                theme=theme,
                parent=parent,
                side_by_side=view is ComparisonView.SIDE_BY_SIDE,
            )
            window.setWindowTitle(_titled("Trace comparison", title_suffix))
            windows.append(window)
        for window in windows:
            _keep_alive(window, parent)
            window.show()
        if len(windows) > 1:
            origin = windows[0].pos()
            for window, (dx, dy) in zip(
                windows[1:], cascade_offsets(len(windows))[1:], strict=True
            ):
                window.move(origin + QtCore.QPoint(dx, dy))
        dialog.comparison_windows = windows  # type: ignore[attr-defined]
        dialog.comparison_window = windows[0] if windows else None  # type: ignore[attr-defined]

    def _open_concatenation(
        arrays: list[tuple[str, TraceData]], neuron_choice: list[int]
    ) -> ConcatenatedTraceWindow:
        concatenation = build_trace_concatenation(arrays, neuron_choice, dialog.gap_ms)
        window = ConcatenatedTraceWindow(concatenation, theme=theme, parent=parent)
        window.setWindowTitle(_titled("Concatenated traces", title_suffix))
        if dialog.add_to_project and concatenation.rows:
            group_name = dialog.new_array_name
            data_name = concatenation_data_name([name for name, _ in arrays])
            trace = concatenation.to_trace_data(data_name)
            experiment = Experiment(
                name=group_name,
                duration=float(trace.traces.shape[1]) * float(trace.dt),
                timepoints=[],
                data=[trace],
                description="Concatenated from: "
                + ", ".join(array_labels([name for name, _ in arrays])),
            )
            try:
                project.add_experiment(experiment)
            except ValueError as error:
                QtWidgets.QMessageBox.warning(
                    parent, "Could not add trace array", str(error)
                )
            else:
                window.add_note(
                    f"Added to the project as '{group_name} :: {data_name}' "
                    "(Undo removes it)."
                )
                if on_data_added is not None:
                    on_data_added(group_name)
        return window

    dialog.accepted.connect(_open_windows_for_choice)
    dialog.open()
    return dialog
