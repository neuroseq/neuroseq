"""Routine selector with an always-on preview for Design dynamic input.

Selecting a dynamic-input routine used to open its (dense) parameter dialog
immediately, giving no chance to understand what the routine does first. This
module provides a single selector window whose routine dropdown sits above a
live preview -- a one-line goal, the parameters the user will set, and a small
illustrative sample plot -- that is always shown and updates as the selection
changes. A Continue button then opens the real routine dialog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq import theme as theme_module


@dataclass(frozen=True)
class RoutineOverview:
    """Static, human-facing summary of a dynamic-input routine."""

    title: str
    goal: str
    parameters: tuple[tuple[str, str], ...]
    plot_title: str
    plot: Callable[[Axes], None]


def _plot_targeted_stdp(ax: Axes) -> None:
    iterations = np.arange(0, 12)
    target = 150.0
    epsp = target - (target - 100.0) * np.exp(-iterations / 3.0)
    ax.plot(iterations, epsp, "-o", color="#1f77b4", markersize=3)
    ax.axhline(target, ls="--", color="#d62728", label="target")
    ax.set_xlabel("STDP pairing iteration")
    ax.set_ylabel("EPSP (% baseline)")
    ax.legend(loc="lower right", fontsize="small")


def _plot_rheobase(ax: Axes) -> None:
    amplitude = np.linspace(0.0, 10.0, 200)
    peak_mv = -70.0 + 90.0 / (1.0 + np.exp(-(amplitude - 5.0) * 2.0))
    threshold = -20.0
    rheobase = amplitude[np.argmax(peak_mv >= threshold)]
    ax.plot(amplitude, peak_mv, color="#1f77b4")
    ax.axhline(threshold, ls="--", color="#d62728", label="spike threshold")
    ax.axvline(rheobase, ls=":", color="#2ca02c", label="rheobase")
    ax.set_xlabel("stimulus amplitude")
    ax.set_ylabel("peak Vm (mV)")
    ax.legend(loc="upper left", fontsize="small")


def _plot_stdp_window(ax: Axes) -> None:
    delta_t = np.linspace(-50.0, 50.0, 400)
    tau = 15.0
    delta_w = np.where(
        delta_t >= 0.0,
        np.exp(-delta_t / tau),
        -np.exp(delta_t / tau),
    )
    ax.plot(delta_t, delta_w, color="#1f77b4")
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.axvline(0.0, color="#888888", lw=0.8)
    ax.text(25.0, 0.6, "LTP", color="#2ca02c", fontsize="small")
    ax.text(-45.0, -0.7, "LTD", color="#d62728", fontsize="small")
    ax.set_xlabel("Δt = t_post − t_pre (ms)")
    ax.set_ylabel("Δ weight")


# Keyed by the routine identifiers in StimEditor.DYNAMIC_INPUT_ROUTINES.
ROUTINE_OVERVIEWS: dict[str, RoutineOverview] = {
    "Targeted STDP synaptic strength": RoutineOverview(
        title="Targeted STDP synaptic strength",
        goal=(
            "Closed-loop tuning of a single synapse's strength. The routine "
            "repeatedly measures the postsynaptic EPSP with five-pulse probes "
            "and applies pre/post STDP induction pairings, adapting the pairing "
            "until the EPSP settles at your chosen target."
        ),
        parameters=(
            ("Pre- and postsynaptic neurons", "the synapse to probe and drive"),
            ("Trace & synapse-weight signals", "what is read back each iteration"),
            ("Target EPSP", "the strength the loop converges toward"),
            ("Post-pulse window", "where the EPSP is measured"),
        ),
        plot_title="Drive the EPSP to a target",
        plot=_plot_targeted_stdp,
    ),
    "Minimal stimulus for spiking (rheobase)": RoutineOverview(
        title="Minimal stimulus for spiking (rheobase)",
        goal=(
            "Find the smallest stimulus that makes a neuron fire — its rheobase. "
            "The routine sweeps the pulse amplitude (or duration) with a bisection "
            "search, checking after each probe whether the neuron crossed the "
            "spike threshold, and reports the minimal value that elicits a spike."
        ),
        parameters=(
            ("Neuron & trace", "the cell to probe and the voltage read back"),
            ("Sweeping strategy", "vary amplitude or duration"),
            ("Amplitude / duration range", "search bounds for the sweep"),
            ("Spike threshold & post-pulse window", "how a spike is detected"),
        ),
        plot_title="Find the minimal spiking stimulus",
        plot=_plot_rheobase,
    ),
    "STDP timing window (LTP/LTD vs Δt)": RoutineOverview(
        title="STDP timing window (LTP/LTD vs Δt)",
        goal=(
            "Measure the STDP timing window: how a synapse's weight change "
            "depends on the pre/post spike interval Δt. The routine sweeps Δt, "
            "inducing one pairing per value and recording the weight change — "
            "potentiation (LTP) for pre-before-post, depression (LTD) otherwise."
        ),
        parameters=(
            ("Trace & synapse weight", "signals read to measure Δ weight"),
            ("Δt range & step", "the timing offsets swept"),
            ("Post-pulse window", "where the response is measured"),
            ("Sweep mode", "reset-to-baseline or continuous"),
        ),
        plot_title="Map LTP/LTD vs spike timing",
        plot=_plot_stdp_window,
    ),
}


def routine_overview(routine_name: str) -> RoutineOverview | None:
    """Return the overview for ``routine_name``, or ``None`` if unknown."""
    return ROUTINE_OVERVIEWS.get(routine_name)


class RoutineSelectorDialog(QtWidgets.QDialog):
    """Pick a dynamic-input routine with its preview always shown below.

    A single window holds the routine dropdown and, beneath it, a live preview
    (goal, parameters, sample plot) of the selected routine -- the preview is
    always on and updates as the selection changes, rather than living in a
    separate step. Emits :attr:`routine_chosen` with the selected routine name
    when the user clicks Continue; the caller opens the real routine dialog.
    """

    routine_chosen = QtCore.Signal(str)

    def __init__(
        self,
        routines: list[str],
        theme: str = theme_module.LIGHT,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowTitle("Design dynamic input")
        self.resize(520, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Design routine:", self))
        self.cmb_routine = QtWidgets.QComboBox(self)
        self.cmb_routine.addItems(routines)
        selector_row.addWidget(self.cmb_routine, stretch=1)
        layout.addLayout(selector_row)

        # --- Preview (always visible, updates with the selection) ---
        self.lbl_title = QtWidgets.QLabel(self)
        title_font = self.lbl_title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 1.5)
        self.lbl_title.setFont(title_font)
        self.lbl_title.setWordWrap(True)
        layout.addWidget(self.lbl_title)

        self.lbl_goal = QtWidgets.QLabel(self)
        self.lbl_goal.setWordWrap(True)
        layout.addWidget(self.lbl_goal)

        params_header = QtWidgets.QLabel("Parameters you'll set:", self)
        params_font = params_header.font()
        params_font.setBold(True)
        params_header.setFont(params_font)
        layout.addWidget(params_header)

        self.lbl_params = QtWidgets.QLabel(self)
        self.lbl_params.setWordWrap(True)
        layout.addWidget(self.lbl_params)

        self.plot_widget = MatplotlibWidget(size=(4.5, 2.6))
        # The sample plot is illustrative, so drop the interactive navigation
        # toolbar (pan/zoom/save) -- it's clutter here and frees vertical space.
        if getattr(self.plot_widget, "toolbar", None) is not None:
            self.plot_widget.toolbar.setVisible(False)
        self._axes = self.plot_widget.getFigure().add_subplot(111)
        layout.addWidget(self.plot_widget, stretch=1)

        sample_note = QtWidgets.QLabel(
            "Sample plot — illustrative shape only, not your data.", self
        )
        sample_note.setWordWrap(True)
        note_font = sample_note.font()
        note_font.setItalic(True)
        sample_note.setFont(note_font)
        layout.addWidget(sample_note)

        button_box = QtWidgets.QDialogButtonBox(self)
        self.btn_continue = button_box.addButton(
            "Continue", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_continue)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.cmb_routine.currentTextChanged.connect(self._update_preview)
        self._update_preview(self.cmb_routine.currentText())

    @property
    def selected_routine(self) -> str:
        return self.cmb_routine.currentText()

    def _update_preview(self, routine_name: str) -> None:
        overview = routine_overview(routine_name)
        if self.btn_continue is not None:
            self.btn_continue.setEnabled(overview is not None)
        if overview is None:
            self.lbl_title.setText(routine_name)
            self.lbl_goal.setText("No preview available for this routine.")
            self.lbl_params.clear()
            self._axes.clear()
            self.plot_widget.draw()
            return
        self.lbl_title.setText(overview.title)
        self.lbl_goal.setText(overview.goal)
        self.lbl_params.setText(
            "\n".join(
                f"• {name} — {description}" for name, description in overview.parameters
            )
        )
        self._axes.clear()
        overview.plot(self._axes)
        self._axes.set_title(overview.plot_title)
        self._recolor_axes()
        self.plot_widget.getFigure().tight_layout()
        self.plot_widget.draw()

    def _on_continue(self) -> None:
        self.routine_chosen.emit(self.selected_routine)
        self.accept()

    def apply_theme(self, theme: str) -> None:
        """Recolor the sample plot to match the application theme."""
        self._theme = theme
        self._recolor_axes()
        self.plot_widget.draw()

    def _recolor_axes(self) -> None:
        face, text = theme_module.mpl_colors(self._theme)
        self.plot_widget.getFigure().set_facecolor(face)
        self._axes.set_facecolor(face)
        self._axes.tick_params(colors=text)
        for spine in self._axes.spines.values():
            spine.set_color(text)
        for axis_label in (self._axes.xaxis.label, self._axes.yaxis.label):
            axis_label.set_color(text)
        self._axes.title.set_color(text)
