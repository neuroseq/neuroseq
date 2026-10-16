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
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtSvg import QSvgRenderer

from neuroseq import theme as theme_module


class _SelfRaisingDialog(QtWidgets.QDialog):
    """A non-modal QDialog that raises itself above its siblings whenever it
    is activated (clicked into focus).

    Several window managers raise an entire transient-dialog chain as one
    unit on click, rather than reordering siblings within it - so clicking
    one of several sibling popups (all transient for the same parent
    routine dialog: the parameters dialog, its plots window, and its
    stim-pattern heatmap window) does not bring it in front of the others;
    the whole group just moves together, at whatever relative order the
    window manager already had. Explicitly re-raising on ActivationChange
    bypasses that window-manager heuristic and enforces the expected
    last-clicked-is-on-top order.
    """

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.raise_()


_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


@dataclass(frozen=True)
class RoutineOverview:
    """Static, human-facing summary of a dynamic-input routine.

    A preview is either a matplotlib sample plot (``plot``) or a static image
    (``image_path``, e.g. a bundled SVG diagram) -- exactly one is set.
    """

    title: str
    goal: str
    parameters: tuple[tuple[str, str], ...]
    plot_title: str = ""
    plot: Callable[[Axes], None] | None = None
    image_path: Path | None = None


def _plot_stdp_synaptic_tuning(ax: Axes) -> None:
    iterations = np.arange(0, 12)
    target = 150.0
    epsp = target - (target - 100.0) * np.exp(-iterations / 3.0)
    ax.plot(iterations, epsp, "-o", color="#1f77b4", markersize=3)
    ax.axhline(target, ls="--", color="#d62728", label="target")
    ax.set_xlabel("STDP pairing iteration")
    ax.set_ylabel("EPSP (% baseline)")
    ax.legend(loc="lower right", fontsize="small")


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


# Keyed by the routine identifiers in DynamicInputController.DYNAMIC_INPUT_ROUTINES.
ROUTINE_OVERVIEWS: dict[str, RoutineOverview] = {
    "STDP synaptic tuning": RoutineOverview(
        title="STDP synaptic tuning",
        goal=(
            "Closed-loop tuning of one synapse's strength: probe the EPSP, "
            "pair pre/post spikes via STDP, and adapt until the EPSP reaches "
            "your target."
        ),
        parameters=(
            ("Pre- and postsynaptic neurons", "the synapse to probe and drive"),
            ("Trace & synapse-weight signals", "what is read back each iteration"),
            ("Target EPSP", "the strength the loop converges toward"),
            ("Post-pulse window", "where the EPSP is measured"),
        ),
        plot_title="Drive the EPSP to a target",
        plot=_plot_stdp_synaptic_tuning,
    ),
    "Minimal input for spiking (rheobase)": RoutineOverview(
        title="Minimal input for spiking (rheobase)",
        goal=(
            "Find a neuron's rheobase, the smallest input that makes it fire. "
            "Bisection-searches pulse amplitude or duration for the minimal "
            "value that triggers a spike."
        ),
        parameters=(
            ("Neuron & trace", "the cell to probe and the voltage read back"),
            ("Sweeping strategy", "vary amplitude or duration"),
            ("Amplitude / duration range", "search bounds for the sweep"),
            ("Spike threshold & post-pulse window", "how a spike is detected"),
        ),
        # Strength-duration curve (rheobase & chronaxie), a public-domain
        # diagram by Savvas Radevic bundled under resources/.
        image_path=_RESOURCES_DIR / "rheobase_chronaxie.svg",
    ),
    "STDP timing window (LTP/LTD vs Δt)": RoutineOverview(
        title="STDP timing window (LTP/LTD vs Δt)",
        goal=(
            "Measure the STDP timing window: how weight change depends on "
            "spike timing Δt. Potentiation (LTP) for pre-before-post, "
            "depression (LTD) otherwise."
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


class _SvgImageWidget(QtWidgets.QWidget):
    """Render an SVG scaled to fit the widget while preserving aspect ratio."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = QSvgRenderer()
        self.setMinimumHeight(180)

    def load(self, path: Path) -> None:
        self._renderer.load(str(path))
        self.update()

    def is_valid(self) -> bool:
        return self._renderer.isValid()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        if not self._renderer.isValid():
            return
        target = self._renderer.defaultSize()
        target.scale(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - target.width()) / 2.0
        y = (self.height() - target.height()) / 2.0
        rect = QtCore.QRectF(x, y, target.width(), target.height())
        painter = QtGui.QPainter(self)
        # The diagram uses fixed dark strokes; paint it on a white card so it
        # stays legible on a dark-theme dialog background.
        painter.fillRect(rect, QtGui.QColor("white"))
        self._renderer.render(painter, rect)


class RoutineSelectorDialog(_SelfRaisingDialog):
    """Pick a dynamic-input routine with its preview always shown below.

    A single window holds the routine dropdown and, beneath it, a live preview
    (goal, parameters, sample plot) of the selected routine -- the preview is
    always on and updates as the selection changes, rather than living in a
    separate step. Emits :attr:`routine_chosen` with the selected routine name
    when the user clicks Continue; the caller opens the real routine dialog.

    No Qt parent (see _SelfRaisingDialog's docstring): a parent sets
    WM_TRANSIENT_FOR, and on X11/KWin a transient dialog can end up sharing
    its owner's "raise group" so clicking it never brings it above another
    sibling popup (e.g. a routine dialog still open from a previous choice).
    """

    routine_chosen = QtCore.Signal(str)

    def __init__(
        self,
        routines: list[str],
        theme: str = theme_module.LIGHT,
    ) -> None:
        super().__init__()
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

        # --- Preview (always visible, updates with the selection) - in its
        # own scroll area so the info and plot both still fit, scrolled, on
        # a small screen instead of getting clipped or squeezed. The selector
        # above and the buttons below stay fixed, out of the scrolled area.
        preview = QtWidgets.QWidget(self)
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 4, 0)
        preview_layout.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel(preview)
        title_font = self.lbl_title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 1.5)
        self.lbl_title.setFont(title_font)
        self.lbl_title.setWordWrap(True)
        preview_layout.addWidget(self.lbl_title)

        self.lbl_goal = QtWidgets.QLabel(preview)
        self.lbl_goal.setWordWrap(True)
        preview_layout.addWidget(self.lbl_goal)

        params_header = QtWidgets.QLabel("Parameters you'll set:", preview)
        params_font = params_header.font()
        params_font.setBold(True)
        params_header.setFont(params_font)
        preview_layout.addWidget(params_header)

        self.lbl_params = QtWidgets.QLabel(preview)
        self.lbl_params.setWordWrap(True)
        preview_layout.addWidget(self.lbl_params)

        self.plot_widget = MatplotlibWidget(size=(4.5, 2.6))
        # The sample plot is illustrative, so drop the interactive navigation
        # toolbar (pan/zoom/save) -- it's clutter here and frees vertical space.
        if getattr(self.plot_widget, "toolbar", None) is not None:
            self.plot_widget.toolbar.setVisible(False)
        self._axes = self.plot_widget.getFigure().add_subplot(111)
        preview_layout.addWidget(self.plot_widget, stretch=1)

        # Some routines preview a static diagram (SVG) rather than a plot; only
        # one of the two preview widgets is visible at a time.
        self.image_widget = _SvgImageWidget(preview)
        self.image_widget.setVisible(False)
        preview_layout.addWidget(self.image_widget, stretch=1)

        sample_note = QtWidgets.QLabel("Illustration only, not your data.", preview)
        sample_note.setWordWrap(True)
        note_font = sample_note.font()
        note_font.setItalic(True)
        sample_note.setFont(note_font)
        preview_layout.addWidget(sample_note)

        preview_scroll = QtWidgets.QScrollArea(self)
        preview_scroll.setWidget(preview)
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(preview_scroll, stretch=1)

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
            self.plot_widget.setVisible(True)
            self.image_widget.setVisible(False)
            return
        self.lbl_title.setText(overview.title)
        self.lbl_goal.setText(overview.goal)
        self.lbl_params.setText(
            "\n".join(
                f"• {name}: {description}" for name, description in overview.parameters
            )
        )
        if overview.image_path is not None:
            self.image_widget.load(overview.image_path)
            self.image_widget.setVisible(True)
            self.plot_widget.setVisible(False)
        elif overview.plot is not None:
            self._axes.clear()
            overview.plot(self._axes)
            self._axes.set_title(overview.plot_title)
            self._recolor_axes()
            self.plot_widget.getFigure().tight_layout()
            self.plot_widget.draw()
            self.plot_widget.setVisible(True)
            self.image_widget.setVisible(False)

    def _on_continue(self) -> None:
        self.routine_chosen.emit(self.selected_routine)
        self.accept()

    def apply_theme(self, theme: str) -> None:
        """No-op for the sample plot's own colors: the preview always stays
        white/black regardless of the app theme (kept for the caller's sake,
        since every other dynamic-input dialog is themed this way)."""
        self._theme = theme
        self._recolor_axes()
        self.plot_widget.draw()

    def _recolor_axes(self) -> None:
        self.plot_widget.getFigure().set_facecolor("white")
        self._axes.set_facecolor("white")
        self._axes.tick_params(colors="black")
        for spine in self._axes.spines.values():
            spine.set_color("black")
        for axis_label in (self._axes.xaxis.label, self._axes.yaxis.label):
            axis_label.set_color("black")
        self._axes.title.set_color("black")
