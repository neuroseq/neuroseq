from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from qtpy import QtCore

from neuroseq.stimulation.dynamic_input.dynamic_input_routine import DynamicInputDialog
from neuroseq.stimulation.dynamic_input.rheobase_dialog import RheobaseDialog
from neuroseq.stimulation.dynamic_input.routine_overview_dialog import (
    RoutineSelectorDialog,
)
from neuroseq.stimulation.dynamic_input.stdp_synaptic_tuning_dialog import (
    StdpSynapticTuningDialog,
)
from neuroseq.stimulation.dynamic_input.stdp_window_dialog import StdpWindowDialog

if TYPE_CHECKING:
    from neuroseq.stimulation.stim_editor import StimEditor


class DynamicInputController(QtCore.QObject):
    """Own dynamic-input routines, using the editor as an editing service."""

    simulation_requested = QtCore.Signal(float)

    # Dynamic-input routines offered when the user opens the designer.
    STDP_SYNAPTIC_TUNING_ROUTINE = "STDP synaptic tuning"
    RHEOBASE_ROUTINE = "Minimal input for spiking (rheobase)"
    STDP_WINDOW_ROUTINE = "STDP timing window (LTP/LTD vs Δt)"
    DYNAMIC_INPUT_ROUTINES = (
        STDP_SYNAPTIC_TUNING_ROUTINE,
        RHEOBASE_ROUTINE,
        STDP_WINDOW_ROUTINE,
    )
    #: Preselected when the designer opens.
    DEFAULT_ROUTINE = RHEOBASE_ROUTINE

    def __init__(
        self, editor: StimEditor, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.editor = editor
        self._stdp_synaptic_tuning_dialog: StdpSynapticTuningDialog | None = None
        self._rheobase_dialog: RheobaseDialog | None = None
        self._stdp_window_dialog: StdpWindowDialog | None = None
        self._routine_selector_dialog: RoutineSelectorDialog | None = None
        self._dynamic_input_theme = "light"

    def open_dynamic_input_designer(self) -> None:
        if self.editor.current_stim_modifier is None or self.editor.project is None:
            return
        # A single selector window shows the routine dropdown with an always-on
        # preview (goal, parameters, sample plot) beneath it; the real routine
        # dialog opens only when the user clicks Continue.
        dialog = RoutineSelectorDialog(
            list(self.DYNAMIC_INPUT_ROUTINES),
            self._dynamic_input_theme,
            default=self.DEFAULT_ROUTINE,
        )
        dialog.routine_chosen.connect(self._open_routine_dialog_for)
        dialog.finished.connect(lambda _result: self._forget_routine_selector(dialog))
        dialog.finished.connect(dialog.deleteLater)
        self._routine_selector_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_routine_selector(self, dialog: RoutineSelectorDialog) -> None:
        # The selector deletes itself once closed; drop the reference so
        # discard_dynamic_input_dialogs does not close a deleted C++ object.
        if self._routine_selector_dialog is dialog:
            self._routine_selector_dialog = None

    def _open_routine_dialog_for(self, routine: str) -> None:
        if routine == self.STDP_SYNAPTIC_TUNING_ROUTINE:
            self._show_routine_dialog(
                "_stdp_synaptic_tuning_dialog", StdpSynapticTuningDialog
            )
        elif routine == self.RHEOBASE_ROUTINE:
            self._show_routine_dialog("_rheobase_dialog", RheobaseDialog)
        elif routine == self.STDP_WINDOW_ROUTINE:
            self._show_routine_dialog("_stdp_window_dialog", StdpWindowDialog)

    def _show_routine_dialog(
        self,
        attr_name: Literal[
            "_stdp_synaptic_tuning_dialog", "_rheobase_dialog", "_stdp_window_dialog"
        ],
        cls: type[StdpSynapticTuningDialog]
        | type[RheobaseDialog]
        | type[StdpWindowDialog],
    ) -> None:
        dialog = getattr(self, attr_name)
        if dialog is None:
            # No Qt parent: a parent sets WM_TRANSIENT_FOR to the main
            # window, and on X11/KWin the routine dialog and its own two
            # popups (also parentless - see StimHeatmapWindow's
            # construction in stdp_synaptic_tuning_dialog.py) would then all
            # share one owner. KWin can treat that whole chain as a single
            # "raise group" tied to the app's own foreground status, so
            # none of them can be individually brought to the front of the
            # others by clicking - not even this one.
            dialog = cls(self.editor, None)
            dialog.simulation_requested.connect(self.simulation_requested)
            setattr(self, attr_name, dialog)
        dialog.apply_theme(self._dynamic_input_theme)
        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def discard_dynamic_input_dialogs(self) -> None:
        """Drop any cached Design dynamic input dialogs.

        Called when the project is replaced so the designer does not carry a
        previous run's state (plots, log, history) into a new project. Dialogs
        are rebuilt fresh the next time they are opened.
        """
        for attr_name in (
            "_stdp_synaptic_tuning_dialog",
            "_rheobase_dialog",
            "_stdp_window_dialog",
            "_routine_selector_dialog",
        ):
            dialog = getattr(self, attr_name)
            if dialog is not None:
                # Release a SysCon2 block still in flight for this routine.
                dispose = getattr(dialog, "dispose", None)
                if dispose is not None:
                    dispose()
                dialog.close()
                dialog.deleteLater()
                setattr(self, attr_name, None)

    def dynamic_input_simulation_finished(self) -> None:
        for dialog in self._dynamic_input_dialogs():
            dialog.simulation_finished()

    def dynamic_input_simulation_failed(self, message: str) -> None:
        for dialog in self._dynamic_input_dialogs():
            dialog.simulation_failed(message)

    def apply_dynamic_input_theme(self, theme: str) -> None:
        self._dynamic_input_theme = theme
        for dialog in self._dynamic_input_dialogs():
            dialog.apply_theme(theme)

    def _dynamic_input_dialogs(
        self,
    ) -> list[DynamicInputDialog]:
        return [
            dialog
            for dialog in (
                self._stdp_synaptic_tuning_dialog,
                self._rheobase_dialog,
                self._stdp_window_dialog,
            )
            if dialog is not None
        ]

    def refresh(self) -> None:
        for dialog in self._dynamic_input_dialogs():
            dialog.refresh()
