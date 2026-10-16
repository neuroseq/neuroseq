from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore

from neuroseq.static_input.stim_pattern_dialog import StimPatternContext
from neuroseq.static_input.stim_pattern_dialog import StimPatternDialog

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim_editor import StimEditor


class StaticInputController(QtCore.QObject):
    """Manage the static-input designer using the editor as an editing service."""

    def __init__(
        self, editor: StimEditor, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.editor = editor
        self._pattern_dialog: StimPatternDialog | None = None
        editor.project_changed.connect(self._on_project_changed)
        editor.current_stim_changed.connect(self._refresh_context)
        editor.current_stim_cleared.connect(self._reset_start_time)
        editor.destroyed.connect(self.discard_dialog)

    def _on_project_changed(self, outgoing: Project | None, incoming: Project) -> None:
        if self._pattern_dialog is None:
            return
        if outgoing is not None:
            outgoing.static_input_state = self._pattern_dialog.capture_state()
        self._pattern_dialog.restore_state(incoming.static_input_state)

    def _current_pattern_context(self) -> StimPatternContext:
        if self.editor.current_stim_modifier is None:
            raise ValueError("No current stim array selected.")
        stim_array = self.editor.current_stim_modifier.stim_array
        return StimPatternContext(
            stim_name=stim_array.name,
            units=stim_array.units,
            dt_seconds=float(stim_array.dt),
            n_neurons=int(stim_array.values.shape[1]),
        )

    def _refresh_context(self, _stim_array: StimArray | None) -> None:
        if self._pattern_dialog is None:
            return
        available = self.editor.current_stim_modifier is not None
        self._pattern_dialog.setEnabled(available)
        if available:
            self._pattern_dialog.set_context(self._current_pattern_context())

    def _reset_start_time(self) -> None:
        if self._pattern_dialog is not None:
            self._pattern_dialog.reset_start_time()

    def open_pattern_designer(self) -> None:
        if self.editor.current_stim_modifier is None or self.editor.project is None:
            return
        if self._pattern_dialog is None:
            self._pattern_dialog = StimPatternDialog(self.editor)
            self._pattern_dialog.restore_state(self.editor.project.static_input_state)
        self._refresh_context(self.editor.current_stim_modifier.stim_array)
        self._pattern_dialog.show()
        self._pattern_dialog.raise_()
        self._pattern_dialog.activateWindow()

    def discard_dialog(self) -> None:
        if self._pattern_dialog is not None:
            if self.editor.project is not None:
                self.editor.project.static_input_state = (
                    self._pattern_dialog.capture_state()
                )
            self._pattern_dialog.close()
            self._pattern_dialog.deleteLater()
            self._pattern_dialog = None
