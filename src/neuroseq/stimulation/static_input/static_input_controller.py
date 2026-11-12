from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore

from neuroseq.stimulation.static_input.stim_pattern_dialog import StimPatternContext
from neuroseq.stimulation.static_input.stim_pattern_dialog import StimPatternDialog

if TYPE_CHECKING:
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
        # The designer's values per input array (#648): a newly selected
        # array starts from the defaults, one selected again gets back what
        # was entered for it. Kept per workspace, never shared across projects.
        self._states: dict[str, dict[str, float | int | str | bool]] = {}
        self._dialog_stim_name: str | None = None
        editor.current_stim_changed.connect(self._refresh_context)
        editor.current_stim_cleared.connect(self._reset_start_time)
        editor.destroyed.connect(self.discard_dialog)

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
        if not available:
            return
        context = self._current_pattern_context()
        self._pattern_dialog.set_context(context)
        if context.stim_name != self._dialog_stim_name:
            self._switch_state(context.stim_name)

    def _switch_state(self, stim_name: str) -> None:
        """Keep the shown array's values and show ``stim_name``'s."""
        assert self._pattern_dialog is not None
        if self._dialog_stim_name is not None:
            self._states[self._dialog_stim_name] = self._pattern_dialog.state()
        # Forget deleted (or renamed) arrays, so a new array that reuses a
        # name starts from the defaults too.
        project = self.editor.project
        if project is not None:
            self._states = {
                name: state
                for name, state in self._states.items()
                if name in project.stim_arrays
            }
        self._pattern_dialog.restore_state(self._states.get(stim_name))
        self._dialog_stim_name = stim_name

    def _reset_start_time(self) -> None:
        if self._pattern_dialog is not None:
            self._pattern_dialog.reset_start_time()

    def open_pattern_designer(self) -> None:
        if self.editor.current_stim_modifier is None or self.editor.project is None:
            return
        if self._pattern_dialog is None:
            self._pattern_dialog = StimPatternDialog(self.editor)
        self._refresh_context(self.editor.current_stim_modifier.stim_array)
        self._pattern_dialog.show()
        self._pattern_dialog.raise_()
        self._pattern_dialog.activateWindow()

    def discard_dialog(self) -> None:
        dialog, self._pattern_dialog = self._pattern_dialog, None
        if dialog is not None:
            try:
                dialog.close()
                dialog.deleteLater()
            except RuntimeError:
                # Also runs from the editor's ``destroyed`` signal: at app
                # shutdown Qt may already have deleted the parentless dialog,
                # leaving only its Python wrapper. Nothing is left to close.
                pass
        self._states = {}
        self._dialog_stim_name = None
