"""Shared prompts for starting dynamic-input routines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtWidgets

if TYPE_CHECKING:
    from neuroseq.stim_editor import StimEditor


def confirm_start_routine(dialog: QtWidgets.QWidget, editor: StimEditor) -> str:
    """Ask how to start a dynamic-input routine, shared by all three routine
    dialogs' Start handlers.

    A stim array with no stimuli at all needs no prompt - just start. When
    it already holds stimuli (including from a previous run of this or
    another routine), ask whether to clear it and start from the
    beginning, or continue appending after the last existing stimulus -
    which keeps a previous run's progress. "Continue" only trims any
    unsimulated tail first (a manually drawn pattern, or a routine stopped
    mid-run), same as reset_stim_array() - it never touches simulated
    content. Returns "clear", "continue", or "cancel".
    """
    if editor.current_stim_modifier is None:
        return "continue"
    if not editor.current_stim_array_has_any_data():
        return "continue"
    name = editor.current_stim_modifier.stim_array.name
    # Parented to whichever top-level window is actually active, not to
    # `dialog` - see clear_stim_array()'s docstring for why: the routine
    # dialogs are created without a Qt parent (_show_routine_dialog), so a
    # box that is not explicitly raised/activated can end up shown behind
    # them, and this box is now shown far more often than the old
    # unsimulated-tail-only check it replaced (every routine start on an
    # array that already has any content, not just a rare edge case) - a
    # modal QMessageBox stuck behind another window blocks the whole
    # application with no visible way to dismiss it.
    box = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow() or dialog)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle("Existing input")
    box.setText(f"'{name}' already contains stimuli.")
    box.setInformativeText(
        "Clear the array and start from the beginning, continue after the "
        "last existing stimulus (keeping progress from a previous run), or "
        "cancel."
    )
    clear_button = box.addButton(
        "Clear and start over", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
    )
    continue_button = box.addButton(
        "Continue after last stimulus", QtWidgets.QMessageBox.ButtonRole.AcceptRole
    )
    cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel_button)
    box.show()
    box.raise_()
    box.activateWindow()
    box.exec()
    clicked = box.clickedButton()
    if clicked is clear_button:
        return "clear"
    if clicked is continue_button:
        return "continue"
    return "cancel"
