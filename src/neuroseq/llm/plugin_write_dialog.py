"""Confirmation dialog shown before saving an LLM-authored analysis plugin."""

from __future__ import annotations

import difflib
import pathlib

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


def plugin_write_preview(
    target: pathlib.Path, content: str, previous: str | None
) -> str:
    """Full code for a new plugin, or a unified diff for an overwrite."""
    if previous is None:
        return content
    diff = difflib.unified_diff(
        previous.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=f"{target} (current)",
        tofile=f"{target} (proposed)",
    )
    return "".join(diff) or "(no changes)"


class PluginWriteConfirmDialog(QtWidgets.QDialog):
    """Read-only preview of the code the assistant wants to save."""

    def __init__(
        self,
        target: pathlib.Path,
        content: str,
        previous: str | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        overwrite = previous is not None
        self.setWindowTitle(
            "Update analysis plugin?" if overwrite else "Save analysis plugin?"
        )
        self.setModal(True)
        self.resize(820, 620)
        layout = QtWidgets.QVBoxLayout(self)

        action = "replace" if overwrite else "create"
        self.message_label = QtWidgets.QLabel(
            f"The chat assistant wants to {action} this analysis plugin:\n"
            f"{target}\n\n"
            "NeuroSeq runs this code when you next open the analysis menu. "
            "Only save code you have reviewed and trust.",
            self,
        )
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(
            self.message_label.textInteractionFlags()
            | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.message_label)

        self.code_view = QtWidgets.QPlainTextEdit(self)
        self.code_view.setReadOnly(True)
        self.code_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.code_view.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.code_view.setPlainText(plugin_write_preview(target, content, previous))
        layout.addWidget(self.code_view, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(self)
        self.save_button = buttons.addButton(
            "Update plugin" if overwrite else "Save plugin",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.cancel_button = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        # Cancel is the default so a stray Enter never saves code.
        assert self.cancel_button is not None
        self.cancel_button.setDefault(True)
        if self.save_button is not None:
            self.save_button.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def confirm_plugin_write(
    target: pathlib.Path,
    content: str,
    previous: str | None,
    parent: QtWidgets.QWidget | None = None,
) -> bool:
    """Ask the user to approve a plugin write; True only on explicit accept."""
    dialog = PluginWriteConfirmDialog(target, content, previous, parent)
    try:
        return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    finally:
        dialog.deleteLater()
