from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.project import Project
from neuroseq.stim import StimType


class Igor2PxpStimExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export stim arrays to Igor2 PXP")
        self._project = project
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Select stim arrays to export.\n"
                "Inhibitory arrays will be negated (values x -1)."
            )
        )

        self._list = QtWidgets.QListWidget()
        for name, stim_array in self._project.stim_arrays.items():
            negate_note = (
                " [negated]" if stim_array.stim_type == StimType.Inhibition else ""
            )
            item = QtWidgets.QListWidgetItem(f"{name}{negate_note}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_array_names(self) -> list[str]:
        result: list[str] = []
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is None:
                continue
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                result.append(str(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        return result
