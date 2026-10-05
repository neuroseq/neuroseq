from __future__ import annotations

from collections.abc import Collection
from collections.abc import Sequence

from qtpy import QtCore
from qtpy import QtWidgets


class NeuronSubsetDialog(QtWidgets.QDialog):
    """A dialog for picking an arbitrary (not necessarily contiguous) subset
    of neuron ids out of a list of candidates, via a checkable list."""

    def __init__(
        self,
        neuron_ids: Sequence[int],
        checked_ids: Collection[int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select neurons")
        layout = QtWidgets.QVBoxLayout(self)

        button_row = QtWidgets.QHBoxLayout()
        select_all_button = QtWidgets.QPushButton("Select all", self)
        select_none_button = QtWidgets.QPushButton("Select none", self)
        select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        button_row.addWidget(select_all_button)
        button_row.addWidget(select_none_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        checked_ids = set(neuron_ids) if checked_ids is None else set(checked_ids)
        self.list_widget = QtWidgets.QListWidget(self)
        for neuron_id in neuron_ids:
            item = QtWidgets.QListWidgetItem(f"Neuron {neuron_id}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, neuron_id)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if neuron_id in checked_ids
                else QtCore.Qt.CheckState.Unchecked
            )
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _set_all_checked(self, checked: bool) -> None:
        state = (
            QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        )
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None:
                item.setCheckState(state)

    @property
    def selected_neuron_ids(self) -> list[int]:
        ids = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None:
                continue
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                ids.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return sorted(ids)
