from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Data
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.project import Project

NAME_COLUMN = 0
KIND_COLUMN = 1
DURATION_COLUMN = 2
DETAILS_COLUMN = 3
DESCRIPTION_COLUMN = 4

_DATA_ROLE = QtCore.Qt.ItemDataRole.UserRole
_DATUM_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


def _data_kind(datum: Data) -> str:
    if isinstance(datum, TraceData):
        return "Traces"
    if isinstance(datum, SpikeData):
        return "Spikes"
    return "Synapses"


def _data_details(datum: Data) -> str:
    if isinstance(datum, TraceData):
        return f"{datum.traces.shape[0]} neurons, {datum.units}"
    if isinstance(datum, SpikeData):
        return f"{len(datum.spike_indices)} spikes"
    if isinstance(datum, SynapseData):
        return f"{datum.values.shape[0]}x{datum.values.shape[1]} matrix"
    return ""


class DataManagerDialog(QtWidgets.QDialog):
    """View, rename, describe and delete data groups and their data sets.

    Every change goes through a :class:`Project` method, so it is undoable and
    saved with the project.
    """

    data_changed = QtCore.Signal()

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Manage Data")
        self.setModal(True)
        self.resize(900, 600)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        title = QtWidgets.QLabel("Data in Project")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        hint = QtWidgets.QLabel(
            "Double-click a name or a group description to edit it. Expand a "
            "group to delete single data sets instead of the whole group."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(
            ["Name", "Kind", "Duration (s)", "Details", "Description"]
        )
        self.tree.header().setStretchLastSection(True)
        self.tree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # Editors are opened by _on_item_double_clicked, only on editable columns.
        self.tree.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self.tree)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.info_label)

        button_layout = QtWidgets.QHBoxLayout()
        self.rename_button = QtWidgets.QPushButton("Rename")
        self.rename_button.clicked.connect(self._on_rename_clicked)
        button_layout.addWidget(self.rename_button)
        self.description_button = QtWidgets.QPushButton("Edit Description")
        self.description_button.clicked.connect(self._on_description_clicked)
        button_layout.addWidget(self.description_button)
        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self._populate()
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
        self._on_selection_changed()

    # ------------------------------------------------------------------ view
    def _populate(self) -> None:
        """Rebuild the tree from the project, keeping expanded groups open."""
        expanded = {
            self.tree.topLevelItem(i).data(NAME_COLUMN, _DATA_ROLE)
            for i in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(i).isExpanded()
        }
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            for key, experiment in self.project.experiments.items():
                group = QtWidgets.QTreeWidgetItem(
                    [
                        key,
                        "Data group",
                        f"{experiment.duration:.3f}",
                        f"{len(experiment.data)} data set(s)",
                        experiment.description,
                    ]
                )
                group.setData(NAME_COLUMN, _DATA_ROLE, key)
                group.setFlags(group.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                group.setToolTip(DESCRIPTION_COLUMN, experiment.description)
                for datum in experiment.data:
                    child = QtWidgets.QTreeWidgetItem(
                        [datum.name, _data_kind(datum), "", _data_details(datum), ""]
                    )
                    child.setData(NAME_COLUMN, _DATA_ROLE, key)
                    child.setData(NAME_COLUMN, _DATUM_ROLE, datum)
                    child.setFlags(child.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    group.addChild(child)
                self.tree.addTopLevelItem(group)
                group.setExpanded(key in expanded)
            for column in range(DESCRIPTION_COLUMN):
                self.tree.resizeColumnToContents(column)
        finally:
            self.tree.blockSignals(False)
        self.info_label.setText(f"Total: {len(self.project.experiments)} data group(s)")
        self._on_selection_changed()

    def group_item(self, key: str) -> QtWidgets.QTreeWidgetItem | None:
        """The tree item of data group ``key`` (for tests and scripting)."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(NAME_COLUMN, _DATA_ROLE) == key:
                return item
        return None

    @staticmethod
    def _is_group(item: QtWidgets.QTreeWidgetItem) -> bool:
        return item.parent() is None

    def _selected_items(self) -> list[QtWidgets.QTreeWidgetItem]:
        return list(self.tree.selectedItems())

    def _on_selection_changed(self) -> None:
        selected = self._selected_items()
        self.delete_button.setEnabled(bool(selected))
        self.rename_button.setEnabled(len(selected) == 1)
        self.description_button.setEnabled(
            len(selected) == 1 and self._is_group(selected[0])
        )

    # --------------------------------------------------------------- editing
    def _editable_column(self, item: QtWidgets.QTreeWidgetItem, column: int) -> bool:
        return column == NAME_COLUMN or (
            column == DESCRIPTION_COLUMN and self._is_group(item)
        )

    def _on_item_double_clicked(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if self._editable_column(item, column):
            self.tree.editItem(item, column)

    def _on_rename_clicked(self) -> None:
        selected = self._selected_items()
        if len(selected) == 1:
            self.tree.editItem(selected[0], NAME_COLUMN)

    def _on_description_clicked(self) -> None:
        selected = self._selected_items()
        if len(selected) == 1 and self._is_group(selected[0]):
            self.tree.editItem(selected[0], DESCRIPTION_COLUMN)

    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if not self._editable_column(item, column):
            return
        key = item.data(NAME_COLUMN, _DATA_ROLE)
        text = item.text(column)
        try:
            if column == DESCRIPTION_COLUMN:
                self.project.set_experiment_description(key, text)
            elif self._is_group(item):
                self.project.rename_experiment(key, text)
            else:
                datum = item.data(NAME_COLUMN, _DATUM_ROLE)
                self.project.rename_experiment_data(key, datum, text)
        except (KeyError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Cannot rename", str(error))
        # Deferred: rebuilding the tree inside itemChanged would delete the
        # item whose editor is still closing.
        QtCore.QTimer.singleShot(0, self._refresh_after_edit)

    def _refresh_after_edit(self) -> None:
        self._populate()
        self.data_changed.emit()

    # -------------------------------------------------------------- deleting
    def _on_delete_clicked(self) -> None:
        selected = self._selected_items()
        if not selected:
            return
        groups = [
            item.data(NAME_COLUMN, _DATA_ROLE)
            for item in selected
            if self._is_group(item)
        ]
        datasets: dict[str, list[Data]] = {}
        labels = [f"data group '{key}'" for key in groups]
        for item in selected:
            key = item.data(NAME_COLUMN, _DATA_ROLE)
            if self._is_group(item) or key in groups:
                continue
            datasets.setdefault(key, []).append(item.data(NAME_COLUMN, _DATUM_ROLE))
            labels.append(f"{item.text(KIND_COLUMN).lower()} '{key} :: {item.text(0)}'")

        message = "Are you sure you want to delete:\n\n" + "\n".join(labels)
        if groups:
            message += "\n\nDeleting a data group deletes all of its data sets."
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.project.remove_data(groups, datasets)
        self._populate()
        self.data_changed.emit()
