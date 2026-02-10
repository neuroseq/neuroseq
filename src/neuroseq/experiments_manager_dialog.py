from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.project import Project


class ExperimentsManagerDialog(QtWidgets.QDialog):
    """A dialog that allows the user to view and manage all experiments in the project."""

    experiments_changed = QtCore.Signal()

    def __init__(
        self,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Manage Experiments")
        self.setModal(True)
        self.resize(800, 600)

        # Main layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QtWidgets.QLabel("Experiments in Project")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Table widget to display experiments
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Duration (s)", "Data Sets", "Details"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        layout.addWidget(self.table)

        # Populate the table
        self._populate_table()

        # Information label
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        self.info_label.setText(f"Total: {len(self.project.experiments)} experiment(s)")
        layout.addWidget(self.info_label)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()

        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        button_layout.addWidget(self.delete_button)

        button_layout.addStretch()

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        # Connect signals
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self._on_selection_changed()

    def _populate_table(self) -> None:
        """Populate the table with experiment data."""
        self.table.setRowCount(len(self.project.experiments))

        for row, (exp_name, experiment) in enumerate(self.project.experiments.items()):
            # Name
            name_item = QtWidgets.QTableWidgetItem(experiment.name)
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, exp_name)
            self.table.setItem(row, 0, name_item)

            # Duration
            duration_item = QtWidgets.QTableWidgetItem(f"{experiment.duration:.3f}")
            duration_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, duration_item)

            # Data sets count
            data_count = len(experiment.data)
            data_item = QtWidgets.QTableWidgetItem(str(data_count))
            data_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, data_item)

            # Details
            details = self._get_experiment_details(experiment)
            details_item = QtWidgets.QTableWidgetItem(details)
            self.table.setItem(row, 3, details_item)

        # Resize columns to content
        self.table.resizeColumnsToContents()

    def _get_experiment_details(self, experiment) -> str:
        """Generate a detailed description of the experiment's data."""
        details_parts = []
        for datum in experiment.data:
            if isinstance(datum, TraceData):
                details_parts.append(
                    f"Trace '{datum.name}' ({datum.traces.shape[0]} neurons)"
                )
            elif isinstance(datum, SpikeData):
                details_parts.append(
                    f"Spike '{datum.name}' ({len(datum.spike_indices)} spikes)"
                )
            elif isinstance(datum, SynapseData):
                details_parts.append(
                    f"Synapse '{datum.name}' ({datum.values.shape[0]}x{datum.values.shape[1]})"
                )
        return "; ".join(details_parts)

    def _get_selected_row_indices(self) -> list[int]:
        """Get the indices of selected rows."""
        selected_indexes = self.table.selectionModel().selectedRows()
        return sorted([index.row() for index in selected_indexes])

    def _on_selection_changed(self) -> None:
        """Update button state based on selection."""
        has_selection = len(self._get_selected_row_indices()) > 0
        self.delete_button.setEnabled(has_selection)

    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        selected_rows = self._get_selected_row_indices()
        if not selected_rows:
            return

        # Get the experiment names to delete
        experiments_to_delete = []
        for row in selected_rows:
            name_item = self.table.item(row, 0)
            exp_key = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            experiments_to_delete.append((row, exp_key, name_item.text()))

        # Show confirmation dialog
        exp_names = ", ".join([name for _, _, name in experiments_to_delete])
        if len(experiments_to_delete) == 1:
            message = f"Are you sure you want to delete the experiment '{exp_names}'?"
        else:
            message = (
                f"Are you sure you want to delete {len(experiments_to_delete)} "
                f"experiments?\n\n{exp_names}"
            )

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Delete experiments (in reverse row order to maintain indices)
            for _, exp_key, _ in reversed(experiments_to_delete):
                del self.project.experiments[exp_key]

            # Refresh the table
            self._populate_table()
            self.info_label.setText(
                f"Total: {len(self.project.experiments)} experiment(s)"
            )
            self.experiments_changed.emit()

            # Show success message
            if len(experiments_to_delete) == 1:
                QtWidgets.QMessageBox.information(
                    self,
                    "Experiment Deleted",
                    f"Experiment '{exp_names}' has been deleted successfully.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Experiments Deleted",
                    f"{len(experiments_to_delete)} experiments have been deleted successfully.",
                )
