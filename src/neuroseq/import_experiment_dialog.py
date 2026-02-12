from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.project import Project


def describe_data(data: Data) -> str:
    if isinstance(data, TraceData):
        return f"Trace data '{data.name}': {data.traces.shape[0]} neurons, {data.traces.shape[1]} timestamps"
    elif isinstance(data, SpikeData):
        return f"Spike data '{data.name}': {len(data.spike_indices)} spikes"
    elif isinstance(data, SynapseData):
        return f"Synapse data '{data.name}': {data.values.shape[0]}x{data.values.shape[1]} synapses, {data.values.shape[2]} timestamps"
    return ""


def describe_experiment(experiment: Experiment) -> str:
    description = f"Importing {experiment.duration}s of experimental data:\n\n"
    for datum in experiment.data:
        description += f"  - {describe_data(datum)}\n"
    return description


class ImportExperimentDialog(QtWidgets.QDialog):
    """A dialog that allows the user to import data, either as a new Experiment or appended to an existing one."""

    def __init__(
        self,
        experiment: Experiment,
        project: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.existing_experiments = list(project.experiments.values())

        self.experiment = experiment
        self.append_to_experiment = False
        self.append_experiment_name: str = ""
        self.offset: float = 0.0

        self.setWindowTitle("Import data")
        self.setModal(True)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.lbl_description = QtWidgets.QLabel(describe_experiment(self.experiment))
        self.layout.addWidget(self.lbl_description)

        self.rad_new = QtWidgets.QRadioButton("Create new experiment")
        self.layout.addWidget(self.rad_new)
        self.rad_append = QtWidgets.QRadioButton("Apply to existing experiment")
        self.layout.addWidget(self.rad_append)
        self.rad_new.setChecked(True)

        self.new_box = QtWidgets.QWidget()
        layout_new_box = QtWidgets.QVBoxLayout(self.new_box)
        layout_new_box.addWidget(QtWidgets.QLabel("Experiment name:"))
        self.txt_new_name = QtWidgets.QLineEdit()
        layout_new_box.addWidget(self.txt_new_name)
        self.layout.addWidget(self.new_box)
        self.txt_new_name.setText(self.experiment.name)

        self.append_box = QtWidgets.QWidget()
        layout_append_box = QtWidgets.QVBoxLayout(self.append_box)
        layout_append_box.addWidget(QtWidgets.QLabel("Existing experiment:"))
        self.cmb_experiments = QtWidgets.QComboBox()
        self.cmb_experiments.addItems(
            [experiment.name for experiment in self.existing_experiments]
        )
        layout_append_box.addWidget(self.cmb_experiments)
        layout_append_box.addWidget(QtWidgets.QLabel("Timepoint to insert data:"))
        self.cmb_offset = QtWidgets.QComboBox()
        layout_append_box.addWidget(self.cmb_offset)
        self.layout.addWidget(self.append_box)
        self.append_box.setVisible(False)

        self.lbl_error = QtWidgets.QLabel("")
        self.lbl_error.setStyleSheet("color: red")
        self.layout.addWidget(self.lbl_error)

        self.layout.addStretch()

        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Import")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(button_layout)

        if not self.existing_experiments:
            self.rad_append.setEnabled(False)

        self.rad_new.toggled.connect(self.rad_new_append_toggled)
        self.txt_new_name.textEdited.connect(self.new_name_text_edited)
        self.cmb_experiments.currentIndexChanged.connect(
            self.cmb_experiments_index_changed
        )
        self.cmb_offset.currentIndexChanged.connect(self.cmb_offset_index_changed)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self._apply_default_selection()
        self.resize(800, 600)

    def _apply_default_selection(self) -> None:
        if not self.existing_experiments:
            self.rad_new.setChecked(True)
            self.cmb_experiments_index_changed(0)
            self.new_name_text_edited(self.txt_new_name.text())
            return

        matching_index = next(
            (
                i
                for i, experiment in enumerate(self.existing_experiments)
                if experiment.name == self.experiment.name
            ),
            -1,
        )
        if matching_index >= 0:
            self.rad_append.setChecked(True)
            self.cmb_experiments.setCurrentIndex(matching_index)
            self.cmb_experiments_index_changed(matching_index)
        else:
            self.rad_new.setChecked(True)
            self.cmb_experiments_index_changed(0)
        self.new_name_text_edited(self.txt_new_name.text())

    def rad_new_append_toggled(self):
        self.append_to_experiment = self.rad_append.isChecked()
        self.new_box.setVisible(not self.append_to_experiment)
        self.txt_new_name.setEnabled(not self.append_to_experiment)
        self.append_box.setVisible(self.append_to_experiment)
        self.cmb_experiments.setEnabled(self.append_to_experiment)
        self.cmb_offset.setEnabled(self.append_to_experiment)
        if self.append_to_experiment:
            self.cmb_experiments_index_changed(self.cmb_experiments.currentIndex())
            self._refresh_append_error()
        else:
            self.new_name_text_edited(self.txt_new_name.text())

    def cmb_offset_index_changed(self, index: int) -> None:
        if index < 0:
            self.offset = 0.0
            return
        experiment_index = self.cmb_experiments.currentIndex()
        if experiment_index < 0 or experiment_index >= len(self.existing_experiments):
            self.offset = 0.0
            return
        self.offset = self.existing_experiments[experiment_index].timepoints[index]
        self._refresh_append_error()

    def cmb_experiments_index_changed(self, index: int) -> None:
        self.cmb_offset.clear()
        if index < 0 or index >= len(self.existing_experiments):
            return
        self.append_experiment_name = self.existing_experiments[index].name
        timepoints = list(self.existing_experiments[index].timepoints)
        can_replace_at_zero = self._can_replace_at_zero(index)
        option_labels = []
        for i, timepoint in enumerate(timepoints):
            label = f"{timepoint * 1000:.5g} ms"
            if i == len(timepoints) - 1:
                label = f"{label} (Append)"
            if len(timepoints) == 2 and i == 0 and can_replace_at_zero:
                label = f"{label} (replace data)"
            option_labels.append(label)
        self.cmb_offset.addItems(option_labels)
        # select the last timepoint by default
        if self.cmb_offset.count() > 0:
            self.cmb_offset.setCurrentIndex(0)
        # check if appending this data to the selected experiment is valid
        self._refresh_append_error()

    def new_name_text_edited(self, text: str) -> None:
        self.experiment.name = text
        self.lbl_error.clear()
        self.save_button.setEnabled(True)

    def _can_replace_at_zero(self, experiment_index: int) -> bool:
        if experiment_index < 0 or experiment_index >= len(self.existing_experiments):
            return False
        existing_experiment = self.existing_experiments[experiment_index]
        existing_trace_count = sum(
            datum.traces.shape[0]
            for datum in existing_experiment.data
            if isinstance(datum, TraceData)
        )
        incoming_trace_count = sum(
            datum.traces.shape[0]
            for datum in self.experiment.data
            if isinstance(datum, TraceData)
        )
        if existing_trace_count != incoming_trace_count:
            return False

        existing_spike_count = sum(
            len(datum.spike_indices)
            for datum in existing_experiment.data
            if isinstance(datum, SpikeData)
        )
        incoming_spike_count = sum(
            len(datum.spike_indices)
            for datum in self.experiment.data
            if isinstance(datum, SpikeData)
        )
        return existing_spike_count == incoming_spike_count

    def _refresh_append_error(self) -> None:
        error = self.project.issues_with_appending_experimental_data(
            self.append_experiment_name,
            self.experiment.data,
        )
        self.lbl_error.setText(error)
        self.save_button.setEnabled(error == "")
