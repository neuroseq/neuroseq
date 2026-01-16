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
        self.rad_append = QtWidgets.QRadioButton("Append to existing experiment")
        self.layout.addWidget(self.rad_append)
        self.rad_new.setChecked(True)
        self.rad_new.setChecked(False)

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
        self.cmb_experiments_index_changed(0)
        self.new_name_text_edited(self.txt_new_name.text())
        self.resize(800, 600)

    def rad_new_append_toggled(self):
        self.append_to_experiment = self.rad_append.isChecked()
        self.new_box.setVisible(not self.append_to_experiment)
        self.txt_new_name.setEnabled(not self.append_to_experiment)
        self.append_box.setVisible(self.append_to_experiment)
        self.cmb_experiments.setEnabled(self.append_to_experiment)
        self.cmb_offset.setEnabled(self.append_to_experiment)
        if self.append_to_experiment:
            self.cmb_experiments_index_changed(self.cmb_experiments.currentIndex())
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

    def cmb_experiments_index_changed(self, index: int) -> None:
        self.cmb_offset.clear()
        if index < 0 or index >= len(self.existing_experiments):
            return
        self.append_experiment_name = self.existing_experiments[index].name
        self.cmb_offset.addItems(
            [
                f"{timepoint * 1000:.5g} ms"
                for timepoint in self.existing_experiments[index].timepoints
            ]
        )
        # select the last timepoint by default
        if self.cmb_offset.count() > 0:
            self.cmb_offset.setCurrentIndex(self.cmb_offset.count() - 1)
        # check if appending this data to the selected experiment is valid
        self.lbl_error.setText(
            self.project.issues_with_appending_experimental_data(
                self.append_experiment_name, self.experiment.data
            )
        )
        if self.lbl_error.text() != "":
            self.save_button.setEnabled(False)
        else:
            self.lbl_error.clear()
            self.save_button.setEnabled(True)

    def new_name_text_edited(self, text: str) -> None:
        self.experiment.name = text
        if self.experiment.name in [
            experiment.name for experiment in self.existing_experiments
        ]:
            self.lbl_error.setText(
                f"An experiment with name '{self.experiment.name}' already exists."
            )
            self.save_button.setEnabled(False)
        else:
            self.lbl_error.clear()
            self.save_button.setEnabled(True)
