from typing import Dict
from typing import Optional

from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model


class NeuronGroupDialog(QtWidgets.QDialog):
    def accept(self) -> None:
        print("Accepted!")
        self.done(1)

    def reject(self) -> None:
        print("Rejected!")
        self.done(0)

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Define a Neurongroup")
        qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


class SynapseDialog(QtWidgets.QDialog):
    def accept(self) -> None:
        print("Accepted!")
        self.done(1)

    def reject(self) -> None:
        print("Rejected!")
        self.done(0)

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Define a synapse")
        qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


class DictionaryWidget(QtWidgets.QWidget):
    def __init__(
        self, dictionary: Dict, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.layout = QtWidgets.QGridLayout()
        self.dictionary = dictionary

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])

        for row, (key, value) in enumerate(dictionary.items()):
            key_item = QtWidgets.QTableWidgetItem(str(key))
            value_item = QtWidgets.QTableWidgetItem(str(value))
            self.table.insertRow(row)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, value_item)

        self.layout.addWidget(self.table)
        self.setLayout(self.layout)


class B2ModelWidget(QtWidgets.QWidget):
    def accept(self) -> None:
        d = QtWidgets.QDialog(self)
        layout = QtWidgets.QGridLayout(d)
        text = QtWidgets.QLabel("Do you want to save the model?")
        save_button = QtWidgets.QPushButton("Yes", d)
        no_button = QtWidgets.QPushButton("No", d)
        layout.addWidget(text, 1, 1, 1, 2)
        layout.addWidget(save_button, 2, 1, 1, 1)
        layout.addWidget(no_button, 2, 2, 1, 1)
        d.setLayout(layout)

        def save_and_close() -> None:
            self.save()
            d.close()

        no_button.clicked.connect(d.close)
        save_button.clicked.connect(save_and_close)

        if d.exec():
            print("Success!")
        else:
            print("Cancel!")

    def reject(self) -> None:
        print("Rejected!")

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.layout = QtWidgets.QGridLayout()

        model_dialog = QtWidgets.QDialog()
        model_dialog.setWindowTitle("Model")

        layout = QtWidgets.QGridLayout(model_dialog)

        timestep_title = QtWidgets.QLabel("Timestep", model_dialog)
        timestep = QtWidgets.QLineEdit()
        timestep_button = QtWidgets.QPushButton("Set", model_dialog)

        solver = QtWidgets.QLineEdit(model_dialog)
        solver_button = QtWidgets.QPushButton("Set", model_dialog)
        solver_title = QtWidgets.QLabel("Solver")

        seed = QtWidgets.QLineEdit(model_dialog)
        seed_button = QtWidgets.QPushButton("Set", model_dialog)
        seet_title = QtWidgets.QLabel("Random Seed")

        parameter_title = QtWidgets.QLabel("Parameter", model_dialog)
        parameter = QtWidgets.QLineEdit(model_dialog)
        parameter_key = QtWidgets.QLineEdit(model_dialog)
        parameter_button = QtWidgets.QPushButton("Set", model_dialog)

        parameter_erase_title = QtWidgets.QLabel("Erase parameter", model_dialog)
        parameter_erase_key = QtWidgets.QLineEdit(model_dialog)
        parameter_erase_button = QtWidgets.QPushButton("Erase", model_dialog)

        parameter_overview_button = QtWidgets.QPushButton(
            "Show parameters", model_dialog
        )
        add_neurongroup_button = QtWidgets.QPushButton("Add Neurongroup", model_dialog)
        add_synapse_button = QtWidgets.QPushButton("Add Synapse", model_dialog)

        layout.addWidget(timestep_title, 1, 1)
        layout.addWidget(timestep, 1, 2)
        layout.addWidget(timestep_button, 1, 4)

        layout.addWidget(solver_title, 2, 1)
        layout.addWidget(solver, 2, 2)
        layout.addWidget(solver_button, 2, 4)

        layout.addWidget(seet_title, 3, 1)
        layout.addWidget(seed, 3, 2)
        layout.addWidget(seed_button, 3, 4)

        layout.addWidget(parameter_title, 4, 1)
        layout.addWidget(parameter, 4, 2)
        layout.addWidget(parameter_key, 4, 3)
        layout.addWidget(parameter_button, 4, 4)

        layout.addWidget(parameter_erase_title, 5, 1)
        layout.addWidget(parameter_erase_key, 5, 2)
        layout.addWidget(parameter_erase_button, 5, 4)

        layout.addWidget(parameter_overview_button, 6, 1)
        layout.addWidget(add_neurongroup_button, 6, 2)
        layout.addWidget(add_synapse_button, 6, 3)

        qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn, model_dialog)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox, 7, 1, 1, 3, QtCore.Qt.AlignCenter)

        model_dialog.setLayout(layout)

        # connect signals to slots
        add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        parameter_overview_button.clicked.connect(self.display_parameters)

        self.layout.addWidget(model_dialog, 0, 0)

        self.setLayout(self.layout)

    def open_add_neurongroup_dialog(self) -> None:
        dlg = NeuronGroupDialog(self)

        if dlg.exec():
            print("Success!")
            dlg.close()
        else:
            print("Cancel!")
            dlg.close()

    def open_add_synapse_dialog(self) -> None:
        dlg = SynapseDialog(self)

        if dlg.exec():
            print("Success!")
            dlg.close()
        else:
            print("Cancel!")
            dlg.close()

    def display_parameters(self) -> None:
        d = self.model.parameters
        if d is None:
            d = {}
        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        dlg = DictionaryWidget(d, self)
        layout.addWidget(dlg)
        dlg.setLayout(layout)
        window.exec()

    def save(self) -> None:
        pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    model = B2Model()
    widget = B2ModelWidget(model)
    widget.show()
    sys.exit(app.exec_())
