from __future__ import annotations

from typing import Dict
from typing import Optional

from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model


class NeuronGroupDialog(QtWidgets.QDialog):


    def display_error(self, message: str):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()

    def accept(self) -> None:
        print("Neurongroup accepted!")
        self.done(1)

    def reject(self) -> None:
        print("Neurongroup rejected!")
        self.done(0)

    def accept_equation(self) -> None:
        if self.name is None:
            self.display_error("Please define a name first!")
            return

        line = self.equation.getText()
        self.model.define_equation(line)

    def accept_name(self) -> None:
        line = self.group_name.getText()
        if line in self.model.neurongroups:
            self.display_error("Name already exists!")
        else: 
            self.model.neurongroups[name] = {}
            self.name = line

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Define a Neurongroup")

        layout = QtWidgets.QGridLayout(self)

        name_title = QtWidgets.QLabel("Name", self)
        self.group_name = QtWidgets.QLineEdit(self)
        name_button = QtWidgets.QPushButton("save", self)
        name_button_show = QtWidgets.QPushButton("show", self)

        equation_title = QtWidgets.QLabel("Equation", self)
        self.equation = QtWidgets.QLineEdit(self)
        equation_button = QtWidgets.QPushButton("save", self)
        equation_button_show = QtWidgets.QPushButton("show", self)
        equation_button.clicked.connect(self.accept_equation)

        threshold_title = QtWidgets.QLabel("Threshold", self)
        self.threshold = QtWidgets.QLineEdit(self)
        threshold_button = QtWidgets.QPushButton("save", self)
        threshold_button_show = QtWidgets.QPushButton("show", self)

        reset_title = QtWidgets.QLabel("Reset", self)
        self.reset = QtWidgets.QLineEdit(self)
        reset_button = QtWidgets.QPushButton("save", self)
        reset_button_show = QtWidgets.QPushButton("show", self)

        refractory_title = QtWidgets.QLabel("Refractory", self)
        self.refractory = QtWidgets.QLineEdit(self)
        refractory_button = QtWidgets.QPushButton("save", self)
        refractory_button_show = QtWidgets.QPushButton("show", self)

        self.solver = QtWidgets.QLineEdit(self)
        solver_button = QtWidgets.QPushButton("save", self)
        solver_title = QtWidgets.QLabel("Solver", self)
        solver_button_show = QtWidgets.QPushButton("show", self)

        self.neuronnumber = QtWidgets.QLineEdit(self)
        neuronnumber_button = QtWidgets.QPushButton("save", self)
        neuronnumber_title = QtWidgets.QLabel("Neuronnumber", self)
        neuronnumber_button_show = QtWidgets.QPushButton("show", self)

        layout.addWidget(name_title, 1, 1)
        layout.addWidget(self.group_name, 1, 2)
        layout.addWidget(name_button, 1, 3)
        layout.addWidget(name_button_show, 1, 4)

        layout.addWidget(equation_title, 2, 1)
        layout.addWidget(self.equation, 2, 2)
        layout.addWidget(equation_button, 2, 3)
        layout.addWidget(equation_button_show, 2, 4)

        layout.addWidget(threshold_title, 3, 1)
        layout.addWidget(self.threshold, 3, 2)
        layout.addWidget(threshold_button, 3, 3)
        layout.addWidget(threshold_button_show, 3, 4)

        layout.addWidget(reset_title, 4, 1)
        layout.addWidget(self.reset, 4, 2)
        layout.addWidget(reset_button, 4, 3)
        layout.addWidget(reset_button_show, 4, 4)

        layout.addWidget(refractory_title, 5, 1)
        layout.addWidget(self.refractory, 5, 2)
        layout.addWidget(refractory_button, 5, 3)
        layout.addWidget(refractory_button_show, 5, 4)

        layout.addWidget(solver_title, 6, 1)
        layout.addWidget(self.solver, 6, 2)
        layout.addWidget(solver_button, 6, 3)
        layout.addWidget(solver_button_show, 6, 4)

        layout.addWidget(neuronnumber_title, 7, 1)
        layout.addWidget(self.neuronnumber, 7, 2)
        layout.addWidget(neuronnumber_button, 7, 3)
        layout.addWidget(neuronnumber_button_show, 7, 4)

        qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox, 7, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(layout)


class SynapseDialog(QtWidgets.QDialog):
    def accept(self) -> None:
        # build up the synapse and set everything to the model -> assemble_synapse
        self.done(1)

    def reject(self) -> None:
        # do nothing.
        self.done(0)

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        layout = QtWidgets.QGridLayout()

        self.setWindowTitle("Define a synapse")

        origin_title = QtWidgets.QLabel("Origin", self)
        origin = QtWidgets.QLineEdit(self)
        origin_button = QtWidgets.QPushButton("save", self)
        origin_button_show = QtWidgets.QPushButton("show", self)

        target_title = QtWidgets.QLabel("Target", self)
        target = QtWidgets.QLineEdit(self)
        target_button = QtWidgets.QPushButton("save", self)
        target_button_show = QtWidgets.QPushButton("show", self)

        equation_title = QtWidgets.QLabel("Equation", self)
        equation = QtWidgets.QLineEdit(self)
        equation_button = QtWidgets.QPushButton("save", self)
        equation_button_show = QtWidgets.QPushButton("show", self)

        pre_title = QtWidgets.QLabel("Pre", self)
        pre = QtWidgets.QLineEdit(self)
        pre_button = QtWidgets.QPushButton("save", self)
        pre_button_show = QtWidgets.QPushButton("show", self)

        post_title = QtWidgets.QLabel("Post", self)
        post = QtWidgets.QLineEdit(self)
        post_button = QtWidgets.QPushButton("save", self)
        post_button_show = QtWidgets.QPushButton("show", self)

        connect_title = QtWidgets.QLabel("Connection def.", self)
        connect = QtWidgets.QLineEdit(self)
        connect_button = QtWidgets.QPushButton("save", self)
        connect_button_show = QtWidgets.QPushButton("show", self)

        connect_prob_title = QtWidgets.QLabel("Connection prob.", self)
        connect_prob = QtWidgets.QLineEdit(self)
        connect_prob_button = QtWidgets.QPushButton("save", self)
        connect_prob_button_show = QtWidgets.QPushButton("show", self)

        layout.addWidget(origin_title, 1, 1)
        layout.addWidget(origin, 1, 2)
        layout.addWidget(origin_button, 1, 3)
        layout.addWidget(origin_button_show, 1, 4)

        layout.addWidget(target_title, 2, 1)
        layout.addWidget(target, 2, 2)
        layout.addWidget(target_button, 2, 3)
        layout.addWidget(target_button_show, 2, 4)

        layout.addWidget(equation_title, 3, 1)
        layout.addWidget(equation, 3, 2)
        layout.addWidget(equation_button, 3, 3)
        layout.addWidget(equation_button_show, 3, 4)

        layout.addWidget(pre_title, 4, 1)
        layout.addWidget(pre, 4, 2)
        layout.addWidget(pre_button, 4, 3)
        layout.addWidget(pre_button_show, 4, 4)

        layout.addWidget(post_title, 5, 1)
        layout.addWidget(post, 5, 2)
        layout.addWidget(post_button, 5, 3)
        layout.addWidget(post_button_show, 5, 4)

        layout.addWidget(connect_title, 6, 1)
        layout.addWidget(connect, 6, 2)
        layout.addWidget(connect_button, 6, 3)
        layout.addWidget(connect_button_show, 6, 4)

        layout.addWidget(connect_prob_title, 7, 1)
        layout.addWidget(connect_prob, 7, 2)
        layout.addWidget(connect_prob_button, 7, 3)
        layout.addWidget(connect_prob_button_show, 7, 4)

        qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox, 8, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(layout)


class DictionaryWidget(QtWidgets.QWidget):
    def __init__(
        self, dictionary: Dict, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        self.dictionary = dictionary

        if dictionary is None or len(dictionary) == 0:
            self.table = QtWidgets.QLabel("No parameters defined")
        else:
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Key", "Value"])

            for row, (key, value) in enumerate(dictionary.items()):
                key_item = QtWidgets.QTableWidgetItem(str(key))
                value_item = QtWidgets.QTableWidgetItem(str(value))
                self.table.insertRow(row)
                self.table.setItem(row, 0, key_item)
                self.table.setItem(row, 1, value_item)

        layout.addWidget(self.table)
        self.setLayout(layout)


class B2ModelWidget(QtWidgets.QDialog):
    def accept(self) -> None:
        model_save_dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QGridLayout(model_save_dialog)
        text = QtWidgets.QLabel("Do you want to save the model?")
        save_button = QtWidgets.QPushButton("Yes", model_save_dialog)
        no_button = QtWidgets.QPushButton("No", model_save_dialog)
        layout.addWidget(text, 1, 1, 1, 2)
        layout.addWidget(save_button, 2, 1, 1, 1)
        layout.addWidget(no_button, 2, 2, 1, 1)
        model_save_dialog.setLayout(layout)

        def save_and_close() -> None:
            self.save()
            model_save_dialog.close()

        def discard_and_close() -> None:
            self.model = B2Model()
            model_save_dialog.close()

        no_button.clicked.connect(discard_and_close)
        save_button.clicked.connect(save_and_close)

        model_save_dialog.exec()

        super().accept()
        self.close()

    def reject(self) -> None:
        super().reject()
        print("Rejected model!")
        self.model = B2Model()
        self.close()

    def __init__(
        self, model: B2Model, parent: Optional[QtWidgets.QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.layout = QtWidgets.QGridLayout()

        self.setWindowTitle("Model")

        layout = QtWidgets.QGridLayout(self)

        timestep_title = QtWidgets.QLabel("Timestep", self)
        timestep = QtWidgets.QLineEdit()
        timestep_button = QtWidgets.QPushButton("save", self)

        seed = QtWidgets.QLineEdit(self)
        seed_button = QtWidgets.QPushButton("save", self)
        seed_title = QtWidgets.QLabel("Random Seed")

        parameter_title = QtWidgets.QLabel("Parameter", self)
        parameter = QtWidgets.QLineEdit(self)
        parameter_key = QtWidgets.QLineEdit(self)
        parameter_button = QtWidgets.QPushButton("save", self)

        parameter_erase_title = QtWidgets.QLabel("Erase parameter", self)
        parameter_erase_key = QtWidgets.QLineEdit(self)
        parameter_erase_button = QtWidgets.QPushButton("erase", self)

        parameter_overview_button = QtWidgets.QPushButton("Show parameters", self)
        add_neurongroup_button = QtWidgets.QPushButton("Add Neurongroup", self)
        add_synapse_button = QtWidgets.QPushButton("Add Synapse", self)

        layout.addWidget(timestep_title, 1, 1)
        layout.addWidget(timestep, 1, 2)
        layout.addWidget(timestep_button, 1, 4)

        layout.addWidget(seed_title, 3, 1)
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

        self.buttonBox = QtWidgets.QDialogButtonBox(qtbtn, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox, 7, 1, 1, 3, QtCore.Qt.AlignCenter)

        self.setLayout(layout)

        # connect signals to slots on the outermost
        add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        parameter_overview_button.clicked.connect(self.display_parameters)

        self.setLayout(self.layout)

    def open_add_neurongroup_dialog(self) -> None:
        dlg = NeuronGroupDialog(self)

        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")

        dlg.close()

    def open_add_synapse_dialog(self) -> None:
        dlg = SynapseDialog(self)

        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")

        dlg.close()

    def display_parameters(self) -> None:
        d = (
            self.model.parameters
            if self.model.parameters is not None
            else {}
            | {
                "dt": self.model.dt,
                "seed": self.model.seed,
            }
        )

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
