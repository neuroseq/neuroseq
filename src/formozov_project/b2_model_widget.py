from __future__ import annotations

from functools import partial
from typing import Callable

from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model


class ElementDialog(QtWidgets.QDialog):
    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.name: str | None = None

        self.layout = QtWidgets.QGridLayout(self)

        self.name_title = QtWidgets.QLabel("Name", self)
        self.group_name = QtWidgets.QLineEdit(self)
        self.name_button = QtWidgets.QPushButton("save", self)
        self.show_parameters = QtWidgets.QPushButton("show parameters", self)

        self.qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(self.qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def display_error(self, message: str) -> None:
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()

    def accept(self) -> None:
        self.done(1)

    def reject(self) -> None:
        self.done(0)

    def accept_parameter(
        self,
        element: QtWidgets.QLineEdit,
        paramname: str,
        model_callback: Callable,
        convert: Callable = lambda x: x,
    ) -> None:
        if self.name is None:
            self.display_error("Please define a name first!")

        line = element.text()

        try:
            var = convert(line)
            model_callback(self.name, var)
        except Exception as e:
            self.display_error(f"Error when saving parameter {paramname}: {e}")

    def accept_name(self, target: str) -> None:
        name = self.group_name.text()

        if name == "":
            self.display_error("Please enter a name!")
        elif name in getattr(self.model, target):
            self.display_error("Name already exists!")
        else:
            self.name = name
            getattr(self.model, target)[name] = {}
            self.name_button.setText("saved")

    def display_parameters(self, target: str) -> None:
        d = getattr(self.model, target)[self.name]
        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        dlg = DictionaryWidget(d, self)
        layout.addWidget(dlg)
        dlg.setLayout(layout)
        window.exec()

    def not_connected(self) -> None:
        self.display_error("Not connected yet!")


class NeuronGroupDialog(ElementDialog):
    def accept(self) -> None:
        if self.name is None or self.name not in self.model.neurongroups:
            self.display_error("Please define a synapse first by giving it a name!")
        else:
            try:
                self.model.assemble_neurongroup(self.name)
                self.done(1)
            except Exception as e:
                self.display_error(
                    f"Error when assembling Neurongroup {self.name}: {e}"
                )

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        super().__init__(model, parent)
        self.setWindowTitle("Define a Neurongroup")

        equation_title = QtWidgets.QLabel("Equation", self)
        equation = QtWidgets.QTextEdit(self)
        equation_button = QtWidgets.QPushButton("save", self)

        threshold_title = QtWidgets.QLabel("Threshold", self)
        threshold = QtWidgets.QLineEdit(self)
        threshold_button = QtWidgets.QPushButton("save", self)

        reset_title = QtWidgets.QLabel("Reset", self)
        reset = QtWidgets.QLineEdit(self)
        reset_button = QtWidgets.QPushButton("save", self)

        refractory_title = QtWidgets.QLabel("Refractory", self)
        refractory = QtWidgets.QLineEdit(self)
        refractory_button = QtWidgets.QPushButton("save", self)

        solver = QtWidgets.QLineEdit(self)
        solver_button = QtWidgets.QPushButton("save", self)
        solver_title = QtWidgets.QLabel("Solver", self)

        neuronnumber = QtWidgets.QLineEdit(self)
        neuronnumber_button = QtWidgets.QPushButton("save", self)
        neuronnumber_title = QtWidgets.QLabel("Neuronnumber", self)

        for qle in [equation, threshold, reset, refractory, solver, neuronnumber]:
            qle.setMinimumWidth(
                qle.sizeHint().width() * 3
            )  # Set the width to twice the default width

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(
            self.group_name,
            1,
            2,
        )
        self.layout.addWidget(self.name_button, 1, 3)

        self.layout.addWidget(equation_title, 2, 1)
        self.layout.addWidget(equation, 2, 2)
        self.layout.addWidget(equation_button, 2, 3)

        self.layout.addWidget(threshold_title, 3, 1)
        self.layout.addWidget(threshold, 3, 2)
        self.layout.addWidget(threshold_button, 3, 3)

        self.layout.addWidget(reset_title, 4, 1)
        self.layout.addWidget(reset, 4, 2)
        self.layout.addWidget(reset_button, 4, 3)

        self.layout.addWidget(refractory_title, 5, 1)
        self.layout.addWidget(refractory, 5, 2)
        self.layout.addWidget(refractory_button, 5, 3)

        self.layout.addWidget(solver_title, 6, 1)
        self.layout.addWidget(solver, 6, 2)
        self.layout.addWidget(solver_button, 6, 3)

        self.layout.addWidget(neuronnumber_title, 7, 1)
        self.layout.addWidget(neuronnumber, 7, 2)
        self.layout.addWidget(neuronnumber_button, 7, 3)

        self.layout.addWidget(self.show_parameters, 8, 1, 1, 3, QtCore.Qt.AlignCenter)

        self.layout.addWidget(self.buttonBox, 9, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        # connect signals and sockets: TODO

        self.show_parameters.clicked.connect(
            partial(self.display_parameters, "neurongroups")
        )

        self.threshold_button.clicked.connect(self.not_connected)
        self.reset_button.clicked.connect(self.not_connected)
        self.refractory_button.clicked.connect(self.not_connected)
        self.solver_button.clicked.connect(self.not_connected)
        self.neuronnumber_button.clicked.connect(self.not_connected)
        self.name_button.clicked.connect(
            partial(self.accept_name, target="neurongroups")
        )

        equation_button.clicked.connect(
            partial(
                self.accept_parameter, equation, "equation", self.model.define_equation
            )
        )


class SynapseDialog(ElementDialog):
    def accept(self) -> None:
        if self.name is None or self.name not in self.model.synapses:
            self.display_error("Please define a synapse first by giving it a name!")
        else:
            self.model.assemble_synapse(self.name)
            self.done(1)

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        super().__init__(model, parent)

        self.setWindowTitle("Define a synapse")

        origin_title = QtWidgets.QLabel("Origin", self)
        origin = QtWidgets.QLineEdit(self)
        origin_button = QtWidgets.QPushButton("save", self)

        target_title = QtWidgets.QLabel("Target", self)
        target = QtWidgets.QLineEdit(self)
        target_button = QtWidgets.QPushButton("save", self)

        equation_title = QtWidgets.QLabel("Equation", self)
        equation = QtWidgets.QTextEdit(self)
        equation_button = QtWidgets.QPushButton("save", self)

        pre_title = QtWidgets.QLabel("Pre", self)
        pre = QtWidgets.QTextEdit(self)
        pre_button = QtWidgets.QPushButton("save", self)

        post_title = QtWidgets.QLabel("Post", self)
        post = QtWidgets.QTextEdit(self)
        post_button = QtWidgets.QPushButton("save", self)

        connect_title = QtWidgets.QLabel("Connection def.", self)
        connect = QtWidgets.QLineEdit(self)
        connect_button = QtWidgets.QPushButton("save", self)

        connect_prob_title = QtWidgets.QLabel("Connection prob.", self)
        connect_prob = QtWidgets.QLineEdit(self)
        connect_prob_button = QtWidgets.QPushButton("save", self)

        for qle in [
            origin,
            target,
            equation,
            pre,
            post,
            connect,
            connect_prob,
        ]:
            qle.setMinimumWidth(
                qle.sizeHint().width() * 3
            )  # Set the width to twice the default width

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.group_name, 1, 2)
        self.layout.addWidget(self.name_button, 1, 3)

        self.layout.addWidget(origin_title, 2, 1)
        self.layout.addWidget(origin, 2, 2)
        self.layout.addWidget(origin_button, 2, 3)

        self.layout.addWidget(target_title, 3, 1)
        self.layout.addWidget(target, 3, 2)
        self.layout.addWidget(target_button, 3, 3)

        self.layout.addWidget(equation_title, 4, 1)
        self.layout.addWidget(equation, 4, 2)
        self.layout.addWidget(equation_button, 4, 3)

        self.layout.addWidget(pre_title, 5, 1)
        self.layout.addWidget(pre, 5, 2)
        self.layout.addWidget(pre_button, 5, 3)

        self.layout.addWidget(post_title, 6, 1)
        self.layout.addWidget(post, 6, 2)
        self.layout.addWidget(post_button, 6, 3)

        self.layout.addWidget(connect_title, 7, 1)
        self.layout.addWidget(connect, 7, 2)
        self.layout.addWidget(connect_button, 7, 3)

        self.layout.addWidget(connect_prob_title, 8, 1)
        self.layout.addWidget(connect_prob, 8, 2)
        self.layout.addWidget(connect_prob_button, 8, 3)

        self.layout.addWidget(self.show_parameters, 9, 1, 1, 3, QtCore.Qt.AlignCenter)

        self.layout.addWidget(self.buttonBox, 10, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        # connect signals and sockets: TODO
        self.origin_button.clicked.connect(self.not_connected)
        self.target_button.clicked.connect(self.not_connected)
        self.equation_button.clicked.connect(self.not_connected)
        self.pre_button.clicked.connect(self.not_connected)
        self.post_button.clicked.connect(self.not_connected)
        self.connect_button.clicked.connect(self.not_connected)
        self.connect_prob_button.clicked.connect(self.not_connected)
        self.name_button.clicked.connect(partial(self.accept_name, "synapses"))


class DictionaryWidget(QtWidgets.QWidget):
    def __init__(
        self, dictionary: dict, parent: QtWidgets.QDialog | None = None
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        self.dictionary = dictionary

        if dictionary is None or len(dictionary) == 0:
            self.table = QtWidgets.QLabel("No parameters defined")
        else:
            self.table = QtWidgets.QTableWidget()
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setRowCount(len(dictionary))
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

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        super().__init__(parent)
        self.model = model
        layout = QtWidgets.QGridLayout()

        self.setWindowTitle("Model")

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

        # connect signals to slots on the outermost
        add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        parameter_overview_button.clicked.connect(self.display_parameters)

        self.setLayout(layout)

    def open_add_neurongroup_dialog(self) -> None:
        dlg = NeuronGroupDialog(self.model, self)

        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")

        dlg.close()

    def open_add_synapse_dialog(self) -> None:
        dlg = SynapseDialog(self.model, self)

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
