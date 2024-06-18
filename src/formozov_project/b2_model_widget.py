from __future__ import annotations

from functools import partial
from typing import Any
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
        self.name_entry = QtWidgets.QLineEdit(self)
        self.name_button = QtWidgets.QPushButton("save", self)
        self.show_parameters_button = QtWidgets.QPushButton("Show parameters", self)

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

        try:
            line = element.text()
        except AttributeError:
            line = element.toPlainText()
        except Exception as e:
            self.display_error(f"Error when getting parameter {paramname}: {e}")

        try:
            var = convert(line)
            model_callback(self.name, var)
        except Exception as e:
            self.display_error(f"Error when saving parameter {paramname}: {e}")

    def accept_name(self, target: str) -> None:
        name = self.name_entry.text()
        if name == "":
            self.display_error("Please enter a name!")
        elif name in getattr(self.model, target):
            self.display_error("Name already exists!")
        else:
            self.name = name
            getattr(self.model, target)[name] = {}
            self.name_button.setText("saved")

    def display_parameters(self, target: str) -> None:
        d = getattr(self.model, target).get(self.name, None)
        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        dlg = DictionaryWidget(d, self)
        layout.addWidget(dlg)
        dlg.setLayout(layout)
        window.resize(800, 600)
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
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(
            self.name_entry,
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

        self.layout.addWidget(
            self.show_parameters_button, 8, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.buttonBox, 9, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        # connect signals and sockets: TODO

        self.show_parameters_button.clicked.connect(
            partial(self.display_parameters, "neurongroups")
        )

        threshold_button.clicked.connect(self.not_connected)
        reset_button.clicked.connect(self.not_connected)
        refractory_button.clicked.connect(self.not_connected)
        solver_button.clicked.connect(self.not_connected)
        neuronnumber_button.clicked.connect(self.not_connected)
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
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.name_entry, 1, 2)
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

        self.layout.addWidget(
            self.show_parameters_button, 9, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.buttonBox, 10, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        # connect signals and sockets: TODO
        origin_button.clicked.connect(self.not_connected)
        target_button.clicked.connect(self.not_connected)
        equation_button.clicked.connect(self.not_connected)
        pre_button.clicked.connect(self.not_connected)
        post_button.clicked.connect(self.not_connected)
        connect_button.clicked.connect(self.not_connected)
        connect_prob_button.clicked.connect(self.not_connected)

        self.name_button.clicked.connect(partial(self.accept_name, "synapses"))
        self.show_parameters_button.clicked.connect(
            partial(self.display_parameters, "synapses")
        )


class ParameterDialog(ElementDialog):
    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        super().__init__(model, parent)
        self.setWindowTitle("Define a parameter")

        value_title = QtWidgets.QLabel("Value", self)
        self.value_entry = QtWidgets.QTextEdit(self)
        self.value_button = QtWidgets.QPushButton("save", self)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.name_entry, 1, 2)
        self.layout.addWidget(self.name_button, 1, 3)

        self.layout.addWidget(value_title, 2, 1)
        self.layout.addWidget(self.value_entry, 2, 2)
        self.layout.addWidget(self.value_button, 2, 3)

        self.layout.addWidget(
            self.show_parameters_button, 3, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.buttonBox, 4, 1, 1, 3, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        for qle in [self.name_entry, self.value_entry]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        self.show_parameters_button.clicked.connect(
            partial(self.display_parameters, "parameters")
        )

        self.value_button.clicked.connect(self.accept_value)

        self.name_button.clicked.connect(partial(self.accept_name, target="parameters"))

        if self.model.parameters is None:
            self.model.parameters = {}

    def accept_value(self) -> None:
        if self.name is None:
            self.display_error("Please define a parameter first by giving it a name!")
        else:
            self.accept_parameter(
                self.value_entry, self.name, self.model.define_parameter
            )
            self.value_button.setText("saved")

    def display_parameters(self, _: str) -> None:
        d = self.model.parameters if self.model.parameters is not None else {}

        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        dlg = DictionaryWidget(d, self)
        layout.addWidget(dlg)
        dlg.setLayout(layout)
        window.resize(800, 600)
        window.exec()


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
            self.table = QtWidgets.QTreeWidget(self)
            self.table.setColumnWidth(
                0, 400
            )  # Set the width of the first column to 200 pixels
            self.table.setColumnWidth(
                1, 400
            )  # Set the width of the second column to 200 pixels
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setHeaderLabels(["Key", "Value"])

            self.add_dict_to_tree(self.table, dictionary)

        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_dict_to_tree(
        self,
        widget: QtWidgets.QTreeWidget,
        dictionary: dict,
        parent: QtWidgets.QDialog | None = None,
    ) -> None:
        if parent is None:
            parent = widget.invisibleRootItem()

        for key, value in dictionary.items():
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, str(key))

            if isinstance(value, dict):
                self.add_dict_to_tree(widget, value, item)
            else:
                item.setText(1, str(value))

            parent.addChild(item)


class B2ModelWidget(ElementDialog):
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
            self.save(self.model_register)
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
        self,
        model: B2Model,
        parent: QtWidgets.QDialog | None = None,
        model_register: dict | None = None,
    ) -> None:
        super().__init__(model, parent)

        self.model_register = model_register

        self.setWindowTitle("Model definition")
        self.show_parameters_button.hide()
        timestep_title = QtWidgets.QLabel("Timestep", self)
        timestep = QtWidgets.QLineEdit()
        timestep_button = QtWidgets.QPushButton("save", self)

        seed = QtWidgets.QLineEdit(self)
        seed_button = QtWidgets.QPushButton("save", self)
        seed_title = QtWidgets.QLabel("Random seed")

        add_parameter_button = QtWidgets.QPushButton("Add parameter", self)
        add_neurongroup_button = QtWidgets.QPushButton("Add neurongroup", self)
        add_synapse_button = QtWidgets.QPushButton("Add synapse", self)
        display_system_button = QtWidgets.QPushButton("Display system", self)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.name_entry, 1, 2, 1, 2)
        self.layout.addWidget(self.name_button, 1, 4)

        self.layout.addWidget(timestep_title, 2, 1)
        self.layout.addWidget(timestep, 2, 2, 1, 2)
        self.layout.addWidget(timestep_button, 2, 4)

        self.layout.addWidget(seed_title, 3, 1)
        self.layout.addWidget(seed, 3, 2, 1, 2)
        self.layout.addWidget(seed_button, 3, 4)

        self.layout.addWidget(add_parameter_button, 5, 1)

        self.layout.addWidget(
            add_neurongroup_button,
            5,
            2,
        )

        self.layout.addWidget(
            add_synapse_button,
            5,
            3,
        )

        self.layout.addWidget(display_system_button, 5, 4)

        self.layout.addWidget(self.buttonBox, 8, 1, 1, 4, QtCore.Qt.AlignCenter)

        # connect signals to slots on the outermost

        add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        add_parameter_button.clicked.connect(self.open_add_parameter_dialog)
        display_system_button.clicked.connect(self.display_system)

        timestep_button.clicked.connect(self.not_connected)
        seed_button.clicked.connect(self.not_connected)
        self.name_button.clicked.connect(self.not_connected)

        self.setLayout(self.layout)

    def display_system(self) -> None:
        d: dict[Any, Any] = {}
        d["neurongroups"] = self.model.neurongroups
        d["synapses"] = self.model.synapses
        d["parameters"] = self.model.parameters
        d["timestep"] = self.model.dt
        d["seed"] = self.model.seed
        d["name"] = self.name

        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        dlg = DictionaryWidget(d, self)
        layout.addWidget(dlg)
        dlg.setLayout(layout)
        window.resize(800, 600)
        window.exec()

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

    def open_add_parameter_dialog(self) -> None:
        dlg = ParameterDialog(self.model, self)

        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")

        dlg.close()

    def save(self, model_dict: dict | None) -> None:
        if self.name is None:
            self.display_error("Please define a name first!")

        elif model_dict is None:
            self.display_error(
                "Please provide a model dictionary. This method is yet supported"
            )
        else:
            model_dict["self.name"] = self


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    model = B2Model()
    widget = B2ModelWidget(model)
    widget.show()
    sys.exit(app.exec_())
