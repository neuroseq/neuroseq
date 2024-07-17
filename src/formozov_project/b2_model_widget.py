from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model


class ModelDialog(QtWidgets.QDialog):
    """
    Base class for all dialog windows in the B2ModelWidget.

    Parameters
    ----------
    QtWidgets : QtWidgets.QDialogN
        General dialog widget from qt.
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QWidget | None = None) -> None:
        """
        Create a new ModelDialog instance.

        Parameters
        ----------
        model : B2Model
            Model to set up using this dialog
        parent : QtWidgets.QDialog | None, optional
            Possible parent widget, by default None
        """
        super().__init__(parent)
        self.model = model
        self.name: str | None = None

        self.layout = QtWidgets.QGridLayout(self)

        self.name_entry, self.name_button = self.add_text_field("Name")
        self.show_parameters_button = QtWidgets.QPushButton("Show parameters", self)

        self.qtbtn = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        # do not add these button_box immediatelly to the layout, as ok and cancel buttons are typically expected to appear on the last row

        self.button_box = QtWidgets.QDialogButtonBox(self.qtbtn)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(self.layout)

    def _convert_string_to_quantity(
        self, input: str, valuetype: type = float
    ) -> b2.Quantity:
        """
        Convert string input to brian2 quantities, comprised of value and unit.

        Parameters
        ----------
        input : str
            String to convert to brian2 quantity. Must be of the form `value * unit` or `value unit`.
        valuetype : type, optional
            _description_, by default float
        Returns
        -------
        b2.Quantity
            Brian2 quantity instance corresponding to the input string
        """

        # quantity is nparray | scalar * unit

        v = input.split("*")

        if len(v) == 1:
            v = input.split(" ")

        if len(v) == 1:
            raise ValueError(
                "Quantities must follow the pattern 'value * unit ' or 'value unit'"
            )

        value, unit = v

        value = value.strip()

        unit = unit.strip()

        # Convert the value to float
        converted_value = valuetype(literal_eval(value))

        # Get the unit from the b2.units module
        b2unit: b2.Unit = getattr(b2.units, unit)

        # Return the quantity
        return converted_value * b2unit

    def _convert_value_optional(
        self, input: str, valuetype: type = float
    ) -> str | b2.Quantity:
        """
        Try to convert the input string into a brian2 quantity with a value of type `type` and
        a unit as given in the string. If that fails, return the raw string.
        This function is mostly useful where brian2 accepts either a quantity or a string.

        Parameters
        ----------
        input : str
            Inputstring to possibly convert
        valuetype : type, optional
            type of the quantities value: int, float, list, np.array etc, by default float

        Returns
        -------
        str | b2.Quantity
            Either the converted quantity if conversion is possible or the input string
        """
        try:
            return self._convert_string_to_quantity(input, valuetype=valuetype)
        except ValueError:
            return input
        except Exception as e:
            raise RuntimeError(
                f"An error occured during conversion of a string to a quantity {e}"
            ) from e

    def add_text_field(
        self,
        title: str,
        textfieldtype: type = QtWidgets.QLineEdit,
        buttontext: str = "save",
        layout: QtWidgets.QGridLayout = None,
    ) -> tuple[
        QtWidgets.QTextEdit | QtWidgets.QLineEdit,
        QtWidgets.QPushButton,
    ]:
        """
        Add an element to the dialog that has, from left to right, a name, a multi- or single line field that allows text input, and a push button displaying a given text.

        Parameters
        ----------
        title : str
            Name of the field, displayed on the leftmost element.
        textfieldtype : type, optional
            Type of the text input field to have on the second elemnet, by default QtWidgets.QLineEdit
        buttontext : str, optional
            Text to display in the rightmost button, by default "save"
        layout : QtWidgets.QLayout | None
            The layout to add the buttons to

        Returns
        -------
        tuple[QtWidgets.QTextEdit | QtWidgets.QLineEdit, QtWidgets.QPushButton, ]
            titlefield, textfield and button widgets
        """

        if layout is None:
            layout = self.layout

        titlefield = QtWidgets.QLabel(title, self)
        textfield = textfieldtype(self)
        button = QtWidgets.QPushButton(buttontext, self)

        row = layout.rowCount()

        layout.addWidget(titlefield, row, 1)
        layout.addWidget(textfield, row, 2)
        layout.addWidget(button, row, 3)

        return textfield, button

    def display_error(self, message: str) -> None:
        """
        Display an error message when something goes wrong in the dialog.
        The message is displayed in a new window that has to be confirmed by the user.

        Parameters
        ----------
        message : str
            Message to display
        """
        self.error_dialog = QtWidgets.QMessageBox(self)
        self.error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        self.error_dialog.setWindowTitle("Error")
        self.error_dialog.setText(message)
        self.error_dialog.exec()

    def accept_parameter(
        self,
        element: QtWidgets.QLineEdit | QtWidgets.QTextEdit,
        paramname: str,
        model_callback: Callable,
        convert: Callable = lambda x: x,
    ) -> None:
        """
        Accept a parameter into the model system.

        Parameters
        ----------
        element : QtWidgets.QLineEdit | QtWidgets.QTextEdit
            Element to read the parameter value from
        paramname : str
            Name of the parameter in the model
        model_callback : Callable
            method of `B2Model` to call to set the parameter
        convert : Callable, optional
            Converter function to convert the value into the correct type. Not implemented yet. by default lambdax:x
        """
        if self.name is None:
            self.display_error("Please define a name first!")
            return

        try:
            line = (
                element.toPlainText()
                if isinstance(element, QtWidgets.QTextEdit)
                else element.text()
            )
        except Exception as e:
            self.display_error(f"Error when getting parameter {paramname}: {e}")
            line = None

        element.clear()

        if line is not None:
            try:
                var = convert(line)
                model_callback(self.name, var)
            except Exception as e:
                self.display_error(f"Error when saving parameter {paramname}: {e}")

    def accept_parameter_with_savebutton(
        self,
        element: QtWidgets.QLineEdit | QtWidgets.QTextEdit,
        name: str,
        model_callback: Callable,
        button: QtWidgets.QPushButton,
        converter: Callable = lambda x: x,
        text: str = "saved",
    ) -> None:
        """
        Wrapper around `ModelDialog.accept_parameter` that additionally changes the text in the `button` argument to the argument 'text'.

        Parameters
        ----------
        element : QtWidgets.QLineEdit | QtWidgets.QTextEdit
            Text input element to read the parameter from to be used
        name : str
            Name of the parameter
        model_callback : Callable
            Function that defines the parameter in the model
        button : QtWidgets.QPushButton
            Button to change text in.
        converter: Callable
            Function to convert types - not implemented yet.
        text: str
            Text to change the button text to
        """
        self.accept_parameter(element, name, model_callback, converter)
        button.setText(text)
        button.setDown(True)

    def accept_name(self, target: str) -> None:
        """
        Accept the name attribute of a derived dialog and add it to the `target` member of the model, e.g., to `neurongroups` or `synapses`.
        Can only be used when the target should be a dictionary.
        Parameters
        ----------
        target : str
            Name of the B2Model member to add the name to
        """
        name = self.name_entry.text()
        if name == "":
            self.display_error("Please enter a name!")
        elif self.name is not None:
            self.display_error("A name has already been defined")
        else:
            if name not in getattr(self.model, target):
                self.name = name
                getattr(self.model, target)[name] = {"name": name}
            else:
                self.display_error(
                    f"The name {name} has already been defined in {target}"
                )
            self.name_button.setText("saved")
            self.name_button.setDown(True)

    def display_parameters(
        self,
        target: str | dict[Any, Any] | None,
        read_only: bool = False,
        blocking: bool = True,
    ) -> None:
        """
        Display the parameters of a target member of the model in a new window.

        Parameters
        ----------
        target : str
            name of the target variable in the model
        read_only: bool
            whether or not the entries should be read only
        blocking: bool
            whether the parameter window should be opened as a blocking window or not
        """
        if isinstance(target, str):
            d = getattr(self.model, target).get(self.name, None)
        else:
            d = target

        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        self.dictwidget = DictionaryWidget(d, window, read_only=read_only)
        layout.addWidget(self.dictwidget)

        window.resize(800, 600)  # FIXME: make this more flexible

        window.move(self.x() + self.width(), self.y())

        if blocking:
            window.exec()
        else:
            window.show()

    def not_connected(self) -> None:
        """
        Dummy function that displays a message for buttons and elements that are not connected to a slot yet.
        """
        self.display_error("Not connected yet!")


class NeuronGroupDialog(ModelDialog):
    """
    Dialog window for defining a Neurongroup for a b2model.

    Parameters
    ----------
    ModelDialog : ModelDialog
        The base class of all dialogs in the b2_model_widget namespace.
    """

    def accept(self) -> None:
        # verify entries
        if self.name not in self.model.neurongroups:
            self.display_error("Error, first define a neurongroup by giving it a name")
        elif (
            "model" not in self.model.neurongroups[self.name]
            or "n" not in self.model.neurongroups[self.name]
        ):
            self.display_error(
                "The model equation and the neuronnumber need to be defined"
            )
        else:
            try:
                self.model.assemble_neurongroup(name=self.name)
                super().accept()
            except Exception as e:
                self.display_error(
                    f"Error when assembling neurongroup {self.name}: {e}"
                )

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        """
        Create a new NeuronGroupDialog instance which is displayed in a new window. Creates a table of
        text fields to fill in and buttons to save the data that has been provided into the respective variable of the model.

        Parameters
        ----------
        model : B2Model
            The model instance to set up using this dialog
        parent : QtWidgets.QDialog | None, optional
            possible parent window of this instance, by default None
        """
        super().__init__(model, parent)
        self.setWindowTitle("Define a Neurongroup")

        self.equation, self.equation_button = self.add_text_field(
            "Equation", QtWidgets.QTextEdit
        )
        self.threshold, self.threshold_button = self.add_text_field(
            "Threshold", QtWidgets.QLineEdit
        )
        self.reset, self.reset_button = self.add_text_field(
            "Reset", QtWidgets.QLineEdit
        )
        self.refractory, self.refractory_button = self.add_text_field(
            "Refractory", QtWidgets.QLineEdit
        )
        self.solver, self.solver_button = self.add_text_field(
            "Solver method", QtWidgets.QLineEdit
        )
        self.neuronnumber, self.neuronnumber_button = self.add_text_field(
            "Neuronnumber", QtWidgets.QLineEdit
        )

        for qle in [
            self.equation,
            self.threshold,
            self.reset,
            self.refractory,
            self.solver,
            self.neuronnumber,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        row = self.layout.rowCount()

        # Add the buttons to the layout
        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 4, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.button_box, row + 1, 1, 1, 4, QtCore.Qt.AlignCenter)

        # connect signals and sockets:

        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters("neurongroups")
        )

        self.threshold_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.threshold,
                "threshold",
                self.model.define_threshold,
                self.threshold_button,
            )
        )

        self.reset_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.reset, "reset", self.model.define_reset, self.reset_button
            )
        )

        self.refractory_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.refractory,
                "refractory",
                self.model.define_refractory,
                self.refractory_button,
                self._convert_value_optional,
            )
        )

        self.solver_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.solver,
                "solver",
                self.model.define_solver_method,
                self.solver_button,
            )
        )

        self.neuronnumber_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.neuronnumber,
                "neuronnumber",
                self.model.define_neuronnumber,
                self.neuronnumber_button,
                lambda v: int(v),
            )
        )

        self.name_button.clicked.connect(
            lambda: self.accept_name(target="neurongroups")
        )

        self.equation_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.equation, "model", self.model.define_equation, self.equation_button
            )
        )


class NeuronSubgroupDialog(ModelDialog):
    """
    Dialog window for defining a subgroup of a neurongroup.

    Parameters
    ----------
    ModelDialog :
        The common baseclass for all the dialog windows used to define a model.
    """

    def accept(self) -> None:
        """
        Accept the current subgroup definition. Displays error message if false
        """

        if self.name is None or self.target_name is None:
            self.display_error(
                "Error, every neuron subgroup needs a name and a defined target"
            )
        else:
            try:
                self.model.define_neuron_subgroup_target(self.name, self.target_name)
                self.model.define_neuron_subgroup_idxs(self.name, self.subgroup_def)
                self.model.assemble_neuronsubgroup(self.name)
                super().accept()
            except Exception as e:
                del self.model.neurongroups[self.name]  # delete erroneous definition
                self.display_error(
                    f"Error when defining a neuronsubgroup on target {self.target_name}: {e}"
                )

    def __init__(self, model: B2Model, parent: QtWidgets.QWidget | None = None):
        """
        Create a new dialog to define a neuronsubgroup of an existing neurongroup. This neurongroup is
        always set to be defined in the parent dialog fo this window.
        Displays an error message when the parent has no name defined

        Parameters
        ----------
        model : B2Model
            Model to define a subgroup for
        parent :QtWidgets.QWidget
            parent window. Defines the parentwindow
        """
        super().__init__(model, parent)

        self.target_name: str | None = None
        self.subgroup_def: int | tuple | list = 0

        self.target, self.target_button = self.add_text_field(
            "Target", QtWidgets.QLineEdit
        )

        self.definition, self.definition_button = self.add_text_field(
            "Indices", QtWidgets.QLineEdit
        )

        row = self.layout.rowCount()

        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 3, QtCore.Qt.AlignCenter
        )
        self.layout.addWidget(self.button_box, row + 1, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.name_button.clicked.connect(
            lambda: self.accept_name(target="neurongroups")
        )

        self.definition_button.clicked.connect(self.get_subgroup_def)

        self.target_button.clicked.connect(self.get_targetname)

        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters("neurongroups")
        )

    def showEvent(self, event: QtCore.QEvent) -> None:
        # TODO: this is here to make the window a bit bigger. Is there a better way to do this?
        # Get the default size
        default_size = self.size()

        # Set the window size relative to the default size
        self.resize(
            int(default_size.width() * 2), int(default_size.height() * 1.2)
        )  # Double the width and height

    def get_subgroup_def(self) -> None:
        """
        parse the input string of the input field into a suitable type for defining neuronsubgroups
        """
        subgrp = self.definition.text().strip()

        # try to convert text input into one of the suitable type
        try:
            if "slice(" in subgrp:
                # README: eval is generally insecure, but literal_eval does not support slices.
                # Perhaps add a custom function in the future to get rid of eval here.
                self.subgroup_def = eval(subgrp)
            else:
                self.subgroup_def = literal_eval(subgrp)
        except Exception as e:
            self.display_error(
                f"Error when trying to build subgroup {self.name} of neurongroup {self.target_name}: {e} with input {subgrp}"
            )

        if type(self.subgroup_def) not in [slice, list, int, tuple]:
            self.display_error(
                f"subgroup {self.name} of neurongroup {self.target_name} is defined with a wrong type: {type(self.subgroup_def)}"
            )

        self.definition_button.setDown(True)
        self.definition_button.setText("saved")

    def get_targetname(self) -> None:
        """
        Accept the given target name. Check that it corresponds to an existing neurongroup first.
        """
        tgt = self.target.text()

        if tgt not in self.model.neurongroups:
            self.display_error(
                f"Define the neurongroup first before you define a subgroup for it. {tgt} is not known"
            )
        else:
            self.target_name = tgt
            self.target_button.setDown(True)
            self.target_button.text = "saved"

    def display_parameters(
        self, _: str | dict[Any, Any] | None = None, __: bool = True, ___: bool = True
    ) -> None:
        """
        Display the parameters of a target member of the model in a new window.

        """

        self.parameterwindow = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(self.parameterwindow)
        table = QtWidgets.QTableWidget(self.parameterwindow)
        table.setRowCount(3)
        table.setColumnCount(2)

        for i, (n, v) in enumerate(
            zip(
                ["name", "target", "indices"],
                [self.name, self.target_name, self.subgroup_def],
                strict=True,
            )
        ):
            nameitem = QtWidgets.QTableWidgetItem(str(n))
            valueitem = QtWidgets.QTableWidgetItem(str(v))

            table.setItem(i, 0, nameitem)
            table.setItem(i, 1, valueitem)

        layout.addWidget(table)
        self.parameterwindow.setLayout(layout)

        default_size = self.parameterwindow.size()

        # Set the window size relative to the default size
        self.parameterwindow.resize(
            default_size.width() * 3, default_size.height() * 6
        )  # Double the width and height

        self.parameterwindow.exec()


class SynapseDialog(ModelDialog):
    """
    Dialog window for defining a Synapse for a b2model.

    Parameters
    ----------
    ModelDialog : ModelDialog
        The base class of all dialogs in the b2_model_widget namespace.
    """

    def accept(self) -> None:
        if self.name not in self.model.synapses:
            self.display_error(
                "Error, first define a synapse system by giving it a name"
            )
        elif (
            "model" not in self.model.synapses[self.name]
            or "target" not in self.model.synapses[self.name]
            or "origin" not in self.model.synapses[self.name]
        ):
            self.display_error(
                "The model equation and the origin- and target neurongroups need to be defined"
            )
        else:
            try:
                self.model.assemble_synapse(name=self.name)
                super().accept()
            except Exception as e:
                self.display_error(
                    f"Error when assembling neurongroup {self.name}: {e}"
                )

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        """
        Create a new SynapseDialog instance and display it in a new window.
        The dialog contains a table of text fields to fill in and buttons to save the data that has been provided into the
         respective element of the synapse dictionary of the model.

        Parameters
        ----------
        model : B2Model
            Model instance to set up using this dialog
        parent : QtWidgets.QDialog | None, optional
            Possible parent window of the window, by default None
        """
        super().__init__(model, parent)

        self.setWindowTitle("Define a synapse")

        self.origin, self.origin_button = self.add_text_field("Origin")

        self.target, self.target_button = self.add_text_field("Target")

        self.equation, self.equation_button = self.add_text_field(
            "Equation", textfieldtype=QtWidgets.QTextEdit
        )

        self.pre, self.pre_button = self.add_text_field(
            "Pre", textfieldtype=QtWidgets.QTextEdit
        )

        self.post, self.post_button = self.add_text_field(
            "Post", textfieldtype=QtWidgets.QTextEdit
        )

        self.connect, self.connect_button = self.add_text_field("Connection def.")

        self.connect_prob, self.connect_prob_button = self.add_text_field(
            "Connection prob."
        )

        for qle in [
            self.origin,
            self.target,
            self.equation,
            self.pre,
            self.post,
            self.connect,
            self.connect_prob,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        row = self.layout.rowCount()

        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.button_box, row + 1, 1, 1, 4, QtCore.Qt.AlignCenter)

        # connect signals and slots
        self.origin_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.origin,
                "origin",
                self.model.define_synapse_origin,
                self.origin_button,
            )
        )
        self.target_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.target,
                "target",
                self.model.define_synapse_target,
                self.target_button,
            )
        )
        self.equation_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.equation,
                "equation",
                self.model.define_synapse_equation,
                self.equation_button,
            )
        )
        self.pre_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.pre, "on_pre", self.model.define_synapse_pre, self.pre_button
            )
        )
        self.post_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.post, "on_post", self.model.define_synapse_post, self.post_button
            )
        )
        self.connect_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.connect,
                "connection_condition",
                self.model.define_synapse_connection,
                self.connect_button,
            )
        )
        self.connect_prob_button.clicked.connect(
            lambda: self.accept_parameter_with_savebutton(
                self.connect_prob,
                "connection_probability",
                self.model.define_synapse_connection_probablitiy,
                self.connect_prob_button,
                converter=lambda input: float(input),
            )
        )

        self.name_button.clicked.connect(lambda: self.accept_name("synapses"))
        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters("synapses")
        )


class ParameterDialog(ModelDialog):
    """
    Dialog window for building up the dictionary of parameters for a B2Model instance.

    Parameters
    ----------
    ModelDialog : ModelDialog
        The base class of all dialogs in the b2_model_widget namespace.
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        """
        Create a new ParameterDialog instance and display it in a new window.
        The dialog contains a table of text fields to fill in and buttons to save the data that has been provided
        into the respective variable of the model.

        Parameters
        ----------
        model : B2Model
            Model instance to set up using this dialog
        parent : QtWidgets.QDialog | None, optional
            Possible parent window of this instance, by default None
        """
        super().__init__(model, parent)
        self.setWindowTitle("Define a parameter")

        self.value_entry, self.value_button = self.add_text_field(
            "Value", textfieldtype=QtWidgets.QTextEdit
        )

        row = self.layout.rowCount()

        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.button_box, row + 1, 1, 1, 3, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        for qle in [self.name_entry, self.value_entry]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters("parameters")
        )

        self.value_button.clicked.connect(self.accept_value)

        self.name_button.clicked.connect(lambda: self.accept_name("parameters"))

        if self.model.parameters is None:
            self.model.parameters = {}

    def accept(self) -> None:
        """
        accept the dialog
        """
        if self.name is None:
            self.display_error("No valid name has been defined")
        else:
            super().accept()

    def accept_value(self) -> None:
        """
        Overridden method to accept the value of a parameter and save it in the model.
        """
        if self.name is None:
            self.display_error("Please define a parameter first by giving it a name!")
        elif self.value_entry.toPlainText() == "":
            self.display_error("Please assign a value or expression before saving!")
        else:
            self.accept_parameter(
                self.value_entry, self.name, self.model.define_parameter, convert=eval
            )
            self.value_button.setText("saved")

    def display_parameters(
        self,
        _: str | dict[Any, Any] | None,
        read_only: bool = False,
        blocking: bool = True,
    ) -> None:
        """
        Display all the parameters in the system in a new window.

        Parameters
        ----------
        _ : str
            dummy for override
        read_only: bool
            Whether to have the parameter table read only or not
        blocking: bool
            Whether to have the parameter window block the rest of the app or not
        """
        d = self.model.parameters if self.model.parameters is not None else {}

        super().display_parameters(d, read_only=read_only, blocking=blocking)


class DictionaryWidget(QtWidgets.QWidget):
    """
    Widget to display possibly nested dictionaries in a tree view.

    Parameters
    ----------
    QtWidgets : QtWidgets.QWidget
        Basic QtWidget class
    """

    def __init__(
        self,
        dictionary: dict,
        parent: QtWidgets.QDialog | None = None,
        read_only: bool = True,
    ) -> None:
        """
        Create a new window to display a dictionary in.

        Parameters
        ----------
        dictionary : dict
            Dictionary to display
        parent : QtWidgets.QDialog | None, optional
            possible parent window, by default None
        """
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        self.dictionary = dictionary

        if dictionary is None or len(dictionary) == 0:
            self.table = QtWidgets.QLabel("No parameters defined")
        else:
            self.table = QtWidgets.QTreeWidget(self)
            self.table.setColumnWidth(0, 300)
            self.table.setColumnWidth(1, 400)
            if read_only:
                self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setHeaderLabels(["Key", "Value"])

            self.add_dict_to_tree(dictionary)

        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_dict_to_tree(
        self,
        dictionary: dict,
        parent: QtWidgets.QTreeWidgetItem | None = None,
    ) -> None:
        """
        Recursive function to add possibly nested dictionaries to the tree view.

        Parameters
        ----------
        widget : QtWidgets.QTreeWidget
            TreeWidget to add the dict to.
        dictionary : dict
            Dictionary to add
        parent : QtWidgets.QTreeWidgetItem | None, optional
            Parent element in the tree view, by default None
        """
        if parent is None:
            parent = self.table.invisibleRootItem()

        existing_elements = {
            parent.child(i).text(0): parent.child(i) for i in range(parent.childCount())
        }

        for key, value in dictionary.items():
            if str(key) in existing_elements:
                continue
            else:
                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, str(key))

                if isinstance(value, dict):
                    self.add_dict_to_tree(value, item)
                else:
                    item.setText(1, str(value))
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

                parent.addChild(item)

    def find_item_by_name(self, name: str) -> QtWidgets.QTreeWidgetItem | None:
        """
        find and return an item by its entry in the name column when it is in the treewidget, otherwise return none

        Parameters
        ----------
        name : str
            name to look up

        Returns
        -------
        QtWidgets.QTreeWidgetItem | None
            The retrieved item, or None if the name does not exist
        """

        found_items = self.table.findItems(
            name,
            QtCore.Qt.MatchFlag.MatchExactly | QtCore.Qt.MatchFlag.MatchRecursive,
            0,
        )

        if len(found_items) > 0:
            return found_items[0]
        else:
            return None


class MonitorDialog(ModelDialog):
    """
    Dialog for defining brian2 monitors

    Parameters
    ----------
    ModelDialog :
        Common baseclass for all model dialogs
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog = None) -> None:
        """
        Open a new dialog to define a monitor

        Parameters
        ----------
        model : B2Model
            The model to define the monitor for
        parent : QtWidgets.QDialog
            The parent dialog
        """
        super().__init__(model, parent)

        self.tp: str | None = None
        self.var: str | None = None
        self.src: str | None = None

        self.type, self.type_button = self.add_text_field(
            "Type",
            QtWidgets.QLineEdit,
        )

        self.source, self.source_button = self.add_text_field(
            "Source",
            QtWidgets.QLineEdit,
        )
        self.variable, self.variable_button = self.add_text_field(
            "Variable",
            QtWidgets.QLineEdit,
        )
        row = self.layout.rowCount()

        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.button_box, row + 1, 1, 1, 4, QtCore.Qt.AlignCenter)

        # connect signals and sockets
        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters("monitors")
        )

        layout = QtWidgets.QGridLayout()

        self.setLayout(layout)
        self.setWindowTitle("Define a Monitor")

        self.name_button.clicked.connect(lambda: self.accept_name(target="monitors"))

        self.type_button.clicked.connect(self.accept_type)

        self.source_button.clicked.connect(self.accept_source)

        self.variable_button.clicked.connect(self.accept_variable)

        for qle in [
            self.type,
            self.source,
            self.variable,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

    def accept_type(self) -> None:
        """
        Accept the type input
        """

        if self.type.text() == "":
            self.display_error("Please define type first")

        tptext = self.type.text()

        if tptext not in ["StateMonitor", "SpikeMonitor"]:
            self.display_error(f"Monitor type {tptext} is invalid")
        else:
            self.tp = tptext

    def accept_source(self) -> None:
        """
        Accept the source input
        """

        if self.source.text() == "":
            self.display_error("Please define source first")

        self.src = self.source.text()
        self.source_button.setText("saved")
        self.source_button.setDown(True)

    def accept_variable(self) -> None:
        """
        Accept the variable input
        """

        if self.variable.text() == "":
            self.display_error("Please define variable first")

        self.var = self.variable.text()
        self.variable_button.setText("saved")
        self.variable_button.setDown(True)

    def accept(self) -> None:
        """
        Accept the monitor data and define a new monitor in the held model
        """
        try:
            self.model.define_monitor_source(self.name, self.src)
            self.model.define_monitor_type(self.name, self.tp)
            self.model.define_monitor_variable(self.name, self.var)
            self.model.assemble_monitor(self.name)

            super().accept()
        except Exception as e:
            del self.model.monitors[self.name]  # type: ignore
            self.display_error(f"Error when defining monitor {self.name}: {e}")

    def display_parameters(
        self,
        _: str | dict[Any, Any] | None = None,
        read_only: bool = False,
        blocking: bool = False,
    ) -> None:
        """
        Display the parameters of a target member of the model in a new window.

        Parameters
        ----------
        target : str dummy
        read_only: bool whether for the elements to be writeable or not
        blocking: bool whether the window should block until ok is clicked or not

        """

        self.parameterwindow = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(self.parameterwindow)
        table = QtWidgets.QTableWidget(self.parameterwindow)
        table.setRowCount(4)
        table.setColumnCount(2)

        for i, (n, v) in enumerate(
            zip(
                ["name", "type", "source", "variable"],
                [self.name, self.tp, self.src, self.var],
                strict=True,
            )
        ):
            nameitem = QtWidgets.QTableWidgetItem(str(n))
            valueitem = QtWidgets.QTableWidgetItem(str(v))

            if read_only:
                nameitem.setFlags(nameitem.flags() & ~QtCore.Qt.ItemIsEditable)
                valueitem.setFlags(valueitem.flags() & ~QtCore.Qt.ItemIsEditable)

            table.setItem(i, 0, nameitem)
            table.setItem(i, 1, valueitem)

        layout.addWidget(table)
        self.parameterwindow.setLayout(layout)

        default_size = self.parameterwindow.size()

        # Set the window size relative to the default size
        self.parameterwindow.resize(
            default_size.width() * 3, default_size.height() * 6
        )  # Double the width and height

        self.parameterwindow.exec()


class InitialConditionWidget(ModelDialog):
    """
    InitialConditionWidget A dialog to define initial conditions for

    Parameters
    ----------
    ModelDialog : General baseclass for the b2 model widgets
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        """
        __init__ Create a new dialog for defining an initial condition

        Parameters
        ----------
        model : B2Model
            The model to define the initial condition for
        parent : QtWidgets.QDialog | None, optional
            Parent window, by default None
        """
        super().__init__(model, parent)

        self.layout.itemAt(0).widget().hide()  # name label hiding
        self.name_entry.hide()
        self.name_button.hide()

        self.params: dict[str, str | b2.Quantity | np.array] = {}

        self.source_name = QtWidgets.QLabel("Source", self)
        self.source = QtWidgets.QComboBox(self)
        self.source_button = QtWidgets.QPushButton("save", self)

        # make sure the objects are ordered the same way every time
        self.source.addItems(
            sorted([object.name for object in self.model.network.objects])
        )

        row = self.layout.rowCount()
        self.layout.addWidget(self.source_name, row, 1, 1, 1)
        self.layout.addWidget(self.source, row, 2, 1, 1)
        self.layout.addWidget(self.source_button, row, 3, 1, 1)

        row += 1

        self.var_name = QtWidgets.QLabel("Variable", self)
        self.var = QtWidgets.QComboBox(self)
        self.var_button = QtWidgets.QPushButton("save", self)

        self.layout.addWidget(self.var_name, row, 1, 1, 1)
        self.layout.addWidget(self.var, row, 2, 1, 1)
        self.layout.addWidget(self.var_button, row, 3, 1, 1)

        row += 1

        self.value, self.value_button = self.add_text_field(
            "Value", textfieldtype=QtWidgets.QTextEdit
        )

        self.layout.addWidget(
            self.show_parameters_button, row, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        row += 1
        self.layout.addWidget(self.button_box, row, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.show_parameters_button.clicked.connect(
            lambda: self.display_parameters(target=self.params, read_only=False)
        )
        self.source_button.clicked.connect(self.accept_source)
        self.var_button.clicked.connect(self.accept_variable)
        self.value_button.clicked.connect(self.accept_value)

        for qle in [
            self.source,
            self.var,
            self.value,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

    def accept(self) -> None:
        """
        accept Accept the initial condition. works only if 'source', 'variable' and 'value' are all set.
        """
        if any(var not in self.params for var in ["elementname", "varname", "value"]):
            self.display_error("Cannot define incompletely specified initial condition")
        elif (
            self.params["varname"]
            not in self.model.network[self.params["elementname"]].variables
        ):
            self.display_error(
                f'Variable name {self.params["varname"]} does not exist in element {self.params["elementname"]}'
            )
        else:
            try:
                self.model.define_init(**self.params)
                super().accept()
            except Exception as e:
                self.display_error(
                    f'Error when defining initial condition for source {self.params["elementname"]}, variable {self.params["varname"]}, with value {self.params["value"]}: {e}'
                )

    def accept_source(self) -> None:
        """
        accept_source Accept the name of a source group a variable to initialze belongs to.
        """
        self.params["elementname"] = self.source.currentText()

        self.source_button.setDown(True)
        self.source_button.setText("saved")
        self.var.addItems(
            sorted(self.model.network[self.params["elementname"]].variables.keys())
        )

    def accept_variable(self) -> None:
        """
        accept_variable Accept a variable to set an initial condition for
        """
        self.params["varname"] = self.var.currentText()
        self.var_button.setDown(True)
        self.var_button.setText("saved")

    def accept_value(self) -> None:
        """
        Overridden method to accept the value of a variable in a group and save it in the model.
        """
        # FIXME: the eval here is very insecure and should be replacde with something more restrictive
        txt = self.value.toPlainText()
        final_value = None
        try:
            # try to make expression
            b2.Expression(txt)
            final_value = txt
        except Exception as _:
            # try to eval code when expression making fails
            try:
                final_value = eval(txt)
            except Exception as e:
                self.display_error(f"Error in evaluating value expression {txt}: {e}")

        self.params["value"] = final_value
        self.value_button.setDown(True)
        self.value_button.setText("saved")


class B2ModelWidget(ModelDialog):
    """
    Main widget to define a B2Model instance in a new window.
    This widget contains buttons to add Neurongroups, Synapses, and Parameters to the model.
    It also contains buttons to display the defined system in a table and to save the model to a dictionary in the overarching gui.

    Parameters
    ----------
    ModelDialog : ModelDialog
        Base class for all dialogs in the b2_model_widget namespace.
    """

    def __init__(
        self,
        model: B2Model,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """
        Create a new instance of the B2ModelWidget and display it in a new window.

        Parameters
        ----------
        model : B2Model
            Model instance to fill with data
        parent : QtWidgets.QDialog | None, optional
            possible parent widget, by default None
        """
        super().__init__(model, parent)

        self.params: dict[Any, Any] = {}
        self.params["neurongroups"] = self.model.neurongroups
        self.params["monitors"] = self.model.monitors
        self.params["synapses"] = self.model.synapses
        self.params["parameters"] = self.model.parameters
        self.params["timestep"] = self.model.dt
        self.params["seed"] = self.model.seed

        self.button_layout = QtWidgets.QVBoxLayout()

        self.entry_layout = QtWidgets.QGridLayout()

        self.layout.itemAt(0).widget().hide()  # name label hiding
        self.name_entry.hide()
        self.name_button.hide()

        self.setWindowTitle("Model definition")
        self.show_parameters_button.hide()

        self.timestep, self.timestep_button = self.add_text_field(
            "Timestep",
            layout=self.entry_layout,
        )
        self.seed, self.seed_button = self.add_text_field(
            "Random seed",
            layout=self.entry_layout,
        )

        self.add_parameter_button = QtWidgets.QPushButton("Add parameter", self)
        self.add_neurongroup_button = QtWidgets.QPushButton("Add neurongroup", self)
        self.add_synapse_button = QtWidgets.QPushButton("Add synapse", self)
        self.monitor_button = QtWidgets.QPushButton("Add monitor", self)
        self.add_subgroup_button = QtWidgets.QPushButton("Add neuron subgroup", self)
        self.init_button = QtWidgets.QPushButton("Add initial conditions", self)

        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.add_parameter_button,
            1,
        )
        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.add_neurongroup_button,
            2,
        )
        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.add_subgroup_button,
            3,
        )
        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.add_synapse_button,
            4,
        )
        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.init_button,
            5,
        )
        self.button_layout.addStretch()

        self.button_layout.addWidget(
            self.monitor_button,
            6,
        )
        self.button_layout.addStretch()

        self.layout.addWidget(self.button_box, 7, 1, 1, 4, QtCore.Qt.AlignCenter)

        for qle in [
            self.name_entry,
            self.timestep,
            self.seed,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        # connect signals to slots on the outermost window
        self.add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        self.add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        self.add_parameter_button.clicked.connect(self.open_add_parameter_dialog)
        self.monitor_button.clicked.connect(self.open_monitor_dialog)
        self.add_subgroup_button.clicked.connect(self.open_subgroup_dialog)

        self.timestep_button.clicked.connect(self.accept_timestep)
        self.seed_button.clicked.connect(self.accept_seed)
        self.init_button.clicked.connect(self.open_initial_condition_dialog)

        # add 'next' button to open the next dialog in the model definition.
        self.nextButton = QtWidgets.QPushButton("Next")

        # Get a standard icon from the style
        style = QtWidgets.QApplication.style()
        icon = style.standardIcon(QtWidgets.QStyle.SP_ArrowRight)
        self.nextButton.setIcon(icon)

        self.button_box.addButton(self.nextButton, QtWidgets.QDialogButtonBox.ApplyRole)

        self.layout.addLayout(self.entry_layout, 1, 1)
        self.layout.addLayout(self.button_layout, 1, 2)
        self.setLayout(self.layout)

        self.display_parameters(self.params, blocking=False)

    def update_param_display(self, name: str, value: Any) -> None:
        """
        Update the parameter display in the main window

        Parameters
        ----------
        name : str
            name of the element to update
        value : Any
            value to update the element wiht
        """
        if isinstance(value, dict):
            startitem = self.dictwidget.find_item_by_name(name)

            if startitem is None:
                return

            if startitem.text(1) == str(None):
                startitem.setText(1, "")
            self.dictwidget.add_dict_to_tree(value, startitem)
        else:
            item = self.dictwidget.find_item_by_name(name)

            # update a leaf
            if item is not None:
                item.setText(1, str(value))

    def open_add_neurongroup_dialog(self) -> None:
        """
        Open a new window to define a neurongroup
        """
        self.neuron_dlg = NeuronGroupDialog(self.model, self)

        if self.neuron_dlg.exec():
            self.update_param_display("neurongroups", self.model.neurongroups)

    def open_add_synapse_dialog(self) -> None:
        """
        Open a new window to define a synapse
        """
        self.synapse_dlg = SynapseDialog(self.model, self)

        if self.synapse_dlg.exec():
            self.update_param_display("synapses", self.model.synapses)

    def open_add_parameter_dialog(self) -> None:
        """
        Open a new window to define a parameter for the model
        """
        self.param_dlg = ParameterDialog(self.model, self)

        if self.param_dlg.exec():
            self.update_param_display("parameters", self.model.parameters)

    def open_monitor_dialog(self) -> None:
        """
        Open a new window to define a new monitor
        """
        self.monitor_dlg = MonitorDialog(self.model, self)
        if self.monitor_dlg.exec():
            self.update_param_display("monitors", self.model.monitors)

    def open_subgroup_dialog(self) -> None:
        """
        Open a new window to define a neuron subgroup for the model
        """
        self.subgroupdialog = NeuronSubgroupDialog(self.model, self)
        if self.subgroupdialog.exec():
            self.update_param_display("neurongroups", self.model.neurongroups)

    def accept_timestep(self) -> None:
        """
        set the global timestep from text input
        """
        txtinput = self.timestep.text()

        dt = None
        try:
            dt = self._convert_string_to_quantity(txtinput)
        except Exception as e:
            self.display_error(f"Error when trying to convert timestep {txtinput}: {e}")

        if dt is not None:
            self.model.define_dt(dt)
            self.timestep_button.setText("saved")
            self.timestep_button.setDown(True)
            self.update_param_display("timestep", self.model.dt)

    def accept_seed(self) -> None:
        """
        set the global random seed
        """
        txtinput = self.seed.text()

        seed = None
        try:
            seed = int(txtinput)
        except Exception as e:
            self.display_error(f"Error when trying to convert seed {txtinput}: {e}")

        if seed is not None:
            self.model.seed = seed
            b2.seed(seed)
            self.seed_button.setText("saved")
            self.seed_button.setDown(True)
            self.update_param_display("seed", self.model.seed)

    def open_initial_condition_dialog(self) -> None:
        """
        open_initial_condition_dialog open a new dialog to add initial conditions
        """
        self.init_widget = InitialConditionWidget(self.model, self)

        if self.init_widget.exec():
            pass  # FIXME: not yet implemented: show values of variables
