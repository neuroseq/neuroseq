from __future__ import annotations

from ast import literal_eval
from functools import partial
from typing import Any
from typing import Callable

from brian2 import Quantity
from brian2 import units
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model


class ModelDialog(QtWidgets.QDialog):
    """
    Base class for all dialog windows in the B2ModelWidget.

    Parameters
    ----------
    QtWidgets : QtWidgets.QDialog
        General dialog widget from qt.
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
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

        self.name_title = QtWidgets.QLabel("Name", self)
        self.name_entry = QtWidgets.QLineEdit(self)
        self.name_button = QtWidgets.QPushButton("save", self)
        self.show_parameters_button = QtWidgets.QPushButton("Show parameters", self)

        self.qtbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(self.qtbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def _convert_string_to_quantity(
        self, input: str, valuetype: type = None
    ) -> str | Quantity:
        """
        Convert string input to brian2 quantities, comprised of value and unit.

        Parameters
        ----------
        input : str
            String to convert to brian2 quantity. Must be of the form `value * unit` or `value unit`.

        Returns
        -------
        str | Quantity
            Brian2 quantity instance corresponding to the input string
        """

        # quantity is nparray | scalar * unit

        v = input.split("*")

        if len(v) == 1:
            v = input.split(" ")

        if len(v) == 1:
            raise ValueError(
                "Qunatities must follow the pattern 'value * unit ' or 'value unit'"
            )

        # Split the string into value and unit
        value, unit = input.split()

        # Convert the value to float
        value = valuetype(literal_eval(value))

        # Get the unit from the units module
        unit = getattr(units, unit)

        # Return the quantity
        return value * unit

    def _convert_value_optional(
        self, input: str, valuetype: type = None
    ) -> str | Quantity:
        """
        _summary_

        Parameters
        ----------
        input : str
            _description_
        valuetype : type, optional
            _description_, by default None

        Returns
        -------
        str | Quantity
            _description_
        """
        try:
            return self._convert_string_to_quantity(input, valuetype=valuetype)
        except ValueError:
            return input
        except Exception as e:
            raise RuntimeError(
                f"An error occured during conversion of a string to a quantity {e}"
            ) from e

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
        self.error_dialog.exec_()

    def accept_parameter(
        self,
        element: QtWidgets.QLineEdit,
        paramname: str,
        model_callback: Callable,
        convert: Callable = lambda x: x,
    ) -> None:
        """
        Accept a parameter into the model system.

        Parameters
        ----------
        element : QtWidgets.QLineEdit
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
            line = element.text()
        except AttributeError:
            line = element.toPlainText()
        except Exception as e:
            self.display_error(f"Error when getting parameter {paramname}: {e}")

        element.clear()

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
        self.accept_parameter(element, name, model_callback, convert=converter)
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
            self.name = name
            getattr(self.model, target)[name] = {}
            self.name_button.setText("saved")
            self.name_button.setDown(True)

    def display_parameters(
        self, target: str | dict[Any, Any], read_only: bool = False
    ) -> None:
        """
        Display the parameters of a target member of the model in a new window.

        Parameters
        ----------
        target : str
            name of the target variable in the model
        """
        if isinstance(target, str):
            d = getattr(self.model, target).get(self.name, None)
        else:
            d = target

        window = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout(window)
        self.dictwidget = DictionaryWidget(d, window, read_only=read_only)
        layout.addWidget(self.dictwidget)
        self.dictwidget.setLayout(layout)
        window.resize(800, 600)
        window.exec_()

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

    def __init__(self, model: B2Model, parent: QtWidgets.QDialog | None = None) -> None:
        """
        Create a new NeuronGroupDialog instance which is displayed in a new window. Creates a table of
        text fiedls to fill in and buttons to save the data that has been provided into the respective variable of the model.

        Parameters
        ----------
        model : B2Model
            The model instance to set up using this dialog
        parent : QtWidgets.QDialog | None, optional
            possible parent window of this instance, by default None
        """
        super().__init__(model, parent)
        self.setWindowTitle("Define a Neurongroup")

        self.equation_title = QtWidgets.QLabel("Equation", self)
        self.equation = QtWidgets.QTextEdit(self)
        self.equation_button = QtWidgets.QPushButton("save", self)

        self.threshold_title = QtWidgets.QLabel("Threshold", self)
        self.threshold = QtWidgets.QLineEdit(self)
        self.threshold_button = QtWidgets.QPushButton("save", self)

        self.reset_title = QtWidgets.QLabel("Reset", self)
        self.reset = QtWidgets.QLineEdit(self)
        self.reset_button = QtWidgets.QPushButton("save", self)

        self.refractory_title = QtWidgets.QLabel("Refractory", self)
        self.refractory = QtWidgets.QLineEdit(self)
        self.refractory_button = QtWidgets.QPushButton("save", self)

        self.solver = QtWidgets.QLineEdit(self)
        self.solver_button = QtWidgets.QPushButton("save", self)
        self.solver_title = QtWidgets.QLabel("Solver", self)

        self.neuronnumber = QtWidgets.QLineEdit(self)
        self.neuronnumber_button = QtWidgets.QPushButton("save", self)
        self.neuronnumber_title = QtWidgets.QLabel("Neuronnumber", self)

        for qle in [
            self.equation,
            self.threshold,
            self.reset,
            self.refractory,
            self.solver,
            self.neuronnumber,
        ]:
            qle.setMinimumWidth(qle.sizeHint().width() * 2)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(
            self.name_entry,
            1,
            2,
        )
        self.layout.addWidget(self.name_button, 1, 3)

        self.layout.addWidget(self.equation_title, 2, 1)
        self.layout.addWidget(self.equation, 2, 2)
        self.layout.addWidget(self.equation_button, 2, 3)

        self.layout.addWidget(self.threshold_title, 3, 1)
        self.layout.addWidget(self.threshold, 3, 2)
        self.layout.addWidget(self.threshold_button, 3, 3)

        self.layout.addWidget(self.reset_title, 4, 1)
        self.layout.addWidget(self.reset, 4, 2)
        self.layout.addWidget(self.reset_button, 4, 3)

        self.layout.addWidget(self.refractory_title, 5, 1)
        self.layout.addWidget(self.refractory, 5, 2)
        self.layout.addWidget(self.refractory_button, 5, 3)

        self.layout.addWidget(self.solver_title, 6, 1)
        self.layout.addWidget(self.solver, 6, 2)
        self.layout.addWidget(self.solver_button, 6, 3)

        self.layout.addWidget(self.neuronnumber_title, 7, 1)
        self.layout.addWidget(self.neuronnumber, 7, 2)
        self.layout.addWidget(self.neuronnumber_button, 7, 3)

        self.layout.addWidget(
            self.show_parameters_button, 8, 1, 1, 3, QtCore.Qt.AlignCenter
        )

        self.layout.addWidget(self.buttonBox, 9, 1, 1, 4, QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

        # connect signals and sockets:

        self.show_parameters_button.clicked.connect(
            partial(self.display_parameters, "neurongroups")
        )

        self.threshold_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.threshold,
                "threshold",
                self.model.define_threshold,
                self.threshold_button,
                lambda v: v,
            )
        )

        self.reset_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.reset,
                "reset",
                self.model.define_reset,
                self.reset_button,
                lambda v: v,
            )
        )

        self.refractory_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.refractory,
                "refractory",
                self.model.define_refractory,
                self.refractory_button,
                self._convert_value_optional,
            )
        )

        self.solver_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.solver,
                "solver",
                self.model.define_solver_method,
                self.solver_button,
                lambda v: v,
            )
        )

        self.neuronnumber_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.neuronnumber,
                "neuronnumber",
                self.model.define_neuronnumber,
                self.neuronnumber_button,
                lambda v: int(v),
            )
        )

        self.name_button.clicked.connect(
            partial(self.accept_name, target="neurongroups")
        )

        self.equation_button.clicked.connect(
            partial(
                self.accept_parameter_with_savebutton,
                self.equation,
                "model",
                self.model.define_equation,
                self.equation_button,
            )
        )


class SynapseDialog(ModelDialog):
    """
    Dialog window for defining a Synapse for a b2model.

    Parameters
    ----------
    ModelDialog : ModelDialog
        The base class of all dialogs in the b2_model_widget namespace.
    """

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

        self.origin_title = QtWidgets.QLabel("Origin", self)
        self.origin = QtWidgets.QLineEdit(self)
        self.origin_button = QtWidgets.QPushButton("save", self)

        self.target_title = QtWidgets.QLabel("Target", self)
        self.target = QtWidgets.QLineEdit(self)
        self.target_button = QtWidgets.QPushButton("save", self)

        self.equation_title = QtWidgets.QLabel("Equation", self)
        self.equation = QtWidgets.QTextEdit(self)
        self.equation_button = QtWidgets.QPushButton("save", self)

        self.pre_title = QtWidgets.QLabel("Pre", self)
        self.pre = QtWidgets.QTextEdit(self)
        self.pre_button = QtWidgets.QPushButton("save", self)

        self.post_title = QtWidgets.QLabel("Post", self)
        self.post = QtWidgets.QTextEdit(self)
        self.post_button = QtWidgets.QPushButton("save", self)

        self.connect_title = QtWidgets.QLabel("Connection def.", self)
        self.connect = QtWidgets.QLineEdit(self)
        self.connect_button = QtWidgets.QPushButton("save", self)

        self.connect_prob_title = QtWidgets.QLabel("Connection prob.", self)
        self.connect_prob = QtWidgets.QLineEdit(self)
        self.connect_prob_button = QtWidgets.QPushButton("save", self)

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

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.name_entry, 1, 2)
        self.layout.addWidget(self.name_button, 1, 3)

        self.layout.addWidget(self.origin_title, 2, 1)
        self.layout.addWidget(self.origin, 2, 2)
        self.layout.addWidget(self.origin_button, 2, 3)

        self.layout.addWidget(self.target_title, 3, 1)
        self.layout.addWidget(self.target, 3, 2)
        self.layout.addWidget(self.target_button, 3, 3)

        self.layout.addWidget(self.equation_title, 4, 1)
        self.layout.addWidget(self.equation, 4, 2)
        self.layout.addWidget(self.equation_button, 4, 3)

        self.layout.addWidget(self.pre_title, 5, 1)
        self.layout.addWidget(self.pre, 5, 2)
        self.layout.addWidget(self.pre_button, 5, 3)

        self.layout.addWidget(self.post_title, 6, 1)
        self.layout.addWidget(self.post, 6, 2)
        self.layout.addWidget(self.post_button, 6, 3)

        self.layout.addWidget(self.connect_title, 7, 1)
        self.layout.addWidget(self.connect, 7, 2)
        self.layout.addWidget(self.connect_button, 7, 3)

        self.layout.addWidget(self.connect_prob_title, 8, 1)
        self.layout.addWidget(self.connect_prob, 8, 2)
        self.layout.addWidget(self.connect_prob_button, 8, 3)

        self.layout.addWidget(
            self.show_parameters_button, 9, 1, 1, 3, QtCore.Qt.AlignCenter
        )

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
        self.show_parameters_button.clicked.connect(
            partial(self.display_parameters, "synapses")
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
        """
        Overridden method to accept the value of a parameter and save it in the model.
        """
        if self.name is None:
            self.display_error("Please define a parameter first by giving it a name!")
        elif self.value_entry.toPlainText() == "":
            self.display_error("Please assign a value or expression before saving!")
        else:
            self.accept_parameter(
                self.value_entry, self.name, self.model.define_parameter
            )
            self.value_button.setText("saved")

    def display_parameters(self, _: str | dict[Any, Any]) -> None:
        """
        Display all the parameters in the system in a new window.

        Parameters
        ----------
        _ : str
            dummy for override
        """
        d = self.model.parameters if self.model.parameters is not None else {}

        super().display_parameters(d)


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
            self.table.setColumnWidth(0, 400)
            self.table.setColumnWidth(1, 400)
            if read_only:
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
        """
        Recursive function to add possibly nested dictionaries to the tree view.

        Parameters
        ----------
        widget : QtWidgets.QTreeWidget
            TreeWidget to add the dict to.
        dictionary : dict
            Dictionary to add
        parent : QtWidgets.QDialog | None, optional
            Parent element in the tree view, by default None
        """
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
        parent: QtWidgets.QDialog | None = None,
        model_register: dict | None = None,
    ) -> None:
        """
        Create a new instance of the B2ModelWidget and display it in a new window.

        Parameters
        ----------
        model : B2Model
            Model instance to fill with data
        parent : QtWidgets.QDialog | None, optional
            possible parent widget, by default None
        model_register : dict | None, optional
            dictionary to store models in, by default None
        """
        super().__init__(model, parent)

        self.model_register = model_register

        self.setWindowTitle("Model definition")
        self.show_parameters_button.hide()
        self.timestep_title = QtWidgets.QLabel("Timestep", self)
        self.timestep = QtWidgets.QLineEdit()
        self.timestep_button = QtWidgets.QPushButton("save", self)

        self.seed = QtWidgets.QLineEdit(self)
        self.seed_button = QtWidgets.QPushButton("save", self)
        self.seed_title = QtWidgets.QLabel("Random seed")

        self.add_parameter_button = QtWidgets.QPushButton("Add parameter", self)
        self.add_neurongroup_button = QtWidgets.QPushButton("Add neurongroup", self)
        self.add_synapse_button = QtWidgets.QPushButton("Add synapse", self)
        self.display_system_button = QtWidgets.QPushButton("Display system", self)

        self.layout.addWidget(self.name_title, 1, 1)
        self.layout.addWidget(self.name_entry, 1, 2, 1, 2)
        self.layout.addWidget(self.name_button, 1, 4)

        self.layout.addWidget(self.timestep_title, 2, 1)
        self.layout.addWidget(self.timestep, 2, 2, 1, 2)
        self.layout.addWidget(self.timestep_button, 2, 4)

        self.layout.addWidget(self.seed_title, 3, 1)
        self.layout.addWidget(self.seed, 3, 2, 1, 2)
        self.layout.addWidget(self.seed_button, 3, 4)

        self.layout.addWidget(self.add_parameter_button, 5, 1)

        self.layout.addWidget(
            self.add_neurongroup_button,
            5,
            2,
        )

        self.layout.addWidget(
            self.add_synapse_button,
            5,
            3,
        )

        self.layout.addWidget(self.display_system_button, 5, 4)

        self.layout.addWidget(self.buttonBox, 8, 1, 1, 4, QtCore.Qt.AlignCenter)

        # connect signals to slots on the outermost

        self.add_neurongroup_button.clicked.connect(self.open_add_neurongroup_dialog)
        self.add_synapse_button.clicked.connect(self.open_add_synapse_dialog)
        self.add_parameter_button.clicked.connect(self.open_add_parameter_dialog)
        self.display_system_button.clicked.connect(self.display_system)

        self.timestep_button.clicked.connect(self.not_connected)
        self.seed_button.clicked.connect(self.not_connected)
        self.name_button.clicked.connect(self.not_connected)

        self.setLayout(self.layout)

    def display_system(self) -> None:
        """
        Display the defined system in a tree view in total.
        Useful for getting an overview over the defined variables
        or for checking which ones are still missing.
        """
        d: dict[Any, Any] = {}
        d["neurongroups"] = self.model.neurongroups
        d["synapses"] = self.model.synapses
        d["parameters"] = self.model.parameters
        d["timestep"] = self.model.dt
        d["seed"] = self.model.seed
        d["name"] = self.name

        super().display_parameters(d, read_only=False)

    def open_add_neurongroup_dialog(self) -> None:
        """
        Open a new window to define a neurongroup
        """
        self.neuron_dlg = NeuronGroupDialog(self.model, self)

        self.neuron_dlg.exec_()

    def open_add_synapse_dialog(self) -> None:
        """
        Open a new window to define a synapse
        """
        self.synapse_dlg = SynapseDialog(self.model, self)

        self.synapse_dlg.exec_()

    def open_add_parameter_dialog(self) -> None:
        """
        Open a new window to define a parameter for the model
        """
        self.param_dlg = ParameterDialog(self.model, self)

        self.param_dlg.exec_()
