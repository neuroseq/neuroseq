from __future__ import annotations

import logging
from typing import Any

import brian2 as b2  # noqa: F401
from brian2 import TimedArray
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree import ParameterTree
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import MonitorDialog
from formozov_project.b2_model_widget import NeuronGroupDialog
from formozov_project.b2_model_widget import NeuronSubgroupDialog
from formozov_project.b2_model_widget import ParameterDialog
from formozov_project.b2_model_widget import SynapseDialog
from formozov_project.reconstructor import Reconstructor


class EditableParameterTree(ParameterTree):
    """
    ParameterTree Subclass with additional context menu entries

    Parameters
    ----------
    ParameterTree :
        pyqtgraph ParameterTree class for pretty-displaying hierarchical parameter trees
    """

    def __init__(
        self,
        model: B2Model,
        item_adder_dialog: type,
        name: str,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        """
        Create a new paramter tree displaying a part of the definition of a B2Model.
        Editing the parameter tree edits the model class. When something is edited, the model will be rebuild

        Parameters
        ----------
        model : B2Model
            model to display a part of
        item_adder_dialog : Dialog to open when right-clicking into empty space.

        name : str
            Name of the displayed parameter tree
        parent : QtWidgets.QWidget | None, optional
            Parent widget, by default None
        showHeader : bool, optional
            Whether the header node shall be shown or not, by default True
        """
        super().__init__(parent=parent, showHeader=showHeader)

        self.model = model
        self.dialogtype = item_adder_dialog
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.states = getattr(self.model, self.name)
        params = self._convert_dict_to_params(self.states)
        self.root = Parameter.create(name=name, type="group", children=params)

        self.setParameters(self.root, showTop=False)

        self._connect_parameters(self.root)

    def _connect_parameters(self, param: Parameter) -> None:
        """
        connect the signals emitted by the entries of the parameter tree to their respective actions.
        Whenever something changes (add/delete child node, change value) the model is set to be rebuilt
        when the dialgo is accepted, and changed parameters are updated.

        Parameters
        ----------
        param : Parameter
            parent paremeter
        """
        if param.type() == "group":
            param.sigChildAdded.connect(self._set_rebuild)
            param.sigChildRemoved.connect(self._set_rebuild)
            for child in param.children():
                self._connect_parameters(child)
        elif param.type() == "text":
            # do something that limits the updating of the text in the field. For editing equations, this is important
            param.sigValueChanged.connect(self._set_textfield)
        else:
            param.sigValueChanged.connect(self._set_param)

    def _trace_path_from(self, param: Parameter) -> list[str]:
        """
        Trace the path in the parametertree from the given node 'param' to the root and record it in reverse order
        to have a traversible path for getting out the dictionary entry to be updated.

        Parameters
        ----------
        param : Parameter
            parameter to trace the path for

        Returns
        -------
        list[str]
            list of keys from the root to the given parameter. Gives a path to be traversed to get the correct element
            in the model dict to updated
        """
        path = []
        current_param = param
        while current_param != self.root:
            path.append(current_param.name())
            current_param = current_param.parent()
        return list(reversed(path))

    def _set_rebuild(self, *_: Any) -> None:
        """
        Sets the `must_be_rebuilt` flag of the model to true
        """
        self.model.must_be_rebuilt = True

    def _set_textfield(self, param: Parameter, value: str) -> None:
        """
        Called when a textfield is edited in the parametertree, typically when equations are edited.
        Updates the respective element in the model with the edited text

        Parameters
        ----------
        param : Parameter
            parameter the value of which is edited
        value : str
            new value
        """
        path = self._trace_path_from(param)
        object_to_modify = self.states
        for p in path[0:-1]:
            object_to_modify = object_to_modify[p]
        object_to_modify[param.name()] = value

    def _set_param(self, param: Parameter, value: Any) -> None:
        """
        Update a model parameter whenever its corresponding representation in the parameter tree is updated.

        Parameters
        ----------
        param : Parameter
            parameter that has been updated
        value : Any
            updated value
        """
        self.logger.info(f"setting parameter {param.name()} to value {value}")
        path = self._trace_path_from(param)

        object_to_modify = self.states
        for p in path[0:-1]:
            object_to_modify = object_to_modify[p]

        try:
            object_to_modify[param.name()] = eval(value)
        except SyntaxError:
            vals = value.split(" ")
            object_to_modify[param.name()] = eval(vals[0]) * getattr(b2, vals[1])
        except Exception as e:
            self.logger.critical(
                f"Could not set parameter {param.name()} with value {value}: {e}, trying to set as string"
            )
            object_to_modify[param.name()] = value
        self.model.must_be_rebuilt = True

    def _get_value(self, v: Any) -> dict[str, Any] | Any:
        """
        Modifier function that represents brian2 TimedArrays as dicts with entries [values, dt] and returns
        everything else as is.

        Parameters
        ----------
        v : Any
            Input value to be represented in the parameter tree

        Returns
        -------
        dict[str, Any] | Any
            Dict when a TimedArray is to be represented, unchanged input otherwise
        """
        if isinstance(v, TimedArray):
            return {"values": v.values, "dt": v.dt}
        else:
            return v

    def _convert_dict_to_params(self, data: dict[Any, Any], key: Any = "") -> list[Any]:
        """
        Recursively go through a nested dict and create a hierarchical pyqtgraph Parameter representation of it
        from which a ParameterTree object can be constructed.

        Parameters
        ----------
        data : dict[Any, Any]
            Possibly nested dict to be represented in the parametertree
        key : Any, optional
            Current key of 'data' that is processed. Needed for keeping track of parameter names, by default ""

        Returns
        -------
        list[Any]
            list of Parameter representations that internally define all the parent-child relationships to create the parametertree.
        """
        params = []
        for key, value in data.items():
            param_type = "str"  # Default type
            value = self._get_value(value)
            if isinstance(value, dict):
                child_params = self._convert_dict_to_params(value, key)
                params.append(
                    {
                        "name": key,
                        "type": "group",
                        "children": child_params,
                        "expanded": False,
                        "removable": True,
                        "renamable": False,
                        "syncExpanded": True,
                    }
                )
            else:
                # Determine the parameter type based on value type
                if isinstance(value, int):
                    param_type = "int"
                elif isinstance(value, float):
                    param_type = "float"
                elif isinstance(value, bool):
                    param_type = "bool"
                elif isinstance(value, str):
                    param_type = "text" if "\n" in value else "str"
                # TODO: add quantity parameter type

                param = {
                    "name": key,
                    "type": param_type,
                    "value": value,
                    "removable": True,
                    "renamable": False,
                    "syncExpanded": True,
                }
                params.append(param)

        return params

    def contextMenuEvent(self, ev: QtCore.QEvent) -> None:
        """
        Modified context menu to allow for adding elements through custom dialogs when a right click in a free area occurs

        Parameters
        ----------
        ev : QtCore.QEvent
            Click event to be processed. Will open the standard context menu when an item is hit, and opens a custom dialog
            given in the callers constructor otherwise.
        """
        if self.itemAt(ev.pos()):
            super().contextMenuEvent(ev)
        else:
            menu = QtWidgets.QMenu(self)
            action_menu_entry = menu.addAction(f"add {self.name}")
            action = menu.exec(self.mapToGlobal(ev.pos()))

            if action == action_menu_entry:
                self.addItem()

    def updateParams(self, new_name: str, new_data: dict[Any, Any]) -> None:
        """
        Update the parameters in the parametertree when a new item is added
        tot he model through the custom dialog.

        Parameters
        ----------
        new_name : str
            Name of the element to be added
        new_data : dict[Any, Any]
            Data dict to be added
        """
        param = self._convert_dict_to_params(
            {
                new_name: new_data,
            }
        )
        self.root.addChildren(param)

    def addItem(self) -> None:
        """
        Call the custom dialog given in the callers constructor to add custom elements and update
        the parametertree accordingly
        """
        # check which name has changed, add this name to the internal dict
        # and to the parameters that are displayed
        old_names = set(getattr(self.model, self.name).keys())
        dialog = self.dialogtype(self.model)

        # update the internal states by
        if dialog.exec():
            new_names = set(getattr(self.model, self.name).keys())
            new_name = list(new_names - old_names)[0]
            self.updateParams(new_name, getattr(self.model, self.name)[new_name])


class EditableNeurongroupTree(EditableParameterTree):
    """
    Subclass for representing neurongroups and neuronsubgroups in the gui using
    a pyqtgraph ParameterTree

    Parameters
    ----------
    EditableParameterTree : EditableParameterTree
    """

    def __init__(
        self,
        model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        """
        Create a new EditableNeurongroupTree element in the gui

        Parameters
        ----------
        model : B2Model
            Model to represent the neurongroups of
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        showHeader : bool, optional
            Show the root node title or not, by default True
        """
        super().__init__(
            model=model,
            item_adder_dialog=NeuronGroupDialog,
            name="neurongroups",
            parent=parent,
            showHeader=showHeader,
        )

    def contextMenuEvent(self, ev: QtCore.Event) -> None:
        """
        Modify the standard context menu opened upon right click to allow for
        the addition of neurongroups and neuronsubgroups when a right click occurs in an empty region

        Parameters
        ----------
        ev : QtCore.Event Click event to process
        """
        if self.itemAt(ev.pos()):
            super().contextMenuEvent(ev)
        else:
            menu = QtWidgets.QMenu(self)

            ng_menu_entry = menu.addAction("add Neurongroup")

            nsg_menu_entry = menu.addAction("add Neuron subgroup")

            action = menu.exec(self.mapToGlobal(ev.pos()))

            old_names = set(getattr(self.model, self.name).keys())

            dialog = None

            if action == ng_menu_entry:
                dialog = NeuronGroupDialog(
                    self.model,
                )

            elif action == nsg_menu_entry:
                dialog = NeuronSubgroupDialog(self.model)

            if dialog is not None and dialog.exec():
                new_names = set(getattr(self.model, self.name).keys())
                new_name = list(new_names - old_names)[0]

                # update the internal states by
                self.updateParams(new_name, getattr(self.model, self.name)[new_name])


class ModelEditorDialog(QtWidgets.QDialog):
    """
    Editor dialog for editing the properties of a model using multiple pyqtgraph ParameterTrees.

    Parameters
    ----------
    QtWidgets : QtWidgets.QDialog
        Basic Qt Dialog class
    """

    def __init__(self, model: B2Model, parent: QtWidgets.QWidget | None = None):
        """
        Create a new modeleditor instance and tie it to a B2Model instance.

        Parameters
        ----------
        model : B2Model
            Model to edit through the gui
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        """
        super().__init__(parent=parent)

        trees = {
            "Neurongroups": EditableNeurongroupTree(model, parent=self),
            "Synapses": EditableParameterTree(
                model,
                SynapseDialog,
                name="synapses",
                parent=self,
            ),
            "Monitors": EditableParameterTree(
                model,
                MonitorDialog,
                name="monitors",
                parent=self,
            ),
            "Parameters": EditableParameterTree(
                model,
                ParameterDialog,
                name="parameters",
                parent=self,
            ),
        }
        self.model = model
        self.model_has_been_edited = False
        self.logger = logging.getLogger(__name__)

        layout = QtWidgets.QGridLayout()

        btns = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box = QtWidgets.QDialogButtonBox(btns)

        i = 0
        for k, t in trees.items():
            nameplate = QtWidgets.QLabel(str(k), parent=self)
            bold_font = QtGui.QFont()
            bold_font.setBold(True)
            nameplate.setFont(bold_font)
            layout.addWidget(nameplate, 0, i)
            layout.addWidget(t, 1, i)
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.VLine)
            separator.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(separator, 1, i + 1)
            i += 2

        layout.addWidget(button_box, layout.rowCount(), 1, 1, 4, QtCore.Qt.AlignCenter)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.setLayout(layout)

    def accept(self):
        """
        Accept the current state of the model and rebuilt it if anything has been edited.
        """
        if self.model.must_be_rebuilt:
            self.logger.info("rebuilding model after change")
            states = self.model.to_dict()

            if self.model.dt is not None:
                states["timestep"] = self.model.dt

            constr = Reconstructor(states)

            # FIXME: this can be a problem when the system shall be continued because
            # the sequence of random number changes due to brian2 caching them.
            if self.model.seed != states["seed"]:
                QtWidgets.QMessageBox.warning(
                    "The system's random number generator will be reseeded. This potentially can change the results when compared to previous ones"
                )
                new_model = B2Model(seed=states["seed"])

            try:
                constr.reconstruct(new_model)
            except Exception as e:
                message = f"Error when rebuilding the model after editing: {e}"
                self.logger.critical(message)
                QtWidgets.QMessageBox.critical(message)
            self.model = new_model

        super().accept()
