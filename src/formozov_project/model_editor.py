from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import brian2 as b2  # noqa: F401
from brian2 import TimedArray
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree import registerParameterItemType
from pyqtgraph.parametertree import registerParameterType
from pyqtgraph.parametertree.parameterTypes import ListParameter
from pyqtgraph.parametertree.parameterTypes import ListParameterItem
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import QEvent
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox

from formozov_project.b2_model import B2Model
from formozov_project.b2_model_widget import MonitorDialog
from formozov_project.b2_model_widget import NeuronGroupDialog
from formozov_project.b2_model_widget import NeuronSubgroupDialog
from formozov_project.b2_model_widget import ParameterDialog
from formozov_project.b2_model_widget import SynapseDialog
from formozov_project.reconstructor import Reconstructor


# derive a new combobox which has a 'clicked' signal
class EagerQComboBox(QComboBox):
    """
    Subclass of QtWidgets.QComboBox that emits a signal when it is clicked.
    Parameters
    ----------
    QComboBox :
        Qt's basic dropdown list widget
    """

    clicked = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        """
        Create a new EagerQCombobox instance which emits 'clicked' signals whenever the 'mousePressEvent' occurs on the widget.

        Parameters
        ----------
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        """
        super().__init__(parent)

    # find a way to emit the 'clicked' signal here
    def mousePressEvent(self, event: QEvent) -> None:
        """
        Override fo the mousePressEvent event handler of QComboBox that emits the 'clicked' signal.

        Parameters
        ----------
        event : QEvent
            Incoming event
        """
        self.clicked.emit()
        super().mousePressEvent(event)


class UpdatingListParameterItem(ListParameterItem):
    """
    Represent an UpdatingListParameter in such a way that it is immediatelly updated when a parameter in the element to be updated changes

    Parameters
    ----------
    ListParameterItem : Non-autoupdating baseclass from pyqtgraph
    """

    def __init__(self, param: Parameter, depth: Any, update: Callable):
        """
        Create a new UpdatingListParameterItem instance that updates the displayed parameter list automatically via the passed 'update' callback function

        Parameters
        ----------
        param : Parameter
            Parameter to display
        depth : Any
            depth in the hiearchy
        update : Callable
            Function that returns a list of strings to be used as options
        """
        self.update = update
        super().__init__(param, depth)

    def makeWidget(self) -> EagerQComboBox:
        """
        Override of ListParameterItem.makeWidget that uses the EagerQComboBox to allow for instant updating of the
        available list of options when the widget is clicked.

        Returns
        -------
        EagerQComboBox
            The updated combobox widget
        """
        # this is a copy of the base class function, but with the EagerComboBox
        # instead of the normal QComboBox
        w = EagerQComboBox()
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit
        w.sigChanged = w.currentIndexChanged
        w.value = self.value
        w.setValue = self.setValue
        self.widget = w  ## needs to be set before limits are changed

        # we also need to reconnect the signal
        self.widget.clicked.connect(self._update_list)

        self.limitsChanged(self.param, self.update())
        if len(self.forward) > 0 and self.param.hasValue():
            self.setValue(self.param.value())
        return w

    def _update_list(self) -> None:
        self.limitsChanged(self.param, self.update())


class UpdatingListParameter(ListParameter):
    """
    Subclass fo pyqtgraph.ListParameter to use with UpdatingListParameterItem to have an automatically updating parameter display

    Parameters
    ----------
    ListParameter :
        non-autoupdating Baseclass from pyqtgraph
    """

    itemClass = UpdatingListParameterItem

    def __init__(self, update: Callable | None = None, **opts):
        """
        Create a new UpdatingListParameter instance. This extends the passed keyword arguments with the 'update' kwarg,
        which expects a callabel that returns an updated list of parameters and is passed to the parameteritem representation

        Parameters
        ----------
        update : Callable | None, optional
            update callback, by default None
        """
        if update is None:
            self.update = lambda: self.param.value()
        else:
            self.update = update
        super().__init__(**opts)

    def makeTreeItem(self, depth: Any) -> UpdatingListParameterItem:
        """
        Override of the ListParameter.makeTreeItem to use `UpdatingListParameterItem`
        instead of the normal ListParameterItem function.

        Parameters
        ----------
        depth : Any
            depth in the parametertree

        Returns
        -------
        UpdatingListParameterItem
            New gui representation of the calling parameter
        """
        itemClass = self.itemClass
        paramitem = itemClass(self, depth, self.update)

        return paramitem


# register the new types with the parametertree implementation
registerParameterType("updatinglist", UpdatingListParameter)
registerParameterItemType(
    "updatinglistitem", UpdatingListParameterItem, UpdatingListParameter
)


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
        b2model: B2Model,
        item_adder_dialog: type,
        domain_name: str,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
        list_items: dict[str, Any] | None = None,
    ):
        """
        Create a new paramter tree displaying a part of the definition of a B2Model.
        Editing the parameter tree edits the b2model class. When something is edited, the b2model will be rebuild

        Parameters
        ----------
        b2model : B2Model
            b2model to display a part of
        item_adder_dialog : Dialog to open when right-clicking into empty space.

        domain_name : str
            Name of the domain of the b2model to be displayed. One of synapses, parameter, or monitors. Neurongroups are treated in their own subclass
        parent : QtWidgets.QWidget | None, optional
            Parent widget, by default None
        showHeader : bool, optional
            Whether the header node shall be shown or not, by default True
        """
        super().__init__(parent=parent, showHeader=showHeader)

        if list_items is None:
            list_items = {}

        self.b2model = b2model
        self.dialogtype = item_adder_dialog
        self.logger = logging.getLogger(__name__)
        self.name = domain_name
        self.states = getattr(self.b2model, self.name)
        self.functions = self.b2model.functions[self.name]

        self.list_items = list_items

        params = self._convert_dict_to_params(self.states, self.functions)
        self.root = Parameter.create(name=domain_name, type="group", children=params)

        self.setParameters(self.root, showTop=False)

        self._connect_parameters(self.root, self.functions)

        self.add_item_action = QtWidgets.QAction(f"add {self.name}")
        self.add_item_action.triggered.connect(self.add_item)

    def _connect_parameters(self, param: Parameter, funcs: dict[Any, Any]) -> None:
        """
        connect the signals emitted by the entries of the parameter tree to their respective actions.
        Whenever something changes (add/delete child node, change value) the model is set to be rebuilt
        when the dialog is accepted, and changed parameters are updated.

        Parameters
        ----------
        param : Parameter
            parent paremeter
        """
        if param.type() == "group":
            # FIXME: not sure what to do here.
            param.sigChildAdded.connect(lambda: self.b2model.set_rebuilt(True))
            param.sigAddNew.connect(lambda: self.b2model.set_rebuilt(True))
            param.sigChildRemoved.connect(lambda: self.b2model.set_rebuilt(True))
            # param.sigNameChanged.connect(funcs["rename"])
            for child in param.children():
                if child.name in funcs:
                    self._connect_parameters(child, funcs[child.name()])
                else:
                    pass
        else:
            if param.name() == "name":
                pass
            else:
                param.sigValueChanged.connect(funcs["modify"])
                param.sigRemoved.connect(funcs["delete"])
            # param.sigNameChanged.connect(funcs["rename"])

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

    def _convert_dict_to_params(
        self, data: dict[Any, Any], functions: dict[Any, Any], key: Any = ""
    ) -> list[Any]:
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
                child_params = self._convert_dict_to_params(
                    value,
                    functions[key],
                    key,
                )

                # FIXME: this thing needs a + button for some things with an activated list to choose from for some of those in turn
                params.append(
                    {
                        "name": key,
                        "type": "group",
                        "children": child_params,
                        "expanded": False,
                        "removable": True,
                        "renamable": False,
                        "syncExpanded": True,
                        "functions": functions,
                    }
                )
            else:
                # Determine the parameter type based on value type
                if isinstance(
                    value, int | float
                ):  # make numbers to float to be consistent with python conversions
                    param_type = "float"
                elif isinstance(value, bool):
                    param_type = "bool"
                elif isinstance(value, str):
                    param_type = "text" if "\n" in value else "str"
                elif value is None:
                    param_type = "str"
                    value = str(value)
                param = {
                    "name": key,
                    "type": param_type,
                    "value": value,
                    "removable": True,
                    "renamable": False,
                    "syncExpanded": True,
                    "functions": functions,
                }

                # need 'limits' for the underlying ListParameter and 'element' for the custom class
                if key in list(self.list_items.keys()):
                    param["type"] = "updatinglist"
                    param["limits"] = self.list_items[key]["update_from"]()
                    param["update"] = self.list_items[key]["update_from"]

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
            menu.addAction(self.add_item_action)
            menu.exec(self.mapToGlobal(ev.pos()))

    def update_params(self, new_name: str, new_data: dict[Any, Any]) -> None:
        """
        Update the parameters in the parametertree when a new item is added
        to the model through the custom dialog.

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
            },
            self.functions,
        )
        self.root.addChildren(param)

    def add_item(self) -> None:
        """
        Call the custom dialog given in the callers constructor to add custom elements and update
        the parametertree accordingly
        """

        # check which name has changed, add this name to the internal dict
        # and to the parameters that are displayed

        old_names = set(getattr(self.b2model, self.name).keys())
        self.dialog = self.dialogtype(self.b2model)

        # update the internal states by
        if self.dialog.exec():
            new_names = set(getattr(self.b2model, self.name).keys())

            new_name = list(new_names - old_names)[0]
            param = self._convert_dict_to_params(
                {
                    new_name: getattr(self.b2model, self.name)[new_name],
                },
                self.functions,
            )
            self.root.addChildren(param)


class EditableSynapseTree(EditableParameterTree):
    """
    Subclass for displaying an auto-updating tree of synapse definitions
    """

    def __init__(
        self,
        b2model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        """
        Create a new EditableSynapseTree to represent the synapse definitions of a b2 model

        Parameters
        ----------
        b2model : B2Model
            model to represent the synapses for
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        showHeader : bool, optional
            Whether to show header name, by default True
        """
        super().__init__(
            b2model,
            parent=parent,
            item_adder_dialog=SynapseDialog,
            showHeader=showHeader,
            domain_name="synapses",
            list_items={
                "method": {
                    "update_from": lambda: [
                        "euler",
                        "exponential_euler",
                        "exact",
                    ],  # methods are static, so we can give a list
                },
                "origin": {
                    "update_from": lambda: list(self.b2model.neurongroups.keys()),
                },
                "target": {
                    "update_from": lambda: list(self.b2model.neurongroups.keys()),
                },
            },
        )


class EditableMonitorTree(EditableParameterTree):
    """
    Subclass for displaying an autoupdating tree of monitor definitions.
    """

    def __init__(
        self,
        b2model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        """
        Create a new EditableMonitorTree instance

        Parameters
        ----------
        b2model : B2Model
            model for which the monitor definitions should be represented
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        showHeader : bool, optional
            Whether the header should be defined here, by default True
        """
        super().__init__(
            b2model,
            parent=parent,
            item_adder_dialog=MonitorDialog,
            showHeader=showHeader,
            domain_name="monitors",
            list_items={
                "type": {
                    "update_from": lambda: [
                        b2.StateMonitor.__name__,
                        b2.SpikeMonitor.__name__,
                    ],
                },
                "source": {
                    "update_from": self._update_object_list,
                },
                "variable": {
                    "update_from": self._update_element_list,
                },
            },
        )

    def _update_object_list(self) -> list[str]:
        """
        update the currently available list of objects to monitor

        Returns
        -------
        list[str]
            list of synapses to choose from
        """
        lst = [
            n
            for n in self.b2model.neurongroups
            if "idx" not in self.b2model.neurongroups[n]
        ] + list(self.b2model.synapses.keys())

        self.logger.debug("updating available object list for monitors to {lst}")
        return lst

    def _make_selection_list(self, typename: str, sourcename: str) -> list[str]:
        """
        Make a list of available parameters for monitoring depeding on the type of monitor and the element of the network selected to be monitored

        Parameters
        ----------
        typename : str
            Name of the Monitortype. Either `StatMonitor` or `SpikeMonitor`
        sourcename : str
            Name of the network element to monitor which provides the variables

        Returns
        -------
        list[str]
            list of variables with which the selectable dropdown for 'variable' will be updated
        """
        # the below is taken from how brian2 internally fills variables to monitor
        # if none are given for the respective monitor, which gives the set of
        # selectable variables
        if typename == "StateMonitor":
            return list(self.b2model.network[sourcename].equations.names) + [
                "True",
            ]  # True means that everything is recorded. Allowed for StateMonitor
        elif typename == "SpikeMonitor":
            return list(self.b2model.network[sourcename].variables.keys()) + [
                "None",
            ]  # None is allowed for spikemonitor
        else:
            message = f"Unknonw monitor type {typename}"
            self.logger.critical(message)
            return []

    def _update_element_list(self) -> list[str]:
        """
        Update the list of selectable variables in the 'variable' dropdown menu
        based on the selected monitor and the selected source network element.

        Returns
        -------
        list[str]
            list of selectable variables to monitor
        """
        selected_items = self.selectedItems()

        if len(selected_items) > 0:
            # find the parent groupitem, then get out the type of monitor and the monitored network element,
            # then construct list of selectable elements from them
            self.logger.info("selected item found")
            selected = selected_items[0]

            parent = selected.parent()
            typename = parent.child(1)
            source = parent.child(2)

            try:
                self.logger.debug("making varialbes as intended")
                return self._make_selection_list(typename.value(), source.value())
            except Exception as e:
                self.logger.critical(
                    f"Something went wrong while forming selection for monitored element {source.text(0)}: {e}"
                )
                vars = list(self.b2model.network[source.value()].get_states().keys())

            return vars
        else:
            self.logger.debug("no selected item found, make full list of variables")
            # when we have no selected items, we make a big list of everything that's available since we don´t yet
            # know which one it will be, and we need a list that contains the actually selected one in the
            # underlying monitor definition in order to display it correctly
            vars = []
            for obj in self.b2model.network.objects:
                if isinstance(obj, b2.StateMonitor | b2.EventMonitor):
                    continue
                else:
                    vars.extend(list(obj.equations.names) + list(obj.variables.keys()))

            return list(set(vars)) + [
                "None",
            ]


class EditableNeurongroupTree(EditableParameterTree):
    """
    Subclass for representing neurongroups and neuronsubgroups in the gui using
    a pyqtgraph ParameterTree
    """

    def __init__(
        self,
        b2model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        """
        Create a new EditableNeurongroupTree element in the gui

        Parameters
        ----------
        b2model : B2Model
            Model to represent the neurongroups of
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        showHeader : bool, optional
            Show the root node title or not, by default True
        """
        super().__init__(
            b2model=b2model,
            item_adder_dialog=NeuronGroupDialog,
            domain_name="neurongroups",
            parent=parent,
            showHeader=showHeader,
            list_items={
                "method": {
                    "update_from": lambda: ["euler", "exponential_euler", "exact"],
                },
                "target": {
                    "update_from": self._update_neurongroup_list,
                },
            },
        )

        self.add_neurongroup_action = self.add_item_action
        self.add_neurongroup_action.setText("add Neurongroup")

        self.add_neuronsubgroup_action = QtWidgets.QAction("add Neuron subgroup")
        self.add_neuronsubgroup_action.triggered.connect(self._add_neuronsubgroup)

    def _add_neuronsubgroup(self) -> None:
        """open neuron subgroup definition dialog and update the displayed parameters"""
        old_names = set(getattr(self.b2model, self.name).keys())
        self.dialog = NeuronSubgroupDialog(self.b2model)

        # update the internal states by
        if self.dialog.exec():
            new_names = set(getattr(self.b2model, self.name).keys())
            new_name = list(new_names - old_names)[0]
            param = self._convert_dict_to_params(
                {
                    new_name: getattr(self.b2model, self.name)[new_name],
                },
                self.functions,
            )
            self.root.addChildren(param)

    def _update_neurongroup_list(self) -> list[str]:
        """
        update the currently available list of neurongroups

        Returns
        -------
        list[str]
            list of neurongroups to choose from
        """
        return [
            n
            for n in self.b2model.neurongroups
            if "idx" not in self.b2model.neurongroups[n]
        ]

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

            menu.addAction(self.add_neurongroup_action)

            menu.addAction(self.add_neuronsubgroup_action)

            menu.exec(self.mapToGlobal(ev.pos()))


class ModelEditorDialog(QtWidgets.QDialog):
    """
    Editor dialog for editing the properties of a model using multiple pyqtgraph ParameterTrees.

    Parameters
    ----------
    QtWidgets : QtWidgets.QDialog
        Basic Qt Dialog class
    """

    def __init__(self, b2model: B2Model, parent: QtWidgets.QWidget | None = None):
        """
        Create a new modeleditor instance and tie it to a B2Model instance.

        Parameters
        ----------
        b2model : B2Model
            Model to edit through the gui
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        """
        super().__init__(parent=parent)

        trees = {
            "Neurongroups": EditableNeurongroupTree(b2model, parent=self),
            "Synapses": EditableSynapseTree(
                b2model,
                parent=self,
            ),
            "Monitors": EditableMonitorTree(
                b2model,
                parent=self,
            ),
            "Parameters": EditableParameterTree(
                b2model,
                ParameterDialog,
                domain_name="parameters",
                parent=self,
            ),
        }
        self.b2model = b2model
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
        if self.b2model.must_be_rebuilt:
            self.logger.debug("rebuilding model after change")
            states = self.b2model.to_dict()

            if self.b2model.dt is not None:
                states["timestep"] = self.b2model.dt

            constr = Reconstructor(states)

            # FIXME: this can be a problem when the system shall be continued because
            # the sequence of random number changes due to brian2 caching them.
            if self.b2model.seed != states["seed"]:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Model rng warning"
                    "The system's random number generator will be reseeded. This potentially can change the results when compared to previous ones",
                )
            new_model = B2Model(seed=states["seed"])

            model_ok = True
            try:
                constr.reconstruct(new_model)
            except Exception as e:
                message = f"Error when rebuilding the model after editing: {e}"
                self.logger.critical(message)
                QtWidgets.QMessageBox.critical(self, "Model rebuilding error", message)
                model_ok = False

            if model_ok:
                self.b2model = new_model
                self.b2model.must_be_rebuilt = False
                super().accept()
        else:
            super().accept()
