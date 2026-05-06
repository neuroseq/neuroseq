from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import brian2 as b2  # noqa: F401
from brian2 import TimedArray
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree import registerParameterItemType
from pyqtgraph.parametertree import registerParameterType
from pyqtgraph.parametertree.parameterTypes import ListParameter
from pyqtgraph.parametertree.parameterTypes import ListParameterItem
from pyqtgraph.parametertree.parameterTypes import TextParameter
from pyqtgraph.parametertree.parameterTypes import TextParameterItem
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import QEvent
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox

from neuroseq.b2_model import B2Model
from neuroseq.b2_units import str_to_quantity

if TYPE_CHECKING:
    from neuroseq.project import Project


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

        self.setValue(opts.get("value", "unknown"))

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


class LargeTextParameterItem(TextParameterItem):
    """
    Subclass of TextParameterItem to adjust the textbox size dynamically to the
    number of lines in the displayed text

    Parameters
    ----------
    TextParameterItem : Original pyqtgraph TextParameterItem that only has fixed adjustment

    """

    def makeWidget(self) -> QtWidgets.QTextWidget:
        """
        Override for the superclass' makeWidget function that includees size adjustment

        Returns
        -------
        QtWidgets.QTextWidget
            QTextWidget that displays the editable text
        """
        w = self.adjust_to_text_size(super().makeWidget())
        w.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        return w

    def adjust_to_text_size(self, w: QtWidgets.QTextWidget) -> QtWidgets.QTextWidget:
        """
        Adjust the height of the displayed text box to be (1+num(newlines)) * fontheight

        Parameters
        ----------
        w : QtWidgets.QTextWidget
            Textwidget to adjust height for

        Returns
        -------
        QtWidgets.QTextWidget
            Input widget with adjusted height
        """
        # adjust height of input box to the number of lines, + 1 to make it
        # look less crammed
        new_height = int(
            (1 + self.param.value().count("\n")) * w.fontMetrics().height()
        )
        w.setMinimumSize(QtCore.QSize(int(w.width()), new_height))

        return w


class LargeTextParameter(TextParameter):
    itemClass = LargeTextParameterItem


# register the new types with the parametertree implementation
registerParameterType("updatinglist", UpdatingListParameter)
registerParameterItemType(
    "updatinglistitem", UpdatingListParameterItem, UpdatingListParameter
)

registerParameterType("largetext", LargeTextParameter)
registerParameterItemType("largetextitem", LargeTextParameterItem, LargeTextParameter)


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
        domain_name: str,
        default_creator: Callable,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
        list_items: dict[str, Any] | None = None,
        add_to_groups: bool = False,
        add_to_groups_text: str = "New",
    ):
        """
        Create a new paramter tree displaying a part of the definition of a B2Model.
        Editing the parameter tree edits the b2model class. When something is edited, the b2model will be rebuild

        Parameters
        ----------
        b2model : B2Model
            b2model to display a part of

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
        self.logger = logging.getLogger(__name__)
        self.name = domain_name
        self.states = getattr(self.b2model, self.name)
        self.functions = self.b2model.functions[self.name]
        self.list_items = list_items
        self.default_creator = default_creator
        self.add_to_groups = add_to_groups
        self.add_to_groups_text = add_to_groups_text
        self.add_item_action = QtWidgets.QAction(f"add {self.name}")
        self.add_item_action.triggered.connect(self._add_item)
        self._build_tree()

    def _call_with_errormessage(self, signal: Callable, *args: Any, **kwargs) -> Any:
        """
        Call the function 'signal' with arguments '*args' and display an error message box
        when it fails with an exception. This function mostly acts as a wrapper around qt slots
        to make possibly occuring errors visible. Alternatively it can be used to make exceptions
        visible in the gui

        Parameters
        ----------
        signal : callable
            Function to call. A qt slot that is to be called with the output from a signal.

        Returns
        -------
        Any
            Any return value from the slot
        """
        try:
            return signal(*args, **kwargs)
        except Exception as e:
            message = f"An error occured: {e}"
            self.logger.critical(message)
            QtWidgets.QMessageBox.critical(self, "Error", message)

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
            if all(f in funcs for f in ["delete", "rename"]):
                param.sigRemoved.connect(
                    lambda: self._call_with_errormessage(funcs["delete"])
                )
                param.sigNameChanged.connect(
                    lambda p: self._call_with_errormessage(funcs["rename"], p.name())
                )

            if "addText" in param.opts:
                # allow for making a new element, utilizing the function stored behind the 'add' key in the
                # supplied dictionary of slots
                param.sigAddNew.connect(
                    lambda: self._call_with_errormessage(funcs["add"])
                )
                # rebuilt the tree to make sure everything is updated correctly. TODO: make this less overreaching
                param.sigAddNew.connect(
                    lambda: self._call_with_errormessage(self._build_tree)
                )

            for child in param.children():
                if child.name() in funcs:
                    self._connect_parameters(child, funcs[child.name()])
                else:
                    # root is not part of the function tree and must be skipped
                    pass
        else:
            param.sigValueChanged.connect(
                lambda p: self._call_with_errormessage(
                    funcs["modify"],
                    self._call_with_errormessage(str_to_quantity, p.value()),
                )
            )
            param.sigRemoved.connect(
                lambda: self._call_with_errormessage(funcs["delete"])
            )
            param.sigNameChanged.connect(
                lambda p: self._call_with_errormessage(funcs["rename"], p.name())
            )

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
        self,
        data: dict[Any, Any],
        functions: dict[Any, Any],
        key: Any = "",
        expanded: bool = False,
        read_only: bool = False,
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
        expanded: bool
            Whether group elements should be expanded on first open or not
        read_only:
            Whether children should be read_only based on a condition evalutated in the parent. Currently used to make TimedArray displays readonly
        Returns
        -------
        list[Any]
            list of Parameter representations that internally define all the parent-child relationships to create the parametertree.
        """
        params = []
        for key, value in data.items():
            readonly = isinstance(value, TimedArray)
            value = self._get_value(value)
            if isinstance(value, dict):
                child_params = self._convert_dict_to_params(
                    value, functions[key], key, expanded=expanded, read_only=readonly
                )

                new_param = {
                    "name": key,
                    "type": "group",
                    "children": child_params,
                    "expanded": expanded,
                    "removable": True,
                    "renamable": True,
                    "syncExpanded": True,
                    "readonly": readonly,
                }

                if self.add_to_groups:
                    new_param["addText"] = self.add_to_groups_text

                params.append(new_param)

            else:
                # Determine the parameter type based on value type
                if isinstance(value, b2.Quantity):
                    value = value.in_best_unit(python_code=True)

                value = str(value)
                param_type = (
                    "largetext" if "=" in value else "str"
                )  # make sure our custom QTextEdit based ParameterItem is used for potentially large equations

                param = {
                    "name": key,
                    "type": param_type,
                    "value": value,
                    "removable": True,
                    "renamable": True,
                    "syncExpanded": True,
                    "expanded": expanded,
                    "readonly": read_only,
                }

                # need 'limits' for the underlying ListParameter and 'element' for the custom class
                if key in list(self.list_items.keys()):
                    param["type"] = "updatinglist"
                    param["limits"] = self.list_items[key]["update_from"]()
                    param["update"] = self.list_items[key]["update_from"]

                params.append(param)

        return params

    def _build_tree(self, expanded: bool = False) -> None:
        """
        Rebuild the tree upon changed
        """
        params = self._call_with_errormessage(
            self._convert_dict_to_params, self.states, self.functions, expanded=expanded
        )

        self.root = self._call_with_errormessage(
            Parameter.create,
            name=self.name,
            type="group",
            children=params,
        )
        self.setParameters(self.root, showTop=False)

        self._call_with_errormessage(
            self._connect_parameters, self.root, self.functions
        )

    def _add_item(self) -> None:
        """
        Call the custom dialog given in the callers constructor to add custom elements and update
        the parametertree accordingly
        """
        # FIXME: the connections are correct when the entire tree is rebuild, but when only the added part is rebuild they fail
        # and it seems that the definition dict is copied.
        # check the recursive functions again for a point where stuff can get copied?
        self._call_with_errormessage(self.default_creator)
        self.clear()
        self._build_tree()

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
            # menu = QtWidgets.QMenu(self)
            # menu.addAction(self.add_item_action)
            # menu.exec(self.mapToGlobal(ev.pos()))
            pass


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
            default_creator=b2model.define_default_synapse,
            showHeader=showHeader,
            domain_name="synapses",
            list_items={
                "method": {
                    "update_from": lambda: [
                        "euler",
                        "exponential_euler",
                        "exact",
                        "rk2",
                        "rk4",
                        "heun",
                        "milstein",
                        "-",
                    ],  # methods are static, so we can give a list
                },
                "origin": {
                    "update_from": lambda: list(self.b2model.neurongroups.keys())
                    + ["-"],
                },
                "target": {
                    "update_from": lambda: list(self.b2model.neurongroups.keys())
                    + ["-"],
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
            default_creator=b2model.define_default_monitor,
            showHeader=showHeader,
            domain_name="monitors",
            list_items={
                "type": {
                    "update_from": lambda: [
                        b2.StateMonitor.__name__,
                        b2.SpikeMonitor.__name__,
                        "-",
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
        lst = (
            [
                n
                for n in self.b2model.neurongroups
                if "idx" not in self.b2model.neurongroups[n]
            ]
            + list(self.b2model.synapses.keys())
            + ["-"]
        )

        self.logger.debug(f"updating available object list for monitors to {lst}")
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
                "-",
            ]
        elif typename == "SpikeMonitor":
            return list(self.b2model.network[sourcename].variables.keys()) + [
                "None",
                "-",
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

            # get the type and source widgets independent of the ordering of the children
            parent = selected.parent()
            typename = [
                parent.child(c)
                for c in range(0, parent.childCount())
                if parent.child(c).text(0) == "type"
            ][0]
            source = [
                parent.child(c)
                for c in range(0, parent.childCount())
                if parent.child(c).text(0) == "source"
            ][0]
            varnames = []
            try:
                varnames = self._make_selection_list(typename.value(), source.value())
            except Exception as e:
                self.logger.critical(
                    f"Something went wrong while forming selection for monitored element {source.text(0)}: {e}"
                )

                if source.value() in self.b2model.network.objects:
                    existing_vars = list(
                        self.b2model.network[source.value()].get_states().keys()
                    )
                else:
                    existing_vars = []
                varnames = existing_vars + ["None", "-"]
            return varnames
        else:
            self.logger.debug("no selected item found, make full list of variables")
            # when we have no selected items, we make a big list of everything that's available since we don´t yet
            # know which one it will be, and we need a list that contains the actually selected one in the
            # underlying monitor definition in order to display it correctly
            varnames = [
                "-",
            ]
            for obj in self.b2model.network.objects:
                if isinstance(obj, b2.StateMonitor | b2.EventMonitor):
                    continue
                else:
                    varnames.extend(
                        list(obj.equations.names) + list(obj.variables.keys())
                    )

            return list(set(varnames)) + [
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
            default_creator=b2model.define_default_neurongroup,
            domain_name="neurongroups",
            parent=parent,
            showHeader=showHeader,
            list_items={
                "method": {
                    "update_from": lambda: [
                        "euler",
                        "exponential_euler",
                        "exact",
                        "rk2",
                        "rk4",
                        "heun",
                        "milstein",
                        "-",
                    ],
                },
                "target": {
                    "update_from": self._update_neurongroup_list,
                },
            },
            add_to_groups=False,
        )

        self.add_neurongroup_action = self.add_item_action
        self.add_neurongroup_action.setText("add neurongroup")

        self.add_neuronsubgroup_action = QtWidgets.QAction("add neuron subgroup")
        self.add_neuronsubgroup_action.triggered.connect(self._add_neuronsubgroup)

    def _add_neuronsubgroup(self) -> None:
        """open neuron subgroup definition dialog and update the displayed parameters"""

        self._call_with_errormessage(self.b2model.define_default_neuronsubgroup)
        self.clear()
        self._build_tree()

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
            if "idxs" not in self.b2model.neurongroups[n]
        ] + [
            "-",
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
            # menu = QtWidgets.QMenu(self)
            # menu.addAction(self.add_neurongroup_action)
            # menu.addAction(self.add_neuronsubgroup_action)
            # menu.exec(self.mapToGlobal(ev.pos()))
            pass


class EditableExternalSimParamsTree(EditableParameterTree):
    """Tree for editing the flat ``Project.external_simulate_params`` schema.

    The schema is ``{name: {value, default, type, group, description}}`` —
    see ``neuroseq.example_models.elise_simulate.DEFAULT_PARAMS`` for the
    canonical shape. Top-level rows are the unique ``group`` values; leaf
    parameters are typed (int/float/bool/str) and edits write through to
    ``params_dict[name]["value"]``.

    Bypasses the b2_model-coupled parts of ``EditableParameterTree.__init__``;
    inherits for IS-A and to reuse the context menu.
    """

    def __init__(
        self,
        params_dict: dict[str, dict],
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = False,
    ) -> None:
        ParameterTree.__init__(self, parent=parent, showHeader=showHeader)
        self.params_dict = params_dict
        self.logger = logging.getLogger(__name__)
        # Parent-class fields kept as no-ops so inherited helpers don't break.
        self.b2model = cast(Any, None)
        self.name = "external_simulate_params"
        self.states = {}
        self.functions = {}
        self.list_items = {}
        self.default_creator = lambda: None
        self.add_to_groups = False
        self.add_to_groups_text = ""
        self.add_item_action = QtWidgets.QAction("")
        self._build_tree()

    @staticmethod
    def _param_type(entry_type: str) -> str:
        return {
            "int": "int",
            "float": "float",
            "bool": "bool",
            "str": "str",
        }.get(entry_type, "str")

    def _build_tree(self, expanded: bool = True) -> None:  # type: ignore[override]
        """Group params by `group`, build typed leaves, wire change signals."""
        # Preserve insertion order of params_dict for stable group ordering.
        groups: dict[str, list[dict]] = {}
        for name, entry in self.params_dict.items():
            group = entry.get("group", "Other")
            groups.setdefault(group, []).append(
                {
                    "name": name,
                    "type": self._param_type(entry.get("type", "str")),
                    "value": entry["value"],
                    "default": entry.get("default", entry["value"]),
                    "tip": entry.get("description", ""),
                    "removable": False,
                    "renamable": False,
                }
            )

        children = [
            {
                "name": group_name,
                "type": "group",
                "children": child_params,
                "expanded": expanded,
                "removable": False,
                "renamable": False,
            }
            for group_name, child_params in groups.items()
        ]
        self.root = Parameter.create(
            name=self.name,
            type="group",
            children=children,
        )
        self.setParameters(self.root, showTop=False)
        self._connect_external_sim_params(self.root)

    def _connect_external_sim_params(self, root: Parameter) -> None:
        """Wire each leaf's `sigValueChanged` back into ``params_dict``.

        ``sigValueChanged`` emits ``(parameter, new_value)`` — two positional
        args. We bind the parameter name in a closure rather than a default
        kwarg so the second positional arg can't overwrite it.
        """
        for group in root.children():
            if group.type() != "group":
                continue
            for leaf in group.children():
                self._connect_leaf(leaf)

    def _connect_leaf(self, leaf: Parameter) -> None:
        name = leaf.name()

        def _handler(_param, value, _name=name):
            self._on_value_changed(_name, value)

        leaf.sigValueChanged.connect(_handler)

    def _on_value_changed(self, name: str, value: Any) -> None:
        if name in self.params_dict:
            self.params_dict[name]["value"] = value
            self.logger.debug(f"external sim param '{name}' set to {value!r}")

    def contextMenuEvent(self, ev: QtCore.QEvent) -> None:
        """Schema is fixed: no add/delete; pass clicks on items through."""
        if self.itemAt(ev.pos()):
            ParameterTree.contextMenuEvent(self, ev)


class ModelEditorDialog(QtWidgets.QDialog):
    """
    Editor dialog for editing the properties of a model using multiple pyqtgraph ParameterTrees.

    Parameters
    ----------
    QtWidgets : QtWidgets.QDialog
        Basic Qt Dialog class
    """

    def __init__(
        self,
        b2model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        project: Project | None = None,
    ):
        """
        Create a new modeleditor instance and tie it to a B2Model instance.

        Parameters
        ----------
        b2model : B2Model
            Model to edit through the gui
        parent : QtWidgets.QWidget | None, optional
            parent widget, by default None
        project : Project | None, optional
            If provided and `project.external_simulate_path` is set, the dialog
            switches to "external simulation" mode and edits
            `external_simulate_params` instead of the Brian2 model.
        """
        super().__init__(parent=parent)

        self.setWindowTitle("Model Editor")

        self.project = project
        self.external_simulate_params: dict[str, dict] | None = None
        self.logger = logging.getLogger(__name__)

        external_mode = (
            project is not None and project.external_simulate_path is not None
        )

        if external_mode:
            assert project is not None  # narrow for type checker
            self._init_external_simulate(project, b2model)
            return

        self.b2model = B2Model.from_dict(
            b2model.to_dict(), b2model.network._stored_state
        )

        trees = {
            "Neurongroups": EditableNeurongroupTree(self.b2model, parent=self),
            "Synapses": EditableSynapseTree(
                self.b2model,
                parent=self,
            ),
            "Monitors": EditableMonitorTree(
                self.b2model,
                parent=self,
            ),
            "Parameters": EditableParameterTree(
                self.b2model,
                domain_name="parameters",
                default_creator=self.b2model.define_default_parameter,
                parent=self,
            ),
            "Initial Conditions": EditableParameterTree(
                self.b2model,
                domain_name="initials",
                default_creator=self.b2model.define_default_init,
                parent=self,
                add_to_groups=True,
            ),
        }

        layout = QtWidgets.QGridLayout()

        btns = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        self.button_box = QtWidgets.QDialogButtonBox(btns)

        i = 0
        for k, tree in trees.items():
            nameplate = QtWidgets.QLabel(str(k), parent=self)
            bold_font = QtGui.QFont()
            bold_font.setBold(True)
            nameplate.setFont(bold_font)
            layout.addWidget(nameplate, 0, i)

            add_group_button = QtWidgets.QPushButton("New " + k[0 : len(k) - 1], self)
            add_group_button.clicked.connect(tree._add_item)
            add_group_button.setVisible(False)
            layout.addWidget(add_group_button, 0, i + 1)
            col_add = 0
            if k == "Neurongroups":
                col_add = 4
                add_group_button_sub = QtWidgets.QPushButton("New Subgroup", self)
                layout.addWidget(add_group_button_sub, 0, i + 2)
                add_group_button_sub.clicked.connect(tree._add_neuronsubgroup)
                add_group_button_sub.setVisible(False)
            else:
                col_add = 3
            layout.addWidget(tree, 1, i, 1, col_add - 1)
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.VLine)
            separator.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(separator, 0, i + col_add - 1, layout.rowCount(), 1)
            i += col_add

        seed_widget = QtWidgets.QWidget(self)
        seed_layout = QtWidgets.QHBoxLayout(seed_widget)
        seed_label = QtWidgets.QLabel("Seed:", parent=self)
        self.seed_spin = QtWidgets.QSpinBox(parent=self)
        self.seed_spin.setRange(0, 2**31 - 1)

        try:
            self.seed_spin.setValue(int(self.b2model.seed))
        except Exception:
            self.seed_spin.setValue(123)

        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_spin)

        # place the seed widget above the dialog button box, spanning columns
        layout.addWidget(
            seed_widget,
            layout.rowCount(),
            0,
            1,
            layout.columnCount(),
        )

        layout.addWidget(
            self.button_box,
            layout.rowCount(),
            1,
            1,
            layout.columnCount() - 1,
            QtCore.Qt.AlignCenter,
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(layout)
        self.showMaximized()

    def _init_external_simulate(self, project: Project, b2model: B2Model) -> None:
        """Build a single-tree layout for editing external-simulation params."""
        # `b2model` is kept on the instance so the calling pattern in
        # `gui.btn_edit_clicked` continues to work; we don't edit it here.
        self.b2model = b2model
        self.external_simulate_params = copy.deepcopy(project.external_simulate_params)

        layout = QtWidgets.QVBoxLayout()

        nameplate = QtWidgets.QLabel("External Simulation Parameters", parent=self)
        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        nameplate.setFont(bold_font)
        layout.addWidget(nameplate)

        path_label = QtWidgets.QLabel(
            f"Simulate function: {project.external_simulate_path}", parent=self
        )
        path_label.setStyleSheet("color: #888;")
        layout.addWidget(path_label)

        self.external_sim_tree = EditableExternalSimParamsTree(
            self.external_simulate_params, parent=self
        )
        layout.addWidget(self.external_sim_tree)

        btns = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box = QtWidgets.QDialogButtonBox(btns)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)
        self.resize(700, 600)

    def accept(self):
        """
        Accept the current state of the model and rebuilt it if anything has been edited.
        """
        # External-sim mode: edits already mutated `external_simulate_params`
        # in place via the tree; the caller will copy it back to the project.
        if self.external_simulate_params is not None:
            QtWidgets.QDialog.accept(self)
            return
        self.logger.info(f"accept model editions {self.b2model.must_be_rebuilt}")
        try:
            # apply any changed seed value to the brian2 RNG and the model
            try:
                new_seed = int(self.seed_spin.value())
                self.b2model.set_seed(new_seed)
            except Exception as e:
                # ignore seed setting errors and proceed with rebuild
                self.logger.debug("Failed to set new seed from widget")
                self.logger.exception(e)

            if self.b2model.must_be_rebuilt:
                self.logger.info("rebuilding model after change")
                self.b2model.assemble_network()  # keep underlying network in sync
                self.b2model.must_be_rebuilt = False
            super().accept()
        except Exception as e:
            message = f"Error when rebuilding the model after editing: {e}"
            self.logger.critical(message)
            QtWidgets.QMessageBox.critical(self, "Model rebuilding error", message)

    def reject(self):
        """
        Reject the edited model specification
        """
        self.logger.info("Edited model specification not accepted, restoring")

        super().reject()
