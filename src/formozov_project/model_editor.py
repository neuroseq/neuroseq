from __future__ import annotations

import logging
from typing import Any

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
    def __init__(
        self,
        model: B2Model,
        item_adder_dialog: type,
        name: str,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
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

    def _connect_parameters(self, param):
        for child in param.children():
            if child.type() == "group":
                self._connect_parameters(child)
                child.sigChildAdded.connect(self._on_change)
                child.sigChildRemoved.connect(self._on_change)
            else:
                child.sigValueChanged.connect(self._on_change)
                child.sigNameChanged.connect(self._on_change)

    def _on_change(self):
        self.model.must_be_rebuild = True

    def _get_value(self, v: Any) -> TimedArray | Any:
        if isinstance(v, TimedArray):
            return {"values": v.values, "dt": v.dt}
        else:
            return v

    def _convert_dict_to_params(self, data: dict[Any, Any], key: Any = "") -> list[Any]:
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
                        "renamable": True,
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
                # Add more type checks as necessary

                param = {
                    "name": key,
                    "type": param_type,
                    "value": value,
                    "removable": True,
                    "renamable": True,
                    "syncExpanded": True,
                }
                params.append(param)

        return params

    def contextMenuEvent(self, ev: QtCore.QEvent) -> None:
        if self.itemAt(ev.pos()):
            super().contextMenuEvent(ev)
        else:
            menu = QtWidgets.QMenu(self)
            action_menu_entry = menu.addAction(f"add {self.name}")
            action = menu.exec(self.mapToGlobal(ev.pos()))

            if action == action_menu_entry:
                self.addItem()

    def setParameters(self, params: Any, showTop: bool = False) -> None:
        super().setParameters(params, showTop=showTop)
        self.params = params

    def updateParams(self, new_name: str, new_data: dict[Any, Any]) -> None:
        param = self._convert_dict_to_params(
            {
                new_name: new_data,
            }
        )
        self.root.addChildren(param)

    def addItem(self) -> None:
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
    def __init__(
        self,
        model: B2Model,
        parent: QtWidgets.QWidget | None = None,
        showHeader: bool = True,
    ):
        super().__init__(
            model=model,
            item_adder_dialog=NeuronGroupDialog,
            name="neurongroups",
            parent=parent,
            showHeader=showHeader,
        )

    def contextMenuEvent(self, ev: QtCore.Event) -> None:
        if self.itemAt(ev.pos()):
            self.logger.info("found item")

            super().contextMenuEvent(ev)
        else:
            self.logger.info("no item found here")

            menu = QtWidgets.QMenu(self)

            ng_menu_entry = menu.addAction("add Neurongroup")

            nsg_menu_entry = menu.addAction("add Neuron subgroup")

            action = menu.exec(self.mapToGlobal(ev.pos()))

            old_names = set(getattr(self.model, self.name).keys())

            if action == ng_menu_entry:
                dialog = NeuronGroupDialog(
                    self.model,
                )

            elif action == nsg_menu_entry:
                dialog = NeuronSubgroupDialog(self.model)

            if dialog.exec():
                new_names = set(getattr(self.model, self.name).keys())
                new_name = list(new_names - old_names)[0]

                # update the internal states by
                self.updateParams(new_name, getattr(self.model, self.name)[new_name])


class ModelEditorDialog(QtWidgets.QDialog):
    def __init__(self, model: B2Model, parent: QtWidgets.QWidget | None = None):
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
        if self.model.must_be_rebuilt:
            self.logger.info("rebuilding model after change")
            states = self.model.to_dict()

            if self.model.dt is not None:
                states["timestep"] = self.model.dt

            constr = Reconstructor(states)

            # FIXME: this can be a problem when the system shall be continued because
            # the sequence of random number changes due to brian2 caching them.
            new_model = B2Model(seed=states["seed"])

            constr.reconstruct(new_model)

            self.model = new_model

        super().accept()
