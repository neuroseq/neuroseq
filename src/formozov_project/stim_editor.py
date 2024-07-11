from __future__ import annotations

import logging

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.b2_model import B2Model
from formozov_project.stim_widget import StimWidget
from formozov_project.utils import add_action


class StimEditor(QtWidgets.QWidget):
    """A GUI to edit 2-d numpy arrays of neuron stimuli interactively using mouse and keyboard input."""

    def __init__(
        self,
        menu: QtWidgets.QMenu | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model: B2Model | None = None
        self.current_stim: b2.TimedArray | None = None
        self.stims_dict: dict[str, b2.TimedArray] = {}
        self.copy_time_slice = slice(0, 0)
        self.copy_neuron_slice = slice(0, 0)
        self.copy_values = np.zeros((0, 0))
        layout = QtWidgets.QVBoxLayout()
        # horizontal row of widgets for name, units, dt
        top_horizontal_layout = QtWidgets.QHBoxLayout()
        self.cmb_stim = QtWidgets.QComboBox()
        top_horizontal_layout.addWidget(self.cmb_stim)
        layout.addLayout(top_horizontal_layout)
        self.stim_widget = StimWidget()
        layout.addWidget(self.stim_widget)
        self.toolbar = QtWidgets.QToolBar()
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self._add_actions(menu)
        self.cmb_stim.currentTextChanged.connect(self.select_stim)
        self.stim_widget.selection_moved.connect(self.move_selected_stim)

    def set_model(self, model: B2Model) -> None:
        self.model = model
        self.stims_dict = {}
        if self.model is not None and self.model.parameters is not None:
            self.stims_dict = {
                k: v
                for k, v in self.model.parameters.items()
                if isinstance(v, b2.TimedArray)
            }
        have_stims = self.model is not None and len(self.stims_dict) > 0
        self.cmb_stim.clear()
        self.stim_widget.clear_data()
        self.stim_widget.setEnabled(have_stims)
        self.cmb_stim.setEnabled(have_stims)
        if not have_stims:
            self.current_stim = None
            return
        self.logger.info(f"stims: {self.stims_dict}")
        self.cmb_stim.addItems(list(self.stims_dict.keys()))
        self.cmb_stim.setCurrentIndex(0)
        self.select_stim(self.cmb_stim.currentText())

    def update_stims_values(self, stims_values_dict: dict[str, np.ndarray]) -> None:
        self.logger.debug(stims_values_dict)
        if self.model is None or self.model.parameters is None:
            return
        for name, array_values in stims_values_dict.items():
            if name not in self.model.parameters:
                self.logger.warning(
                    f"Stimuli '{name}' not found in model parameters: ignoring"
                )
            else:
                old_timed_array = self.model.parameters[name]
                self.model.parameters[name] = b2.TimedArray(
                    b2.Quantity(array_values, old_timed_array.dim),
                    old_timed_array.dt * b2.second,
                )
        self.set_model(self.model)

    def select_stim(self, stim_name: str) -> None:
        self.logger.debug(f"Selected stim: {stim_name}")
        if stim_name in self.stims_dict:
            self.current_stim = self.stims_dict[stim_name]
            self.stim_widget.set_data(self.current_stim)

    def toggle_edit_stims(self) -> None:
        editing_enabled = not self.stim_widget.capture_mouse_events
        if not editing_enabled:
            self.stim_widget.unselect_neurons()
        self.stim_widget.capture_mouse_events = editing_enabled
        for action in self.toolbar.actions()[1:]:
            action.setEnabled(editing_enabled)

    def copy_selected_stim(self) -> None:
        if (
            self.current_stim is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        self.copy_time_slice = self.stim_widget.selected_time_slice
        self.copy_neuron_slice = self.stim_widget.selected_neuron_slice
        self.copy_values = self.current_stim.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ].copy()

    def cut_selected_stim(self) -> None:
        self.copy_selected_stim()
        self.delete_selected_stim()

    def paste_selected_stim(self) -> None:
        if self.current_stim is None:
            return
        time_offset = (
            int(self.stim_widget.timeline.x() / self.current_stim.dt)
            - self.copy_time_slice.start
        )
        new_time_slice = slice(
            self.copy_time_slice.start + time_offset,
            self.copy_time_slice.stop + time_offset,
        )
        self.current_stim.values[new_time_slice, self.copy_neuron_slice] = (
            self.copy_values
        )
        self.stim_widget.set_data(self.current_stim)
        self.stim_widget.select_neurons(self.copy_neuron_slice, new_time_slice)

    def delete_selected_stim(self) -> None:
        if self.current_stim is None:
            return
        self.current_stim.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ] = 0
        self.stim_widget.unselect_neurons()
        self.stim_widget.set_data(self.current_stim)

    def move_selected_stim(
        self, neuron_index_offset: int = 0, time_index_offset: int = 0
    ) -> None:
        if (
            self.current_stim is None
            or self.stim_widget.selected_neuron_slice is None
            or self.stim_widget.selected_time_slice is None
        ):
            return
        new_neuron_slice = slice(
            self.stim_widget.selected_neuron_slice.start + neuron_index_offset,
            self.stim_widget.selected_neuron_slice.stop + neuron_index_offset,
        )
        new_time_slice = slice(
            self.stim_widget.selected_time_slice.start + time_index_offset,
            self.stim_widget.selected_time_slice.stop + time_index_offset,
        )
        self.cut_selected_stim()
        self.current_stim.values[new_time_slice, new_neuron_slice] = self.copy_values
        self.stim_widget.set_data(self.current_stim)
        self.stim_widget.select_neurons(new_neuron_slice, new_time_slice)

    def move_selected_stim_up(self) -> None:
        self.move_selected_stim(neuron_index_offset=1)

    def move_selected_stim_down(self) -> None:
        self.move_selected_stim(neuron_index_offset=-1)

    def move_selected_stim_left(self) -> None:
        self.move_selected_stim(time_index_offset=-1)

    def move_selected_stim_right(self) -> None:
        self.move_selected_stim(time_index_offset=1)

    def _add_actions(self, menu: QtWidgets.QMenu) -> None:
        add_action(
            "Toggle edit mode",
            self.toggle_edit_stims,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+E",
            icon=qta.icon("mdi6.playlist-edit"),
        )
        self.toolbar.actions()[0].setCheckable(True)
        self.toolbar.actions()[0].setChecked(False)
        add_action(
            "Copy",
            self.copy_selected_stim,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+C",
            icon=qta.icon("mdi6.content-copy"),
        )
        add_action(
            "Cut",
            self.cut_selected_stim,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+X",
            icon=qta.icon("mdi6.content-cut"),
        )
        add_action(
            "Paste",
            self.paste_selected_stim,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+V",
            icon=qta.icon("mdi6.content-paste"),
        )
        add_action(
            "Delete",
            self.delete_selected_stim,
            menu,
            self.toolbar,
            self,
            shortcut="Delete",
            icon=qta.icon("mdi6.delete"),
        )
        add_action(
            "Move up",
            self.move_selected_stim_up,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+Up",
            icon=qta.icon("mdi6.arrow-up"),
        )
        add_action(
            "Move down",
            self.move_selected_stim_down,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+Down",
            icon=qta.icon("mdi6.arrow-down"),
        )
        add_action(
            "Move left",
            self.move_selected_stim_left,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+Left",
            icon=qta.icon("mdi6.arrow-left"),
        )
        add_action(
            "Move right",
            self.move_selected_stim_right,
            menu,
            self.toolbar,
            self,
            shortcut="Ctrl+Right",
            icon=qta.icon("mdi6.arrow-right"),
        )
        for action in self.toolbar.actions()[1:]:
            action.setEnabled(False)
