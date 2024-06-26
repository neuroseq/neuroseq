from __future__ import annotations

import brian2 as b2
import numpy as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets

from formozov_project.stim_console_widget import StimConsoleWidget
from formozov_project.stim_widget import StimWidget
from formozov_project.utils import add_action


class StimDialog(QtWidgets.QDialog):
    """A dialog to edit 2-d numpy arrays of neuron stimuli interactively using a Python console.

    The input stimuli are not modified, and the edited stimuli are available as self.stim.
    """

    def __init__(
        self,
        stims: np.ndarray | None = None,
        dt: b2.Quantity = 1.0 * b2.ms,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Stimuli")
        self.name = "stim"
        self.stims = stims.copy() if stims is not None else np.zeros((1, 1))
        self.copy_time_slice = slice(0, 0)
        self.copy_neuron_slice = slice(0, 0)
        self.copy_values = np.zeros((0, 0))
        self.units = b2.nS
        self.dt = dt / b2.second
        layout = QtWidgets.QVBoxLayout()
        # horizontal row of widgets for name, units, dt
        top_horizontal_layout = QtWidgets.QHBoxLayout()
        self.text_name = QtWidgets.QLineEdit()
        self.text_name.setText(self.name)
        top_horizontal_layout.addWidget(QtWidgets.QLabel("Name:"))
        top_horizontal_layout.addWidget(self.text_name)
        top_horizontal_layout.addWidget(QtWidgets.QLabel(f"Units: {self.units}"))
        top_horizontal_layout.addWidget(QtWidgets.QLabel(f"dt: {self.dt} s"))
        layout.addLayout(top_horizontal_layout)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        split_left_right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left_parent_widget = QtWidgets.QWidget()
        left_vertical_layout = QtWidgets.QVBoxLayout(left_parent_widget)
        self.stim_widget = StimWidget()
        left_vertical_layout.addWidget(self.stim_widget)
        self.toolbar = QtWidgets.QToolBar()
        left_vertical_layout.addWidget(self.toolbar)
        split_left_right.addWidget(left_parent_widget)
        self.stim_console_widget = StimConsoleWidget(stims=self.stims)
        split_left_right.addWidget(self.stim_console_widget)
        layout.addWidget(split_left_right)
        self.setLayout(layout)
        layout.addWidget(self.button_box)
        self.stim_console_widget.stims_updated.connect(self._update_stims)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        add_action(
            "Toggle edit mode",
            self.toggle_edit_stims,
            None,
            self.toolbar,
            shortcut="Ctrl+E",
            icon=qta.icon("mdi6.playlist-edit"),
        )
        self.toolbar.actions()[0].setCheckable(True)
        self.toolbar.actions()[0].setChecked(False)
        add_action(
            "Copy",
            self.copy_selected_stim,
            None,
            self.toolbar,
            shortcut="Ctrl+C",
            icon=qta.icon("mdi6.content-copy"),
        )
        add_action(
            "Cut",
            self.cut_selected_stim,
            None,
            self.toolbar,
            shortcut="Ctrl+X",
            icon=qta.icon("mdi6.content-cut"),
        )
        add_action(
            "Paste",
            self.paste_selected_stim,
            None,
            self.toolbar,
            shortcut="Ctrl+V",
            icon=qta.icon("mdi6.content-paste"),
        )
        add_action(
            "Delete",
            self.delete_selected_stim,
            None,
            self.toolbar,
            shortcut="Delete",
            icon=qta.icon("mdi6.delete"),
        )
        add_action(
            "Move up",
            self.move_selected_stim_up,
            None,
            self.toolbar,
            shortcut="Ctrl+Up",
            icon=qta.icon("mdi6.arrow-up"),
        )
        add_action(
            "Move down",
            self.move_selected_stim_down,
            None,
            self.toolbar,
            shortcut="Ctrl+Down",
            icon=qta.icon("mdi6.arrow-down"),
        )
        add_action(
            "Move left",
            self.move_selected_stim_left,
            None,
            self.toolbar,
            shortcut="Ctrl+Left",
            icon=qta.icon("mdi6.arrow-left"),
        )
        add_action(
            "Move right",
            self.move_selected_stim_right,
            None,
            self.toolbar,
            shortcut="Ctrl+Right",
            icon=qta.icon("mdi6.arrow-right"),
        )

    def toggle_edit_stims(self) -> None:
        if self.stim_widget.capture_mouse_events:
            self.stim_widget.unselect_neurons()
        self.stim_widget.capture_mouse_events = (
            not self.stim_widget.capture_mouse_events
        )

    def copy_selected_stim(self) -> None:
        if (
            self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        self.copy_time_slice = self.stim_widget.selected_time_slice
        self.copy_neuron_slice = self.stim_widget.selected_neuron_slice
        self.copy_values = self.stims[
            self.stim_widget.selected_neuron_slice, self.stim_widget.selected_time_slice
        ].copy()

    def cut_selected_stim(self) -> None:
        self.copy_selected_stim()
        self.delete_selected_stim()

    def paste_selected_stim(self) -> None:
        time_offset = (
            int(self.stim_widget.timeline.x() / self.dt) - self.copy_time_slice.start
        )
        new_time_slice = slice(
            self.copy_time_slice.start + time_offset,
            self.copy_time_slice.stop + time_offset,
        )
        self.stims[self.copy_neuron_slice, new_time_slice] += self.copy_values
        self.stim_widget.set_data(self.stims, dt=self.dt)
        self.stim_widget.select_neurons(self.copy_neuron_slice, new_time_slice)

    def delete_selected_stim(self) -> None:
        self.stims[
            self.stim_widget.selected_neuron_slice, self.stim_widget.selected_time_slice
        ] = 0
        self.stim_widget.unselect_neurons()
        self.stim_widget.set_data(self.stims, dt=self.dt)

    def move_selected_stim(
        self, neuron_index_offset: int = 0, time_index_offset: int = 0
    ) -> None:
        if (
            self.stim_widget.selected_neuron_slice is None
            or self.stim_widget.selected_time_slice is None
        ):
            return
        new_y_slice = slice(
            self.stim_widget.selected_neuron_slice.start + neuron_index_offset,
            self.stim_widget.selected_neuron_slice.stop + neuron_index_offset,
        )
        new_x_slice = slice(
            self.stim_widget.selected_time_slice.start + time_index_offset,
            self.stim_widget.selected_time_slice.stop + time_index_offset,
        )
        slice_data = self.stims[
            self.stim_widget.selected_neuron_slice, self.stim_widget.selected_time_slice
        ].copy()
        self.stims[
            self.stim_widget.selected_neuron_slice, self.stim_widget.selected_time_slice
        ] = 0
        self.stims[new_y_slice, new_x_slice] = slice_data
        self.stim_widget.set_data(self.stims, dt=self.dt)
        self.stim_widget.select_neurons(new_y_slice, new_x_slice)

    def move_selected_stim_up(self) -> None:
        self.move_selected_stim(neuron_index_offset=1)

    def move_selected_stim_down(self) -> None:
        self.move_selected_stim(neuron_index_offset=-1)

    def move_selected_stim_left(self) -> None:
        self.move_selected_stim(time_index_offset=-1)

    def move_selected_stim_right(self) -> None:
        self.move_selected_stim(time_index_offset=1)

    def text_name_changed(self, text: str) -> None:
        self.name = text

    def _update_stims(self, stims: np.ndarray, valid: bool) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            valid
        )
        if valid:
            self.stim_widget.set_data(stims, dt=self.dt)
            self.stims = stims
