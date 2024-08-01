from __future__ import annotations

import logging
import math
from enum import Enum

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq.b2_model import B2Model
from neuroseq.stim_widget import StimWidget
from neuroseq.timed_array_modifer import TimedArrayModifier
from neuroseq.utils import add_action
from neuroseq.utils import timed_array_with_new_data


class DrawMode(Enum):
    REGION = 1
    SINGLE_STIM = 2
    SINGLE_STIM_ERASE = 3


class StimEditor(QtWidgets.QWidget):
    """A GUI to edit 2-d numpy arrays of neuron stimuli interactively using mouse and keyboard input."""

    def __init__(
        self,
        cmb_stim: QtWidgets.QComboBox,
        toolbar: QtWidgets.QToolBar,
        menu: QtWidgets.QMenu | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.toolbar = toolbar
        self.model: B2Model | None = None
        self.cmb_stim = cmb_stim
        self.current_stim_modifier: TimedArrayModifier | None = None
        self.draw_mode = DrawMode.REGION
        self.draw_stim_value = 1.0
        self.stims_dict: dict[str, TimedArrayModifier] = {}
        self.copy_time_slice = slice(0, 0)
        self.copy_neuron_slice = slice(0, 0)
        self.copy_values = np.zeros((0, 0))
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stim_widget = StimWidget()
        layout.addWidget(self.stim_widget)
        self.setLayout(layout)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        self._add_actions(self.toolbar, menu)
        self.cmb_stim.currentTextChanged.connect(self.select_stim)
        self.stim_widget.mouse_clicked.connect(self.stim_widget_mouse_clicked)
        self.stim_widget.selection_moved.connect(self.move_selected_stim)
        self.stim_widget.selection_made.connect(self.stim_widget_selection_made)

    def set_model(self, model: B2Model) -> None:
        self.model = model
        self.stims_dict = {}
        if self.model is not None and self.model.parameters is not None:
            self.stims_dict = {
                k: TimedArrayModifier(v)
                for k, v in self.model.parameters.items()
                if isinstance(v, b2.TimedArray)
            }
        have_stims = self.model is not None and len(self.stims_dict) > 0
        self.cmb_stim.clear()
        self.stim_widget.clear_data()
        self.stim_widget.setEnabled(have_stims)
        self.cmb_stim.setEnabled(have_stims)
        if not have_stims:
            self.current_stim_modifier = None
            return
        self.logger.info(f"stims: {self.stims_dict}")
        self.cmb_stim.addItems(list(self.stims_dict.keys()))
        self.cmb_stim.setCurrentIndex(0)
        self.select_stim(self.cmb_stim.currentText())

    def update_stim_values_dict(self, stims_values_dict: dict[str, np.ndarray]) -> None:
        if self.model is None or self.model.parameters is None:
            return
        for name, array_values in stims_values_dict.items():
            if name not in self.model.parameters:
                self.logger.warning(
                    f"Stimuli '{name}' not found in model parameters: ignoring"
                )
            else:
                old_timed_array = self.model.parameters[name]
                self.model.parameters[name] = timed_array_with_new_data(
                    old_timed_array, array_values
                )
        self.set_model(self.model)

    def _stim_values_updated(self) -> None:
        if self.current_stim_modifier is None:
            self.action_undo.setEnabled(False)
            self.action_redo.setEnabled(False)
            return
        self.action_undo.setEnabled(self.current_stim_modifier.can_undo())
        self.action_redo.setEnabled(self.current_stim_modifier.can_redo())
        self.stim_widget.set_data(self.current_stim_modifier.timed_array)

    def select_stim(self, stim_name: str) -> None:
        self.logger.debug(f"Selected stim: {stim_name}")
        if stim_name in self.stims_dict:
            self.current_stim_modifier = self.stims_dict[stim_name]
            self.stim_widget.set_data(self.current_stim_modifier.timed_array)
            self.draw_stim_value = np.max(self.current_stim_modifier.timed_array.values)
            if self.draw_stim_value == 0:
                self.draw_stim_value = 1
            self._stim_values_updated()

    def toggle_draw_stims(self, enabled: bool) -> None:
        self.logger.debug(f"Draw stims {'enabled' if enabled else 'disabled'}")
        if enabled:
            self.stim_widget.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.stim_widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        for action in self.actions_draw:
            action.setEnabled(enabled)
        for action in self.actions_edit:
            action.setEnabled(not enabled)

    def draw_mode_updated(self, draw_mode: DrawMode) -> None:
        self.logger.debug(f"Draw mode updated: {draw_mode}")
        self.draw_mode = draw_mode

    def draw_stim_region(
        self, neuron_slice: slice, time_slice: slice, erase=False
    ) -> None:
        if self.current_stim_modifier is None:
            return
        self.logger.debug(f"Draw stim for neurons {neuron_slice} and time {time_slice}")
        stim_value = 0 if erase else self.draw_stim_value
        self.current_stim_modifier.do(neuron_slice, time_slice, stim_value)
        self._stim_values_updated()
        self.stim_widget.select_neurons(neuron_slice, time_slice)

    def stim_widget_selection_made(
        self, neuron_slice: slice, time_slice: slice
    ) -> None:
        self.logger.debug(
            f"Selection made: neuron slice {neuron_slice}, time slice {time_slice}"
        )
        if self.action_toggle_draw.isChecked() and self.draw_mode == DrawMode.REGION:
            self.draw_stim_region(neuron_slice, time_slice)

    def stim_widget_mouse_clicked(self, neuron_index: int, time_index: int) -> None:
        self.logger.debug(f"Mouse clicked: neuron {neuron_index}, time {time_index}")
        if self.current_stim_modifier is None:
            return
        if self.action_toggle_draw.isChecked():
            if self.draw_mode not in [DrawMode.SINGLE_STIM, DrawMode.SINGLE_STIM_ERASE]:
                return
            neuron_slice = slice(neuron_index, neuron_index + 1)
            time_length_s = 0.001
            time_slice = slice(
                time_index,
                math.floor(
                    time_index
                    + time_length_s / self.current_stim_modifier.timed_array.dt
                ),
            )
            self.logger.debug(f"Time slice: {time_slice}")
            if not self._is_valid_slice(neuron_slice, time_slice):
                return
            self.draw_stim_region(
                neuron_slice,
                time_slice,
                erase=self.draw_mode == DrawMode.SINGLE_STIM_ERASE,
            )
        else:
            self.stim_widget.unselect_neurons()
            self.stim_widget.timeline.setPos(
                time_index * self.current_stim_modifier.timed_array.dt
            )

    def copy_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        self.copy_time_slice = self.stim_widget.selected_time_slice
        self.copy_neuron_slice = self.stim_widget.selected_neuron_slice
        self.copy_values = self.current_stim_modifier.timed_array.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ].copy()

    def cut_selected_stim(self) -> None:
        self.copy_selected_stim()
        self.delete_selected_stim()

    def paste_selected_stim(self, invert=False) -> None:
        if self.current_stim_modifier is None:
            return
        time_offset = (
            int(
                self.stim_widget.timeline.x()
                / self.current_stim_modifier.timed_array.dt
            )
            - self.copy_time_slice.start
        )
        new_time_slice = slice(
            self.copy_time_slice.start + time_offset,
            self.copy_time_slice.stop + time_offset,
        )
        new_neuron_slice = self.copy_neuron_slice
        if not self._is_valid_slice(new_neuron_slice, new_time_slice):
            return
        if invert:
            max_value = np.max(self.copy_values)
            if max_value == 0:
                max_value = self.draw_stim_value
            new_values = max_value - self.copy_values
        else:
            new_values = self.copy_values
        self.current_stim_modifier.do(new_neuron_slice, new_time_slice, new_values)
        self._stim_values_updated()
        self.stim_widget.select_neurons(self.copy_neuron_slice, new_time_slice)

    def undo(self) -> None:
        if self.current_stim_modifier is None:
            return
        self.current_stim_modifier.undo()
        self._stim_values_updated()

    def redo(self) -> None:
        if self.current_stim_modifier is None:
            return
        self.current_stim_modifier.redo()
        self._stim_values_updated()

    def save_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        filename, _ = getsavefilename(
            self,
            "Save stim selection",
            "stim.npy",
            "NumPy npy file (*.npy)",
        )
        if filename == "":
            return
        arr = self.current_stim_modifier.timed_array.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ]
        self.logger.info(f"Saving {arr.shape} stim selection to file {filename}")
        try:
            np.save(filename, arr, allow_pickle=False, fix_imports=False)
        except Exception as e:
            msg = f"Error saving stim selection to file: {e}"
            self.logger.error(msg)
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def load_selected_stim(self) -> None:
        if self.current_stim_modifier is None:
            return
        filename, _ = getopenfilename(
            self,
            "Load stim selection",
            "stim.npy",
            "NumPy npy file (*.npy)",
        )
        if filename == "":
            return
        try:
            arr = np.load(filename, allow_pickle=False, fix_imports=False)
            self.logger.info(f"Loaded {arr.shape} stim selection from file {filename}")
            if arr.ndim > 2:
                raise ValueError("Loaded array has wrong shape")
            if arr.shape[0] > self.current_stim_modifier.timed_array.values.shape[0]:
                raise ValueError("Loaded array has too many time points")
            if (
                arr.ndim == 2
                and arr.shape[1]
                > self.current_stim_modifier.timed_array.values.shape[1]
            ):
                raise ValueError("Loaded array has too many neurons")
            time_offset = int(
                self.stim_widget.timeline.x()
                / self.current_stim_modifier.timed_array.dt
            )
            new_time_slice = slice(time_offset, time_offset + arr.shape[0])
            new_neuron_slice = slice(0, arr.shape[1])
            self.current_stim_modifier.do(new_neuron_slice, new_time_slice, arr)
            self._stim_values_updated()
            self.stim_widget.select_neurons(new_neuron_slice, new_time_slice)
        except Exception as e:
            msg = f"Error loading stim selection from file: {e}"
            self.logger.error(msg)
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def delete_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_neuron_slice is None
            or self.stim_widget.selected_time_slice is None
        ):
            return
        self.current_stim_modifier.do(
            self.stim_widget.selected_neuron_slice,
            self.stim_widget.selected_time_slice,
            0,
        )
        self._stim_values_updated()

    def move_selected_stim(
        self, neuron_index_offset: int = 0, time_index_offset: int = 0
    ) -> None:
        if (
            self.current_stim_modifier is None
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
        if not self._is_valid_slice(new_neuron_slice, new_time_slice):
            self.stim_widget.unselect_neurons()
            return
        self.cut_selected_stim()
        self.current_stim_modifier.do(
            new_neuron_slice, new_time_slice, self.copy_values
        )
        self._stim_values_updated()
        self.stim_widget.select_neurons(new_neuron_slice, new_time_slice)

    def move_selected_stim_up(self) -> None:
        self.move_selected_stim(neuron_index_offset=1)

    def move_selected_stim_down(self) -> None:
        self.move_selected_stim(neuron_index_offset=-1)

    def move_selected_stim_left(self) -> None:
        self.move_selected_stim(time_index_offset=-1)

    def move_selected_stim_right(self) -> None:
        self.move_selected_stim(time_index_offset=1)

    def _is_valid_slice(self, neuron_slice: slice, time_slice: slice) -> bool:
        return (
            neuron_slice.start >= 0
            and neuron_slice.stop <= self.stim_widget.n_neurons
            and time_slice.start >= 0
            and time_slice.stop <= self.stim_widget.n_timestamps
        )

    def _add_actions(self, toolbar: QtWidgets.QToolBar, menu: QtWidgets.QMenu) -> None:
        self.action_toggle_draw = add_action(
            "Toggle draw mode",
            self.toggle_draw_stims,
            menu,
            toolbar,
            self,
            shortcut="Ctrl+D",
            icon=qta.icon("mdi6.cursor-default-gesture"),
            checkable=True,
            is_checked=False,
        )
        self.toolbar.addSeparator()
        self.action_undo = add_action(
            "Undo",
            self.undo,
            menu,
            toolbar,
            self,
            shortcut="Ctrl+Z",
            icon=qta.icon("mdi6.undo"),
        )
        self.action_undo.setEnabled(False)
        self.action_redo = add_action(
            "Redo",
            self.redo,
            menu,
            toolbar,
            self,
            shortcut="Ctrl+Shift+Z",
            icon=qta.icon("mdi6.redo"),
        )
        self.action_redo.setEnabled(False)
        self.toolbar.addSeparator()
        self.actions_draw = []
        self.actions_draw.append(
            add_action(
                "Draw stimulus region",
                lambda: self.draw_mode_updated(DrawMode.REGION),
                menu,
                toolbar,
                self,
                icon=qta.icon("mdi6.select"),
                checkable=True,
                is_checked=True,
            )
        )
        self.actions_draw.append(
            add_action(
                "Draw single 1ms stimulus",
                lambda: self.draw_mode_updated(DrawMode.SINGLE_STIM),
                menu,
                toolbar,
                self,
                icon=qta.icon("mdi6.pencil"),
                checkable=True,
                is_checked=False,
            )
        )
        self.actions_draw.append(
            add_action(
                "Erase single 1ms stimulus",
                lambda: self.draw_mode_updated(DrawMode.SINGLE_STIM_ERASE),
                menu,
                toolbar,
                self,
                icon=qta.icon("mdi6.eraser"),
                checkable=True,
                is_checked=False,
            )
        )
        self.actions_draw_group = QtWidgets.QActionGroup(self)
        self.actions_draw_group.setExclusive(True)
        self.toolbar.addSeparator()
        for action in self.actions_draw:
            self.actions_draw_group.addAction(action)
            action.setEnabled(False)
        self.actions_edit = []
        self.actions_edit.append(
            add_action(
                "Select all",
                lambda: self.stim_widget.select_neurons(None, None),
                menu,
                toolbar,
                self,
                shortcut="Ctrl+A",
                icon=qta.icon("mdi6.select-all"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Copy",
                self.copy_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+C",
                icon=qta.icon("mdi6.content-copy"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Cut",
                self.cut_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+X",
                icon=qta.icon("mdi6.content-cut"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Paste",
                self.paste_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+V",
                icon=qta.icon("mdi6.content-paste"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Paste inverted",
                lambda: self.paste_selected_stim(invert=True),
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Shift+V",
                icon=qta.icon("fa5s.paste"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Save to file",
                self.save_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Shift+S",
                icon=qta.icon("mdi6.application-export"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Load from file",
                self.load_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Shift+O",
                icon=qta.icon("mdi6.application-import"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Delete",
                self.delete_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Delete",
                icon=qta.icon("mdi6.delete"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Move up",
                self.move_selected_stim_up,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Up",
                icon=qta.icon("mdi6.arrow-up"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Move down",
                self.move_selected_stim_down,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Down",
                icon=qta.icon("mdi6.arrow-down"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Move left",
                self.move_selected_stim_left,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Left",
                icon=qta.icon("mdi6.arrow-left"),
            )
        )
        self.actions_edit.append(
            add_action(
                "Move right",
                self.move_selected_stim_right,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Right",
                icon=qta.icon("mdi6.arrow-right"),
            )
        )
