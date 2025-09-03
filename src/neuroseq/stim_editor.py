from __future__ import annotations

import logging
from enum import Enum

import brian2 as b2
import brian2.numpy_ as np
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq.b2_model import B2Model
from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.stim_widget import StimWidget
from neuroseq.timed_array_modifer import TimedArrayModifier
from neuroseq.utils import InteractMode
from neuroseq.utils import add_action
from neuroseq.utils import timed_array_with_new_data


class DrawMode(Enum):
    DRAW = 0
    ERASE = 1


class SelectMode(Enum):
    SELECT = 0
    PASTE = 1
    PASTE_INVERTED = 2


class StimEditor(QtWidgets.QWidget):
    """A GUI to edit 2-d numpy arrays of neuron stimuli interactively using mouse and keyboard input."""

    def __init__(
        self,
        controller: ConstrainedPanZoomController,
        cmb_stim: QtWidgets.QComboBox,
        toolbar: QtWidgets.QToolBar,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.toolbar = toolbar
        self.model: B2Model | None = None
        self.cmb_stim = cmb_stim
        self.current_stim_modifier: TimedArrayModifier | None = None
        self.mode = InteractMode.DRAW
        self.draw_mode = DrawMode.DRAW
        self.select_mode = SelectMode.SELECT
        self.draw_stim_value = 1.0
        self.draw_stim_length = 1  # in milliseconds
        self.stims_dict: dict[str, TimedArrayModifier] = {}
        self.copy_values = np.zeros((0, 0))
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.context_menu = QtWidgets.QMenu("Actions", self)
        self._add_actions(self.toolbar, self.context_menu)
        self.stim_widget = StimWidget(controller=controller, menu=self.context_menu)
        layout.addWidget(self.stim_widget, 0, 0, 3, 3)
        # y scrollbar and zoom buttons
        self.y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        controller.connect_y_scroll_bar(self.y_scrollbar)
        layout.addWidget(self.y_scrollbar, 0, 3)
        self.y_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.y_btn_zoom_in.setFixedSize(14, 24)
        self.y_btn_zoom_in.setToolTip("Zoom in (Shift + Mouse Wheel Up)")
        self.y_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.0, 0.5), ()))
        layout.addWidget(self.y_btn_zoom_in, 1, 3)
        self.stim_widget.set_mode(self.mode)
        self.y_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.y_btn_zoom_out.setFixedSize(14, 24)
        self.y_btn_zoom_out.setToolTip("Zoom out (Shift + Mouse Wheel Down)")
        self.y_btn_zoom_out.clicked.connect(lambda: controller.zoom((0.0, -0.5), ()))
        layout.addWidget(self.y_btn_zoom_out, 2, 3)
        # x scrollbar and zoom buttons
        self.x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        controller.connect_x_scroll_bar(self.x_scrollbar)
        layout.addWidget(self.x_scrollbar, 3, 0)
        self.x_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.x_btn_zoom_out.setFixedSize(24, 14)
        self.x_btn_zoom_out.setToolTip("Zoom out (Ctrl + Mouse Wheel Down)")
        self.x_btn_zoom_out.clicked.connect(lambda: controller.zoom((-0.5, 0.0), ()))
        layout.addWidget(self.x_btn_zoom_out, 3, 1)
        self.x_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.x_btn_zoom_in.setFixedSize(24, 14)
        self.x_btn_zoom_in.setToolTip("Zoom in (Ctrl + Mouse Wheel Up)")
        self.x_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.5, 0.0), ()))
        layout.addWidget(self.x_btn_zoom_in, 3, 2)
        self.setLayout(layout)
        self.cmb_stim.currentTextChanged.connect(self.select_stim)
        self.stim_widget.mouse_clicked.connect(self.stim_widget_mouse_clicked)
        self.stim_widget.mouse_dragged.connect(self.stim_widget_mouse_dragged)
        self.stim_widget.escape_pressed.connect(self.cancel_paste_selected_stim)
        self.resize(800, 600)

    def set_model(self, model: B2Model) -> None:
        self.model = model
        self.stims_dict = {}
        if self.model is not None and self.model.parameters is not None:
            self.stims_dict = {
                k: TimedArrayModifier(v, self.stim_widget)
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

    def select_stim(self, stim_name: str) -> None:
        self.logger.debug(f"Selected stim: {stim_name}")
        if (
            stim_name in self.stims_dict
            and self.current_stim_modifier != self.stims_dict[stim_name]
        ):
            self.current_stim_modifier = self.stims_dict[stim_name]
            self.stim_widget.set_data(self.current_stim_modifier.timed_array)
            self.set_mode(InteractMode.DRAW)
            self.set_draw_mode(DrawMode.DRAW)
            self._stim_values_updated()

    def set_mode(self, mode: InteractMode) -> None:
        self.mode = mode
        self.stim_widget.set_mode(mode)
        if self.mode == InteractMode.DRAW:
            self.action_draw_mode.setChecked(True)
            self.action_select_mode.setChecked(False)
            self.stim_widget.set_hover_slice(
                self.draw_stim_value * np.ones((self.draw_stim_length, 1))
            )
        elif self.mode == InteractMode.SELECT:
            self.action_draw_mode.setChecked(False)
            self.action_select_mode.setChecked(True)
            self.stim_widget.set_hover_slice(None)
        for action in self.actions_draw:
            action.setEnabled(self.mode == InteractMode.DRAW)
        for action in self.actions_edit:
            action.setEnabled(self.mode == InteractMode.SELECT)

    def set_draw_mode(self, draw_mode: DrawMode) -> None:
        self.draw_mode = draw_mode

    def draw_stim_region(
        self, neuron_slice: slice, time_slice: slice, erase=False
    ) -> None:
        if self.current_stim_modifier is None:
            return
        self.logger.debug(f"Draw stim for neurons {neuron_slice} and time {time_slice}")
        stim_value = 0 if erase else self.draw_stim_value
        time_slice = slice(time_slice.start, time_slice.stop)
        if not self._is_valid_slice(neuron_slice, time_slice):
            return
        self.current_stim_modifier.do(neuron_slice, time_slice, stim_value)
        self._stim_values_updated()

    def stim_widget_mouse_clicked(
        self, neuron_index: int, time_index: int, button_index: int
    ) -> None:
        self.logger.debug(
            f"Mouse clicked: neuron {neuron_index}, time {time_index}, button {button_index}"
        )
        if self.current_stim_modifier is None:
            return
        if self.mode == InteractMode.SELECT:
            if (
                self.select_mode == SelectMode.PASTE
                or self.select_mode == SelectMode.PASTE_INVERTED
            ):
                self.logger.debug(
                    f"Paste stim at neuron {neuron_index}, time {time_index}"
                )
                self.paste_selected_stim(neuron_index, time_index)
            else:
                self.stim_widget.unselect_neurons()
            return

    def stim_widget_mouse_dragged(
        self,
        initial_neuron_index: int,
        current_neuron_index: int,
        initial_time_index: int,
        current_time_index: int,
        button_index: int,
    ) -> None:
        self.logger.debug(
            f"Mouse dragged from neuron {initial_neuron_index}, time index {initial_time_index} to neuron {current_neuron_index}, time index {current_time_index}, with button: {button_index}"
        )
        if self.current_stim_modifier is None or button_index > 2:
            return
        erase = (self.draw_mode == DrawMode.DRAW and button_index == 2) or (
            self.draw_mode == DrawMode.ERASE and button_index == 1
        )
        if erase:
            # when erasing use the current neuron index
            neuron_slice = slice(current_neuron_index, current_neuron_index + 1)
            if current_neuron_index == initial_neuron_index:
                # if we are still on the initial neuron index, erase all neurons between the two time indices
                time_slice = slice(
                    min(initial_time_index, current_time_index),
                    max(initial_time_index, current_time_index) + self.draw_stim_length,
                )
            else:
                # otherwise just erase the current time index
                time_slice = slice(
                    current_time_index, current_time_index + self.draw_stim_length
                )
        else:
            # when drawing keep the neuron index fixed to the initial neuron index
            neuron_slice = slice(initial_neuron_index, initial_neuron_index + 1)
            # and draw between the two time indices
            time_slice = slice(
                min(initial_time_index, current_time_index),
                max(initial_time_index, current_time_index) + self.draw_stim_length,
            )
        if not self._is_valid_slice(neuron_slice, time_slice):
            return
        if self.mode == InteractMode.DRAW:
            self.draw_stim_region(
                neuron_slice,
                time_slice,
                erase=erase,
            )

    def copy_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        self.copy_values = self.current_stim_modifier.timed_array.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ].copy()

    def cut_selected_stim(self) -> None:
        self.copy_selected_stim()
        self.delete_selected_stim()

    def init_paste_selected_stim(self, invert=False) -> None:
        """Initialize the paste operation by setting the hover slice to the copied values."""
        if self.copy_values.shape[0] == 0 or self.copy_values.shape[1] == 0:
            return
        self.stim_widget.set_hover_slice(self.copy_values)
        self.stim_widget.set_mode(InteractMode.DRAW)
        self.select_mode = SelectMode.PASTE_INVERTED if invert else SelectMode.PASTE

    def cancel_paste_selected_stim(self) -> None:
        if self.select_mode not in (SelectMode.PASTE, SelectMode.PASTE_INVERTED):
            return
        self.stim_widget.set_hover_slice(None)
        self.stim_widget.set_mode(InteractMode.SELECT)
        self.select_mode = SelectMode.SELECT

    def paste_selected_stim(self, neuron_index: int, time_index: int) -> None:
        if self.current_stim_modifier is None:
            return
        new_time_slice = slice(time_index, time_index + self.copy_values.shape[0])
        new_neuron_slice = slice(neuron_index, neuron_index + self.copy_values.shape[1])
        self.logger.debug(
            f"Paste stim with shape {self.copy_values.shape} at neuron {neuron_index}, time {time_index}"
        )
        if not self._is_valid_slice(new_neuron_slice, new_time_slice):
            return
        if self.select_mode == SelectMode.PASTE_INVERTED:
            max_value = np.max(self.copy_values)
            if max_value == 0:
                max_value = self.draw_stim_value
            new_values = max_value - self.copy_values
        else:
            new_values = self.copy_values
        self.current_stim_modifier.do(new_neuron_slice, new_time_slice, new_values)
        self.stim_widget.set_hover_slice(None)
        self.stim_widget.set_mode(InteractMode.SELECT)
        self.select_mode = SelectMode.SELECT
        self._stim_values_updated()

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
            self.copy_values = arr
            self.init_paste_selected_stim()

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

    def choose_stim_length(self) -> None:
        new_stim_length, ok = QtWidgets.QInputDialog.getText(
            self,
            "Stimulus length",
            "Length of stimulus in milliseconds",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.draw_stim_length}",
        )
        if not ok:
            return
        try:
            self.draw_stim_length = max(1, int(new_stim_length))
            self.set_mode(self.mode)  # Reset mode to update hover slice
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse '{new_stim_length}' as a number of milliseconds.",
            )
            return
        self.logger.debug(f"Setting draw stim length to {self.draw_stim_length}ms")

    def _is_valid_slice(self, neuron_slice: slice, time_slice: slice) -> bool:
        return (
            neuron_slice.start >= 0
            and neuron_slice.stop <= self.stim_widget.n_neurons
            and time_slice.start >= 0
            and time_slice.stop <= self.stim_widget.n_timestamps
        )

    def _add_actions(self, toolbar: QtWidgets.QToolBar, menu: QtWidgets.QMenu) -> None:
        self.action_draw_mode = add_action(
            "Draw mode",
            lambda: self.set_mode(InteractMode.DRAW),
            menu,
            toolbar,
            self,
            shortcut="1",
            icon=qta.icon("mdi6.cursor-default-gesture"),
            checkable=True,
            is_checked=True,
        )
        self.action_select_mode = add_action(
            "Select mode",
            lambda: self.set_mode(InteractMode.SELECT),
            menu,
            toolbar,
            self,
            shortcut="2",
            icon=qta.icon("mdi6.select"),
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
                "Draw stimulus",
                lambda: self.set_draw_mode(DrawMode.DRAW),
                menu,
                toolbar,
                self,
                icon=qta.icon("mdi6.pencil"),
                checkable=True,
                is_checked=True,
            )
        )
        self.actions_draw.append(
            add_action(
                "Erase stimulus",
                lambda: self.set_draw_mode(DrawMode.ERASE),
                menu,
                toolbar,
                self,
                icon=qta.icon("mdi6.eraser"),
                checkable=True,
                is_checked=False,
            )
        )
        self.action_stim_length = add_action(
            "Modify stimulus length",
            self.choose_stim_length,
            menu,
            toolbar,
            self,
            icon=qta.icon("mdi6.arrow-expand-horizontal"),
        )
        self.actions_draw.append(self.action_stim_length)
        self.actions_draw_group = QtWidgets.QActionGroup(self)
        self.actions_draw_group.setExclusive(True)
        self.toolbar.addSeparator()
        for action in self.actions_draw:
            self.actions_draw_group.addAction(action)
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
                self.init_paste_selected_stim,
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
                lambda: self.init_paste_selected_stim(invert=True),
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
                "Ctrl+Delete",
                self.delete_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Delete",
                icon=qta.icon("mdi6.delete"),
            )
        )
        for action in self.actions_edit:
            action.setEnabled(False)
