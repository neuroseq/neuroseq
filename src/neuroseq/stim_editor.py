from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Literal
from typing import overload

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename
from qtpy.compat import getsavefilename

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.dynamic_input_routine import DynamicInputDialog
from neuroseq.project import Project
from neuroseq.rheobase_dialog import RheobaseDialog
from neuroseq.routine_overview_dialog import RoutineSelectorDialog
from neuroseq.stdp_synaptic_tuning_dialog import StdpSynapticTuningDialog
from neuroseq.stdp_window_dialog import StdpWindowDialog
from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_array_modifier import PadReversibleAction
from neuroseq.stim_array_modifier import ResizableReversibleAction
from neuroseq.stim_array_modifier import StimArrayModifier
from neuroseq.stim_array_modifier import TrimReversibleAction
from neuroseq.stim_dialog import EditStimDialog
from neuroseq.stim_intensity_dialog import StimIntensityDialog
from neuroseq.stim_loop_dialog import StimLoopDialog
from neuroseq.stim_pattern_dialog import StimPatternContext
from neuroseq.stim_pattern_dialog import StimPatternDialog
from neuroseq.stim_timing import DEFAULT_STIM_DT_SECONDS
from neuroseq.stim_timing import format_milliseconds
from neuroseq.stim_timing import samples_to_milliseconds
from neuroseq.stim_timing import samples_to_seconds
from neuroseq.stim_timing import seconds_to_samples_ceil
from neuroseq.stim_timing import seconds_to_samples_nearest
from neuroseq.stim_timing import stim_dt_matches
from neuroseq.stim_widget import StimWidget
from neuroseq.utils import InteractMode
from neuroseq.utils import add_action
from neuroseq.utils import get_stim_type


class DrawMode(Enum):
    DRAW = 0
    ERASE = 1


class SelectMode(Enum):
    SELECT = 0
    PASTE = 1
    PASTE_INVERTED = 2


class StimEditor(QtWidgets.QWidget):
    """A GUI to edit 2-d numpy arrays of neuron stimuli interactively using mouse and keyboard input."""

    limits_modified = QtCore.Signal()
    simulation_requested = QtCore.Signal(float)
    time_range_focus_requested = QtCore.Signal(float, float)
    current_stim_changed = QtCore.Signal(object)  # StimArray | None
    trim_to_mapped_neurons_requested = QtCore.Signal()

    # Dynamic-input routines offered when the user opens the designer.
    STDP_SYNAPTIC_TUNING_ROUTINE = "STDP synaptic tuning"
    RHEOBASE_ROUTINE = "Minimal input for spiking (rheobase)"
    STDP_WINDOW_ROUTINE = "STDP timing window (LTP/LTD vs Δt)"
    DYNAMIC_INPUT_ROUTINES = (
        STDP_SYNAPTIC_TUNING_ROUTINE,
        RHEOBASE_ROUTINE,
        STDP_WINDOW_ROUTINE,
    )

    def __init__(
        self,
        controller: ConstrainedPanZoomController,
        cmb_stim: QtWidgets.QComboBox,
        toolbar: QtWidgets.QToolBar,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self.toolbar = toolbar
        self.project: Project | None = None
        self.cmb_stim = cmb_stim
        self.current_stim_modifier: StimArrayModifier | None = None
        self.mode = InteractMode.DRAW
        self.draw_mode = DrawMode.DRAW
        self.select_mode = SelectMode.SELECT
        self.draw_stim = DrawStim(DrawStimType.Constant, 1.0, 1.0)
        self.draw_stim_length = 1  # in samples
        self.pad_stim_array_secs = 5.0
        self.trim_stim_array_secs = 5.0
        self.draw_stim_units = ""
        self.stim_type = StimType.Excitation
        self.stims_dict: dict[str, StimArrayModifier] = {}
        self.copy_values = np.zeros((0, 0))
        self._pattern_dialog: StimPatternDialog | None = None
        self._stdp_synaptic_tuning_dialog: StdpSynapticTuningDialog | None = None
        self._rheobase_dialog: RheobaseDialog | None = None
        self._stdp_window_dialog: StdpWindowDialog | None = None
        self._routine_selector_dialog: RoutineSelectorDialog | None = None
        self._dynamic_input_theme = "light"
        # Captures the time index the user clicked at, so a Ramp drag can
        # use it as the ramp's left edge for every subsequent mouse_dragged
        # emit (the signal's `initial_time_index` is the *previous* position
        # during a drag, not the original click).
        self._draw_drag_origin_time_index: int | None = None
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.context_menu = QtWidgets.QMenu("Actions", self)
        self._add_actions(self.toolbar, self.context_menu)
        self.stim_widget = StimWidget(controller=controller, menu=self.context_menu)
        self.stim_widget.selection_changed.connect(self._update_amplitude_action)
        layout.addWidget(self.stim_widget, 0, 0, 3, 3)
        # y scrollbar and zoom buttons
        self.y_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        controller.connect_y_scroll_bar(self.y_scrollbar)
        layout.addWidget(self.y_scrollbar, 0, 3)
        self.y_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.y_btn_zoom_in.setFixedSize(14, 24)
        self.y_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_in.setToolTip("Zoom in (Shift + Mouse Wheel Up)")
        self.y_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.0, 0.5), ()))
        layout.addWidget(self.y_btn_zoom_in, 1, 3)
        self.stim_widget.set_mode(self.mode)
        self.y_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.y_btn_zoom_out.setFixedSize(14, 24)
        self.y_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.y_btn_zoom_out.setToolTip("Zoom out (Shift + Mouse Wheel Down)")
        self.y_btn_zoom_out.clicked.connect(lambda: controller.zoom((0.0, -0.5), ()))
        layout.addWidget(self.y_btn_zoom_out, 2, 3)
        # x scrollbar and zoom buttons
        self.x_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        controller.connect_x_scroll_bar(self.x_scrollbar)
        layout.addWidget(self.x_scrollbar, 3, 0)
        self.x_btn_zoom_out = QtWidgets.QPushButton("-", self)
        self.x_btn_zoom_out.setFixedSize(24, 14)
        self.x_btn_zoom_out.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_out.setToolTip("Zoom out (Ctrl + Mouse Wheel Down)")
        self.x_btn_zoom_out.clicked.connect(lambda: controller.zoom((-0.5, 0.0), ()))
        layout.addWidget(self.x_btn_zoom_out, 3, 1)
        self.x_btn_zoom_in = QtWidgets.QPushButton("+", self)
        self.x_btn_zoom_in.setFixedSize(24, 14)
        self.x_btn_zoom_in.setStyleSheet("QPushButton { padding: 0px; }")
        self.x_btn_zoom_in.setToolTip("Zoom in (Ctrl + Mouse Wheel Up)")
        self.x_btn_zoom_in.clicked.connect(lambda: controller.zoom((0.5, 0.0), ()))
        layout.addWidget(self.x_btn_zoom_in, 3, 2)
        self.setLayout(layout)
        self.cmb_stim.currentTextChanged.connect(self.select_stim)
        self.stim_widget.mouse_clicked.connect(self.stim_widget_mouse_clicked)
        self.stim_widget.mouse_dragged.connect(self.stim_widget_mouse_dragged)
        self.stim_widget.escape_pressed.connect(self.cancel_paste_selected_stim)
        self.resize(800, 600)

    def closeEvent(self, event):
        # explicitly close the fastplotlib figure to avoid the renderer trying
        # to access it after Qt has destroyed this container (same pattern as
        # Gui.closeEvent, DataWidget.closeEvent, CellOverlayWidget.closeEvent).
        self.stim_widget.close()
        super().closeEvent(event)

    def set_project(
        self, project: Project, keep_undo_history=False, stim_to_select=""
    ) -> None:
        self.project = project
        self.project.update_limits()
        if self.current_stim_modifier is not None and keep_undo_history:
            for k in self.stims_dict:
                if k in self.project.stim_arrays:
                    self.logger.debug(
                        f"Updating StimArray for stim '{k}' without clearing undo history"
                    )
                    self.stims_dict[k].stim_array = self.project.stim_arrays[k]
            self.current_stim_modifier = None
            self.select_stim(self.cmb_stim.currentText())
            return
        self.stims_dict = {}
        if self.project is not None and self.project.stim_arrays is not None:
            self.stims_dict = {
                k: StimArrayModifier(v, self.stim_widget)
                for k, v in self.project.stim_arrays.items()
            }
        have_stims = self.project is not None and len(self.stims_dict) > 0
        self.cmb_stim.clear()
        self.stim_widget.clear_data()
        self.stim_widget.setEnabled(have_stims)
        self.cmb_stim.setEnabled(have_stims)
        self.controller.set_limits(
            float(self.project.max_time_secs),
            max(1, int(self.project.max_neurons)),
        )
        if not have_stims:
            self.current_stim_modifier = None
            self.current_stim_changed.emit(None)
            return
        self.limits_modified.emit()
        self.cmb_stim.addItems(list(self.stims_dict.keys()))
        if stim_to_select in self.stims_dict:
            self.cmb_stim.setCurrentText(stim_to_select)
        else:
            self.cmb_stim.setCurrentIndex(0)
        self.select_stim(self.cmb_stim.currentText())

    def update_stim_values_dict(self, stims_values_dict: dict[str, np.ndarray]) -> None:
        if self.project is None or self.project.stim_arrays is None:
            return
        for name, array_values in stims_values_dict.items():
            if name not in self.project.stim_arrays:
                self.logger.warning(f"Stimuli '{name}' not found in project: ignoring")
            else:
                self.project.stim_arrays[name].values = array_values
        self.set_project(self.project)

    def replace_stim_values_with_undo(
        self,
        stim_name: str,
        new_values: np.ndarray,
    ) -> None:
        """Replace one stim array content via the undoable modifier pathway."""
        if stim_name not in self.stims_dict:
            raise ValueError(f"Input '{stim_name}' not found in stim editor.")
        if self.project is None or stim_name not in self.project.stim_arrays:
            raise ValueError(f"Input '{stim_name}' not found in project.")

        modifier = self.stims_dict[stim_name]
        target_values = modifier.stim_array.values
        if target_values.shape != new_values.shape:
            raise ValueError(
                "Input replacement requires matching shape for undo support: "
                f"current={target_values.shape}, new={new_values.shape}"
            )

        previous_selected_stim = self.cmb_stim.currentText()
        if self.current_stim_modifier is not modifier:
            self.cmb_stim.setCurrentText(stim_name)
        if self.current_stim_modifier is None:
            raise ValueError("No selected input is available for modification.")

        n_timepoints, n_neurons = target_values.shape
        self.current_stim_modifier.do(
            slice(0, n_neurons),
            slice(0, n_timepoints),
            np.array(new_values, copy=True),
        )
        self._stim_data_modified()

        if (
            previous_selected_stim
            and previous_selected_stim in self.stims_dict
            and previous_selected_stim != stim_name
        ):
            self.cmb_stim.setCurrentText(previous_selected_stim)

    def pad_stim_array(self) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        padding_seconds, ok = QtWidgets.QInputDialog.getText(
            self,
            "Pad input arrays",
            "Padding to add to input arrays in seconds",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{self.pad_stim_array_secs}",
        )
        if not ok:
            return
        try:
            paddings_seconds_as_float = float(padding_seconds)
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse '{padding_seconds}' as a number of seconds.",
            )
            return
        if paddings_seconds_as_float <= 0.0:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                "Padding seconds must be positive.",
            )
            return
        self.pad_stim_array_secs = paddings_seconds_as_float
        stim_array = self.current_stim_modifier.stim_array
        current_length_secs = stim_array.dt * stim_array.values.shape[0]
        try:
            self._apply_pad(current_length_secs + self.pad_stim_array_secs)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error padding input arrays: '{e}'",
            )
            return

    def _apply_pad(self, total_length_secs: float) -> None:
        """Pad every stim array up to ``total_length_secs`` as one undoable step.

        Like :meth:`_apply_trim`, this is routed through the modifier's undo
        stack (rather than clearing it) so Ctrl+Z removes the padding again --
        and, because the stim array is the authoritative timeline, growing it
        re-exposes any simulated data a prior trim had windowed out.
        """
        if self.current_stim_modifier is None or self.project is None:
            return
        action = PadReversibleAction(
            self.project,
            total_length_secs,
            self.stim_widget,
            self.current_stim_modifier.stim_array.name,
        )
        self._push_timeline_action(action)

    def trim_stim_array(self) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        trim_seconds, ok = QtWidgets.QInputDialog.getText(
            self,
            "Trim input arrays",
            "Seconds to trim from the input arrays",
            QtWidgets.QLineEdit.EchoMode.Normal,
            "5.0",
        )
        if not ok:
            return
        try:
            trim_seconds_as_float = float(trim_seconds)
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Could not parse '{trim_seconds}' as a number of seconds.",
            )
            return
        if trim_seconds_as_float <= 0.0:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                "Seconds to trim must be positive.",
            )
            return
        stim_array = self.current_stim_modifier.stim_array
        current_length_secs = stim_array.dt * stim_array.values.shape[0]
        new_length_secs = current_length_secs - trim_seconds_as_float
        if new_length_secs <= 0.0:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid input",
                f"Trim seconds {trim_seconds_as_float} is greater than or equal to current input length {current_length_secs}.",
            )
            return
        caveats: list[str] = []
        if self.project.trim_stim_array_would_delete_data(new_length_secs):
            caveats.append("• Trimming would delete non-zero input data.")
        if self.project.b2_model.get_sim_time_secs() > new_length_secs:
            # The trim cuts into the recorded simulation. Trimming only windows
            # the traces (see Project.get_data); it does not shorten the run,
            # so a later "Continue" resumes from the full untrimmed state.
            caveats.append(
                "• A simulated trace extends past the new length. Trimming only "
                "hides the simulated data beyond it (undo restores it) — it does "
                "not shorten the simulation. Continuing the simulation resumes "
                "from the full, untrimmed end of the last run, not the trimmed "
                "point, and re-extends the input so the hidden data reappears."
            )
        if caveats:
            continue_with_trim = QtWidgets.QMessageBox.question(
                self,
                "Trim input arrays",
                "Trim anyway?\n\n" + "\n\n".join(caveats),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if continue_with_trim != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.trim_stim_array_secs = trim_seconds_as_float
        self._apply_trim(new_length_secs)

    def _apply_trim(
        self, new_length_secs: float, only_stim_name: str | None = None
    ) -> None:
        """Trim stim arrays to ``new_length_secs`` as one undoable step.

        Trimming is routed through the modifier's undo stack (rather than
        clearing it) so Ctrl+Z restores the stim length -- and, because the
        stim array is the authoritative timeline, the simulated data that was
        windowed out by the trim reappears with it. ``only_stim_name`` limits
        the trim to a single array (used by :meth:`reset_stim_array`).
        """
        if self.current_stim_modifier is None or self.project is None:
            return
        action = TrimReversibleAction(
            self.project,
            new_length_secs,
            self.stim_widget,
            self.current_stim_modifier.stim_array.name,
            only_stim_name=only_stim_name,
        )
        self._push_timeline_action(action)

    def reset_stim_array(self) -> None:
        """Trim the current stim array back to the simulation head.

        Design routines append their probe/pairing blocks after the end of the
        current stim array, so a long left-over array makes each run simulate a
        huge empty tail for nothing. Resetting trims the array to the current
        simulation time (or to zero when nothing has run yet), so blocks are
        appended right after the simulated state. It is undoable and never cuts
        into the recorded simulation (it only removes the unused trailing tail).
        """
        if self.current_stim_modifier is None or self.project is None:
            return
        stim = self.current_stim_modifier.stim_array
        current_secs = float(stim.dt) * stim.values.shape[0]
        # Keep at least one sample so the array never becomes empty (an empty
        # stim would break the widget's min/max scaling); when nothing has run
        # this collapses the array to a single baseline sample.
        target_secs = max(self.project.b2_model.get_sim_time_secs(), float(stim.dt))
        if current_secs <= target_secs + float(stim.dt) * 0.5:
            return  # already at (or within a sample of) the simulation head
        self._apply_trim(target_secs, only_stim_name=stim.name)

    def current_stim_array_has_unsimulated_tail(self) -> bool:
        """Would reset_stim_array() actually discard nonzero content?

        Every design routine calls reset_stim_array() right before it
        starts. Content at or before the simulation head is never touched
        by that trim, so stacking routines back to back - each appending
        after the previous one's already-simulated output - never loses
        anything and needs no confirmation. The one case that does is
        nonzero content *past* the simulation head (a manually drawn
        pattern, or a routine stopped mid-run): reset_stim_array() would
        silently drop it.
        """
        if self.current_stim_modifier is None or self.project is None:
            return False
        stim = self.current_stim_modifier.stim_array
        target_secs = max(self.project.b2_model.get_sim_time_secs(), float(stim.dt))
        n_keep = seconds_to_samples_nearest(target_secs, float(stim.dt))
        return bool(np.any(stim.values[n_keep:, :]))

    def _push_timeline_action(
        self, action: TrimReversibleAction | PadReversibleAction
    ) -> None:
        """Run a whole-timeline resize action and push it onto the undo stack.

        Trim and pad both resize every stim array at once, so this applies the
        action on the current modifier's stack (preserving undo history) and
        clears the other per-array stacks, whose indices no longer line up with
        the shared timeline (as ``apply_pattern`` does for the same reason).
        """
        if self.current_stim_modifier is None or self.project is None:
            return
        modifier = self.current_stim_modifier
        modifier.action_stack = modifier.action_stack[: modifier.current_index + 1]
        modifier.action_stack.append(action)
        action(modifier.stim_array, self.stim_widget)
        modifier.current_index = len(modifier.action_stack) - 1
        for other in self.stims_dict.values():
            if other is not modifier:
                other.action_stack = []
                other.current_index = -1
        self.controller.set_limits(
            float(self.project.max_time_secs),
            max(1, int(self.project.max_neurons)),
        )
        self.limits_modified.emit()
        self._stim_data_modified()

    def loop_stim_array(self) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        dialog = StimLoopDialog(self.project, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self.set_project(self.project, keep_undo_history=False)
            return
        if dialog.loop_count > 1:
            try:
                self.project.loop_stim_arrays(dialog.loop_count)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Error looping input arrays: '{e}'",
                )
                return
        self.set_project(self.project, keep_undo_history=False)

    def _stim_values_updated(self) -> None:
        self._update_amplitude_action()
        if self.current_stim_modifier is None:
            self.action_undo.setEnabled(False)
            self.action_redo.setEnabled(False)
            return
        self.action_undo.setEnabled(self.current_stim_modifier.can_undo())
        self.action_redo.setEnabled(self.current_stim_modifier.can_redo())

    def _stim_data_modified(self) -> None:
        """Call after any real edit to the current stim array's values -

        draw/erase/paste/delete/pad/trim/loop/undo/redo - as opposed to
        _stim_values_updated() alone, which also runs on a plain stim
        *selection* change and must not mark the project dirty.
        """
        if self.project is not None:
            self.project.mark_dirty()
        self._stim_values_updated()

    def select_stim(self, stim_name: str) -> None:
        self.logger.debug(f"Selected stim: {stim_name}")
        if (
            stim_name in self.stims_dict
            and self.current_stim_modifier != self.stims_dict[stim_name]
        ):
            old_dt_seconds = self._current_stim_dt_seconds()
            new_dt_seconds = float(self.stims_dict[stim_name].stim_array.dt)
            old_draw_duration_seconds = samples_to_seconds(
                self.draw_stim_length, old_dt_seconds
            )
            self.draw_stim_length = max(
                1, seconds_to_samples_nearest(old_draw_duration_seconds, new_dt_seconds)
            )
            self.current_stim_modifier = self.stims_dict[stim_name]
            self.stim_type = get_stim_type(stim_name)
            self.draw_stim.draw_stim_type = DrawStimType.Constant
            self.draw_stim.value1 = np.max(self.current_stim_modifier.stim_array.values)
            if self.draw_stim.value1 == 0:
                self.draw_stim.value1 = 1.0
            self.draw_stim.value2 = self.draw_stim.value1
            self.stim_widget.set_data(self.current_stim_modifier.stim_array)
            self.current_stim_changed.emit(self.current_stim_modifier.stim_array)
            self.controller.focus_neuron_range(
                0,
                int(self.current_stim_modifier.stim_array.values.shape[1]),
            )
            for modifier in self.stims_dict.values():
                if modifier is not self.current_stim_modifier:
                    self.stim_widget.set_background_data(modifier.stim_array)
            self.set_mode(InteractMode.DRAW)
            self.set_draw_mode(DrawMode.DRAW)
            self._stim_values_updated()
        pattern_dialog = getattr(self, "_pattern_dialog", None)
        if pattern_dialog is not None and self.current_stim_modifier is not None:
            pattern_dialog.set_context(self._current_pattern_context())

    @overload
    def _current_draw_stim_values(
        self,
        force_array: Literal[True],
        erase: bool = ...,
        length: int | None = ...,
        reverse: bool = ...,
    ) -> np.ndarray: ...
    @overload
    def _current_draw_stim_values(
        self,
        force_array: bool = ...,
        erase: bool = ...,
        length: int | None = ...,
        reverse: bool = ...,
    ) -> np.ndarray | float: ...
    def _current_draw_stim_values(
        self,
        force_array: bool = False,
        erase: bool = False,
        length: int | None = None,
        reverse: bool = False,
    ) -> np.ndarray | float:
        n_samples = self.draw_stim_length if length is None else max(1, length)
        if erase:
            if force_array:
                return np.zeros((n_samples, 1))
            else:
                return 0.0
        match self.draw_stim.draw_stim_type:
            case DrawStimType.Constant:
                if force_array:
                    return np.full((n_samples, 1), self.draw_stim.value1)
                else:
                    return self.draw_stim.value1
            case DrawStimType.Linear:
                start = self.draw_stim.value2 if reverse else self.draw_stim.value1
                stop = self.draw_stim.value1 if reverse else self.draw_stim.value2
                return np.linspace(start, stop, n_samples)[:, np.newaxis]
            case _:
                # Legacy types (Logarithmic / Random) are no longer surfaced
                # via the designer dialog; coerce to a Constant fallback so
                # restored designs that still carry them don't crash.
                if force_array:
                    return np.full((n_samples, 1), self.draw_stim.value1)
                return self.draw_stim.value1

    def _update_amplitude_action(self) -> None:
        modifier = self.current_stim_modifier
        neuron_slice = self.stim_widget.selected_neuron_slice
        time_slice = self.stim_widget.selected_time_slice
        self.action_set_amplitude.setEnabled(
            self.mode == InteractMode.SELECT
            and modifier is not None
            and neuron_slice is not None
            and time_slice is not None
            and modifier.stim_array.values[time_slice, neuron_slice].size > 0
        )

    def set_mode(self, mode: InteractMode) -> None:
        self.mode = mode
        self.stim_widget.set_mode(mode)
        if self.mode == InteractMode.DRAW:
            self.action_draw_mode.setChecked(True)
            self.action_select_mode.setChecked(False)
            self.stim_widget.set_hover_slice(
                self._current_draw_stim_values(force_array=True)
            )
        elif self.mode == InteractMode.SELECT:
            self.action_draw_mode.setChecked(False)
            self.action_select_mode.setChecked(True)
            self.stim_widget.set_hover_slice(None)
        for action in self.actions_draw:
            action.setVisible(self.mode == InteractMode.DRAW)
        for action in self.actions_edit:
            action.setVisible(self.mode == InteractMode.SELECT)
        self._update_amplitude_action()

    def set_draw_mode(self, draw_mode: DrawMode) -> None:
        self.draw_mode = draw_mode

    def _ensure_time_capacity_for_slice(self, time_slice: slice) -> bool:
        if self.current_stim_modifier is None:
            return False
        if time_slice.stop is None:
            return False
        if time_slice.stop <= self.stim_widget.n_timestamps:
            return True
        if self.project is None:
            return False
        stim_name = self.current_stim_modifier.stim_array.name
        required_total_time_seconds = (
            float(self.current_stim_modifier.stim_array.dt) * time_slice.stop
        )
        self.project.pad_stim_arrays(required_total_time_seconds)
        self.set_project(
            self.project,
            keep_undo_history=False,
            stim_to_select=stim_name,
        )
        return time_slice.stop <= self.stim_widget.n_timestamps

    def draw_stim_region(
        self,
        neuron_slice: slice,
        time_slice: slice,
        erase: bool = False,
        reverse: bool = False,
    ) -> None:
        if self.current_stim_modifier is None:
            return
        time_slice = slice(time_slice.start, time_slice.stop)
        if not self._is_valid_slice(neuron_slice, time_slice) and (
            erase or not self._ensure_time_capacity_for_slice(time_slice)
        ):
            return
        if not self._is_valid_slice(neuron_slice, time_slice):
            return
        slice_length = max(1, time_slice.stop - time_slice.start)
        values = self._current_draw_stim_values(
            erase=erase, length=slice_length, reverse=reverse
        )
        self.current_stim_modifier.do(neuron_slice, time_slice, values)
        self._stim_data_modified()

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
        if self.mode == InteractMode.DRAW and button_index <= 2:
            erase = (self.draw_mode == DrawMode.DRAW and button_index == 2) or (
                self.draw_mode == DrawMode.ERASE and button_index == 1
            )
            # A Ramp drag has already committed the final shape from origin
            # → current via the trailing mouse_dragged emit, so the click
            # handler must not stamp a fresh draw_stim_length-wide ramp on
            # top of its right end. Clear the origin marker either way.
            ramp_drag_in_progress = (
                self._draw_drag_origin_time_index is not None
                and not erase
                and self.draw_stim.draw_stim_type == DrawStimType.Linear
            )
            self._draw_drag_origin_time_index = None
            if ramp_drag_in_progress:
                return
            neuron_slice = slice(neuron_index, neuron_index + 1)
            time_slice = slice(time_index, time_index + self.draw_stim_length)
            if not self._is_valid_slice(neuron_slice, time_slice):
                return
            self.draw_stim_region(
                neuron_slice,
                time_slice,
                erase=erase,
            )
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
            # For a Ramp draw, anchor the left edge of the slice to the
            # ORIGINAL click position so successive emits during a drag
            # overwrite one coherent ramp rather than stamping a short
            # ramp between each pair of consecutive sampled positions.
            # The signal's `initial_time_index` is the previous emit's
            # position once a drag is underway, so we capture the origin
            # from the very first emit (where initial == current ==
            # pointer-down position) and reuse it until release.
            if self.draw_stim.draw_stim_type == DrawStimType.Linear:
                if (
                    self._draw_drag_origin_time_index is None
                    or initial_time_index == current_time_index
                ):
                    self._draw_drag_origin_time_index = initial_time_index
                left_anchor = self._draw_drag_origin_time_index
            else:
                left_anchor = initial_time_index
            # and draw between the two time indices
            time_slice = slice(
                min(left_anchor, current_time_index),
                max(left_anchor, current_time_index) + self.draw_stim_length,
            )
        if not self._is_valid_slice(neuron_slice, time_slice):
            return
        # When a Ramp drag goes backward (current < origin), flip the linspace
        # so value1 stays at the user's click position and value2 lands at the
        # release position.
        reverse_ramp = (
            not erase
            and self.draw_stim.draw_stim_type == DrawStimType.Linear
            and self._draw_drag_origin_time_index is not None
            and current_time_index < self._draw_drag_origin_time_index
        )
        if self.mode == InteractMode.DRAW:
            self.draw_stim_region(
                neuron_slice,
                time_slice,
                erase=erase,
                reverse=reverse_ramp,
            )

    def copy_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        self.copy_values = self.current_stim_modifier.stim_array.values[
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
                max_value = max(self.draw_stim.value1, self.draw_stim.value2)
            new_values = max_value - self.copy_values
        else:
            new_values = self.copy_values
        self.current_stim_modifier.do(new_neuron_slice, new_time_slice, new_values)
        self.stim_widget.set_hover_slice(None)
        self.stim_widget.set_mode(InteractMode.SELECT)
        self.select_mode = SelectMode.SELECT
        self._stim_data_modified()

    def undo(self) -> None:
        if self.current_stim_modifier is None:
            return
        self.current_stim_modifier.undo()
        if self.project is not None:
            self.controller.set_limits(
                float(self.project.max_time_secs),
                max(1, int(self.project.max_neurons)),
            )
            self.limits_modified.emit()
        self._stim_data_modified()

    def redo(self) -> None:
        if self.current_stim_modifier is None:
            return
        self.current_stim_modifier.redo()
        if self.project is not None:
            self.controller.set_limits(
                float(self.project.max_time_secs),
                max(1, int(self.project.max_neurons)),
            )
            self.limits_modified.emit()
        self._stim_data_modified()

    def save_selected_stim(self) -> None:
        if (
            self.current_stim_modifier is None
            or self.stim_widget.selected_time_slice is None
            or self.stim_widget.selected_neuron_slice is None
        ):
            return
        filename, _ = getsavefilename(
            self,
            "Save input selection",
            "stim.npy",
            "NumPy npy file (*.npy)",
        )
        if filename == "":
            return
        arr = self.current_stim_modifier.stim_array.values[
            self.stim_widget.selected_time_slice, self.stim_widget.selected_neuron_slice
        ]
        self.logger.info(f"Saving {arr.shape} stim selection to file {filename}")
        try:
            np.save(filename, arr, allow_pickle=False)
        except Exception as e:
            msg = f"Error saving stim selection to file: {e}"
            self.logger.error(msg)
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def load_selected_stim(self) -> None:
        if self.current_stim_modifier is None:
            return
        filename, _ = getopenfilename(
            self,
            "Load input selection",
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
            if arr.shape[0] > self.current_stim_modifier.stim_array.values.shape[0]:
                raise ValueError("Loaded array has too many time points")
            if (
                arr.ndim == 2
                and arr.shape[1] > self.current_stim_modifier.stim_array.values.shape[1]
            ):
                raise ValueError("Loaded array has too many neurons")
            self.copy_values = arr
            self.init_paste_selected_stim()

        except Exception as e:
            msg = f"Error loading stim selection from file: {e}"
            self.logger.error(msg)
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def set_selected_stim_amplitude(self) -> None:
        """Set non-zero samples in the selection to an amplitude, with undo."""
        modifier = self.current_stim_modifier
        neuron_slice = self.stim_widget.selected_neuron_slice
        time_slice = self.stim_widget.selected_time_slice
        if modifier is None or neuron_slice is None or time_slice is None:
            return
        values = modifier.stim_array.values[time_slice, neuron_slice].copy()
        nonzero = values != 0
        if not np.any(nonzero):
            return
        units = modifier.stim_array.units
        amplitude_text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set amplitude",
            f"Amplitude ({units}) for non-zero values in the selected region"
            if units
            else "Amplitude for non-zero values in the selected region",
            QtWidgets.QLineEdit.EchoMode.Normal,
            str(float(values[nonzero][0])),
        )
        if not ok:
            return
        try:
            amplitude = float(amplitude_text)
            if not math.isfinite(amplitude):
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self, "Invalid amplitude", "Enter a finite number for the amplitude."
            )
            return
        if np.all(values[nonzero] == amplitude):
            return
        values[nonzero] = amplitude
        modifier.do(neuron_slice, time_slice, values)
        self._stim_values_updated()

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
        self._stim_data_modified()

    def toggle_show_all_stim_arrays(self) -> None:
        self.stim_widget.set_background_data_visible(
            not self.stim_widget.show_background_graphics
        )

    def choose_stim_intensity(self) -> None:
        dialog = StimIntensityDialog(
            self.draw_stim,
            self.stim_widget.units,
            self.stim_type,
            self.draw_stim_length,
            self._current_stim_dt_seconds(),
            self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.draw_stim = dialog.draw_stim
            self.draw_stim_length = dialog.draw_stim_length
            self.logger.debug(
                "Setting draw stim length to %s samples (%sms)",
                self.draw_stim_length,
                format_milliseconds(self._draw_stim_length_milliseconds()),
            )
            self.set_mode(self.mode)  # Reset mode to update hover slice

    def _is_valid_slice(self, neuron_slice: slice, time_slice: slice) -> bool:
        return (
            neuron_slice.start >= 0
            and neuron_slice.stop <= self.stim_widget.n_neurons_total
            and time_slice.start >= 0
            and time_slice.stop <= self.stim_widget.n_timestamps
        )

    def _current_stim_dt_seconds(self) -> float:
        if self.current_stim_modifier is not None:
            return float(self.current_stim_modifier.stim_array.dt)
        return DEFAULT_STIM_DT_SECONDS

    def _draw_stim_length_milliseconds(self) -> float:
        return samples_to_milliseconds(
            self.draw_stim_length, self._current_stim_dt_seconds()
        )

    def create_stim_array(self):
        if self.project is None:
            return
        stim_type = StimType.Excitation
        n_neurons = 10
        units = "nS"
        dt = self._current_stim_dt_seconds()
        if self.current_stim_modifier is not None:
            stim_type = self.current_stim_modifier.stim_array.stim_type
            n_neurons = self.current_stim_modifier.stim_array.values.shape[1]
            units = self.current_stim_modifier.stim_array.units
        dt = float(dt)
        stim_array = StimArray(
            "new_stim",
            stim_type,
            np.zeros((0, n_neurons)),
            units,
            dt,
        )
        dialog = EditStimDialog(
            stim_array,
            existing_names=list(self.project.stim_arrays.keys()),
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            n_timestamps = seconds_to_samples_ceil(
                float(self.project.max_time_secs), float(dialog.dt)
            )
            self.project.add_stim_array(
                StimArray(
                    dialog.name,
                    dialog.stim_type,
                    np.zeros((n_timestamps, dialog.n_neurons)),
                    dialog.units,
                    dialog.dt,
                )
            )
            self.set_project(self.project, stim_to_select=dialog.name)

    def edit_stim_array(self):
        if self.current_stim_modifier is None:
            return
        old_name = self.current_stim_modifier.stim_array.name
        dialog = EditStimDialog(
            self.current_stim_modifier.stim_array,
            existing_names=[k for k in self.project.stim_arrays if k != old_name],
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            old_draw_duration_seconds = samples_to_seconds(
                self.draw_stim_length, self._current_stim_dt_seconds()
            )
            if not stim_dt_matches(dialog.dt, self.current_stim_modifier.stim_array.dt):
                _other = self.project.remappable_data_description(dialog.dt)
                _extra = (
                    f" All other project data ({_other}) will also be remapped"
                    " using zero-order hold."
                    if _other
                    else ""
                )
                continue_with_remap = QtWidgets.QMessageBox.question(
                    self,
                    "Change project sample interval",
                    f"Changing the input timestep will remap input array "
                    f"'{old_name}' from "
                    f"{format_milliseconds(self.current_stim_modifier.stim_array.dt * 1000.0)} ms "
                    f"to {format_milliseconds(dialog.dt * 1000.0)} ms."
                    f"{_extra} "
                    "Timing may shift by up to one sample. Continue?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                )
                if continue_with_remap != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                self.draw_stim_length = max(
                    1,
                    seconds_to_samples_nearest(old_draw_duration_seconds, dialog.dt),
                )
                self.project.remap_all_data_to_dt(dialog.dt)
            if dialog.name != old_name:
                self.project.rename_stim_array(old_name, dialog.name)
            self.project.update_stim_array_properties(
                dialog.name,
                n_neurons=dialog.n_neurons,
                stim_type=dialog.stim_type,
                units=dialog.units,
                dt=dialog.dt,
            )
            self.set_project(self.project, stim_to_select=dialog.name)

    def delete_stim_array(self):
        if self.current_stim_modifier is None:
            return
        if (
            QtWidgets.QMessageBox.question(
                self,
                "Delete input array",
                f"Are you sure you want to delete input array '{self.current_stim_modifier.stim_array.name}'? (This action cannot be undone.)",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            == QtWidgets.QMessageBox.StandardButton.Yes
        ):
            self.project.delete_stim_array(self.current_stim_modifier.stim_array.name)
            self.set_project(self.project)

    def clear_stim_array(self) -> None:
        if self.current_stim_modifier is None:
            return
        name = self.current_stim_modifier.stim_array.name
        # Parented to whichever top-level window is actually active, not to
        # this widget: the routine dialogs' "Clear input array" button calls
        # straight through to this method, and those dialogs are created
        # without a Qt parent (see _show_routine_dialog's docstring). A box
        # parented to self would hand focus back to the main window once
        # closed, leaving the parentless routine dialog behind it.
        if (
            QtWidgets.QMessageBox.question(
                QtWidgets.QApplication.activeWindow() or self,
                "Clear input array",
                f"Set all values in '{name}' to zero?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        new_values = np.zeros_like(self.current_stim_modifier.stim_array.values)
        self.replace_stim_values_with_undo(name, new_values)
        pattern_dialog = getattr(self, "_pattern_dialog", None)
        if pattern_dialog is not None:
            pattern_dialog.reset_start_time()

    def _current_pattern_context(self) -> StimPatternContext:
        if self.current_stim_modifier is None:
            raise ValueError("No current stim array selected.")
        stim_array = self.current_stim_modifier.stim_array
        return StimPatternContext(
            stim_name=stim_array.name,
            units=stim_array.units,
            dt_seconds=float(stim_array.dt),
            n_neurons=int(stim_array.values.shape[1]),
        )

    def open_pattern_designer(self) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        if self._pattern_dialog is None:
            self._pattern_dialog = StimPatternDialog(self)
        self._pattern_dialog.set_context(self._current_pattern_context())
        self._pattern_dialog.show()
        self._pattern_dialog.raise_()
        self._pattern_dialog.activateWindow()

    def open_dynamic_input_designer(self) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        # A single selector window shows the routine dropdown with an always-on
        # preview (goal, parameters, sample plot) beneath it; the real routine
        # dialog opens only when the user clicks Continue.
        dialog = RoutineSelectorDialog(
            list(self.DYNAMIC_INPUT_ROUTINES), self._dynamic_input_theme
        )
        dialog.routine_chosen.connect(self._open_routine_dialog_for)
        dialog.finished.connect(dialog.deleteLater)
        self._routine_selector_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _open_routine_dialog_for(self, routine: str) -> None:
        if routine == self.STDP_SYNAPTIC_TUNING_ROUTINE:
            self._show_routine_dialog(
                "_stdp_synaptic_tuning_dialog", StdpSynapticTuningDialog
            )
        elif routine == self.RHEOBASE_ROUTINE:
            self._show_routine_dialog("_rheobase_dialog", RheobaseDialog)
        elif routine == self.STDP_WINDOW_ROUTINE:
            self._show_routine_dialog("_stdp_window_dialog", StdpWindowDialog)

    def _show_routine_dialog(
        self,
        attr_name: Literal[
            "_stdp_synaptic_tuning_dialog", "_rheobase_dialog", "_stdp_window_dialog"
        ],
        cls: type[StdpSynapticTuningDialog]
        | type[RheobaseDialog]
        | type[StdpWindowDialog],
    ) -> None:
        dialog = getattr(self, attr_name)
        if dialog is None:
            # No Qt parent: a parent sets WM_TRANSIENT_FOR to the main
            # window, and on X11/KWin the routine dialog and its own two
            # popups (also parentless - see StimHeatmapWindow's
            # construction in stdp_synaptic_tuning_dialog.py) would then all
            # share one owner. KWin can treat that whole chain as a single
            # "raise group" tied to the app's own foreground status, so
            # none of them can be individually brought to the front of the
            # others by clicking - not even this one.
            dialog = cls(self, None)
            dialog.simulation_requested.connect(self.simulation_requested)
            setattr(self, attr_name, dialog)
        dialog.apply_theme(self._dynamic_input_theme)
        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def discard_dynamic_input_dialogs(self) -> None:
        """Drop any cached Design dynamic input dialogs.

        Called when the project is replaced so the designer does not carry a
        previous run's state (plots, log, history) into a new project. Dialogs
        are rebuilt fresh the next time they are opened.
        """
        for attr_name in (
            "_stdp_synaptic_tuning_dialog",
            "_rheobase_dialog",
            "_stdp_window_dialog",
            "_routine_selector_dialog",
        ):
            dialog = getattr(self, attr_name)
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
                setattr(self, attr_name, None)

    def focus_time_range(self, start_seconds: float, stop_seconds: float) -> None:
        """Keep stimulation and linked data views focused on the same interval."""
        self.controller.focus_time_range(start_seconds, stop_seconds)
        self.time_range_focus_requested.emit(start_seconds, stop_seconds)

    def dynamic_input_simulation_finished(self) -> None:
        for dialog in self._dynamic_input_dialogs():
            dialog.simulation_finished()

    def dynamic_input_simulation_failed(self, message: str) -> None:
        for dialog in self._dynamic_input_dialogs():
            dialog.simulation_failed(message)

    def apply_dynamic_input_theme(self, theme: str) -> None:
        self._dynamic_input_theme = theme
        for dialog in self._dynamic_input_dialogs():
            dialog.apply_theme(theme)

    def _dynamic_input_dialogs(
        self,
    ) -> list[DynamicInputDialog]:
        return [
            dialog
            for dialog in (
                self._stdp_synaptic_tuning_dialog,
                self._rheobase_dialog,
                self._stdp_window_dialog,
            )
            if dialog is not None
        ]

    def apply_pattern(
        self,
        neuron_slice: slice | list[int],
        time_slice: slice,
        new_values: np.ndarray,
    ) -> None:
        if self.current_stim_modifier is None or self.project is None:
            return
        modifier = self.current_stim_modifier
        target_name = modifier.stim_array.name
        action = ResizableReversibleAction(
            self.project,
            target_name,
            neuron_slice,
            time_slice,
            new_values,
            self.stim_widget,
        )
        modifier.action_stack = modifier.action_stack[: modifier.current_index + 1]
        modifier.action_stack.append(action)
        action(modifier.stim_array, self.stim_widget)
        modifier.current_index = len(modifier.action_stack) - 1
        for other in self.stims_dict.values():
            if other is not modifier:
                other.action_stack = []
                other.current_index = -1
        self.controller.set_limits(
            float(self.project.max_time_secs),
            max(1, int(self.project.max_neurons)),
        )
        self.limits_modified.emit()
        self._stim_data_modified()

    def _add_actions(self, toolbar: QtWidgets.QToolBar, menu: QtWidgets.QMenu) -> None:
        self.action_undo = add_action(
            "Undo (Ctrl+Z)",
            self.undo,
            menu,
            toolbar,
            self,
            shortcut="Ctrl+Z",
            icon="mdi6.undo",
        )
        self.action_undo.setEnabled(False)
        self.action_redo = add_action(
            "Redo (Ctrl+Shift+Z)",
            self.redo,
            menu,
            toolbar,
            self,
            shortcut="Ctrl+Shift+Z",
            icon="mdi6.redo",
        )
        self.action_redo.setEnabled(False)
        toolbar.addSeparator()
        self.action_draw_mode = add_action(
            "Draw mode (1)",
            lambda: self.set_mode(InteractMode.DRAW),
            menu,
            toolbar,
            self,
            shortcut="1",
            icon="mdi6.cursor-default-gesture",
            checkable=True,
            is_checked=True,
        )
        self.action_select_mode = add_action(
            "Select mode (2)",
            lambda: self.set_mode(InteractMode.SELECT),
            menu,
            toolbar,
            self,
            shortcut="2",
            icon="mdi6.select",
            checkable=True,
            is_checked=False,
        )
        toolbar.addSeparator()
        # Show drawing settings only while Draw mode is active.
        self.actions_draw = []
        self.actions_draw.append(
            add_action(
                "Draw input",
                lambda: self.set_draw_mode(DrawMode.DRAW),
                menu,
                toolbar,
                self,
                icon="mdi6.pencil",
                checkable=True,
                is_checked=True,
            )
        )
        self.actions_draw.append(
            add_action(
                "Erase input",
                lambda: self.set_draw_mode(DrawMode.ERASE),
                menu,
                toolbar,
                self,
                icon="mdi6.eraser",
                checkable=True,
                is_checked=False,
            )
        )
        self.actions_draw.append(
            add_action(
                "Set input value/length…",
                self.choose_stim_intensity,
                menu,
                toolbar,
                self,
                icon="mdi6.network-strength-3",
            )
        )
        self.actions_draw_group = QtWidgets.QActionGroup(self)
        self.actions_draw_group.setExclusive(True)
        for action in self.actions_draw:
            self.actions_draw_group.addAction(action)
        # Show selection settings only while Select mode is active.
        self.actions_edit = []
        self.action_set_amplitude = add_action(
            "Set amplitude to selected input",
            self.set_selected_stim_amplitude,
            menu,
            toolbar,
            self,
            icon="mdi6.tune-vertical",
        )
        self.actions_edit.append(self.action_set_amplitude)
        self.action_set_amplitude.setEnabled(False)
        self.actions_edit.append(
            add_action(
                "Select all (Ctrl+A)",
                lambda: self.stim_widget.select_neurons(None, None),
                menu,
                toolbar,
                self,
                shortcut="Ctrl+A",
                icon="mdi6.select-all",
            )
        )
        self.actions_edit.append(
            add_action(
                "Copy (Ctrl+C)",
                self.copy_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+C",
                icon="mdi6.content-copy",
            )
        )
        self.actions_edit.append(
            add_action(
                "Cut (Ctrl+X)",
                self.cut_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+X",
                icon="mdi6.content-cut",
            )
        )
        self.actions_edit.append(
            add_action(
                "Paste (Ctrl+V)",
                self.init_paste_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Ctrl+V",
                icon="mdi6.content-paste",
            )
        )
        self.actions_edit.append(
            add_action(
                "Paste inverted (Ctrl+Shift+V)",
                lambda: self.init_paste_selected_stim(invert=True),
                menu,
                toolbar,
                self,
                shortcut="Ctrl+Shift+V",
                icon="fa5s.paste",
            )
        )
        self.actions_edit.append(
            add_action(
                "Save to file",
                self.save_selected_stim,
                menu,
                toolbar,
                self,
                icon="mdi6.application-export",
            )
        )
        self.actions_edit.append(
            add_action(
                "Load from file",
                self.load_selected_stim,
                menu,
                toolbar,
                self,
                icon="mdi6.application-import",
            )
        )
        self.actions_edit.append(
            add_action(
                "Delete selected input region",
                self.delete_selected_stim,
                menu,
                toolbar,
                self,
                shortcut="Delete",
                icon="mdi6.delete",
            )
        )
        for action in self.actions_edit:
            action.setVisible(False)
        # Stim array lifecycle operations moved to a single dropdown button
        # next to the stim array combobox in the top Data toolbar (issue
        # #535) - see Gui._add_stim_array_ops_button_to_data_toolbar(), which
        # adds these same actions into that button's menu. They stay
        # registered on `menu` (the "Actions" context menu) and on `self`
        # (for their keyboard shortcuts, where applicable) but not on the
        # left toolbar.
        self.action_create_stim_array = add_action(
            "Create input array",
            self.create_stim_array,
            menu,
            None,
            self,
            icon="mdi6.text-box-plus",
        )
        self.action_edit_stim_array = add_action(
            "Edit input array",
            self.edit_stim_array,
            menu,
            None,
            self,
            icon="mdi6.text-box-edit",
        )
        self.action_delete_stim_array = add_action(
            "Delete input array",
            self.delete_stim_array,
            menu,
            None,
            self,
            icon="mdi6.text-box-remove",
        )
        self.action_clear_stim_array = add_action(
            "Clear input array",
            self.clear_stim_array,
            menu,
            None,
            self,
            icon="mdi6.text-box-minus",
        )
        self.action_trim_to_mapped_neurons = add_action(
            "Trim to Mapped Neurons…",
            self.trim_to_mapped_neurons_requested.emit,
            menu,
            None,
            self,
            icon="mdi6.content-cut",
        )
        self.action_trim_to_mapped_neurons.setEnabled(False)
        self.action_trim_to_mapped_neurons.setToolTip(
            "Permanently keep only the neurons with a Cellpose cell mapped to "
            "them, removing the rest from every input array and experiment in "
            "this project."
        )
        # Pad/Trim/Loop stim array also moved to the stim array ops dropdown
        # in the top Data toolbar (issue #535) - see
        # Gui._add_stim_array_ops_button_to_data_toolbar().
        self.action_pad_stim_array = add_action(
            "Pad input array (Ctrl+Shift+Plus)",
            self.pad_stim_array,
            menu,
            None,
            self,
            shortcut="Ctrl++",
            icon="mdi6.timeline-plus",
        )
        self.action_trim_stim_array = add_action(
            "Trim input array (Ctrl+Shift+Minus)",
            self.trim_stim_array,
            menu,
            None,
            self,
            shortcut="Ctrl+Shift+-",
            icon="mdi6.timeline-minus",
        )
        self.action_loop_stim_array = add_action(
            "Loop input array",
            self.loop_stim_array,
            menu,
            None,
            self,
            icon="mdi6.repeat-variant",
        )
        # Design static/dynamic input and Show all stim arrays also moved out
        # of this toolbar (issue #535): the Design buttons become rectangular
        # icon+text QPushButtons in the top View toolbar
        # (Gui._add_stim_design_buttons_to_view_toolbar), and Show all stim
        # arrays - a genuine toggle - joins the other view-toggle QActions
        # there directly (Gui._create_actions_view).
        add_action(
            "Design static input",
            self.open_pattern_designer,
            menu,
            None,
            self,
            icon="mdi6.pencil-box-outline",
        )
        add_action(
            "Design dynamic input",
            self.open_dynamic_input_designer,
            menu,
            None,
            self,
            icon="mdi6.pencil-box-multiple-outline",
        )
        self.action_show_all_stim_arrays = add_action(
            "Show all input arrays",
            self.toggle_show_all_stim_arrays,
            menu,
            None,
            self,
            icon="mdi6.view-list",
            checkable=True,
            is_checked=True,
        )
