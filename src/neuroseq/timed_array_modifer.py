from __future__ import annotations

import logging

import brian2 as b2
import brian2.numpy_ as np

from neuroseq.stim_widget import StimWidget


class ReversibleAction:
    """An action that can be done and undone on a (slice of a) 2-d brian2 TimedArray."""

    def __init__(
        self,
        timed_array: b2.TimedArray,
        neuron_slice: slice,
        time_slice: slice,
        new_values: np.ndarray | float,
    ) -> None:
        self.neuron_slice = neuron_slice
        self.time_slice = time_slice
        self.old_values = timed_array.values[time_slice, neuron_slice].copy()
        self.new_values = (
            new_values.copy() if isinstance(new_values, np.ndarray) else new_values
        )

    def __call__(self, timed_array: b2.TimedArray, stim_widget: StimWidget) -> None:
        self._asign_values(timed_array, stim_widget, self.new_values)

    def undo(self, timed_array: b2.TimedArray, stim_widget: StimWidget) -> None:
        self._asign_values(timed_array, stim_widget, self.old_values)

    def _asign_values(
        self,
        timed_array: b2.TimedArray,
        stim_widget: StimWidget,
        values: np.ndarray | float,
    ) -> None:
        """Assign the new values to the TimedArray and update the StimWidget accordingly."""
        timed_array.values[self.time_slice, self.neuron_slice] = values
        stim_widget.update_data_slice(self.neuron_slice, self.time_slice, values)


class TimedArrayModifier:
    """A class to track changes to a 2-d brian2 TimedArray and allow undoing and redoing them."""

    def __init__(self, timed_array: b2.TimedArray, stim_widget: StimWidget) -> None:
        self.logger = logging.getLogger(__name__)
        self.timed_array = timed_array
        self.stim_widget = stim_widget
        self.action_stack: list[ReversibleAction] = []
        self.current_index = -1

    def __len__(self) -> int:
        return len(self.action_stack)

    def do(
        self,
        neuron_slice: slice,
        time_slice: slice,
        new_values: np.ndarray | float,
    ) -> None:
        action_index = self.current_index + 1
        self.logger.debug(
            f"Creating action {action_index}: neuron slice {neuron_slice}, time slice {time_slice}, TimedArray {self.timed_array.name}"
        )
        action = ReversibleAction(
            self.timed_array, neuron_slice, time_slice, new_values
        )
        action(self.timed_array, self.stim_widget)
        self.action_stack = self.action_stack[:action_index]
        self.action_stack.append(action)
        self.current_index = action_index

    def undo(self) -> None:
        if not self.can_undo():
            return
        action_index = self.current_index
        action = self.action_stack[action_index]
        self.logger.debug(
            f"Undoing action {action_index}: neuron slice {action.neuron_slice}, time slice {action.time_slice}, TimedArray {self.timed_array.name}"
        )
        action.undo(self.timed_array, self.stim_widget)
        self.current_index -= 1

    def redo(self) -> None:
        if not self.can_redo():
            return
        self.current_index += 1
        action_index = self.current_index
        action = self.action_stack[action_index]
        action(self.timed_array, self.stim_widget)
        self.logger.debug(
            f"Redoing action {self.current_index}: neuron slice {action.neuron_slice}, time slice {action.time_slice}, TimedArray {self.timed_array.name}"
        )

    def can_undo(self) -> bool:
        return self.current_index >= 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.action_stack) - 1
