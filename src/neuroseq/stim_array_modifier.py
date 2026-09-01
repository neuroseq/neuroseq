from __future__ import annotations

import logging

import numpy as np

from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim_timing import seconds_to_samples_nearest
from neuroseq.stim_widget import StimWidget


class ReversibleAction:
    """An action that can be done and undone on a (slice of a) numpy array."""

    def __init__(
        self,
        stim_array: StimArray,
        neuron_slice: slice,
        time_slice: slice,
        new_values: np.ndarray | float,
    ) -> None:
        self.neuron_slice = neuron_slice
        self.time_slice = time_slice
        self.old_values = stim_array.values[time_slice, neuron_slice].copy()
        self.new_values = (
            new_values.copy() if isinstance(new_values, np.ndarray) else new_values
        )

    def __call__(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        self._asign_values(stim_array, stim_widget, self.new_values)

    def undo(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        self._asign_values(stim_array, stim_widget, self.old_values)

    def _asign_values(
        self,
        stim_array: StimArray,
        stim_widget: StimWidget,
        values: np.ndarray | float,
    ) -> None:
        """Assign the new values to the StimArray and update the StimWidget accordingly."""
        if (
            self.time_slice.stop is not None
            and self.time_slice.stop > stim_array.values.shape[0]
        ) or (
            self.neuron_slice.stop is not None
            and self.neuron_slice.stop > stim_array.values.shape[1]
        ):
            return
        stim_array.values[self.time_slice, self.neuron_slice] = values
        stim_widget.update_data_slice(self.neuron_slice, self.time_slice, values)


class ResizableReversibleAction:
    """One-step undo for pad + overwrite operations on a project stim array."""

    def __init__(
        self,
        project: Project,
        target_stim_name: str,
        neuron_slice: slice | list[int],
        time_slice: slice,
        new_values: np.ndarray,
        stim_widget: StimWidget,
    ) -> None:
        self.project = project
        self.target_stim_name = target_stim_name
        self.neuron_slice = neuron_slice
        self.time_slice = time_slice
        self.new_values = np.array(new_values, copy=True)
        self.stim_widget = stim_widget
        self.before_lengths: dict[str, int] = {
            name: arr.values.shape[0] for name, arr in project.stim_arrays.items()
        }
        self.before_target_values: np.ndarray | None = None
        self.new_length = time_slice.stop
        if self.new_length is None:
            raise ValueError("ResizableReversibleAction requires bounded time_slice")

    def __call__(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        self._ensure_padded()
        target = self.project.stim_arrays[self.target_stim_name]
        self.before_target_values = target.values[
            self.time_slice, self.neuron_slice
        ].copy()
        target.values[self.time_slice, self.neuron_slice] = self.new_values
        self._refresh_stim_widget()
        self.project.update_limits()

    def undo(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        target = self.project.stim_arrays[self.target_stim_name]
        if self.before_target_values is not None:
            target.values[self.time_slice, self.neuron_slice] = (
                self.before_target_values
            )
        for name, old_length in self.before_lengths.items():
            if name not in self.project.stim_arrays:
                continue
            arr = self.project.stim_arrays[name]
            if arr.values.shape[0] > old_length:
                arr.values = arr.values[:old_length, :]
        self._refresh_stim_widget()
        self.project.update_limits()

    def _ensure_padded(self) -> None:
        target_dt = float(self.project.stim_arrays[self.target_stim_name].dt)
        self.project.pad_stim_arrays(self.new_length * target_dt)

    def _refresh_stim_widget(self) -> None:
        target = self.project.stim_arrays[self.target_stim_name]
        self.stim_widget.set_data(target)
        for name, arr in self.project.stim_arrays.items():
            if name != self.target_stim_name:
                self.stim_widget.set_background_data(arr)


class TrimReversibleAction:
    """One-step undo for trimming every project stim array to a target length.

    The stim array is the project's authoritative timeline, so trimming it
    also windows out any simulated data past the new length (handled on read in
    :meth:`neuroseq.project.Project.get_data`). Undo re-appends the removed stim
    tails, which restores the timeline and, with it, the simulated data.
    """

    def __init__(
        self,
        project: Project,
        new_length_seconds: float,
        stim_widget: StimWidget,
        target_stim_name: str,
        only_stim_name: str | None = None,
    ) -> None:
        self.project = project
        self.new_length_seconds = float(new_length_seconds)
        self.stim_widget = stim_widget
        self.target_stim_name = target_stim_name
        # When set, only this array is trimmed (used by the per-array "reset"
        # action); otherwise every stim array is trimmed to the shared length.
        self.only_stim_name = only_stim_name
        # Populated on ``__call__``: the trailing rows removed from each array,
        # keyed by stim name, so undo can re-append exactly what was cut.
        self.removed_tails: dict[str, np.ndarray] = {}
        # The modifier logs these slice attributes; they are informational only.
        self.neuron_slice = slice(None)
        self.time_slice = slice(None)

    def __call__(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        self.removed_tails = {}
        for name, arr in self.project.stim_arrays.items():
            if self.only_stim_name is not None and name != self.only_stim_name:
                continue
            n_keep = seconds_to_samples_nearest(self.new_length_seconds, arr.dt)
            if n_keep < arr.values.shape[0]:
                self.removed_tails[name] = arr.values[n_keep:, :].copy()
                arr.values = arr.values[:n_keep, :]
        self._refresh_stim_widget()
        self.project.update_limits()

    def undo(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        for name, tail in self.removed_tails.items():
            if name not in self.project.stim_arrays:
                continue
            arr = self.project.stim_arrays[name]
            if tail.shape[1] != arr.values.shape[1]:
                # Neuron count changed since the trim; can't safely re-append.
                continue
            arr.values = np.concatenate([arr.values, tail], axis=0)
        self._refresh_stim_widget()
        self.project.update_limits()

    def _refresh_stim_widget(self) -> None:
        target = self.project.stim_arrays.get(self.target_stim_name)
        if target is not None:
            self.stim_widget.set_data(target)
        for name, arr in self.project.stim_arrays.items():
            if name != self.target_stim_name:
                self.stim_widget.set_background_data(arr)


class PadReversibleAction:
    """One-step undo for padding every project stim array to a target length.

    Padding appends trailing zeros to grow each stim array up to
    ``total_length_seconds`` (the stim array is the project's authoritative
    timeline). Undo trims each array back to the length it had before the pad;
    because padding only adds zeros, restoring the original lengths restores the
    original content exactly, so the removed rows need not be stored.
    """

    def __init__(
        self,
        project: Project,
        total_length_seconds: float,
        stim_widget: StimWidget,
        target_stim_name: str,
    ) -> None:
        self.project = project
        self.total_length_seconds = float(total_length_seconds)
        self.stim_widget = stim_widget
        self.target_stim_name = target_stim_name
        # Populated on ``__call__``: each array's length before the pad, so undo
        # can slice the trailing zeros back off.
        self.before_lengths: dict[str, int] = {}
        # The modifier logs these slice attributes; they are informational only.
        self.neuron_slice = slice(None)
        self.time_slice = slice(None)

    def __call__(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        self.before_lengths = {
            name: int(arr.values.shape[0])
            for name, arr in self.project.stim_arrays.items()
        }
        self.project.pad_stim_arrays(self.total_length_seconds)
        self._refresh_stim_widget()
        self.project.update_limits()

    def undo(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        del stim_array, stim_widget
        for name, old_length in self.before_lengths.items():
            if name not in self.project.stim_arrays:
                continue
            arr = self.project.stim_arrays[name]
            if arr.values.shape[0] > old_length:
                arr.values = arr.values[:old_length, :]
        self._refresh_stim_widget()
        self.project.update_limits()

    def _refresh_stim_widget(self) -> None:
        target = self.project.stim_arrays.get(self.target_stim_name)
        if target is not None:
            self.stim_widget.set_data(target)
        for name, arr in self.project.stim_arrays.items():
            if name != self.target_stim_name:
                self.stim_widget.set_background_data(arr)


class StimArrayModifier:
    """A class to track changes to a StimArray and allow undoing and redoing them."""

    def __init__(self, stim_array: StimArray, stim_widget: StimWidget) -> None:
        self.logger = logging.getLogger(__name__)
        self.stim_array = stim_array
        self.stim_widget = stim_widget
        self.action_stack: list[
            ReversibleAction
            | ResizableReversibleAction
            | TrimReversibleAction
            | PadReversibleAction
        ] = []
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
            f"Creating action {action_index}: neuron slice {neuron_slice}, time slice {time_slice}, StimArray {self.stim_array.name}"
        )
        action = ReversibleAction(self.stim_array, neuron_slice, time_slice, new_values)
        action(self.stim_array, self.stim_widget)
        self.action_stack = self.action_stack[:action_index]
        self.action_stack.append(action)
        self.current_index = action_index

    def undo(self) -> None:
        if not self.can_undo():
            return
        action_index = self.current_index
        action = self.action_stack[action_index]
        self.logger.debug(
            f"Undoing action {action_index}: neuron slice {action.neuron_slice}, time slice {action.time_slice}, StimArray {self.stim_array.name}"
        )
        action.undo(self.stim_array, self.stim_widget)
        self.current_index -= 1

    def redo(self) -> None:
        if not self.can_redo():
            return
        self.current_index += 1
        action_index = self.current_index
        action = self.action_stack[action_index]
        action(self.stim_array, self.stim_widget)
        self.logger.debug(
            f"Redoing action {self.current_index}: neuron slice {action.neuron_slice}, time slice {action.time_slice}, StimArray {self.stim_array.name}"
        )

    def can_undo(self) -> bool:
        return self.current_index >= 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.action_stack) - 1
