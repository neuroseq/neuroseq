"""Session-local undo/redo commands shared by all of a project's data editors.

Commands retain the data needed for replay, never simulator or device state.
The history is independent of Qt and is deliberately excluded from project files.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from inspect import ismethod


@dataclass
class ProjectCommand:
    label: str
    redo: Callable[[], None]
    undo: Callable[[], None]


class ProjectHistory:
    """A chronological command stack; a new edit discards the redo branch.

    A command must finish successfully before the history position changes.
    Commands should validate before changing data, or roll back partial changes
    themselves. Listeners run after the position changes, and cannot break replay.
    Bound-method listeners are weak so a project does not retain closed editors.
    """

    def __init__(self) -> None:
        self.action_stack: list[ProjectCommand] = []
        self.current_index = -1
        self._replaying = False
        self._listeners: list[Callable[[str], None] | weakref.WeakMethod] = []

    def __len__(self) -> int:
        return len(self.action_stack)

    def subscribe(self, callback: Callable[[str], None]) -> Callable[[], None]:
        listener = weakref.WeakMethod(callback) if ismethod(callback) else callback
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self, event: str) -> None:
        for listener in tuple(self._listeners):
            callback = (
                listener() if isinstance(listener, weakref.WeakMethod) else listener
            )
            if callback is None:
                self._listeners.remove(listener)
                continue
            try:
                callback(event)
            except Exception:
                logging.getLogger(__name__).exception("Project history listener failed")

    def execute(self, command: ProjectCommand) -> None:
        if self._replaying:
            raise RuntimeError("Cannot execute a new command during undo/redo")
        self._replaying = True
        try:
            command.redo()
        finally:
            self._replaying = False
        self.record_applied(command)

    def record_applied(self, command: ProjectCommand) -> None:
        """Record an edit already applied successfully by a data mutator."""
        if self._replaying:
            raise RuntimeError("Cannot record a new command during undo/redo")
        del self.action_stack[self.current_index + 1 :]
        self.action_stack.append(command)
        self.current_index += 1
        self._notify("execute")

    def can_undo(self) -> bool:
        return self.current_index >= 0

    def can_redo(self) -> bool:
        return self.current_index + 1 < len(self.action_stack)

    @property
    def undo_label(self) -> str:
        return self.action_stack[self.current_index].label if self.can_undo() else ""

    @property
    def redo_label(self) -> str:
        return (
            self.action_stack[self.current_index + 1].label if self.can_redo() else ""
        )

    def undo(self) -> None:
        if self._replaying:
            raise RuntimeError("Cannot undo while a command is running")
        if not self.can_undo():
            return
        self._replaying = True
        try:
            self.action_stack[self.current_index].undo()
            self.current_index -= 1
        finally:
            self._replaying = False
        self._notify("undo")

    def redo(self) -> None:
        if self._replaying:
            raise RuntimeError("Cannot redo while a command is running")
        if not self.can_redo():
            return
        self._replaying = True
        try:
            self.action_stack[self.current_index + 1].redo()
            self.current_index += 1
        finally:
            self._replaying = False
        self._notify("redo")

    def clear(self) -> None:
        """Release retained data at a deliberate, non-undoable edit boundary."""
        if self._replaying:
            raise RuntimeError("Cannot clear history while a command is running")
        self.action_stack.clear()
        self.current_index = -1
        self._notify("clear")
