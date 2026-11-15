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
    # Whether the command can change input (stim) arrays; listeners that
    # only redraw inputs skip the others (e.g. data group edits).
    affects_inputs: bool = True


class StaleCommandError(RuntimeError):
    """Raised by a command whose target data changed outside the history.

    The command must raise before changing any data. The history then drops
    every retained command, because the remaining ones may be stale too.
    """


DEFAULT_MAX_COMMANDS = 100


class ProjectHistory:
    """A chronological command stack; a new edit discards the redo branch.

    A command must finish successfully before the history position changes.
    Commands should validate before changing data, or roll back partial changes
    themselves. Listeners run after the position changes, and cannot break replay.
    Bound-method listeners are weak so a project does not retain closed editors.
    At most ``max_commands`` commands are retained; the oldest are dropped first.
    """

    def __init__(self, max_commands: int = DEFAULT_MAX_COMMANDS) -> None:
        if max_commands < 1:
            raise ValueError("max_commands must be at least 1")
        self.max_commands = int(max_commands)
        self.action_stack: list[ProjectCommand] = []
        self.current_index = -1
        self._replaying = False
        self._listeners: list[Callable[[str], None] | weakref.WeakMethod] = []
        # The command the latest "execute"/"undo"/"redo" event applied.
        self.last_command: ProjectCommand | None = None

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
        overflow = len(self.action_stack) - self.max_commands
        if overflow > 0:
            del self.action_stack[:overflow]
            self.current_index -= overflow
        self.last_command = command
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
            command = self.action_stack[self.current_index]
            command.undo()
            self.current_index -= 1
        except StaleCommandError:
            self._discard_stale("undo")
            return
        finally:
            self._replaying = False
        self.last_command = command
        self._notify("undo")

    def redo(self) -> None:
        if self._replaying:
            raise RuntimeError("Cannot redo while a command is running")
        if not self.can_redo():
            return
        self._replaying = True
        try:
            command = self.action_stack[self.current_index + 1]
            command.redo()
            self.current_index += 1
        except StaleCommandError:
            self._discard_stale("redo")
            return
        finally:
            self._replaying = False
        self.last_command = command
        self._notify("redo")

    def _discard_stale(self, operation: str) -> None:
        self._replaying = False
        logging.getLogger(__name__).warning(
            "Cannot %s: the affected data was changed outside the edit history. "
            "The undo history has been cleared.",
            operation,
            exc_info=True,
        )
        self.clear()

    def clear(self) -> None:
        """Release retained data at a deliberate, non-undoable edit boundary."""
        if self._replaying:
            raise RuntimeError("Cannot clear history while a command is running")
        self.action_stack.clear()
        self.current_index = -1
        self.last_command = None
        self._notify("clear")
