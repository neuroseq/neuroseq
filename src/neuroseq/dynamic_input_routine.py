from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import brian2.numpy_ as np


class SimulationHost(Protocol):
    def request_simulation(self, seconds: float) -> None: ...

    def append_block(self, block: np.ndarray) -> int: ...


class DynamicInputDialog(Protocol):
    def simulation_finished(self) -> None: ...
    def simulation_failed(self, message: str) -> None: ...
    def refresh(self) -> None: ...
    def apply_theme(self, theme: str) -> None: ...


class RoutineBackend:
    """Qt-free run/pause/finish plumbing for closed-loop input routines."""

    def __init__(self, host: SimulationHost) -> None:
        self._host = host
        self.running = False
        self.awaiting_simulation = False
        self.paused = False
        self.pending_simulation_seconds: float | None = None
        self.phase = "idle"
        self.on_log: Callable[[str], None] | None = None
        self.on_finished: Callable[[str], None] | None = None
        self.on_advance: Callable[[], None] | None = None
        self.on_measurement: Callable[[], None] | None = None

    def begin(self) -> None:
        self.running = True
        self.awaiting_simulation = False
        self.paused = False
        self.pending_simulation_seconds = None

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._log(
            "Paused - the routine will hold after its current step."
            if self.paused
            else "Resumed."
        )
        if not self.paused and self.pending_simulation_seconds is not None:
            seconds = self.pending_simulation_seconds
            self.pending_simulation_seconds = None
            self.request_simulation(seconds)

    def request_simulation(self, seconds: float) -> None:
        if self.paused:
            self.pending_simulation_seconds = seconds
            self._log("Paused - next iteration queued; click Resume to continue.")
            return
        self.awaiting_simulation = True
        self._host.request_simulation(seconds)

    def simulation_finished(self) -> None:
        if not self.running or not self.awaiting_simulation:
            return
        self.awaiting_simulation = False
        self._advance()

    def simulation_failed(self, message: str) -> None:
        if self.running and self.awaiting_simulation:
            self.finish(f"Simulation failed: {message}")

    def finish(self, message: str) -> None:
        self.running = False
        self.awaiting_simulation = False
        self.paused = False
        self.pending_simulation_seconds = None
        self.phase = "idle"
        self._log(message)
        if self.on_finished is not None:
            self.on_finished(message)

    def _log(self, message: str) -> None:
        if self.on_log is not None:
            self.on_log(message)

    def _measurement_changed(self) -> None:
        if self.on_measurement is not None:
            self.on_measurement()

    def _advance(self) -> None:
        if self.on_advance is not None:
            self.on_advance()


def backend_property(name: str) -> property:
    """Expose backend-owned state through a dialog compatibility property."""
    return property(
        lambda dialog: getattr(dialog._backend, name),
        lambda dialog, value: setattr(dialog._backend, name, value),
    )
