from __future__ import annotations

import brian2.numpy_ as np

from neuroseq.dynamic_input_routine import RoutineBackend


class _Host:
    def __init__(self) -> None:
        self.requests: list[float] = []

    def request_simulation(self, seconds: float) -> None:
        self.requests.append(seconds)

    def append_block(self, block: np.ndarray) -> int:
        del block
        return 0


class _Backend(RoutineBackend):
    def __init__(self, host: _Host) -> None:
        super().__init__(host)
        self.advances = 0

    def _advance(self) -> None:
        self.advances += 1


def test_pause_defers_and_resume_dispatches_request() -> None:
    host = _Host()
    backend = _Backend(host)
    backend.begin()
    backend.toggle_pause()
    backend.request_simulation(0.25)
    assert host.requests == []
    assert backend.pending_simulation_seconds == 0.25
    backend.toggle_pause()
    assert host.requests == [0.25]
    assert backend.awaiting_simulation


def test_finish_resets_state_and_notifies_callback() -> None:
    host = _Host()
    backend = _Backend(host)
    finished: list[str] = []
    backend.on_finished = finished.append
    backend.begin()
    backend.request_simulation(0.1)
    backend.finish("done")
    assert not backend.running
    assert not backend.awaiting_simulation
    assert not backend.paused
    assert backend.pending_simulation_seconds is None
    assert backend.phase == "idle"
    assert finished == ["done"]
