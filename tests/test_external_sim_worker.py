from __future__ import annotations

import threading

from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.background_jobs import ExternalSimWorker
from neuroseq.data import Experiment


class _ProgressReceiver(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.fractions: list[float] = []
        self.thread_ids: list[int | None] = []

    @QtCore.Slot(float)
    def record(self, fraction: float) -> None:
        self.fractions.append(float(fraction))
        self.thread_ids.append(threading.current_thread().ident)


def _fake_simulate_success(stims_dict, simulation_time, params, progress):
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        progress(fraction)
    return Experiment(
        name="fake_success",
        duration=float(simulation_time),
        timepoints=[],
        data=[],
    )


def _fake_simulate_failure(stims_dict, simulation_time, params, progress):
    progress(0.5)
    raise ValueError("kaboom")


def _spawn_worker(simulate_fn, *, accepts_progress=True):
    thread = QtCore.QThread()
    cancel_event = threading.Event()
    worker = ExternalSimWorker(
        simulate_fn=simulate_fn,
        stims_dict={},
        simulation_time=1.0,
        params={},
        accepts_progress=accepts_progress,
        cancel_event=cancel_event,
    )
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
    return thread, worker, cancel_event


def test_worker_runs_on_a_non_main_thread(qtbot: QtBot) -> None:
    seen_thread_ids: list[int | None] = []

    def _simulate(stims_dict, simulation_time, params, progress):
        seen_thread_ids.append(threading.current_thread().ident)
        progress(1.0)
        return Experiment("non_main", float(simulation_time), [], [])

    thread, worker, _ = _spawn_worker(_simulate)
    try:
        with qtbot.waitSignal(worker.finished, timeout=2000):
            thread.start()
        assert seen_thread_ids
        assert seen_thread_ids[0] != threading.main_thread().ident
    finally:
        thread.quit()
        thread.wait(2000)


def test_worker_progress_signals_arrive_in_main_thread(qtbot: QtBot) -> None:
    thread, worker, _ = _spawn_worker(_fake_simulate_success)
    receiver = _ProgressReceiver()
    worker.progress.connect(receiver.record)

    try:
        with qtbot.waitSignal(worker.finished, timeout=2000):
            thread.start()
        assert receiver.fractions == sorted(receiver.fractions)
        assert receiver.thread_ids
        assert receiver.thread_ids == [threading.main_thread().ident] * len(
            receiver.thread_ids
        )
    finally:
        thread.quit()
        thread.wait(2000)


def test_worker_emits_finished_with_experiment(qtbot: QtBot) -> None:
    thread, worker, _ = _spawn_worker(_fake_simulate_success)
    try:
        with qtbot.waitSignal(worker.finished, timeout=2000) as blocker:
            thread.start()
        experiment = blocker.args[0]
        assert isinstance(experiment, Experiment)
        assert experiment.name == "fake_success"
    finally:
        thread.quit()
        thread.wait(2000)


def test_worker_emits_failed_on_simulator_exception(qtbot: QtBot) -> None:
    thread, worker, _ = _spawn_worker(_fake_simulate_failure)
    finished_emitted: list[bool] = []
    worker.finished.connect(lambda *_: finished_emitted.append(True))

    try:
        with qtbot.waitSignal(worker.failed, timeout=2000) as blocker:
            thread.start()
        message = blocker.args[0]
        assert "ValueError" in message
        assert "kaboom" in message
        assert not finished_emitted
    finally:
        thread.quit()
        thread.wait(2000)
