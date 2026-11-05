from __future__ import annotations

import threading
import time

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _wire_fake_external_simulator(gui: ProjectWorkspace, fake_simulate_fn) -> None:
    gui.project.external_simulate_path = "test:fake"
    gui.project.external_simulate_function = fake_simulate_fn


def _make_blocking_simulation(block: threading.Event, name: str):
    def _simulate(stims_dict, simulation_time, params, progress):
        progress(0.0)
        while not block.is_set():
            time.sleep(0.05)
            progress(0.0)
        return Experiment(name=name, duration=simulation_time, timepoints=[], data=[])

    return _simulate


@pytest.fixture
def gui(qtbot: QtBot):
    widget = ProjectWorkspace()
    qtbot.addWidget(widget)
    yield widget
    try:
        widget.close()
    except RuntimeError:
        return
    qtbot.waitUntil(
        lambda: (
            widget._background_job.worker_thread is None
            or not widget._background_job.worker_thread.isRunning()
        ),
        timeout=WAIT_TIMEOUT_MS,
    )


def test_cancel_via_progress_dialog_raises_and_emits_cancelled(
    qtbot: QtBot, gui: ProjectWorkspace
) -> None:
    # runs until cancelled, so it can't finish before the cancel arrives
    block = threading.Event()
    _wire_fake_external_simulator(
        gui, _make_blocking_simulation(block, "should_not_finish")
    )
    original_experiments = dict(gui.project.experiments)

    gui._do_external_simulation(1.0)
    worker = gui._background_job.worker
    assert worker is not None
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    gui.progress_dialog.canceled.emit()

    try:
        qtbot.waitUntil(
            lambda: (
                bool(cancelled)
                and gui._background_job.worker is None
                and QtWidgets.QApplication.overrideCursor() is None
            ),
            timeout=WAIT_TIMEOUT_MS,
        )
    finally:
        block.set()
    assert gui.project.experiments == original_experiments
    assert gui.continue_simulation_action.isEnabled()
    assert gui._do_external_simulation_action.isEnabled()
    assert gui._background_job.worker is None
    assert QtWidgets.QApplication.overrideCursor() is None


def test_simulation_actions_disabled_during_run(
    qtbot: QtBot, gui: ProjectWorkspace
) -> None:
    block = threading.Event()
    _wire_fake_external_simulator(
        gui, _make_blocking_simulation(block, "disabled_during_run")
    )

    gui._do_external_simulation(1.0)
    worker = gui._background_job.worker
    assert worker is not None
    qtbot.waitUntil(lambda: gui.progress_dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    assert not gui.continue_simulation_action.isEnabled()
    assert not gui._do_external_simulation_action.isEnabled()

    finished: list[bool] = []
    worker.finished.connect(lambda *_: finished.append(True))
    block.set()
    qtbot.waitUntil(
        lambda: bool(finished) and gui._background_job.worker_thread is None,
        timeout=WAIT_TIMEOUT_MS,
    )

    assert gui.continue_simulation_action.isEnabled()
    assert gui._do_external_simulation_action.isEnabled()


def test_dispatch_is_idempotent_while_running(
    qtbot: QtBot, gui: ProjectWorkspace
) -> None:
    block = threading.Event()
    call_count = 0

    def _simulate(stims_dict, simulation_time, params, progress):
        nonlocal call_count
        call_count += 1
        progress(0.0)
        while not block.is_set():
            time.sleep(0.05)
            progress(0.0)
        return Experiment("idempotent", simulation_time, [], [])

    _wire_fake_external_simulator(gui, _simulate)

    gui._do_external_simulation(1.0)
    original_worker = gui._background_job.worker
    assert original_worker is not None
    qtbot.waitUntil(lambda: call_count == 1, timeout=WAIT_TIMEOUT_MS)

    gui._do_external_simulation(1.0)

    assert gui._background_job.worker is original_worker
    assert call_count == 1

    finished: list[bool] = []
    original_worker.finished.connect(lambda *_: finished.append(True))
    block.set()
    qtbot.waitUntil(lambda: bool(finished), timeout=WAIT_TIMEOUT_MS)


def test_close_event_cancels_inflight_sim(qtbot: QtBot, gui: ProjectWorkspace) -> None:
    def _simulate(stims_dict, simulation_time, params, progress):
        for i in range(1_000_000):
            progress(i / 1_000_000)
        return Experiment("should_not_finish", simulation_time, [], [])

    _wire_fake_external_simulator(gui, _simulate)

    gui._do_external_simulation(1.0)
    assert gui._background_job.worker is not None

    gui.close()

    qtbot.waitUntil(
        lambda: (
            gui._background_job.worker_thread is None
            or not gui._background_job.worker_thread.isRunning()
        ),
        timeout=WAIT_TIMEOUT_MS,
    )


def test_experiment_appended_to_its_own_workspace(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch
) -> None:
    block = threading.Event()
    _wire_fake_external_simulator(gui, _make_blocking_simulation(block, "dispatched"))
    original_project = gui.project
    monkeypatch.setattr(
        gui, "_append_experimental_data", original_project.add_experiment
    )

    gui._do_external_simulation(1.0)
    worker = gui._background_job.worker
    assert worker is not None

    other_project = Project()
    other = ProjectWorkspace(other_project)
    qtbot.addWidget(other)

    finished: list[bool] = []
    worker.finished.connect(lambda *_: finished.append(True))
    block.set()
    qtbot.waitUntil(lambda: bool(finished), timeout=WAIT_TIMEOUT_MS)
    qtbot.waitUntil(lambda: not gui._background_job.busy, timeout=WAIT_TIMEOUT_MS)

    assert "dispatched" in original_project.experiments
    assert "dispatched" not in other_project.experiments


def test_progress_dialog_cancel_button_text_lifecycle(
    qtbot: QtBot, gui: ProjectWorkspace
) -> None:
    block = threading.Event()
    _wire_fake_external_simulator(
        gui, _make_blocking_simulation(block, "text_lifecycle")
    )

    cancel_button = gui.progress_dialog.findChild(QtWidgets.QPushButton)
    assert cancel_button is None or not cancel_button.isVisible()

    gui._do_external_simulation(1.0)
    worker = gui._background_job.worker
    assert worker is not None
    qtbot.waitUntil(lambda: gui.progress_dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)

    cancel_button = gui.progress_dialog.findChild(QtWidgets.QPushButton)
    assert cancel_button is not None
    assert cancel_button.isEnabled()

    block.set()
    qtbot.waitUntil(lambda: not gui._background_job.busy, timeout=WAIT_TIMEOUT_MS)

    cancel_button = gui.progress_dialog.findChild(QtWidgets.QPushButton)
    assert cancel_button is None or not cancel_button.isVisible()
