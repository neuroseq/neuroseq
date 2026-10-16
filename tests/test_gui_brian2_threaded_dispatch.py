from __future__ import annotations

import threading
import time

import pytest
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot

import neuroseq.gui
from neuroseq.gui import Gui
from neuroseq.project import Project


@pytest.fixture
def gui_with_b2_project(qtbot: QtBot, test_project: Project):
    widget = Gui()
    qtbot.addWidget(widget)
    test_project.external_simulate_path = None
    test_project.external_simulate_function = None
    widget.load_project(test_project)
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
        timeout=10000,
    )


def _set_simulation_time(monkeypatch: MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(
        neuroseq.gui,
        "getSimulationTime",
        lambda *args, **kwargs: seconds,
    )


def test_brian2_sim_runs_on_worker_thread(
    qtbot: QtBot, gui_with_b2_project: Gui, monkeypatch: MonkeyPatch
) -> None:
    seen_thread_ids: list[int | None] = []

    def _spy_run_simulation(time_seconds, report=None, stim_array_padding_seconds=0.0):
        seen_thread_ids.append(threading.current_thread().ident)
        if report is not None:
            report(0.1, 1.0, 0.0, 1.0)

    monkeypatch.setattr(
        gui_with_b2_project.project, "run_simulation", _spy_run_simulation
    )
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is None, timeout=5000
    )

    assert seen_thread_ids
    assert seen_thread_ids[0] != threading.main_thread().ident


def test_brian2_progress_dialog_shows_label_transition(
    qtbot: QtBot, gui_with_b2_project: Gui, monkeypatch: MonkeyPatch
) -> None:
    allow_report = threading.Event()

    def _spy_run_simulation(time_seconds, report=None, stim_array_padding_seconds=0.0):
        allow_report.wait(timeout=5.0)
        if report is not None:
            report(0.1, 0.25, 0.0, 1.0)

    monkeypatch.setattr(
        gui_with_b2_project.project, "run_simulation", _spy_run_simulation
    )
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()

    qtbot.waitUntil(
        lambda: (
            gui_with_b2_project.progress_dialog.labelText()
            == "Preparing Brian2 simulation…"
        ),
        timeout=2000,
    )
    allow_report.set()
    qtbot.waitUntil(
        lambda: (
            gui_with_b2_project.progress_dialog.labelText() == "Simulating Brian2..."
        ),
        timeout=5000,
    )
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is None, timeout=5000
    )


def test_brian2_finished_calls_update_data(
    qtbot: QtBot, gui_with_b2_project: Gui, monkeypatch: MonkeyPatch
) -> None:
    calls: list[bool] = []

    def _spy_run_simulation(time_seconds, report=None, stim_array_padding_seconds=0.0):
        if report is not None:
            report(0.1, 1.0, 0.0, 1.0)

    def _spy_update_data(*, keep_undo_history=False):
        calls.append(bool(keep_undo_history))

    monkeypatch.setattr(
        gui_with_b2_project.project, "run_simulation", _spy_run_simulation
    )
    monkeypatch.setattr(gui_with_b2_project, "_update_data", _spy_update_data)
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is None, timeout=5000
    )
    qtbot.waitUntil(lambda: calls == [True], timeout=2000)

    assert calls == [True]


def test_brian2_failure_surfaces_critical_dialog(
    qtbot: QtBot,
    gui_with_b2_project: Gui,
    monkeypatch: MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    def _failing_run_simulation(
        time_seconds, report=None, stim_array_padding_seconds=0.0
    ):
        raise RuntimeError("brian2 boom")

    monkeypatch.setattr(
        gui_with_b2_project.project,
        "run_simulation",
        _failing_run_simulation,
    )
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is None, timeout=5000
    )

    assert any(
        call["name"] == "critical"
        and call["title"] == "Simulation error"
        and "brian2 boom" in call["text"].lower()
        for call in qmessagebox_calls
    )
    assert gui_with_b2_project.menu_file.isEnabled()
    assert gui_with_b2_project.continue_simulation_action.isEnabled()
    assert gui_with_b2_project._do_external_simulation_action.isEnabled()


def test_brian2_run_button_blocks_reentrancy(
    qtbot: QtBot,
    gui_with_b2_project: Gui,
    monkeypatch: MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    block = threading.Event()
    run_count = 0

    def _blocking_run_simulation(
        time_seconds, report=None, stim_array_padding_seconds=0.0
    ):
        nonlocal run_count
        run_count += 1
        while not block.is_set():
            time.sleep(0.05)

    monkeypatch.setattr(
        gui_with_b2_project.project,
        "run_simulation",
        _blocking_run_simulation,
    )
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is not None, timeout=2000
    )
    qtbot.waitUntil(lambda: run_count == 1, timeout=2000)
    original_worker = gui_with_b2_project._background_job.worker
    assert original_worker is not None
    assert not gui_with_b2_project.menu_file.isEnabled()
    assert not gui_with_b2_project.continue_simulation_action.isEnabled()
    try:
        gui_with_b2_project.btn_continue_simulation_clicked()

        assert gui_with_b2_project._background_job.worker is original_worker
        assert run_count == 1
        assert any(
            call["name"] == "information" and call["title"] == "Busy"
            for call in qmessagebox_calls
        )
    finally:
        block.set()
        qtbot.waitUntil(
            lambda: gui_with_b2_project._background_job.worker is None, timeout=5000
        )

    assert gui_with_b2_project.menu_file.isEnabled()
    assert gui_with_b2_project.continue_simulation_action.isEnabled()


def test_brian2_close_event_drains_inflight_sim(
    qtbot: QtBot, gui_with_b2_project: Gui, monkeypatch: MonkeyPatch
) -> None:
    block = threading.Event()

    def _blocking_run_simulation(
        time_seconds, report=None, stim_array_padding_seconds=0.0
    ):
        block.wait(timeout=5.0)

    monkeypatch.setattr(
        gui_with_b2_project.project,
        "run_simulation",
        _blocking_run_simulation,
    )
    _set_simulation_time(monkeypatch, 0.05)

    gui_with_b2_project.btn_continue_simulation_clicked()
    qtbot.waitUntil(
        lambda: gui_with_b2_project._background_job.worker is not None, timeout=2000
    )

    def _delayed_release() -> None:
        time.sleep(0.2)
        block.set()

    releaser = threading.Thread(target=_delayed_release)
    releaser.start()
    try:
        gui_with_b2_project.close()
    finally:
        releaser.join(timeout=2.0)

    qtbot.waitUntil(
        lambda: (
            gui_with_b2_project._background_job.worker_thread is None
            or not gui_with_b2_project._background_job.worker_thread.isRunning()
        ),
        timeout=10000,
    )
