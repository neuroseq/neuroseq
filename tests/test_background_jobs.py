from __future__ import annotations

import threading

import pytest
from qtpy import QtCore

from neuroseq.background_jobs import BackgroundIOWorker
from neuroseq.background_jobs import BackgroundJob
from neuroseq.background_jobs import EliseProtocolWorker
from neuroseq.background_jobs import wait_for_jobs


@pytest.fixture
def job(qapp):
    owner = BackgroundJob()
    yield owner
    assert owner.stop(5000)


def test_completion_runs_on_main_thread_after_worker_stops(qtbot, job):
    release = threading.Event()
    results = []
    worker = BackgroundIOWorker(lambda report: release.wait(5))

    def on_finished(result):
        assert thread.isFinished()
        assert not job.busy
        assert job.worker is None
        assert (
            QtCore.QThread.currentThread()
            is QtCore.QCoreApplication.instance().thread()
        )
        results.append(result)

    job.start(
        worker,
        on_progress=lambda _: None,
        on_finished=on_finished,
        on_failed=pytest.fail,
    )
    thread = job.worker_thread
    assert thread is not None
    release.set()
    qtbot.waitUntil(lambda: bool(results))
    assert results == [True]


def test_completion_can_start_the_next_job(qtbot, job):
    results = []

    def first_finished(result):
        results.append(result)
        job.start(
            BackgroundIOWorker(lambda report: "second"),
            on_progress=lambda _: None,
            on_finished=results.append,
            on_failed=pytest.fail,
        )

    job.start(
        BackgroundIOWorker(lambda report: "first"),
        on_progress=lambda _: None,
        on_finished=first_finished,
        on_failed=pytest.fail,
    )
    qtbot.waitUntil(lambda: len(results) == 2)
    assert results == ["first", "second"]
    assert not job.busy


def test_failure_releases_owner_and_delivers_error_on_main_thread(qtbot, job):
    errors = []

    def task(report):
        raise OSError("failed save")

    def failed(message):
        assert not job.busy
        assert (
            QtCore.QThread.currentThread()
            is QtCore.QCoreApplication.instance().thread()
        )
        errors.append(message)

    job.start(
        BackgroundIOWorker(task),
        on_progress=lambda _: None,
        on_finished=lambda _: pytest.fail("unexpected success"),
        on_failed=failed,
    )
    qtbot.waitUntil(lambda: bool(errors))
    assert errors == ["OSError: failed save"]


def test_cancellation_retains_ownership_until_task_returns(qtbot, job):
    cancel_event = threading.Event()
    release = threading.Event()
    cancelled = []

    def run(progress):
        assert release.wait(5)
        progress(0.5)

    worker = EliseProtocolWorker(run, {}, cancel_event)
    job.start(
        worker,
        on_progress=lambda _: None,
        on_finished=lambda _: pytest.fail("unexpected success"),
        on_failed=pytest.fail,
        on_cancelled=lambda: cancelled.append(True),
        cancel=cancel_event.set,
    )
    job.cancel()
    assert cancel_event.is_set()
    assert job.busy
    assert job.worker is worker
    release.set()
    qtbot.waitUntil(lambda: bool(cancelled))
    assert not job.busy


def test_rejects_overlapping_jobs(qtbot, job):
    release = threading.Event()
    job.start(
        BackgroundIOWorker(lambda report: release.wait(5)),
        on_progress=lambda _: None,
        on_finished=lambda _: None,
        on_failed=pytest.fail,
    )
    worker = job.worker
    try:
        with pytest.raises(RuntimeError, match="already running"):
            job.start(
                BackgroundIOWorker(lambda report: None),
                on_progress=lambda _: None,
                on_finished=lambda _: None,
                on_failed=pytest.fail,
            )
        assert job.worker is worker
    finally:
        release.set()
        qtbot.waitUntil(lambda: not job.busy)


def test_stop_cancels_protocol_without_delivering_ui_callbacks(qtbot, job):
    cancel_event = threading.Event()
    entered = threading.Event()
    callbacks = []

    def run(progress):
        entered.set()
        assert cancel_event.wait(5)
        progress(0.5)

    job.start(
        EliseProtocolWorker(run, {}, cancel_event),
        on_progress=callbacks.append,
        on_finished=callbacks.append,
        on_failed=callbacks.append,
        on_cancelled=lambda: callbacks.append("cancelled"),
        cancel=cancel_event.set,
    )
    assert entered.wait(5)
    assert job.stop(5000)
    assert cancel_event.is_set()
    assert not job.busy
    QtCore.QCoreApplication.processEvents()
    assert callbacks == []


def test_shutdown_timeout_keeps_ownership(qtbot, job):
    release = threading.Event()
    results = []
    job.start(
        BackgroundIOWorker(lambda report: release.wait(5)),
        on_progress=lambda _: None,
        on_finished=results.append,
        on_failed=pytest.fail,
    )
    worker, thread = job.worker, job.worker_thread
    try:
        assert not job.stop(1)
        assert job.busy
        assert job.worker is worker
        assert job.worker_thread is thread
    finally:
        release.set()
        qtbot.waitUntil(lambda: not job.busy)
    assert results == [True]


def test_idle_follows_callbacks_and_wait_for_jobs_defers_to_event_loop(qtbot, job):
    events = []
    release = threading.Event()
    job.idle.connect(lambda: events.append("idle"))
    job.start(
        BackgroundIOWorker(lambda report: release.wait(5)),
        on_progress=lambda _: None,
        on_finished=lambda _result: events.append("finished"),
        on_failed=pytest.fail,
        cancel=release.set,
    )
    assert job.cancellable
    owner = QtCore.QObject()
    wait_for_jobs(
        [job],
        on_idle=lambda: events.append("all idle"),
        on_timeout=pytest.fail,
        timeout_ms=5000,
        parent=owner,
    )
    job.cancel()
    qtbot.waitUntil(lambda: "all idle" in events)
    assert events == ["finished", "idle", "all idle"]
    assert not job.cancellable


def test_wait_for_jobs_times_out_while_a_job_keeps_running(qtbot, job):
    release = threading.Event()
    outcome = []
    job.start(
        BackgroundIOWorker(lambda report: release.wait(5)),
        on_progress=lambda _: None,
        on_finished=lambda _result: None,
        on_failed=pytest.fail,
    )
    owner = QtCore.QObject()
    wait_for_jobs(
        [job],
        on_idle=lambda: outcome.append("idle"),
        on_timeout=lambda: outcome.append("timeout"),
        timeout_ms=20,
        parent=owner,
    )
    qtbot.waitUntil(lambda: bool(outcome))
    release.set()
    qtbot.waitUntil(lambda: not job.busy)
    qtbot.wait(20)
    assert outcome == ["timeout"]


def test_raising_completion_callback_is_reported_not_fatal(
    qtbot, job, qmessagebox_calls
):
    idle = []
    job.idle.connect(lambda: idle.append(True))

    def on_finished(_result):
        raise ValueError("bad result")

    job.start(
        BackgroundIOWorker(lambda report: "first"),
        on_progress=lambda _: None,
        on_finished=on_finished,
        on_failed=pytest.fail,
    )
    qtbot.waitUntil(lambda: bool(idle))
    assert not job.busy
    assert qmessagebox_calls[-1]["name"] == "critical"
    assert "ValueError: bad result" in qmessagebox_calls[-1]["text"]
    # The owner remains usable for the next job.
    results = []
    job.start(
        BackgroundIOWorker(lambda report: "second"),
        on_progress=lambda _: None,
        on_finished=results.append,
        on_failed=pytest.fail,
    )
    qtbot.waitUntil(lambda: results == ["second"])
