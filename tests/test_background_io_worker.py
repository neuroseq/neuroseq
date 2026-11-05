"""Tests for the background IO worker used by save / load / import."""

from __future__ import annotations

import threading
import time

from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.background_jobs import BackgroundIOWorker
from tests.wait_timeout import WAIT_TIMEOUT_MS


class _ProgressReceiver(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.percents: list[int] = []
        self.thread_ids: list[int | None] = []

    @QtCore.Slot(int)
    def record(self, percent: int) -> None:
        self.percents.append(int(percent))
        self.thread_ids.append(threading.current_thread().ident)


def _spawn_worker(task):
    thread = QtCore.QThread()
    worker = BackgroundIOWorker(task)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker


def test_io_worker_runs_off_main_thread(qtbot: QtBot) -> None:
    seen: list[int | None] = []

    def _task(report):
        seen.append(threading.current_thread().ident)
        report(50)
        return "ok"

    thread, worker = _spawn_worker(_task)
    try:
        with qtbot.waitSignal(worker.finished, timeout=WAIT_TIMEOUT_MS) as blocker:
            thread.start()
        assert seen
        assert seen[0] != threading.main_thread().ident
        assert blocker.args[0] == "ok"
    finally:
        thread.quit()
        thread.wait(2000)


def test_io_worker_progress_signal_arrives_in_main_thread(qtbot: QtBot) -> None:
    receiver = _ProgressReceiver()

    def _task(report):
        for p in (0, 25, 50, 75, 100):
            report(p)
        return None

    thread, worker = _spawn_worker(_task)
    worker.progress.connect(receiver.record)
    try:
        with qtbot.waitSignal(worker.finished, timeout=WAIT_TIMEOUT_MS):
            thread.start()
        assert receiver.percents == sorted(receiver.percents)
        assert receiver.thread_ids
        assert all(tid == threading.main_thread().ident for tid in receiver.thread_ids)
    finally:
        thread.quit()
        thread.wait(2000)


def test_io_worker_emits_failed_on_exception(qtbot: QtBot) -> None:
    def _task(report):
        report(10)
        raise RuntimeError("disk full")

    thread, worker = _spawn_worker(_task)
    finished_seen: list[bool] = []
    worker.finished.connect(lambda *_: finished_seen.append(True))
    try:
        with qtbot.waitSignal(worker.failed, timeout=WAIT_TIMEOUT_MS) as blocker:
            thread.start()
        message = blocker.args[0]
        assert "RuntimeError" in message
        assert "disk full" in message
        assert not finished_seen
    finally:
        thread.quit()
        thread.wait(2000)


def test_io_worker_finished_payload_is_task_return_value(qtbot: QtBot) -> None:
    sentinel = {"path": "/tmp/x.nseq", "bytes_written": 12345}

    def _task(_report):
        time.sleep(0.05)
        return sentinel

    thread, worker = _spawn_worker(_task)
    try:
        with qtbot.waitSignal(worker.finished, timeout=WAIT_TIMEOUT_MS) as blocker:
            thread.start()
        assert blocker.args[0] is sentinel
    finally:
        thread.quit()
        thread.wait(2000)
