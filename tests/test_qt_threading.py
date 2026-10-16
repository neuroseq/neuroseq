from __future__ import annotations

import threading

from pytestqt.qtbot import QtBot
from qtpy import QtCore

from neuroseq.gui import Gui
from neuroseq.gui import returns_to_main_thread


class _Worker(QtCore.QObject):
    done = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.thread_during_run: QtCore.QThread | None = None

    @QtCore.Slot()
    @returns_to_main_thread
    def run(self) -> None:
        self.thread_during_run = QtCore.QThread.currentThread()
        self.done.emit(1)


def test_worker_is_back_on_the_main_thread_after_run(qtbot: QtBot) -> None:
    """A worker that stays on its finished thread is destroyed there by
    deleteLater, which can deadlock against the GIL (see returns_to_main_thread)."""
    thread = QtCore.QThread()
    worker = _Worker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(thread.quit)

    thread.start()
    qtbot.waitUntil(thread.isFinished, timeout=5000)

    assert worker.thread_during_run is thread
    assert worker.thread() == QtCore.QCoreApplication.instance().thread()


def test_worker_returns_to_main_thread_even_when_run_raises(qtbot: QtBot) -> None:
    class _Failing(QtCore.QObject):
        @QtCore.Slot()
        @returns_to_main_thread
        def run(self) -> None:
            raise RuntimeError("boom")

    thread = QtCore.QThread()
    worker = _Failing()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    thread.start()
    # The raising slot never calls quit(); the finally has already run by
    # the time the worker is back on the main thread.
    qtbot.waitUntil(
        lambda: worker.thread() == QtCore.QCoreApplication.instance().thread(),
        timeout=5000,
    )
    thread.quit()
    assert thread.wait(5000)


def test_gui_io_worker_ends_up_on_the_main_thread(qtbot: QtBot) -> None:
    """Checked at thread.finished - the moment deleteLater is queued - since
    the worker is deleted shortly afterwards and can't be inspected later."""
    gui = Gui()
    qtbot.addWidget(gui)
    release = threading.Event()

    def task(report):
        # Hold the task open until the test has hooked thread.finished.
        release.wait(5)
        return 1

    assert gui._dispatch_io_task(task, "t", lambda _r: None)
    worker = gui._io_worker
    thread = gui._io_thread
    assert worker is not None and thread is not None
    worker_thread_at_finish: list[QtCore.QThread] = []
    thread.finished.connect(lambda: worker_thread_at_finish.append(worker.thread()))
    release.set()

    qtbot.waitUntil(lambda: bool(worker_thread_at_finish), timeout=5000)
    qtbot.waitUntil(lambda: gui._io_worker is None, timeout=5000)

    assert worker_thread_at_finish[0] == QtCore.QCoreApplication.instance().thread()
    gui.close()
