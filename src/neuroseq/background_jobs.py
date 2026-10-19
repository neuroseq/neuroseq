"""Workers and lifecycle ownership for main-window background jobs."""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
from collections.abc import Callable
from typing import Any

from qtpy import QtCore
from qtpy import QtWidgets

logger = logging.getLogger(__name__)


def wait_for_jobs(
    jobs: list[BackgroundJob],
    *,
    on_idle: Callable[[], None],
    on_timeout: Callable[[], None],
    timeout_ms: int,
    parent: QtCore.QObject,
) -> QtCore.QObject:
    """Call ``on_idle`` once every job has ended, or ``on_timeout`` first.

    Neither callback runs inside a job's completion callbacks: both are
    deferred to the event loop so a caller may safely dispose the jobs' owners.
    Returns the waiter; deleting it abandons the wait.
    """
    waiter = QtCore.QObject(parent)
    timer = QtCore.QTimer(waiter)
    timer.setSingleShot(True)

    def _finish(callback: Callable[[], None]) -> None:
        timer.stop()
        for job in jobs:
            with contextlib.suppress(TypeError, RuntimeError):
                job.idle.disconnect(_check)
        waiter.deleteLater()
        QtCore.QTimer.singleShot(0, callback)

    def _check() -> None:
        if not any(job.busy for job in jobs):
            _finish(on_idle)

    for job in jobs:
        job.idle.connect(_check)
    timer.timeout.connect(lambda: _finish(on_timeout))
    timer.start(timeout_ms)
    _check()
    return waiter


def returns_to_main_thread[F: Callable[..., Any]](run: F) -> F:
    """Decorate a worker's ``run`` slot: once it returns, move the worker
    back to the main thread.

    Workers are normally cleaned up with ``thread.finished -> worker.
    deleteLater``, which queues the deletion on the *worker's* thread - the
    one that is shutting down. Destroying a PyQt-wrapped QObject there takes
    the GIL (sip checks for Python reimplementations of
    ``disconnectNotify``) while Qt already holds one of its signal/slot
    mutexes. Qt shares those mutexes between unrelated objects (they are
    picked from a small pool by address), so the main thread - holding the
    GIL and connecting some unrelated signal - can wait on the same mutex:
    a lock-order deadlock, intermittent by nature, that froze CI runs until
    the job timeout. pytest-timeout's thread method cannot fire either,
    since the main thread holds the GIL. With the worker back on the main
    thread, ``deleteLater`` runs there instead and the cycle cannot form.

    Signals the worker emitted while running are unaffected (they were
    queued to the main thread already).
    """

    @functools.wraps(run)
    def wrapper(self: QtCore.QObject, *args: Any, **kwargs: Any) -> Any:
        try:
            return run(self, *args, **kwargs)
        finally:
            app = QtCore.QCoreApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())

    return wrapper  # type: ignore[return-value]


class SimulationGuard:
    """Process-wide owner token for simulations that touch Brian2 globals.

    Brian2 keeps its random seed, code-generation preferences and device in
    process-global state, so two workspaces simulating on worker threads at
    once would corrupt each other's runs. A workspace acquires the guard
    before dispatching such a simulation and releases it when the job ends or
    the workspace closes. There is no queue: a refused start is reported.
    """

    _owner: object | None = None
    _description = ""

    @classmethod
    def acquire(cls, owner: object, description: str) -> str | None:
        """Take the guard for ``owner``; return the busy owner's description
        if another owner holds it."""
        if cls._owner is not None and cls._owner is not owner:
            return cls._description
        cls._owner = owner
        cls._description = description
        return None

    @classmethod
    def release(cls, owner: object) -> None:
        """Release the guard if ``owner`` holds it; otherwise do nothing."""
        if cls._owner is owner:
            cls._owner = None
            cls._description = ""

    @classmethod
    def held_by(cls) -> object | None:
        return cls._owner


class SimulationCancelledError(Exception):
    """Raised from a progress callback when the user clicks Cancel."""


class BackgroundIOWorker(QtCore.QObject):
    """Runs a blocking IO task on a worker QThread.

    The task is a callable taking a single argument: a
    ``report(percent: int)`` callback that the worker translates into the
    ``progress(int)`` signal. Tasks that don't natively report progress can
    ignore the callback. The task's return value is delivered to the main
    thread via ``finished(result)``; any exception is stringified and
    delivered via ``failed(str)``.
    """

    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, task: Callable[[Callable[[int], None]], object]) -> None:
        super().__init__()
        self._task = task

    @QtCore.Slot()
    @returns_to_main_thread
    def run(self) -> None:
        def _report(percent: int) -> None:
            with contextlib.suppress(Exception):
                self.progress.emit(int(percent))

        try:
            result = self._task(_report)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class ExternalSimWorker(QtCore.QObject):
    """Runs an external simulation on a worker QThread."""

    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(
        self,
        simulate_fn: Callable,
        stims_dict: dict,
        simulation_time: float,
        params: dict,
        accepts_progress: bool,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._simulate_fn = simulate_fn
        self._stims_dict = stims_dict
        self._simulation_time = float(simulation_time)
        self._params = params
        self._accepts_progress = accepts_progress
        self._cancel_event = cancel_event

    @QtCore.Slot()
    @returns_to_main_thread
    def run(self) -> None:
        def _progress_cb(fraction: float) -> None:
            if self._cancel_event.is_set():
                raise SimulationCancelledError("user cancelled")
            with contextlib.suppress(Exception):
                self.progress.emit(float(fraction))

        try:
            if self._accepts_progress:
                experiment = self._simulate_fn(
                    self._stims_dict,
                    self._simulation_time,
                    params=self._params,
                    progress=_progress_cb,
                )
            else:
                experiment = self._simulate_fn(
                    self._stims_dict,
                    self._simulation_time,
                    params=self._params,
                )
        except SimulationCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(experiment)


class EliseProtocolWorker(QtCore.QObject):
    """Runs the ELiSe train/validation/replay protocol on a worker QThread."""

    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(
        self,
        run_fn: Callable,
        kwargs: dict,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._run_fn = run_fn
        self._kwargs = kwargs
        self._cancel_event = cancel_event

    @QtCore.Slot()
    @returns_to_main_thread
    def run(self) -> None:
        def _progress_cb(fraction: float) -> None:
            if self._cancel_event.is_set():
                raise SimulationCancelledError("user cancelled")
            with contextlib.suppress(Exception):
                self.progress.emit(float(fraction))

        try:
            result = self._run_fn(progress=_progress_cb, **self._kwargs)
        except SimulationCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class CancellableWorker(QtCore.QObject):
    """Base for job-specific workers with cooperative cancellation.

    Subclasses implement ``run`` (decorated with ``returns_to_main_thread``)
    and emit exactly one of ``finished``, ``failed`` or ``cancelled``.
    ``progress`` carries a task-defined payload; ``cancel`` may be called from
    any thread and should make the task stop at its next checkpoint.
    """

    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        """Ask the task to stop at its next progress checkpoint."""
        self._cancel_requested.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def run(self) -> None:
        raise NotImplementedError


type JobWorker = (
    BackgroundIOWorker | ExternalSimWorker | EliseProtocolWorker | CancellableWorker
)


class BackgroundJob(QtCore.QObject):
    """Own one worker and its thread until both execution and cleanup finish.

    Workers retain their task-specific signals and cancellation contracts.
    Completion callbacks run on the main thread after the worker thread exits,
    with the owner idle so a callback can immediately dispatch the next job.
    ``idle`` is emitted once a job has ended and its callbacks have returned.
    """

    idle = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.worker: JobWorker | None = None
        self.worker_thread: QtCore.QThread | None = None
        self._on_progress: Callable[[Any], None] | None = None
        self._on_finished: Callable[[Any], None] | None = None
        self._on_failed: Callable[[str], None] | None = None
        self._on_cancelled: Callable[[], None] | None = None
        self._cancel: Callable[[], None] | None = None
        self._outcome: tuple[str, Any] | None = None

    @property
    def busy(self) -> bool:
        return self.worker_thread is not None

    @property
    def cancellable(self) -> bool:
        return self._cancel is not None

    def start(
        self,
        worker: JobWorker,
        *,
        on_progress: Callable[[Any], None],
        on_finished: Callable[[Any], None],
        on_failed: Callable[[str], None],
        on_cancelled: Callable[[], None] | None = None,
        cancel: Callable[[], None] | None = None,
    ) -> None:
        if self.busy:
            raise RuntimeError("A background job is already running.")
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_cancelled = on_cancelled
        self._cancel = cancel
        self._outcome = None
        thread = QtCore.QThread(self)
        self.worker = worker
        self.worker_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finished)
        worker.failed.connect(self._failed)
        terminal_signals = [worker.finished, worker.failed]
        if isinstance(
            worker, (ExternalSimWorker, EliseProtocolWorker, CancellableWorker)
        ):
            worker.cancelled.connect(self._cancelled)
            terminal_signals.append(worker.cancelled)
        for signal in terminal_signals:
            # QThread.quit is thread-safe. Calling it directly also lets stop()
            # wait on the GUI thread without blocking a queued quit request.
            signal.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @QtCore.Slot(int)
    @QtCore.Slot(float)
    @QtCore.Slot(object)
    def _progress(self, value: object) -> None:
        if self.sender() is self.worker and self._on_progress is not None:
            self._on_progress(value)

    @QtCore.Slot(object)
    def _finished(self, result: object) -> None:
        if self.sender() is self.worker:
            self._outcome = ("finished", result)

    @QtCore.Slot(str)
    def _failed(self, message: str) -> None:
        if self.sender() is self.worker:
            self._outcome = ("failed", message)

    @QtCore.Slot()
    def _cancelled(self) -> None:
        if self.sender() is self.worker:
            self._outcome = ("cancelled", None)

    @QtCore.Slot()
    def _thread_finished(self) -> None:
        if self.sender() is self.worker_thread:
            self._complete(deliver=True)

    def _complete(self, *, deliver: bool) -> None:
        outcome = self._outcome
        on_finished, on_failed, on_cancelled = (
            self._on_finished,
            self._on_failed,
            self._on_cancelled,
        )
        self.worker = None
        self.worker_thread = None
        self._outcome = None
        self._on_progress = None
        self._on_finished = None
        self._on_failed = None
        self._on_cancelled = None
        self._cancel = None
        try:
            if not deliver or outcome is None:
                return
            kind, value = outcome
            if kind == "finished" and on_finished is not None:
                on_finished(value)
            elif kind == "failed" and on_failed is not None:
                on_failed(value)
            elif kind == "cancelled" and on_cancelled is not None:
                on_cancelled()
        except Exception as exc:
            # This runs from a Qt slot: an escaping exception aborts PyQt6
            # (qFatal) when no sys.excepthook is installed.
            logger.exception("Handling a background job's %s outcome failed", kind)
            parent = self.parent()
            QtWidgets.QMessageBox.critical(
                parent if isinstance(parent, QtWidgets.QWidget) else None,
                "Unexpected error",
                "A background task ended, but handling its result failed:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            self.idle.emit()

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel()

    def stop(self, timeout_ms: int) -> bool:
        """Cancel and drain the job before its owner closes.

        On timeout retain ownership so the window can remain open safely.
        On success suppress UI callbacks that would reopen dialogs at shutdown.
        """
        if self.worker_thread is None:
            return True
        self.cancel()
        self.worker_thread.quit()
        if not self.worker_thread.wait(timeout_ms):
            return False
        self._complete(deliver=False)
        return True
