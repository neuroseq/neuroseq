"""Helpers for QObject workers that run on a QThread."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from qtpy import QtCore


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
