import gc

from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from tests.wait_timeout import WAIT_TIMEOUT_MS


class _CollectGarbageOnPaint(QtWidgets.QWidget):
    def paintEvent(self, event) -> None:
        # Stands in for any Python code run while painting that happens to
        # trigger the garbage collector.
        gc.collect()
        super().paintEvent(event)


def test_window_left_open_is_not_freed_while_it_is_painted(qtbot: QtBot) -> None:
    # A visible window left only reachable through a reference cycle must not
    # be painted after the test: the collector would then delete it in the
    # middle of the paint, and Qt would crash. tests/conftest.py closes it first.
    window = QtWidgets.QWidget()
    window.cycle = window  # type: ignore[attr-defined]
    _CollectGarbageOnPaint(window).resize(50, 50)
    window.show()
    qtbot.waitExposed(window, timeout=WAIT_TIMEOUT_MS)
    window.update()
