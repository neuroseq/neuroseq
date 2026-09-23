from __future__ import annotations

import random

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import console_art
from neuroseq import theme
from neuroseq.console_widget import SAMPLE_CODE
from neuroseq.console_widget import ConsoleWidget
from neuroseq.project import Project


def test_console(qtbot: QtBot, test_project: Project) -> None:
    signals: list[bool] = []
    console = ConsoleWidget()
    qtbot.addWidget(console)
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    console.console_command_entered.connect(lambda: signals.append(True))
    assert "np" in console.namespace()
    assert "b2" in console.namespace()
    assert "plt" in console.namespace()
    assert console.namespace()["project"] is None
    # start with a valid project
    console.set_project(test_project)
    assert "np" in console.namespace()
    assert console.namespace()["project"] is not None
    # define a new variable in the console namespace
    console.input.setText("x = 5")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert console.namespace()["x"] == 5
    assert len(signals) == 1
    # modify a stim array in the project in a valid way
    target_a = np.ones((5, 20))
    console.input.setText("project.stim_arrays['stimulus'].values = np.ones((5,20))")
    qtbot.keyPress(console.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(test_project.stim_arrays["stimulus"].values, target_a)
    assert len(signals) == 2


def test_retro_theme_preserves_console_banner_character_spacing(qtbot: QtBot) -> None:
    console = ConsoleWidget()
    qtbot.addWidget(console)
    original_family = console.output.font().family()
    console.setStyleSheet(theme.app_stylesheet(theme.RETRO))
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    assert console.output.font().family() == original_family
    assert (
        console.output.document().defaultTextOption().flags()
        & QtGui.QTextOption.Flag.IncludeTrailingSpaces
    )
    metrics = QtGui.QFontMetricsF(console.output.font())
    assert metrics.horizontalAdvance("#") == pytest.approx(
        metrics.horizontalAdvance(" ")
    )
    console.setStyleSheet("")
    assert console.output.font().family() == original_family


def test_console_scrolls_back_to_welcome_splash_on_startup(qtbot: QtBot) -> None:
    console = ConsoleWidget()
    qtbot.addWidget(console)
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    # the intro text is long enough to auto-scroll the view to the bottom as
    # it's inserted - it must land back at the top so the "Welcome to
    # NeuroSeq" splash (the very first line) is what's actually visible.
    qtbot.waitUntil(lambda: console.output.verticalScrollBar().value() == 0)


def test_console_retints_the_brain_image_on_construction_and_every_show(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The brain should get a freshly randomized tint at launch, and again
    every time the console is closed and reopened - not just once."""
    calls: list[int] = []
    original = console_art.tinted_brain_image_bytes

    def _tracked(rng: random.Random | None = None) -> bytes:
        calls.append(1)
        return original(rng)

    monkeypatch.setattr(console_art, "tinted_brain_image_bytes", _tracked)

    console = ConsoleWidget()
    qtbot.addWidget(console)
    # __init__ (_print_intro) already retints once, before the widget is
    # ever shown.
    assert len(calls) == 1

    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    after_first_show = len(calls)
    assert after_first_show > 1

    console.hide()
    console.show()
    qtbot.waitUntil(lambda: console.isVisible())
    assert len(calls) > after_first_show


@pytest.mark.parametrize("method", ["_show_sample_code", "_show_console_history"])
def test_console_help_dialogs_allow_console_interaction(qtbot: QtBot, method) -> None:
    console = ConsoleWidget()
    qtbot.addWidget(console)
    console.show()
    getattr(console, method)()
    dialog = console.findChild(QtWidgets.QDialog)
    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert QtWidgets.QApplication.activeModalWidget() is None
    qtbot.keyClicks(console.input, "1 + 1")
    assert console.input.text() == "1 + 1"
    with qtbot.waitSignal(dialog.destroyed):
        dialog.close()


def test_sample_code_mentions_sequential_pulse_snippet() -> None:
    """The console help dialog must surface the 100 ms / 18 ms snippet."""
    assert "Sequential 100 ms pulses" in SAMPLE_CODE
    assert "18 ms" in SAMPLE_CODE
    assert "pulse_n" in SAMPLE_CODE
    assert "gap_n" in SAMPLE_CODE


def test_sequential_pulse_snippet_is_repl_safe() -> None:
    """The pyqtgraph REPL accumulates lines into a compound statement until
    it sees what looks like the end of a block. ``if/else`` followed by a
    same-indent statement triggers a SyntaxError. Keep the snippet free of
    control flow that needs a trailing blank line to terminate."""
    pulse_block_start = SAMPLE_CODE.index("Sequential 100 ms pulses")
    # Snippet runs from its banner up to the next blank line.
    block_end = SAMPLE_CODE.index("\n\n", pulse_block_start)
    pulse_block = SAMPLE_CODE[pulse_block_start:block_end]
    # The only acceptable compound statement is the trailing for-loop,
    # which the REPL terminates on its own next blank line.
    lines = pulse_block.splitlines()
    compound_lines = [line for line in lines if line.endswith(":")]
    assert compound_lines == ["for i in range(n_neurons):"]


def _run_sequential_pulse_snippet(
    initial_shape: tuple[int, int], dt: float
) -> np.ndarray:
    """Mirror the snippet's logic against a fresh array — single-space
    snippet indentation isn't valid Python, so we transcribe it here."""
    sa = type("StimArray", (), {})()
    sa.values = np.zeros(initial_shape, dtype=np.float32)
    sa.dt = dt

    pulse_n = int(round(0.100 / sa.dt))
    gap_n = int(round(0.018 / sa.dt))
    n_neurons = sa.values.shape[1]
    total_n = n_neurons * (pulse_n + gap_n)
    amp = 1.0
    new_len = max(sa.values.shape[0], total_n)
    sa.values = np.zeros((new_len, n_neurons), dtype=sa.values.dtype)
    for i in range(n_neurons):
        t0 = i * (pulse_n + gap_n)
        sa.values[t0 : t0 + pulse_n, i] = amp
    return sa.values


def test_sequential_pulse_snippet_produces_expected_pattern() -> None:
    """One 100 ms pulse per neuron with an 18 ms inter-pulse gap, only one
    neuron active at any time."""
    n_neurons = 4
    # Start with an array that's already long enough — snippet keeps the
    # existing length and zeros the tail.
    values = _run_sequential_pulse_snippet((500, n_neurons), dt=0.001)

    # 100 + 18 = 118 samples per (pulse + gap) at 1 ms dt.
    assert values.shape == (500, n_neurons)
    for i in range(n_neurons):
        t0 = i * 118
        assert np.all(values[t0 : t0 + 100, i] == 1.0)
        if t0 > 0:
            assert np.all(values[t0 - 18 : t0, i] == 0.0)
        for j in range(n_neurons):
            if j != i:
                assert np.all(values[t0 : t0 + 100, j] == 0.0)
    # Tail past the last pulse stays zero.
    assert np.all(values[n_neurons * 118 :, :] == 0.0)


def test_sequential_pulse_snippet_resizes_when_too_short() -> None:
    """If the existing array is too short to fit all pulses, the snippet
    reallocates to exactly ``n_neurons * (pulse_n + gap_n)`` samples."""
    n_neurons = 4
    values = _run_sequential_pulse_snippet((50, n_neurons), dt=0.001)
    assert values.shape == (n_neurons * 118, n_neurons)
