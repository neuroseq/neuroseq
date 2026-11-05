from __future__ import annotations

import numpy as np
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq
import neuroseq.stimulation.stim
import neuroseq.stimulation.stim_editor
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.stimulation.stim_editor import StimEditor
from neuroseq.stimulation.stim_loop_dialog import StimLoopDialog
from neuroseq.utils import detect_period
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _make_project(*stim_arrays: StimArray) -> Project:
    project = Project()
    for stim_array in stim_arrays:
        project.add_stim_array(stim_array)
    return project


def _make_stim(
    name: str,
    values: np.ndarray,
    dt: float = 0.001,
) -> StimArray:
    return StimArray(
        name=name,
        stim_type=StimType.Excitation,
        values=np.asarray(values, dtype=np.float32),
        units="mV",
        dt=dt,
    )


def test_loop_stim_arrays_tiles_values() -> None:
    values = np.arange(400, dtype=np.float32).reshape(100, 4)
    project = _make_project(_make_stim("stim", values))

    project.loop_stim_arrays(3)

    assert project.stim_arrays["stim"].values.shape == (300, 4)
    assert np.allclose(project.stim_arrays["stim"].values, np.tile(values, (3, 1)))


def test_loop_stim_arrays_loops_all_stims_simultaneously() -> None:
    project = _make_project(
        _make_stim("a", np.ones((100, 4), dtype=np.float32)),
        _make_stim("b", np.ones((100, 2), dtype=np.float32) * 2.0),
    )

    project.loop_stim_arrays(2)

    assert project.stim_arrays["a"].values.shape == (200, 4)
    assert project.stim_arrays["b"].values.shape == (200, 2)


def test_loop_stim_arrays_n_equals_one_is_noop() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    project = _make_project(_make_stim("stim", values.copy()))

    project.loop_stim_arrays(1)

    assert project.stim_arrays["stim"].values.shape == values.shape
    assert np.allclose(project.stim_arrays["stim"].values, values)


def test_loop_stim_arrays_rejects_zero_and_negative() -> None:
    project = _make_project(_make_stim("stim", np.zeros((10, 3), dtype=np.float32)))

    for n_loops in [0, -3]:
        try:
            project.loop_stim_arrays(n_loops)
        except ValueError as exc:
            assert "n_loops must be >= 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_detect_period_finds_minimum() -> None:
    rng = np.random.default_rng(123)
    block = rng.normal(size=(7, 3)).astype(np.float32)
    values = np.tile(block, (5, 1))

    assert detect_period(values) == 7

    values[10, 1] += 0.25
    assert detect_period(values) is None


def test_detect_period_rejects_non_dividing() -> None:
    rng = np.random.default_rng(42)
    block = rng.normal(size=(240, 13)).astype(np.float32)
    periodic_values = np.tile(block, (8, 1))
    padded_values = np.vstack(
        [
            periodic_values,
            np.zeros((80, periodic_values.shape[1]), dtype=np.float32),
        ]
    )

    assert padded_values.shape == (2000, 13)
    assert detect_period(padded_values) is None
    assert detect_period(periodic_values) == 240


def test_detect_period_handles_short_arrays() -> None:
    assert detect_period(np.zeros((0, 3), dtype=np.float32)) is None
    assert detect_period(np.ones((1, 3), dtype=np.float32)) is None


def test_stim_loop_dialog_accept_returns_loop_count(qtbot: QtBot) -> None:
    project = _make_project(_make_stim("stim", np.zeros((10, 2), dtype=np.float32)))
    dialog = StimLoopDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible, timeout=WAIT_TIMEOUT_MS)

    dialog.spinbox.setValue(4)
    dialog.accept()

    assert dialog.loop_count == 4
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_stim_editor_loop_stim_array_accepts_and_tiles(
    monkeypatch,
    qtbot: QtBot,
) -> None:
    class _AcceptedDialog:
        def __init__(self, project: Project, parent=None) -> None:
            self.loop_count = 3

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    project = _make_project(
        _make_stim("stim", np.arange(12, dtype=np.float32).reshape(4, 3))
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(stim_editor.isVisible, timeout=WAIT_TIMEOUT_MS)
    stim_editor.set_project(project)
    monkeypatch.setattr(
        neuroseq.stimulation.stim_editor, "StimLoopDialog", _AcceptedDialog
    )

    stim_editor.loop_stim_array()

    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.values.shape == (12, 3)


def test_stim_editor_loop_stim_array_rejected_refreshes_trim(
    monkeypatch,
    qtbot: QtBot,
) -> None:
    class _RejectedDialog:
        def __init__(self, project: Project, parent=None) -> None:
            project.trim_stim_arrays(0.002)
            self.loop_count = 2

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Rejected)

    project = _make_project(
        _make_stim("stim", np.arange(12, dtype=np.float32).reshape(4, 3))
    )
    cmb_stim = QtWidgets.QComboBox()
    toolbar = QtWidgets.QToolBar()
    controller = ConstrainedPanZoomController()
    stim_editor = StimEditor(controller, cmb_stim, toolbar)
    qtbot.addWidget(stim_editor)
    stim_editor.show()
    qtbot.waitUntil(stim_editor.isVisible, timeout=WAIT_TIMEOUT_MS)
    stim_editor.set_project(project)
    monkeypatch.setattr(
        neuroseq.stimulation.stim_editor, "StimLoopDialog", _RejectedDialog
    )

    stim_editor.loop_stim_array()

    assert stim_editor.current_stim_modifier is not None
    assert stim_editor.current_stim_modifier.stim_array.values.shape == (2, 3)
