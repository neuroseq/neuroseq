from __future__ import annotations

import numpy as np
from formozov_project.stim_dialog import StimDialog
from pytest import approx
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets


def test_stim_dialog_no_data(qtbot: QtBot) -> None:
    stim_dialog = StimDialog()
    qtbot.addWidget(stim_dialog)
    stim_dialog.show()
    qtbot.waitUntil(lambda: stim_dialog.isVisible())
    stim_dialog.close()
    qtbot.waitUntil(lambda: not stim_dialog.isVisible())


def test_stim_dialog_data(qtbot: QtBot) -> None:
    n_neurons = 20
    n_timestamps = 100
    dt = 0.01
    stim = np.zeros((n_neurons, n_timestamps))
    stim_dialog = StimDialog(stim, dt)
    qtbot.addWidget(stim_dialog)
    stim_dialog.show()
    qtbot.waitUntil(lambda: stim_dialog.isVisible())
    assert stim_dialog.stim is not None
    assert np.allclose(stim_dialog.stim, np.zeros_like(stim))
    assert stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    # set all stimuli to 1: valid stim, StimWidget updated, ok button enabled
    stim_dialog.stim_console_widget.input.setText("stim[:,:] = 1")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_dialog.stim, np.ones_like(stim))
    assert stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == n_neurons
    assert stim_dialog.stim_widget.neuron_plots[0].getData()[0][-1] == approx(
        dt * (n_timestamps - 1)
    )
    # orginal data should not be altered
    assert np.allclose(stim, np.zeros_like(stim))
    # set stim to an invalid value: StimWidget not updated, ok button disabled
    stim_dialog.stim_console_widget.input.setText("stim = 5")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert not stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == n_neurons
    assert np.allclose(stim_dialog.stim, np.ones_like(stim))
    assert np.allclose(stim, np.zeros_like(stim))
    # assign random valid stim array with different shape, StimWidget updated, ok button enabled
    stim_dialog.stim_console_widget.input.setText("stim = np.random.rand(83, 127)")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert stim_dialog.stim.shape == (83, 127)
    assert stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == 83
