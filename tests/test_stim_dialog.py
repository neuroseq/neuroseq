from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
from formozov_project.stim_dialog import StimDialog
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
    stims = np.zeros((n_neurons, n_timestamps))
    expected_timestamps = dt * np.arange(n_timestamps)
    stim_dialog = StimDialog(stims, dt)
    qtbot.addWidget(stim_dialog)
    stim_dialog.show()
    qtbot.waitUntil(lambda: stim_dialog.isVisible())

    assert stim_dialog.stims is not None
    assert np.allclose(stim_dialog.stims, np.zeros_like(stims))
    assert stim_dialog.buttonBox.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    # set all stimuli to 1: valid stims, StimWidget updated, ok button enabled
    stim_dialog.stim_console_widget.input.setText("stims[:,:] = 1")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)

    assert np.allclose(stim_dialog.stims, np.ones_like(stims))
    assert stim_dialog.buttonBox.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == n_neurons
    neuron_plot = cast(pg.BarGraphItem, stim_dialog.stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], expected_timestamps)
    # orginal data should not be altered
    assert np.allclose(stims, np.zeros_like(stims))
    # set stims to an invalid value: StimWidget not updated, ok button disabled
    stim_dialog.stim_console_widget.input.setText("stims = 5")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert not stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == n_neurons
    assert np.allclose(stim_dialog.stims, np.ones_like(stims))
    assert np.allclose(stims, np.zeros_like(stims))
    # assign random valid stim array with different shape, StimWidget updated, ok button enabled
    stim_dialog.stim_console_widget.input.setText("stims = np.random.rand(83, 127)")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)

    assert stim_dialog.stims.shape == (83, 127)
    assert stim_dialog.buttonBox.button(

        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == 83
