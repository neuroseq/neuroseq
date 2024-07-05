from __future__ import annotations

from typing import cast

import brian2 as b2
import brian2.numpy_ as np
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


def test_stim_dialog_console(qtbot: QtBot) -> None:
    n_neurons = 20
    n_timestamps = 100
    dt = 10 * b2.ms
    stims = np.zeros((n_neurons, n_timestamps))
    # bar plot x-values are the center of the bars, so shifted forwards by 0.5dt compared to timestamps
    expected_timestamps_array = dt * (np.arange(n_timestamps) + 0.5) / b2.second
    stim_dialog = StimDialog(stims, dt)
    qtbot.addWidget(stim_dialog)
    stim_dialog.show()
    qtbot.waitUntil(lambda: stim_dialog.isVisible())
    assert stim_dialog.stims is not None
    assert np.allclose(stim_dialog.stims, np.zeros_like(stims))
    assert not stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    # set all stimuli to 1: valid stims, StimWidget updated, ok button enabled
    stim_dialog.stim_console_widget.input.setText("stims[:,:] = 1")
    qtbot.keyPress(stim_dialog.stim_console_widget.input, QtCore.Qt.Key.Key_Enter)
    assert np.allclose(stim_dialog.stims, np.ones_like(stims))
    assert stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == n_neurons
    neuron_plot = cast(pg.BarGraphItem, stim_dialog.stim_widget.neuron_plots[0])
    assert len(neuron_plot.getData()[0]) == n_timestamps
    assert np.allclose(neuron_plot.getData()[0], expected_timestamps_array)
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

    assert stim_dialog.button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert len(stim_dialog.stim_widget.neuron_plots) == 83


def test_stim_dialog_edit(qtbot: QtBot) -> None:
    n_neurons = 10
    n_timestamps = 20
    dt = 10 * b2.ms
    stims = np.ones((n_neurons, n_timestamps))
    selection = np.random.randint(3, size=(2, 5))
    stims[1:3, 2:7] = selection
    stim_dialog = StimDialog(stims, dt)
    qtbot.addWidget(stim_dialog)
    stim_dialog.show()
    qtbot.waitUntil(lambda: stim_dialog.isVisible())
    assert stim_dialog.stims is not None
    assert np.allclose(stim_dialog.stims, stims)
    # delete a slice
    stim_dialog.stim_widget.select_neurons(slice(1, 3), slice(10, 20))
    stim_dialog.delete_selected_stim()
    assert np.allclose(stim_dialog.stims[1:3, 10:20], np.zeros((2, 10)))
    # move a slice up
    stim_dialog.stim_widget.select_neurons(slice(1, 3), slice(2, 7))
    stim_dialog.move_selected_stim_up()
    assert np.allclose(stim_dialog.stims[2:4, 2:7], selection)
    # move a slice down
    stim_dialog.move_selected_stim_down()
    assert np.allclose(stim_dialog.stims[1:3, 2:7], selection)
    # move a slice left
    stim_dialog.move_selected_stim_left()
    assert np.allclose(stim_dialog.stims[1:3, 1:6], selection)
    # move a slice right
    stim_dialog.move_selected_stim_right()
    assert np.allclose(stim_dialog.stims[1:3, 2:7], selection)
    # copy a slice
    stim_dialog.copy_selected_stim()
    assert np.allclose(stim_dialog.stims[1:3, 2:7], selection)
    # cut a slice
    stim_dialog.cut_selected_stim()
    assert np.allclose(stim_dialog.stims[1:3, 2:7], np.zeros_like(selection))
    # paste a slice into empty region starting at timeslice 10
    stim_dialog.stim_widget.timeline.setX(0.5 * n_timestamps * dt / b2.second)
    assert np.allclose(stim_dialog.stims[1:3, 10:15], np.zeros_like(selection))
    stim_dialog.paste_selected_stim()
    assert np.allclose(stim_dialog.stims[1:3, 10:15], selection)
