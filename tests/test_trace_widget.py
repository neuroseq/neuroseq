from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.experiment import TraceData
from neuroseq.trace_widget import TraceNormDialog
from neuroseq.trace_widget import TraceWidget


def test_trace_widget_zero_traces(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 0.0991
    data = TraceData(
        name="traces",
        traces=np.zeros(shape=(n_neurons, n_timestamps), dtype=np.float32),
        dt=dt,
        units="mV",
    )
    trace_widget.set_data(data)
    # if all traces are zero we don't normalize by the max value (which is zero),
    # so the plotted values should still be zero
    # (as we are testing values from the gpu allow for some floating point rounding errors)
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.0, abs=0.01
    )
    assert np.min(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.0, abs=0.01
    )


def test_trace_widget(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)
    trace_widget.show()
    trace_norm_dialog.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    n_neurons = 20
    n_timestamps = 100
    dt = 0.0991
    data = TraceData(
        name="traces",
        traces=np.random.rand(n_neurons, n_timestamps),
        dt=dt,
        units="mV",
    )
    trace_widget.set_data(data)
    assert trace_widget.n_neurons == n_neurons
    assert trace_widget.n_timestamps == n_timestamps
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    # default normalisation: traces should be in range [-0.5, 0.5]
    # (as we are testing values from the gpu allow for some floating point rounding errors)
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # update normalisation using the dialog to 500% with clipping
    trace_norm_dialog.clip = True
    trace_norm_dialog.update_norm_input(500)
    assert trace_widget.clip is True
    assert trace_widget.trace_norm_percent == 500
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # diable clipping
    trace_norm_dialog.check_clip_clicked(False)
    assert trace_widget.clip is False
    assert trace_widget.trace_norm_percent == 500
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        2.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 2.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -2.50001
    # reset to default
    trace_norm_dialog.reset_norm()
    assert trace_widget.trace_norm_percent == 100
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # initially no neurons are selected
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
    trace_widget.select_neurons(slice(1, 3), slice(10, 20))
    # wait for the widget to update
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice == slice(1, 3)
    assert trace_widget.selected_time_slice == slice(10, 20)
    # de-selecting neurons reverts to displaying all of them as selected
    trace_widget.unselect_neurons()
    qtbot.wait(20)
    assert trace_widget.selected_neuron_slice is None
    assert trace_widget.selected_time_slice is None
