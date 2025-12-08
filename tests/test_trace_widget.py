from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import TraceData
from neuroseq.trace_widget import TraceNormDialog
from neuroseq.trace_widget import TraceWidget
from neuroseq.utils import neuron_indices_to_colors


def test_trace_widget_zero_traces(qtbot: QtBot) -> None:
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
        traces=np.zeros(shape=(n_neurons, n_timestamps), dtype=np.float32),
        indices=np.arange(n_neurons, dtype=np.uint32),
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
    # changing the normalization doesn't change zero
    trace_norm_dialog.update_norm_input(200)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # nor does disabling the offset, since the median is also zero
    trace_norm_dialog.chk_offset_clicked(False)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # nor does disabling clipping
    trace_norm_dialog.chk_clip_clicked(False)
    assert np.mean(trace_widget.graphic.data[:, 1]) == pytest.approx(0.0, abs=0.01)
    trace_widget.close()


def test_trace_widget_median_offset(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    trace_norm_dialog = TraceNormDialog(trace_widget)
    qtbot.addWidget(trace_widget)
    qtbot.addWidget(trace_norm_dialog)
    trace_widget.show()
    trace_norm_dialog.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    data = TraceData(
        name="traces",
        traces=np.array(
            [np.zeros(100), np.ones(100), -2 * np.ones(100)], dtype=np.float32
        ),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    # default normalisation with median offset: all three constant traces should be 0
    assert trace_widget.offset_by_median is True
    assert np.array(trace_widget.graphic.data[:]).shape == (3, 100, 3)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    # disable median offset: traces should now be 0.0, 0.25, -0.5
    trace_norm_dialog.chk_offset_clicked(False)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.25, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-0.5, abs=0.01)
    # increase normalization to 200%: traces should now be 0.0, 0.5, -1.0 (clipped to -0.5)
    trace_norm_dialog.update_norm_input(200)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.5, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-0.5, abs=0.01)
    # disable clipping: traces should now be 0.0, 0.5, -1.0
    trace_norm_dialog.chk_clip_clicked(False)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.5, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(-1.0, abs=0.01)
    # re-enable median offset: all three constant traces should be 0
    trace_norm_dialog.chk_offset_clicked(True)
    assert np.mean(trace_widget.graphic[0].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[1].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    assert np.mean(trace_widget.graphic[2].data[:, 1]) == pytest.approx(0.0, abs=0.01)
    trace_widget.close()


def test_trace_widget_indices(qtbot: QtBot) -> None:
    controller = ConstrainedPanZoomController()
    trace_widget = TraceWidget(controller)
    qtbot.addWidget(trace_widget)
    trace_widget.show()
    qtbot.waitUntil(lambda: trace_widget.isVisible())
    data = TraceData(
        name="traces",
        traces=np.random.random_sample((5, 100)).astype(np.float32),
        indices=np.array([0, 1, 6, 11, 15], dtype=np.uint32),
        dt=0.001,
        units="mV",
    )
    colormap = {i: neuron_indices_to_colors(np.array([i]))[0] for i in data.indices}
    data.traces[0, :] = 0.0
    trace_widget.set_data(data)
    assert trace_widget.graphic is not None
    for i, line in enumerate(trace_widget.graphic.graphics):
        idx = data.indices[i]
        assert line.offset[1] == pytest.approx(idx)
        assert np.allclose(line.colors[0], colormap[data.indices[i]])
    # now set same data with same indices in a different order
    data.indices = np.array([15, 6, 11, 1, 0], dtype=np.uint32)
    trace_widget.set_data(data)
    for i, line in enumerate(trace_widget.graphic.graphics):
        idx = data.indices[i]
        assert line.offset[1] == pytest.approx(idx)
        assert np.allclose(line.colors[0], colormap[data.indices[i]])


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
        indices=np.arange(n_neurons, dtype=np.uint32),
        dt=dt,
        units="mV",
    )
    trace_widget.offset_by_median = False
    trace_widget.set_data(data)
    assert trace_widget.n_neurons == n_neurons
    assert trace_widget.n_timestamps == n_timestamps
    assert trace_widget.graphic is not None
    assert trace_widget.graphic.data is not None
    # default normalisation with median offset disabled: all traces should be in range [-0.5, 0.5]
    # and the max abs value should be 0.5
    # (as we are testing values from the gpu allow for some floating point rounding errors)
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.51
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.51
    # update normalisation using the dialog to 500% with clipping, no offset
    trace_norm_dialog.clip = True
    trace_norm_dialog.offset_by_median = False
    trace_norm_dialog.update_norm_input(500)
    assert trace_widget.trace_norm_percent == 500
    assert trace_widget.clip is True
    assert trace_widget.offset_by_median is False
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        0.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 0.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -0.50001
    # diable clipping
    trace_norm_dialog.chk_clip_clicked(False)
    assert trace_widget.clip is False
    assert trace_widget.trace_norm_percent == 500
    assert np.max(np.abs(trace_widget.graphic.data[:, 1])) == pytest.approx(
        2.5, abs=0.01
    )
    assert np.max(trace_widget.graphic.data[:, 1]) <= 2.50001
    assert np.min(trace_widget.graphic.data[:, 1]) >= -2.50001
    # reset to default
    trace_norm_dialog.reset_defaults()
    assert trace_widget.trace_norm_percent == 100
    assert trace_widget.clip is True
    assert trace_widget.offset_by_median is True
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
    trace_widget.close()
