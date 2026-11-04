from __future__ import annotations

import math

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.visualization.heatmap_widget import HeatmapNormDialog
from neuroseq.visualization.heatmap_widget import HeatmapWidget
from neuroseq.visualization.time_slider_widget import TimeSliderWidget


def test_heatmap_widget_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    heatmap_norm_dialog = HeatmapNormDialog(heatmap_widget)
    qtbot.addWidget(heatmap_norm_dialog)
    heatmap_norm_dialog.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    # no data, so normalization options should be disabled
    assert not heatmap_norm_dialog.chk_subtract.isChecked()
    assert not heatmap_norm_dialog.chk_subtract.isEnabled()
    assert not heatmap_norm_dialog.slider_idx.isEnabled()


def test_heatmap_widget_context_menu_offers_normalization(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)

    heatmap_widget._rebuild_context_menu()
    action_texts = [action.text() for action in heatmap_widget.context_menu.actions()]
    assert "Normalization…" in action_texts


@pytest.mark.parametrize(
    "relative_playback_speed", [0.25, 0.66, 1.0, 1.71234, 2.0, 3.0]
)
def test_heatmap_widget_3d_data_with_timeline(
    qtbot: QtBot, relative_playback_speed: float, make_synapse_data
) -> None:
    heatmap_widget = HeatmapWidget()
    time_slider = TimeSliderWidget()
    qtbot.addWidget(heatmap_widget)
    qtbot.addWidget(time_slider)
    heatmap_widget.show()
    time_slider.show()
    time_slider.connect_widget(heatmap_widget)
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    n_i = 23
    n_j = 10
    n_timestamps = 10
    time_s = 0.5
    time_slider.set_max_time_secs(time_s)
    data = make_synapse_data("synapse", n_i, n_j, n_timestamps, max_time_secs=time_s)
    # SynapseData("synapse", values, timepoints_s)
    heatmap_widget.set_data(data)
    time_slider.set_relative_playback_speed(relative_playback_speed)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape == (n_timestamps, n_j, n_i)
    assert heatmap_widget.timestamps is not None
    assert heatmap_widget.timestamps.shape == (n_timestamps,)

    # use mock timer as real timing can be very variable in CI
    class FakeElapsedTimer:
        def __init__(self) -> None:
            self.ms = 0

        def restart(self) -> None:
            self.ms = 0

        def elapsed(self) -> int:
            return self.ms

    fake_timer = FakeElapsedTimer()
    time_slider.elapsed_timer = fake_timer  # type: ignore[assignment]
    time_slider.toggle_pause()  # start playback from t=0
    assert time_slider.timer.isActive()

    total_playback_ms = math.ceil(time_s * 1000 / relative_playback_speed)
    for fraction in (0.25, 0.5, 0.75):
        fake_timer.ms = int(total_playback_ms * fraction)
        time_slider._tick()
        assert heatmap_widget.idx == pytest.approx(fraction * (n_timestamps - 1), abs=1)
        assert time_slider.timer.isActive()
    fake_timer.ms = total_playback_ms
    time_slider._tick()
    assert heatmap_widget.idx == n_timestamps - 1
    assert not time_slider.timer.isActive()

    heatmap_widget.close()
    qtbot.waitUntil(lambda: not heatmap_widget.isVisible())


def test_heatmap_widget_clear_data(qtbot: QtBot, make_synapse_data) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    data = make_synapse_data()
    heatmap_widget.set_data(data)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape[0] == data.values.shape[2]
    heatmap_norm_dialog = HeatmapNormDialog(heatmap_widget)
    qtbot.addWidget(heatmap_norm_dialog)
    heatmap_norm_dialog.show()
    assert heatmap_norm_dialog.chk_subtract.isEnabled()
    assert not heatmap_norm_dialog.chk_subtract.isChecked()
    heatmap_norm_dialog.close()
    qtbot.waitUntil(lambda: not heatmap_norm_dialog.isVisible())
    heatmap_widget.clear_data()
    assert heatmap_widget.data is None
    heatmap_norm_dialog = HeatmapNormDialog(heatmap_widget)
    qtbot.addWidget(heatmap_norm_dialog)
    heatmap_norm_dialog.show()
    assert not heatmap_norm_dialog.chk_subtract.isEnabled()
    assert not heatmap_norm_dialog.chk_subtract.isChecked()


@pytest.mark.parametrize("data_shape", [(15, 12, 8), (20, 5, 10), (8, 8, 4)])
def test_heatmap_widget_set_data_with_subtraction(
    qtbot: QtBot, data_shape: tuple[int], make_synapse_data
) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    heatmap_norm_dialog = HeatmapNormDialog(heatmap_widget)
    qtbot.addWidget(heatmap_norm_dialog)
    heatmap_norm_dialog.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    data = make_synapse_data("synapse", *data_shape)
    # give each timepoint a 1 in a different position to ensure non-zero data & non-zero subtraction
    data.values[0, 0, 0] = 1.0
    data.values[0, 1, 0] = 1.0
    data.values[1, 0, 0] = 1.0
    heatmap_widget.set_data(data)
    assert heatmap_widget.data is not None
    assert heatmap_widget.data.shape == data.values.T.shape
    assert heatmap_widget.timestamps is not None
    assert heatmap_widget.timestamps.shape == data_shape[2:]
    assert heatmap_widget.idx == 0
    assert heatmap_widget.subtract_idx is None
    assert heatmap_widget.ig is not None
    assert np.sum(np.abs(heatmap_widget.ig.data[:])) > 1.0
    heatmap_norm_dialog.chk_subtract_clicked(True)
    assert heatmap_widget.subtract_idx == 0
    # subtracting the first timepoint should make the first graphic all zeros
    assert np.sum(np.abs(heatmap_widget.ig.data[:])) == pytest.approx(0)
    heatmap_norm_dialog.slider_idx.setValue(1)
    assert heatmap_widget.subtract_idx == 1
    # subtracting the second timepoint should make the first graphic non-zero again
    assert np.sum(np.abs(heatmap_widget.ig.data[:])) > 1.0
    # but now the second timepoint should be zero
    heatmap_widget.set_time_ms(1000 * data.timestamps[1])
    assert np.sum(np.abs(heatmap_widget.ig.data[:])) == pytest.approx(0)
    # reset defaults should disable subtraction
    heatmap_norm_dialog.reset_defaults()
    assert heatmap_widget.subtract_idx is None
    assert np.sum(np.abs(heatmap_widget.ig.data[:])) > 1.0


def test_heatmap_widget_set_data_no_data(qtbot: QtBot) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    heatmap_widget.set_data(None)
    assert heatmap_widget.data is None


def test_heatmap_widget_set_time_ms(qtbot: QtBot, make_synapse_data) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    data = make_synapse_data(
        "synapse", n_i=10, n_j=10, n_timestamps=5, max_time_secs=0.5
    )
    heatmap_widget.set_data(data)
    for idx in range(data.values.shape[2]):
        time_ms = 1000 * data.timestamps[idx]
        heatmap_widget.set_time_ms(time_ms)
        assert heatmap_widget.idx == idx


def test_heatmap_widget_shows_xy_index_ticks(qtbot: QtBot, make_synapse_data) -> None:
    heatmap_widget = HeatmapWidget()
    qtbot.addWidget(heatmap_widget)
    heatmap_widget.show()
    qtbot.waitUntil(lambda: heatmap_widget.isVisible())
    data = make_synapse_data("synapse", n_i=6, n_j=4, n_timestamps=3)
    heatmap_widget.set_data(data)
    x_ticks = np.asarray(heatmap_widget.subplot.axes.x.ticks)
    y_ticks = np.asarray(heatmap_widget.subplot.axes.y.ticks)
    assert np.array_equal(x_ticks, np.array([0, 5]))
    assert np.array_equal(y_ticks, np.array([0, 3]))
    assert heatmap_widget.subplot.axes.x.line.visible
    assert heatmap_widget.subplot.axes.y.line.visible


@pytest.mark.parametrize(
    ("scale_factor", "expected_step"),
    [(1.0, 100), (4.99, 100), (5.0, 10), (9.99, 10), (10.0, 1), (20.0, 1)],
)
def test_heatmap_tick_step_from_scale(scale_factor: float, expected_step: int) -> None:
    assert HeatmapWidget._tick_step_from_scale(scale_factor) == expected_step


def test_heatmap_tick_positions_include_last_index() -> None:
    ticks = HeatmapWidget._tick_positions(size=106, step=100)
    assert np.array_equal(ticks, np.array([0, 100, 105]))
