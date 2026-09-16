import numpy as np
import pytest
from matplotlib.colors import to_hex
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.input_designer_dialog import StimHeatmapWindow
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def _make_stim(
    n_time: int,
    n_neurons: int,
    dt: float = 0.001,
    stim_type: StimType = StimType.Excitation,
) -> StimArray:
    return StimArray(
        "stim",
        stim_type,
        np.zeros((n_time, n_neurons), dtype=np.float32),
        "mV",
        dt,
    )


def test_stim_heatmap_window_shows_whole_history_without_panning(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    stim = _make_stim(10, 3)
    window.update_from_stim(stim)
    image = window._image
    assert image is not None
    # always starts at t=0 and ends at the array's current length - never a
    # fixed-width window that pans as more samples are appended.
    assert image.get_extent()[0] == 0.0
    assert image.get_extent()[1] == pytest.approx(10 * stim.dt * 1000.0)
    assert window.axes.get_xlabel() == "Time (ms)"
    assert window.axes.get_ylabel() == "Neuron ID"

    # growing the array in place (as a routine appending blocks would) keeps
    # the same image artist, updated rather than rebuilt, and shows every
    # sample - old ones are never dropped/panned away.
    stim.values = np.vstack([stim.values, np.zeros((5, 3), dtype=np.float32)])
    window.update_from_stim(stim)
    assert window._image is image
    assert image.get_extent()[1] == pytest.approx(15 * stim.dt * 1000.0)
    array = image.get_array()
    assert array is not None
    assert array.shape[1] == 15


def test_stim_heatmap_window_ignores_absent_or_empty_stim(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    window.update_from_stim(None)
    assert window._image is None

    window.update_from_stim(_make_stim(0, 3))
    assert window._image is None


def test_stim_heatmap_window_recolors_when_stim_type_changes(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    window.update_from_stim(_make_stim(5, 2, stim_type=StimType.Excitation))
    assert window._image is not None
    excitatory_cmap = window._image.get_cmap()

    window.update_from_stim(_make_stim(5, 2, stim_type=StimType.Inhibition))
    assert window._image is not None
    assert window._image.get_cmap() is not excitatory_cmap


def test_stim_heatmap_window_follows_theme(qtbot) -> None:
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    window.apply_theme(theme.DARK)
    assert to_hex(window.axes.get_facecolor()) == "#1e1e1e"
    window.apply_theme(theme.LIGHT)
    assert to_hex(window.axes.get_facecolor()) == "#ffffff"


def test_stim_heatmap_window_position_below_clamped_to_screen(qtbot) -> None:
    reference = QtWidgets.QDialog()
    qtbot.addWidget(reference)
    reference.resize(720, 640)
    window = StimHeatmapWindow("Test — stimulation pattern")
    qtbot.addWidget(window)

    screen = reference.screen() or QtWidgets.QApplication.primaryScreen()
    available = screen.availableGeometry()
    # Force the reference window near the bottom of the screen: placing the
    # heatmap window naively beneath it (its y + its own height) would then
    # land past the bottom edge, off-screen.
    reference.move(available.left(), available.bottom() - 100)

    window.position_below(reference)

    pos = window.pos()
    size = window.size()
    assert pos.x() >= available.left()
    assert pos.y() >= available.top()
    assert pos.x() + size.width() <= available.right() + 1
    assert pos.y() + size.height() <= available.bottom() + 1
