from __future__ import annotations

from qtpy import QtWidgets

from neuroseq.project_workspace import DetachedPanelWindow


class _CloseTrackingWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> bool:
        self.close_calls += 1
        return super().close()


def test_close_event_calls_widget_close_and_on_close_callback(qtbot) -> None:
    widget = _CloseTrackingWidget()
    callbacks: list[None] = []
    window = DetachedPanelWindow(
        widget, "Title", on_close=lambda: callbacks.append(None)
    )
    qtbot.addWidget(window)

    window.close()

    assert widget.close_calls == 1
    assert callbacks == [None]


def test_scroll_area_wraps_widget_and_is_resizable(qtbot) -> None:
    widget = QtWidgets.QWidget()
    window = DetachedPanelWindow(widget, "Title")
    qtbot.addWidget(window)

    scroll_area = window.findChild(QtWidgets.QScrollArea)

    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert scroll_area.widget() is widget


def test_detached_plot_panels_get_and_keep_the_project_limits(qtbot) -> None:
    import numpy as np

    from neuroseq.data import Experiment
    from neuroseq.data import SpikeData
    from neuroseq.data import TraceData
    from neuroseq.project import Project
    from neuroseq.project_workspace import ProjectWorkspace
    from neuroseq.visualization.data_widget import DataWidget

    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    project = Project()
    trace = TraceData("trace", np.zeros((3, 40)), np.arange(3), "mV", 0.001)
    spikes = SpikeData("spikes", np.array([0, 1]), np.array([0.001, 0.002]))
    project.add_experiment(Experiment("E", 0.04, [], [trace, spikes]))
    gui.update_project(project)
    gui._apply_trace_selection("E :: trace")
    gui.cmb_spike.setCurrentText("E :: spikes")
    gui.cmb_data.setCurrentText("E :: trace [trace]")

    def limits(widget):
        controller = widget.controller
        return controller.max_width, controller.max_height

    expected = limits(gui.trace_widget)
    detached = []
    for panel in (gui.trace_widget, gui.spike_widget, gui.data_widget):
        gui.detach_panel(panel, keep_original=True)
        detached.append(gui._detached_panel_windows[-1]._widget)
    data_widget = detached[-1]
    assert isinstance(data_widget, DataWidget)
    # The detached data panel links to the main view, not a private dummy.
    assert data_widget.default_controller is gui.controller
    for widget in detached:
        assert limits(widget) == expected

    project.experiments["E"].duration = 0.5
    gui.project.update_limits()
    gui.update_limits()
    expected = limits(gui.trace_widget)
    assert expected[0] > 40
    for widget in detached:
        assert limits(widget) == expected
    for window in list(gui._detached_panel_windows):
        window.close()
