from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.device_control.stimulation_setup_widget import SYSCON2_TAB_TITLE
from neuroseq.device_control.stimulation_setup_widget import StimulationSetupWidget
from neuroseq.project_workspace import ProjectWorkspace


def test_tabs_are_added_and_selected_by_title(qtbot: QtBot) -> None:
    host = StimulationSetupWidget()
    qtbot.addWidget(host)
    first = QtWidgets.QLabel("first")
    second = QtWidgets.QLabel("second")
    second.setMinimumSize(400, 300)
    assert host.add_setup(first, "One") == 0
    assert host.add_setup(second, "Two") == 1
    assert host.currentWidget() is first
    # The size hint covers the largest page, so the splitter does not reflow.
    assert host.minimumSizeHint().width() >= 400
    assert host.sizeHint().height() >= 300

    assert host.show_setup("Two")
    assert host.currentWidget() is second
    assert not host.show_setup("Missing")
    assert host.currentWidget() is second


def test_workspace_hosts_syscon2_panel_in_stimulation_setup_tab(qtbot: QtBot) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    host = gui.stimulation_setup
    assert host.tabText(0) == SYSCON2_TAB_TITLE
    assert host.widget(0) is gui.syscon2_panel
    assert not host.isVisibleTo(gui)

    action = gui._view_actions[host]
    action.trigger()
    assert host.isVisibleTo(gui)
    assert gui.syscon2_panel.isVisibleTo(gui)
    action.trigger()
    assert not host.isVisibleTo(gui)
