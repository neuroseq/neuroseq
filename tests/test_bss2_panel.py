from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.device_control import bss2_protocol as bp
from neuroseq.device_control.bss2_connection import STATE_DISCONNECTED
from neuroseq.device_control.bss2_panel import Bss2Panel
from neuroseq.device_control.stimulation_setup_widget import BSS2_TAB_TITLE
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.workspace_devices import WorkspaceDevices
from tests.test_bss2_connection import TOKEN
from tests.test_bss2_connection import fake_launcher
from tests.wait_timeout import WAIT_TIMEOUT_MS

CSV = 'Host,Port,Identifier\n"a.host",11717,"chip_a"\n"b.host",11718,"chip_b"\n'
QUESTION = "neuroseq.device_control.bss2_panel.QtWidgets.QMessageBox.question"


@pytest.fixture()
def devices():
    devices = WorkspaceDevices()
    devices.bss2.launcher = fake_launcher()
    yield devices
    devices.shutdown()


def _stim(values: np.ndarray | None = None) -> StimArray:
    if values is None:
        values = np.zeros((10, 3), np.float32)
        values[2:6, 1] = 2.0
    return StimArray("opsin", StimType.External, values, "a.u.", 0.1)


@pytest.fixture()
def panel(qtbot: QtBot, devices, tmp_config_dir, tmp_path) -> Iterator[Bss2Panel]:
    image = tmp_path / "fake.sif"
    image.write_bytes(b"")
    settings = GuiSettings()
    settings.bss2_settings = {"container_image": str(image), "container_app": ""}
    widget = Bss2Panel(devices.bss2_session(devices), settings=settings)
    qtbot.addWidget(widget)
    widget.set_project(Project())
    widget.set_current_stim_provider(_stim)
    yield widget
    widget.shutdown()


def _connect(qtbot: QtBot, panel: Bss2Panel) -> None:
    panel.host_edit.setText("quiggeldy.test")
    panel.token_edit.setText(TOKEN)
    with qtbot.waitSignal(panel._session.connected, timeout=WAIT_TIMEOUT_MS):
        panel.btn_connect.click()
    qtbot.waitUntil(lambda: panel.btn_deliver.isEnabled(), timeout=WAIT_TIMEOUT_MS)


def test_settings_persist_without_token(qtbot, panel, tmp_config_dir) -> None:
    panel.token_edit.setText(TOKEN)
    panel.madc_spin.setValue(2)
    panel.speedup_spin.setValue(500.0)
    panel.mode_combo.setCurrentIndex(1)
    panel.app_edit.setText("dls")
    panel.app_edit.editingFinished.emit()
    stored = panel._gui_settings.bss2_settings
    assert stored["madc_neuron"] == 2 and stored["speedup"] == 500.0
    assert stored["input_mode"] == bp.INPUT_SPIKES
    assert stored["container_app"] == "dls"
    text = (Path(tmp_config_dir) / "config.json").read_text()
    assert TOKEN not in text and "token" not in text.lower()

    reloaded = GuiSettings()
    assert reloaded.bss2_settings["madc_neuron"] == 2
    again = Bss2Panel(panel._session, settings=reloaded)
    qtbot.addWidget(again)
    assert again.madc_spin.value() == 2
    assert again.spike_box.isVisibleTo(again)
    assert not again.current_box.isVisibleTo(again)
    assert again.token_edit.text() == ""


def test_gui_settings_drop_token_keys(tmp_config_dir) -> None:
    config = Path(tmp_config_dir) / "config.json"
    config.write_text(json.dumps({"bss2_settings": {"token": "x", "speedup": 10}}))
    settings = GuiSettings()
    assert settings.bss2_settings == {"speedup": 10}
    config.write_text(json.dumps({"bss2_settings": [1, 2]}))
    assert GuiSettings().bss2_settings == {}


def test_refresh_setups_picks_default_and_fills_fields(
    qtbot, panel, monkeypatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    panel.setup_fetcher = lambda url: CSV
    panel.host_edit.setText("unlisted.host")  # not in the list: default pick
    panel.refresh_setups()
    # Two downloaded setups followed by the three built-in KIP instances.
    qtbot.waitUntil(lambda: panel.setup_combo.count() == 5, timeout=WAIT_TIMEOUT_MS)
    assert panel.setup_combo.itemText(4) == ("quiggeldy.bioai.eu:11780 (special)")
    index = bp.default_setup_index("tester", 2)
    assert panel.setup_combo.currentIndex() == index
    expected = ["a.host", "b.host"][index]
    assert panel.host_edit.text() == expected
    assert panel._gui_settings.bss2_settings["setup_host"] == expected
    panel._on_setup_chosen(1 - index)
    assert panel.identifier_edit.text() == ["chip_a", "chip_b"][1 - index]
    panel._on_setup_chosen(7)  # out of range: ignored

    def fail(_url):
        raise OSError("offline")

    panel.setup_fetcher = fail
    panel.refresh_setups()
    qtbot.waitUntil(
        lambda: "offline" in panel.log_view.toPlainText(), timeout=WAIT_TIMEOUT_MS
    )
    assert panel.btn_refresh_setups.isEnabled()
    panel._on_setups_downloaded("Host,Port\n")
    panel._on_setups_downloaded("Host,Port,Identifier\n")
    assert "empty" in panel.log_view.toPlainText()


def test_connect_requires_image(qtbot, panel, qmessagebox_calls) -> None:
    panel.image_edit.setText("")
    panel.connect_hardware()
    assert qmessagebox_calls and "container image" in qmessagebox_calls[0]["text"]
    assert panel._session.state == STATE_DISCONNECTED


def test_connect_deliver_adds_undoable_experiment(qtbot, panel, mocker) -> None:
    project = Project()
    panel.set_project(project)
    _connect(qtbot, panel)
    assert panel.identifier_label.text().startswith("fakechip_quiggeldy.test")
    assert not panel.image_edit.isEnabled()
    assert TOKEN not in panel.log_view.toPlainText()

    sent: list[dict] = []
    original_run = panel._session.run

    def capture(request_id, arguments):
        sent.append(arguments)
        return original_run(request_id, arguments)

    panel._session.run = capture  # type: ignore[method-assign]
    question = mocker.patch(
        QUESTION, return_value=QtWidgets.QMessageBox.StandardButton.Yes
    )
    panel.madc_spin.setValue(1)
    with qtbot.waitSignal(panel.experiment_added, timeout=WAIT_TIMEOUT_MS) as blocker:
        panel.btn_deliver.click()
        assert panel.run_in_progress
        assert not panel.btn_deliver.isEnabled() and panel.btn_stop.isEnabled()
    summary = question.call_args.args[2]
    assert "Neurons: 3" in summary and "3 segment" in summary
    arguments = sent[0]
    assert arguments["mode"] == bp.INPUT_CONSTANT_CURRENT
    assert arguments["n_neurons"] == 3 and arguments["record_v_neuron"] == 1
    assert arguments["duration_ms"] == pytest.approx(1.0)
    offsets = bp.decode_array(arguments["segments"]["offset"])
    assert offsets[:, 1].tolist() == [0, 500, 0]

    name = blocker.args[0]
    assert name.startswith("BSS-2 opsin ")
    experiment = project.experiments[name]
    assert experiment.duration == pytest.approx(1.0)
    spikes, trace = experiment.data
    assert isinstance(spikes, SpikeData) and isinstance(trace, TraceData)
    # Only neuron 1 is driven (0.2-0.6 s); the fake chip fires it mid-pulse.
    assert spikes.spike_indices.tolist() == [1]
    assert spikes.spike_times.tolist() == pytest.approx([0.4])
    assert trace.indices.tolist() == [1] and trace.units == "a.u."
    assert not panel.run_in_progress

    project.history.undo()
    assert name not in project.experiments
    project.history.redo()
    assert name in project.experiments

    # A second readback with the same name gets a suffix.
    panel._run = ("again", "opsin", 1000.0)
    result = {
        "duration_ms": 1.0,
        "spikes": {
            "indices": bp.encode_array(np.zeros(0), "<u4"),
            "times_ms": bp.encode_array(np.zeros(0), "<f4"),
        },
    }
    mocker.patch(
        "neuroseq.device_control.bss2_protocol.readback_experiment",
        return_value=Experiment(name, 1.0, [], []),
    )
    panel._on_run_finished("again", result)
    assert f"{name} (2)" in project.experiments


def test_deliver_declined_or_invalid(qtbot, panel, mocker, qmessagebox_calls) -> None:
    _connect(qtbot, panel)
    mocker.patch(QUESTION, return_value=QtWidgets.QMessageBox.StandardButton.No)
    panel.deliver()
    assert not panel.run_in_progress

    panel.set_current_stim_provider(lambda: None)
    panel.deliver()
    assert "no current stim array" in qmessagebox_calls[-1]["text"]

    panel.set_current_stim_provider(_stim)
    panel.madc_spin.setValue(10)
    panel.deliver()
    assert "recorded neuron" in qmessagebox_calls[-1]["text"]


def test_deliver_spikes_from_project_data(qtbot, panel, mocker) -> None:
    project = Project()
    project.add_experiment(
        Experiment(
            "recording",
            2.0,
            [],
            [SpikeData("events", np.array([0, 1, 1]), np.array([0.1, 0.5, 1.5]))],
        )
    )
    project.add_experiment(
        Experiment(
            "traces only",
            1.0,
            [],
            [TraceData("v", np.zeros((1, 3)), np.array([0]), "mV", 0.1)],
        )
    )
    panel.set_project(project)
    assert [
        panel.experiment_combo.itemText(i)
        for i in range(panel.experiment_combo.count())
    ] == ["recording"]
    assert panel.dataset_combo.currentText() == "events"
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData(bp.INPUT_SPIKES))
    panel.spike_source_combo.setCurrentIndex(
        panel.spike_source_combo.findData(bp.SPIKE_SOURCE_EXPERIMENT)
    )
    assert panel.experiment_combo.isEnabled() and not panel.threshold_spin.isEnabled()
    panel.weight_spin.setValue(30)
    _connect(qtbot, panel)
    sent: list[dict] = []

    def fake_run(_request_id: str, arguments: dict) -> bool:
        sent.append(arguments)
        return True

    panel._session.run = fake_run  # type: ignore[method-assign]
    mocker.patch(QUESTION, return_value=QtWidgets.QMessageBox.StandardButton.Yes)
    panel.deliver()
    assert panel.run_in_progress
    arguments = sent[0]
    assert arguments["mode"] == bp.INPUT_SPIKES
    assert arguments["n_neurons"] == 2
    assert arguments["spikes"]["weight"] == 30
    assert arguments["duration_ms"] == pytest.approx(2.0)
    assert panel._run is not None and panel._run[1] == "recording events"

    # Results for other request ids are ignored; a failure ends the run.
    panel._on_run_finished("someone else", {})
    panel._on_run_failed("someone else", "x")
    assert panel.run_in_progress
    panel._on_run_failed(panel._run[0], "chip gone")
    assert not panel.run_in_progress
    assert "chip gone" in panel.log_view.toPlainText()

    panel.dataset_combo.clear()
    with pytest.raises(ValueError, match="spike dataset"):
        panel._plan(panel.current_settings())


def test_deliver_spikes_from_stim_threshold(panel) -> None:
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData(bp.INPUT_SPIKES))
    panel.event_ms_spin.setValue(100.0)
    plan, name = panel._plan(panel.current_settings())
    assert name == "opsin"
    assert isinstance(plan, bp.SpikeInputPlan)
    assert plan.indices.tolist() == [1, 1, 1, 1]


def test_bad_readback_is_reported(qtbot, panel, qmessagebox_calls) -> None:
    panel._run = ("rid", "opsin", 1000.0)
    panel._on_run_finished("rid", {"duration_ms": 1.0})
    assert "could not be added" in qmessagebox_calls[-1]["text"]
    assert not panel.run_in_progress


def test_stop_aborts_run_and_disconnect(qtbot, panel, devices, mocker) -> None:
    devices.bss2.launcher = fake_launcher("slow_run")
    _connect(qtbot, panel)
    mocker.patch(QUESTION, return_value=QtWidgets.QMessageBox.StandardButton.Yes)
    panel.deliver()
    assert panel.run_in_progress
    assert not panel.btn_disconnect.isEnabled()
    with qtbot.waitSignal(panel._session.disconnected, timeout=WAIT_TIMEOUT_MS):
        panel.btn_stop.click()
    assert not panel.run_in_progress
    assert panel.btn_connect.isEnabled()
    assert "Stopped by the user" in panel.log_view.toPlainText()

    devices.bss2.launcher = fake_launcher()
    _connect(qtbot, panel)
    with qtbot.waitSignal(panel._session.disconnected, timeout=WAIT_TIMEOUT_MS):
        panel.btn_disconnect.click()
    assert panel.identifier_label.text() == "–"


def test_workspace_hosts_bss2_tab_and_blocks_close_during_run(
    qtbot: QtBot, monkeypatch
) -> None:
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    host = gui.stimulation_setup
    assert host.show_setup(BSS2_TAB_TITLE)
    assert host.currentWidget() is gui.bss2_panel
    assert gui.close_blockers() == []
    gui.bss2_panel._run = ("rid", "opsin", 1000.0)
    blockers = gui.close_blockers()
    assert any("BrainScaleS-2" in b for b in blockers)
    gui.bss2_panel._run = None

    gui.project.add_experiment(
        Experiment(
            "BSS-2 x",
            1.0,
            [],
            [SpikeData("BSS-2 spikes", np.array([0]), np.array([0.5]))],
        )
    )
    gui.bss2_panel.experiment_added.emit("BSS-2 x")
    assert gui.cmb_spike.currentText().startswith("BSS-2 x ::")
    gui.dispose()
    assert gui.bss2_panel._closed


def test_browse_image_and_busy_hardware(
    qtbot, panel, mocker, qmessagebox_calls, tmp_path
) -> None:
    mocker.patch(
        "neuroseq.device_control.bss2_panel.getopenfilename",
        return_value=("/images/lab.sif", ""),
    )
    panel.btn_browse_image.menu().actions()[0].trigger()
    assert panel.image_edit.text() == "/images/lab.sif"
    assert panel._gui_settings.bss2_settings["container_image"] == "/images/lab.sif"

    panel.image_edit.setText(str(tmp_path / "fake.sif"))
    _connect(qtbot, panel)
    panel._session.run = lambda rid, args: False  # type: ignore[method-assign]
    mocker.patch(QUESTION, return_value=QtWidgets.QMessageBox.StandardButton.Yes)
    panel.deliver()
    assert "busy" in qmessagebox_calls[-1]["text"]
    assert not panel.run_in_progress


def test_debug_log_traces_connection_and_copies_with_context(qtbot, panel) -> None:
    _connect(qtbot, panel)
    log = panel.log_view.toPlainText()
    assert "Launch: " in log and "Helper process started" in log
    assert '→ {"id": 1, "command": "connect"' in log
    assert '← {"id": null, "event": "ready"' in log
    assert TOKEN not in log
    panel.copy_log()
    copied = QtWidgets.QApplication.clipboard().text()
    assert copied.startswith("=== NeuroSeq BrainScaleS-2 debug log ===")
    assert "token:       given" in copied and "Launch: " in copied
    assert TOKEN not in copied
    panel.clear_log()
    assert panel.log_view.toPlainText() == ""


def test_debug_log_can_hide_protocol_traffic(qtbot, panel) -> None:
    panel.chk_protocol_trace.setChecked(False)
    _connect(qtbot, panel)
    log = panel.log_view.toPlainText()
    assert "Launch: " not in log and "→ " not in log
    assert "Connected to" in log  # status lines are always logged


def test_singularity_version_reports_missing_program(monkeypatch) -> None:
    from neuroseq.device_control import bss2_panel

    monkeypatch.setattr(bss2_panel.shutil, "which", lambda _name: None)
    assert bss2_panel._singularity_version() == "not found on PATH"


def test_connect_refuses_a_missing_image(qtbot, panel, qmessagebox_calls) -> None:
    panel.image_edit.setText("/containers/stable/latest-missing")
    panel.connect_hardware()
    assert qmessagebox_calls and "does not exist" in qmessagebox_calls[0]["text"]
    assert "does not exist" in panel.log_view.toPlainText()
    assert panel._session.state == STATE_DISCONNECTED


def test_activation_cache_dir_is_created(monkeypatch, tmp_path) -> None:
    from qtpy import QtCore as _QtCore

    from neuroseq.device_control import bss2_panel

    monkeypatch.setattr(
        _QtCore.QStandardPaths, "writableLocation", lambda _location: str(tmp_path)
    )
    directory = bss2_panel._activation_cache_dir()
    assert directory == tmp_path / "neuroseq" / "bss2" and directory.is_dir()
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(
        _QtCore.QStandardPaths, "writableLocation", lambda _location: str(blocker)
    )
    assert bss2_panel._activation_cache_dir() is None


def test_browse_sandbox_folder(qtbot, panel, mocker, tmp_path) -> None:
    mocker.patch(
        "neuroseq.device_control.bss2_panel.getexistingdirectory",
        return_value=str(tmp_path),
    )
    actions = [a.text() for a in panel.btn_browse_image.menu().actions()]
    assert actions == ["Image file (.sif)…", "Sandbox folder…"]
    panel.btn_browse_image.menu().actions()[1].trigger()
    assert panel.image_edit.text() == str(tmp_path)
    assert panel._gui_settings.bss2_settings["container_image"] == str(tmp_path)


def test_connect_drops_an_app_the_image_does_not_define(qtbot, panel, mocker) -> None:
    from neuroseq.device_control import bss2_panel

    mocker.patch.object(bss2_panel, "_container_apps", return_value=set())
    panel.app_edit.setText("dls")
    _connect(qtbot, panel)
    assert panel.app_edit.text() == ""
    assert "defines no app 'dls'" in panel.log_view.toPlainText()
    assert panel._gui_settings.bss2_settings["container_app"] == ""


def test_connect_keeps_a_defined_or_unknown_app(qtbot, panel, mocker) -> None:
    from neuroseq.device_control import bss2_panel

    mocker.patch.object(bss2_panel, "_container_apps", return_value={"dls"})
    panel.app_edit.setText("dls")
    _connect(qtbot, panel)
    assert panel.app_edit.text() == "dls"


def test_container_apps_parses_singularity_output(mocker) -> None:
    import subprocess

    from neuroseq.device_control import bss2_panel

    mocker.patch.object(bss2_panel.shutil, "which", return_value="/usr/bin/singularity")
    run = mocker.patch.object(
        bss2_panel.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, "INFO: x\ndls\nother\n", ""),
    )
    assert bss2_panel._container_apps("/img") == {"dls", "other"}
    assert run.call_args.args[0] == [
        "/usr/bin/singularity",
        "inspect",
        "--list-apps",
        "/img",
    ]
    run.return_value = subprocess.CompletedProcess([], 255, "", "FATAL")
    assert bss2_panel._container_apps("/img") is None
    run.side_effect = subprocess.TimeoutExpired("x", 60)
    assert bss2_panel._container_apps("/img") is None
    mocker.patch.object(bss2_panel.shutil, "which", return_value=None)
    assert bss2_panel._container_apps("/img") is None


def test_connect_without_host_loads_setups_then_connects(qtbot, panel) -> None:
    panel.setup_fetcher = lambda url: CSV
    panel.host_edit.setText("")
    panel.token_edit.setText(TOKEN)
    with qtbot.waitSignal(panel._session.connected, timeout=WAIT_TIMEOUT_MS):
        panel.btn_connect.click()
    assert panel.host_edit.text() in {"a.host", "b.host"}
    assert "loading the setup list first" in panel.log_view.toPlainText()


def test_connect_without_host_reports_a_failed_setup_download(
    qtbot, panel, qmessagebox_calls
) -> None:
    def fail(url):
        raise OSError("offline")

    panel.setup_fetcher = fail
    panel.host_edit.setText("")
    panel.connect_hardware()
    qtbot.waitUntil(lambda: bool(qmessagebox_calls), timeout=WAIT_TIMEOUT_MS)
    assert "Enter the quiggeldy host" in qmessagebox_calls[-1]["text"]
    assert panel._session.state == STATE_DISCONNECTED


def test_connect_without_token_does_not_ask(qtbot, panel, mocker, monkeypatch) -> None:
    monkeypatch.delenv("QUIGGELDY_TOKEN", raising=False)
    panel.host_edit.setText("quiggeldy.test")
    panel.token_edit.setText("")
    question = mocker.patch(QUESTION)
    with qtbot.waitSignal(panel._session.connected, timeout=WAIT_TIMEOUT_MS):
        panel.connect_hardware()
    question.assert_not_called()
    assert "Connecting without an EBRAINS token." in panel.log_view.toPlainText()


def test_picker_lists_the_kip_instances_and_keeps_a_saved_one(
    qtbot, devices, tmp_config_dir
) -> None:
    labels = [setup.label for setup in bp.KIP_SETUPS]
    assert labels == [
        "quiggeldy.bioai.eu:11717 (stable)",
        "quiggeldy.bioai.eu:11750 (experimental)",
        "quiggeldy.bioai.eu:11780 (special)",
    ]
    settings = GuiSettings()
    settings.bss2_settings = {"setup_host": "", "setup_port": 11717}
    fresh = Bss2Panel(devices.bss2_session(devices), settings=settings)
    qtbot.addWidget(fresh)
    assert [fresh.setup_combo.itemText(i) for i in range(3)] == labels
    # Nothing saved: the experimental instance is preselected and filled in.
    assert fresh.setup_combo.currentIndex() == bp.DEFAULT_KIP_SETUP == 1
    assert (fresh.host_edit.text(), fresh.port_spin.value()) == (
        "quiggeldy.bioai.eu",
        11750,
    )
    fresh._on_setup_chosen(2)
    assert fresh.port_spin.value() == 11780
    fresh._on_setup_chosen(1)
    assert (fresh.host_edit.text(), fresh.port_spin.value()) == (
        "quiggeldy.bioai.eu",
        11750,
    )
    fresh.shutdown()

    settings.bss2_settings = {
        "setup_host": "quiggeldy.bioai.eu",
        "setup_port": 11750,
    }
    kip = Bss2Panel(devices.bss2_session(devices), settings=settings)
    qtbot.addWidget(kip)
    assert kip.setup_combo.count() == 3
    assert kip.setup_combo.currentText().endswith("(experimental)")
    kip.shutdown()

    settings.bss2_settings = {
        "setup_host": "other.host",
        "setup_port": 11717,
        "setup_identifier": "chip_a",
    }
    other = Bss2Panel(devices.bss2_session(devices), settings=settings)
    qtbot.addWidget(other)
    assert other.setup_combo.count() == 4
    assert other.setup_combo.currentText() == "chip_a (other.host:11717)"
    other.shutdown()


def test_combined_setups_skips_duplicates() -> None:
    downloaded = [bp.QuiggeldySetup("quiggeldy.bioai.eu", 11717, "chip")]
    combined = bp.combined_setups(downloaded)
    assert [(s.port, s.identifier) for s in combined] == [
        (11717, "chip"),
        (11750, ""),
        (11780, ""),
    ]
    assert bp.QuiggeldySetup("h", 1, "").label == "h:1"


def test_refresh_keeps_the_selected_instance(qtbot, panel) -> None:
    panel.setup_fetcher = lambda url: CSV
    assert (panel.host_edit.text(), panel.port_spin.value()) == (
        bp.DEFAULT_QUIGGELDY_HOST,
        bp.DEFAULT_QUIGGELDY_PORT,
    )
    panel.refresh_setups()
    qtbot.waitUntil(lambda: panel.setup_combo.count() == 5, timeout=WAIT_TIMEOUT_MS)
    assert panel.setup_combo.currentText().endswith(":11750 (experimental)")
    assert panel.port_spin.value() == 11750


def test_defaults_are_the_working_quiggeldy_instance() -> None:
    defaults = bp.Bss2Settings()
    assert (defaults.setup_host, defaults.setup_port) == ("quiggeldy.bioai.eu", 11750)
    assert defaults.container_app == ""
