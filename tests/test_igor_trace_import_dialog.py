from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.exporter_igor2 import _write_ibw_v5
from neuroseq.igor_trace_import_dialog import IgorTraceImportDialog
from neuroseq.importer_igor2 import IgorPxpLoadReport
from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import SkippedPxpRecord
from neuroseq.project import Project


def _find_trace_data(
    project: Project,
    experiment_name: str,
    trace_name: str,
) -> TraceData:
    experiment = project.experiments[experiment_name]
    for data in experiment.data:
        if isinstance(data, TraceData) and data.name == trace_name:
            return data
    raise AssertionError(f"TraceData '{trace_name}' not found in '{experiment_name}'")


def _find_spike_data(project: Project, experiment_name: str) -> list[SpikeData]:
    experiment = project.experiments[experiment_name]
    return [data for data in experiment.data if isinstance(data, SpikeData)]


def test_trace_dialog_creates_new_trace_in_new_experiment(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n", encoding="utf-8")

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog._import_file()

    trace_data = _find_trace_data(project, "Experiment A", "igor_wave")
    assert trace_data.traces.shape == (1, 3)
    np.testing.assert_allclose(trace_data.traces[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(trace_data.indices, np.array([0], dtype=np.uint32))
    assert dialog.selected_experiment_name == "Experiment A"
    assert dialog.selected_trace_name == "igor_wave"
    assert qmessagebox_calls[-1]["name"] == "information"
    assert "import complete" in qmessagebox_calls[-1]["title"].lower()


def test_trace_dialog_wraps_content_in_scroll_area(qtbot: QtBot) -> None:
    dialog = IgorTraceImportDialog(Project())
    qtbot.addWidget(dialog)

    assert isinstance(dialog.scroll_area, QtWidgets.QScrollArea)
    assert dialog.scroll_area.widget() is not None
    assert dialog.scroll_area.widgetResizable() is True


def test_trace_dialog_wave_plot_hidden_initially(qtbot: QtBot) -> None:
    dialog = IgorTraceImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert not dialog.wave_plot.isVisible()


def test_trace_dialog_wave_plot_shown_after_wave_load(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    ibw = _write_ibw_v5(
        np.array([0.0, 1.0, 0.0], dtype="<f4"),
        wave_name="test",
        dt=0.001,
    )
    path = tmp_path / "test.ibw"
    path.write_bytes(ibw)
    dialog = IgorTraceImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog._load_file(str(path))

    qtbot.waitUntil(lambda: dialog.wave_plot.isVisible())


def test_trace_dialog_creates_new_trace_in_existing_experiment(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n", encoding="utf-8")

    project = Project()
    project.add_experiment(
        Experiment(name="Existing", duration=0.0, timepoints=[], data=[])
    )

    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_combo.setCurrentText("Existing")
    dialog._import_file()

    trace_data = _find_trace_data(project, "Existing", "igor_wave")
    assert trace_data.traces.shape == (1, 3)
    assert project.experiments["Existing"].duration == pytest.approx(0.003)
    assert dialog.selected_experiment_name == "Existing"
    assert dialog.selected_trace_name == "igor_wave"
    assert qmessagebox_calls[-1]["name"] == "information"


def test_trace_dialog_imports_into_existing_trace_data(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n10.0\n20.0\n30.0\n", encoding="utf-8")

    project = Project()
    project.add_experiment(
        Experiment(
            name="Experiment A",
            duration=0.002,
            timepoints=[],
            data=[
                TraceData(
                    name="Existing Trace",
                    traces=np.array([[1.0, 2.0]], dtype=np.float32),
                    indices=np.array([0], dtype=np.uint32),
                    units="mV",
                    dt=0.001,
                )
            ],
        )
    )

    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.import_existing_radio.setChecked(True)
    dialog.existing_experiment_combo.setCurrentText("Experiment A")
    dialog.existing_trace_combo.setCurrentText("Existing Trace")
    dialog.existing_neuron_id_spin.setValue(5)
    dialog.existing_replace_checkbox.setChecked(True)
    dialog.existing_extend_checkbox.setChecked(True)
    dialog._import_file()

    trace_data = _find_trace_data(project, "Experiment A", "Existing Trace")
    assert trace_data.traces.shape == (2, 3)
    np.testing.assert_array_equal(trace_data.indices, np.array([0, 5], dtype=np.uint32))
    np.testing.assert_allclose(trace_data.traces[1], np.array([10.0, 20.0, 30.0]))
    assert dialog.selected_experiment_name == "Experiment A"
    assert dialog.selected_trace_name == "Existing Trace"
    assert qmessagebox_calls[-1]["name"] == "information"


def test_trace_dialog_loads_pxp_and_shows_wave_picker(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0, 2.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    wave_b = IgorWave(
        name="WaveB",
        values=np.array([3.0, 4.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a, "root/WaveB": wave_b}

    monkeypatch.setattr("neuroseq.igor_trace_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))

    assert dialog.wave_picker_group.isVisible()
    assert dialog.pxp_wave_list.count() == 2
    items = [
        dialog.pxp_wave_list.item(i).text()
        for i in range(dialog.pxp_wave_list.count())
        if dialog.pxp_wave_list.item(i) is not None
    ]
    assert "root/WaveA" in items
    assert "root/WaveB" in items
    assert dialog.loaded_wave is None
    assert not dialog.import_button.isEnabled()


def test_trace_dialog_stays_open_for_multiple_pxp_imports(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pxp_path = tmp_path / "multi.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    wave_b = IgorWave(
        name="WaveB",
        values=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a, "root/WaveB": wave_b}

    monkeypatch.setattr("neuroseq.igor_trace_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))
    dialog.new_experiment_name_edit.setText("Experiment A")

    items = [dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())]
    dialog.pxp_wave_list.setCurrentItem(
        next(item for item in items if item is not None and item.text() == "root/WaveA")
    )
    assert dialog.loaded_wave is wave_a
    dialog._import_file()

    assert dialog.isVisible()
    assert dialog.cancel_button.text() == "Close"
    assert _find_trace_data(project, "Experiment A", "WaveA").traces.shape == (1, 3)

    wave_a_item = next(
        item
        for item in (
            dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())
        )
        if item is not None and "WaveA" in item.text()
    )
    assert wave_a_item.text().startswith("✓ ")
    assert not (wave_a_item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable)

    dialog.new_experiment_combo.setCurrentText("Experiment A")
    wave_b_item = next(
        item
        for item in (
            dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())
        )
        if item is not None and item.text() == "root/WaveB"
    )
    dialog.pxp_wave_list.setCurrentItem(wave_b_item)
    assert dialog.new_trace_name_edit.text() == "WaveB"
    dialog._import_file()

    assert dialog.isVisible()
    assert _find_trace_data(project, "Experiment A", "WaveB").traces.shape == (1, 3)
    assert dialog.selected_experiment_name == "Experiment A"
    assert dialog.selected_trace_name == "WaveB"

    with qtbot.waitSignal(dialog.finished, timeout=1000):
        dialog.cancel_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_trace_dialog_reimport_same_pxp_wave_is_blocked(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pxp_path = tmp_path / "multi.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    wave_b = IgorWave(
        name="WaveB",
        values=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a, "root/WaveB": wave_b}

    monkeypatch.setattr("neuroseq.igor_trace_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))
    dialog.new_experiment_name_edit.setText("Experiment A")

    wave_a_item = next(
        dialog.pxp_wave_list.item(i)
        for i in range(dialog.pxp_wave_list.count())
        if dialog.pxp_wave_list.item(i) is not None
        and dialog.pxp_wave_list.item(i).text() == "root/WaveA"
    )
    dialog.pxp_wave_list.setCurrentItem(wave_a_item)
    dialog._import_file()

    imported_wave_a_item = next(
        dialog.pxp_wave_list.item(i)
        for i in range(dialog.pxp_wave_list.count())
        if dialog.pxp_wave_list.item(i) is not None
        and "WaveA" in dialog.pxp_wave_list.item(i).text()
    )
    dialog.pxp_wave_list.setCurrentItem(imported_wave_a_item)
    dialog._update_import_button_state()

    assert dialog.pxp_wave_list.currentItem() is imported_wave_a_item
    assert dialog.import_button.isEnabled() is False


def test_trace_dialog_warns_when_pxp_records_are_skipped(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0, 2.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a}
    skipped = [
        SkippedPxpRecord(
            record_index=55,
            record_type="WaveRecord",
            reason="TypeError: buffer is too small for requested array",
        )
    ]

    monkeypatch.setattr("neuroseq.igor_trace_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=skipped),
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))

    assert dialog.wave_picker_group.isVisible()
    assert dialog.pxp_wave_list.count() == 1
    assert dialog.loaded_wave is wave_a

    warning_calls = [call for call in qmessagebox_calls if call["name"] == "warning"]
    assert len(warning_calls) == 1
    assert warning_calls[0]["title"] == "Some waves were skipped"
    assert "could not be parsed and were skipped" in warning_calls[0]["text"]


def test_trace_dialog_optional_spike_detection_adds_spikedata(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n", encoding="utf-8")

    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.detect_spikes_voltage_trace",
        lambda *args, **kwargs: SpikeData(
            name="Detected spikes",
            spike_indices=np.array([0], dtype=np.int32),
            spike_times=np.array([0.001], dtype=np.float32),
        ),
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog.spike_detect_checkbox.setChecked(True)
    dialog._import_file()

    spike_data = _find_spike_data(project, "Experiment A")
    assert len(spike_data) == 1
    assert spike_data[0].name == "Igor Patch Clamp Spikes"
    assert any(
        isinstance(data, TraceData) for data in project.experiments["Experiment A"].data
    )
    assert "detected 1 spike(s)" in qmessagebox_calls[-1]["text"].lower()


def test_trace_dialog_spike_detection_off_adds_no_spikedata(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n", encoding="utf-8")

    def _unexpected_spike_detection(*args, **kwargs):
        raise AssertionError("Spike detection should not run when checkbox is off")

    monkeypatch.setattr(
        "neuroseq.igor_trace_import_dialog.detect_spikes_voltage_trace",
        _unexpected_spike_detection,
    )

    project = Project()
    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog.spike_detect_checkbox.setChecked(False)
    dialog._import_file()

    assert _find_spike_data(project, "Experiment A") == []


def test_trace_dialog_offers_remap_on_dt_change_and_remaps_on_yes(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """When importing a trace at a non-default dt and the project already has a
    stim array, a remap confirmation must be shown and all data remapped on Yes."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("new_trace\n1.0\n2.0\n", encoding="utf-8")

    from neuroseq.stim import StimArray
    from neuroseq.stim import StimType

    project = Project()
    project.add_stim_array(
        StimArray(
            "stim1", StimType.External, np.zeros((10, 1), dtype=np.float32), "pA", 0.001
        )
    )

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == pytest.approx(0.0005)
    assert project.stim_arrays["stim1"].dt == pytest.approx(0.0005)
    assert project.stim_arrays["stim1"].values.shape[0] == 20
    trace_data = _find_trace_data(project, "Experiment A", "new_trace")
    assert trace_data.dt == pytest.approx(0.0005)


def test_trace_dialog_aborts_import_on_remap_decline(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the remap confirmation must leave the project untouched."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("new_trace\n1.0\n2.0\n", encoding="utf-8")

    from neuroseq.stim import StimArray
    from neuroseq.stim import StimType

    project = Project()
    project.add_stim_array(
        StimArray(
            "stim1", StimType.External, np.zeros((10, 1), dtype=np.float32), "pA", 0.001
        )
    )

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    dialog = IgorTraceImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == pytest.approx(0.001)
    assert "Experiment A" not in project.experiments
    assert project.stim_arrays["stim1"].dt == pytest.approx(0.001)
    assert project.stim_arrays["stim1"].values.shape[0] == 10
