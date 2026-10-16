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
from neuroseq.export.exporter_igor2 import _write_ibw_v5
from neuroseq.importer_patch_clamp_data import IgorPxpLoadReport
from neuroseq.importer_patch_clamp_data import IgorWave
from neuroseq.importer_patch_clamp_data import SkippedPxpRecord
from neuroseq.patch_clamp_data_output_import_dialog import (
    PatchClampDataOutputImportDialog,
)
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
    dialog = PatchClampDataOutputImportDialog(project)
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


@pytest.mark.parametrize("rate", [500.0, 2000.0])
@pytest.mark.parametrize("header", ["", "recorded_voltage\n"])
def test_csv_output_sampling_rate(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    rate: float,
    header: str,
) -> None:
    path = tmp_path / "voltage.csv"
    path.write_text(header + "1\n2\n3\n4\n", encoding="utf-8")
    project = Project()
    dialog = PatchClampDataOutputImportDialog(project, csv_only=True)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.new_experiment_name_edit.setText("CSV recording")
    dialog.sampling_rate_spin.setValue(rate)
    assert dialog.source_dt_spin.value() == pytest.approx(1.0 / rate)
    dialog._import_file()

    name = "recorded_voltage" if header else "voltage"
    trace = _find_trace_data(project, "CSV recording", name)
    assert trace.dt == 0.001
    expected = [1, 3] if rate == 2000.0 else [1, 1, 2, 2, 3, 3, 4, 4]
    np.testing.assert_allclose(trace.traces[0], expected)
    assert qmessagebox_calls[-1]["name"] == "information"


@pytest.mark.parametrize("suffix", [".csv", ".txt"])
@pytest.mark.parametrize("header", ["", "cell_a,cell_b\n"])
def test_csv_table_imports_all_columns(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    header: str,
    suffix: str,
) -> None:
    path = tmp_path / f"recording{suffix}"
    path.write_text(header + "1,10\n2,20\n3,30\n4,40\n", encoding="utf-8")
    project = Project()
    dialog = PatchClampDataOutputImportDialog(project, csv_only=True)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.new_experiment_name_edit.setText("Table")
    dialog.new_neuron_id_spin.setValue(5)
    dialog.sampling_rate_spin.setValue(2000)
    assert not dialog.import_existing_radio.isEnabled()
    dialog._import_file()
    names = ["cell_a", "cell_b"] if header else ["recording_1", "recording_2"]
    for index, name in enumerate(names):
        trace = _find_trace_data(project, "Table", name)
        assert trace.dt == 0.001
        np.testing.assert_array_equal(trace.indices, [5 + index])
        np.testing.assert_allclose(trace.traces[0], np.array([1, 3]) * (10**index))
    assert len(project.experiments["Table"].data) == 2
    assert qmessagebox_calls[-1]["name"] == "information"


def test_csv_table_name_collision_does_not_partially_import(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    project = Project()
    existing = TraceData("cell_b", np.ones((1, 2)), np.array([0]), "mV", 0.001)
    project.add_experiment(Experiment("Table", 0.002, [], [existing]))
    path = tmp_path / "recording.csv"
    path.write_text("cell_a,cell_b\n1,10\n2,20\n", encoding="utf-8")
    dialog = PatchClampDataOutputImportDialog(project, csv_only=True)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.new_experiment_combo.setCurrentText("Table")
    dialog._import_file()
    assert project.experiments["Table"].data == [existing]
    assert qmessagebox_calls[-1]["name"] == "critical"


def test_trace_dialog_wraps_content_in_scroll_area(qtbot: QtBot) -> None:
    dialog = PatchClampDataOutputImportDialog(Project())
    qtbot.addWidget(dialog)

    assert isinstance(dialog.scroll_area, QtWidgets.QScrollArea)
    assert dialog.scroll_area.widget() is not None
    assert dialog.scroll_area.widgetResizable() is True


def test_trace_dialog_wave_plot_hidden_initially(qtbot: QtBot) -> None:
    dialog = PatchClampDataOutputImportDialog(Project())
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
    dialog = PatchClampDataOutputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog._load_file(str(path))

    qtbot.waitUntil(lambda: dialog.wave_plot.isVisible())


def test_trace_dialog_shows_multi_sweep_info(qtbot: QtBot) -> None:
    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.loaded_wave = IgorWave(
        name="multi",
        values=np.arange(12, dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="ibw",
        n_sweeps=3,
        samples_per_sweep=4,
    )
    dialog._update_preview()

    label = dialog.findChild(QtWidgets.QLabel, "multi_sweep_info")
    assert label is not None
    text = label.text()
    assert "3 sweeps" in text
    assert "4 source samples" in text
    assert "12 total source samples" in text
    parent_group = label.parent()
    assert isinstance(parent_group, QtWidgets.QGroupBox)
    assert parent_group.title() == "Wave Preview"


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

    dialog = PatchClampDataOutputImportDialog(project)
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

    dialog = PatchClampDataOutputImportDialog(project)
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

    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.is_pxp_file", lambda _: True
    )
    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))

    assert dialog.wave_picker_group.isVisible()
    assert dialog.pxp_wave_list.count() == 2
    items: list[str] = []
    for i in range(dialog.pxp_wave_list.count()):
        item = dialog.pxp_wave_list.item(i)
        assert item is not None
        items.append(item.text())
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

    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.is_pxp_file", lambda _: True
    )
    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
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
    assert wave_a_item.text().startswith("âœ“ ")
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

    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.is_pxp_file", lambda _: True
    )
    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))
    dialog.new_experiment_name_edit.setText("Experiment A")

    wave_a_item = None
    for i in range(dialog.pxp_wave_list.count()):
        item = dialog.pxp_wave_list.item(i)
        assert item is not None
        if item.text() == "root/WaveA":
            wave_a_item = item
            break
    assert wave_a_item is not None
    dialog.pxp_wave_list.setCurrentItem(wave_a_item)
    dialog._import_file()

    imported_wave_a_item = None
    for i in range(dialog.pxp_wave_list.count()):
        item = dialog.pxp_wave_list.item(i)
        assert item is not None
        if "WaveA" in item.text():
            imported_wave_a_item = item
            break
    assert imported_wave_a_item is not None
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

    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.is_pxp_file", lambda _: True
    )
    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=skipped),
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
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
        "neuroseq.patch_clamp_data_output_import_dialog.detect_spikes_voltage_trace",
        lambda *args, **kwargs: SpikeData(
            name="Detected spikes",
            spike_indices=np.array([0], dtype=np.int32),
            spike_times=np.array([0.001], dtype=np.float32),
        ),
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
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
        "neuroseq.patch_clamp_data_output_import_dialog.detect_spikes_voltage_trace",
        _unexpected_spike_detection,
    )

    project = Project()
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.new_experiment_name_edit.setText("Experiment A")
    dialog.spike_detect_checkbox.setChecked(False)
    dialog._import_file()

    assert _find_spike_data(project, "Experiment A") == []


@pytest.mark.parametrize("multiple_columns", [False, True])
@pytest.mark.parametrize("match_source", [False, True])
def test_trace_import_preserves_independent_dataset_timing(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    multiple_columns: bool,
    match_source: bool,
) -> None:
    from neuroseq.stim import StimArray
    from neuroseq.stim import StimType

    path = tmp_path / "wave.csv"
    path.write_text(
        "first,second\n1,10\n2,20\n" if multiple_columns else "first\n1\n2\n",
        encoding="utf-8",
    )
    project = Project()
    stim_values = np.arange(10, dtype=np.float32).reshape(10, 1)
    project.add_stim_array(
        StimArray("stim1", StimType.External, stim_values.copy(), "pA", 0.0005208125)
    )
    original_project_dt = project.dt
    existing_trace = TraceData(
        "calcium",
        np.array([[1, 2, 3]], dtype=np.float32),
        np.array([0], dtype=np.uint32),
        "units",
        1 / 30,
    )
    project.add_experiment(
        Experiment(name="Existing", duration=0.1, timepoints=[], data=[existing_trace])
    )

    def unexpected_question(*args, **kwargs):
        pytest.fail("Importing independent traces must not prompt to remap data")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", unexpected_question)
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.new_experiment_name_edit.setText("Imported")
    dialog.match_source_grid_checkbox.setChecked(match_source)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == original_project_dt
    assert project.stim_arrays["stim1"].dt == pytest.approx(0.0005208125)
    np.testing.assert_array_equal(project.stim_arrays["stim1"].values, stim_values)
    assert existing_trace.dt == pytest.approx(1 / 30)
    np.testing.assert_array_equal(existing_trace.traces, [[1, 2, 3]])
    names = ["first", "second"] if multiple_columns else ["first"]
    for name in names:
        trace = _find_trace_data(project, "Imported", name)
        assert trace.dt == pytest.approx(0.0005 if match_source else 0.001)


def test_new_experiment_is_named_after_the_wave(tmp_path: Path, qtbot: QtBot) -> None:
    """Issue #619: the experiment name is filled in, so a wave can be imported
    without typing one."""
    first = tmp_path / "first.csv"
    first.write_text("sweep_A\n1.0\n2.0\n", encoding="utf-8")
    second = tmp_path / "second.csv"
    second.write_text("sweep_B\n1.0\n2.0\n", encoding="utf-8")
    project = Project()
    project.add_experiment(
        Experiment(name="sweep_B", duration=0.002, timepoints=[], data=[])
    )
    dialog = PatchClampDataOutputImportDialog(project)
    qtbot.addWidget(dialog)

    dialog._load_file(str(first))
    assert dialog.new_experiment_name_edit.text() == "sweep_A"
    assert dialog.import_button.isEnabled()
    # An existing experiment's name is not suggested again.
    dialog._load_file(str(second))
    assert dialog.new_experiment_name_edit.text() == "sweep_B 2"
    # A name the user typed is kept when another wave is loaded.
    dialog.new_experiment_name_edit.setText("My cell")
    dialog._load_file(str(first))
    assert dialog.new_experiment_name_edit.text() == "My cell"


def test_output_browse_formats(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = PatchClampDataOutputImportDialog(Project())
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Import patch clamp output"
    labels = "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
    assert "VOLTAGE is the trace" in labels
    assert "CURRENT is the trace" in labels
    filters = []

    def choose_file(parent, title, directory, file_filter):
        filters.append(file_filter)
        return "", ""

    monkeypatch.setattr(
        "neuroseq.patch_clamp_data_output_import_dialog.getopenfilename", choose_file
    )
    dialog._browse_file()
    assert filters == [
        "Packed Experiment (*.pxp);;Binary Wave (*.ibw);;CSV files (*.csv);;Text files (*.txt);;All files (*)"
    ]


def test_unsupported_output_format(
    qtbot: QtBot, qmessagebox_calls: list[dict[str, str]], tmp_path: Path
) -> None:
    dialog = PatchClampDataOutputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.load_file(str(tmp_path / "output.unsupported"))
    assert dialog.loaded_wave is None
    assert dialog.file_path == ""
    assert qmessagebox_calls[-1]["name"] == "critical"
    assert "Unsupported" in qmessagebox_calls[-1]["text"]
    assert ".unsupported" in qmessagebox_calls[-1]["text"]
