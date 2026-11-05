from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data import Experiment
from neuroseq.data import TraceData
from neuroseq.export.exporter_igor2 import _write_ibw_v5
from neuroseq.importers.importer_patch_clamp_data import IgorPxpLoadReport
from neuroseq.importers.importer_patch_clamp_data import IgorWave
from neuroseq.importers.importer_patch_clamp_data import SkippedPxpRecord
from neuroseq.importers.patch_clamp_data_input_import_dialog import DtMismatchChoice
from neuroseq.importers.patch_clamp_data_input_import_dialog import (
    PatchClampDataInputImportDialog,
)
from neuroseq.importers.patch_clamp_data_input_import_dialog import (
    ask_dt_mismatch_choice,
)
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.wait_timeout import WAIT_TIMEOUT_MS


def test_input_browse_formats(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = PatchClampDataInputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Import patch clamp input"
    assert "patch-clamp input waveforms" in dialog.intro.text()
    assert "Igor" not in dialog.intro.text()
    assert dialog.file_label.text() == "No input waveform selected"
    labels = "\n".join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
    assert "CURRENT is the input" in labels
    assert "VOLTAGE is the input" in labels
    filters = []

    def choose_file(parent, title, directory, file_filter):
        filters.append(file_filter)
        return "", ""

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.getopenfilename",
        choose_file,
    )
    assert not hasattr(dialog, "format_combo")
    dialog._browse_file()
    assert filters == [
        "Patch-clamp files (*.pxp *.ibw *.csv *.txt);;Packed Experiment (*.pxp);;"
        "Binary Wave (*.ibw);;CSV files (*.csv);;Text files (*.txt);;All files (*)"
    ]


def test_unsupported_input_format(
    qtbot: QtBot, qmessagebox_calls: list[dict[str, str]], tmp_path: Path
) -> None:
    dialog = PatchClampDataInputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.load_file(str(tmp_path / "input.unsupported"))
    assert dialog.loaded_wave is None
    assert dialog.file_path == ""
    assert qmessagebox_calls[-1]["name"] == "critical"
    assert "Unsupported" in qmessagebox_calls[-1]["text"]
    assert ".unsupported" in qmessagebox_calls[-1]["text"]


@pytest.mark.parametrize("suffix", [".csv", ".txt", ".TXT"])
@pytest.mark.parametrize("rate", [500.0, 2000.0])
@pytest.mark.parametrize("header", ["", "injected_current\n"])
def test_csv_input_sampling_rate(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    rate: float,
    header: str,
    suffix: str,
) -> None:
    path = tmp_path / f"current{suffix}"
    path.write_text(header + "1\n2\n3\n4\n", encoding="utf-8")
    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.sampling_rate_spin.setValue(rate)

    assert dialog.source_dt_spin.value() == pytest.approx(1.0 / rate)
    dialog._import_file()

    name = "injected_current" if header else "current"
    stim = project.stim_arrays[name]
    assert stim.dt == pytest.approx(1.0 / rate)
    expected = [1, 2, 3, 4]
    np.testing.assert_allclose(stim.values[:, 0], expected)
    assert qmessagebox_calls[-1]["name"] == "information"


def test_patch_clamp_data_input_import_dialog_creates_new_stim_array(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert "igor_wave" in project.stim_arrays
    stim_array = project.stim_arrays["igor_wave"]
    assert stim_array.stim_type == StimType.External
    assert stim_array.dt == pytest.approx(0.0005)
    assert stim_array.values.shape == (4, 1)
    np.testing.assert_allclose(stim_array.values[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
    assert dialog.selected_stim_name == "igor_wave"
    assert qmessagebox_calls[-1]["name"] == "information"
    assert "import complete" in qmessagebox_calls[-1]["title"].lower()


def test_patch_clamp_data_input_import_dialog_wraps_content_in_scroll_area(
    qtbot: QtBot,
) -> None:
    dialog = PatchClampDataInputImportDialog(Project())
    qtbot.addWidget(dialog)

    assert isinstance(dialog.scroll_area, QtWidgets.QScrollArea)
    assert dialog.scroll_area.widget() is not None
    assert dialog.scroll_area.widgetResizable() is True
    labels = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
    assert "Imported sample interval:" not in labels
    assert not hasattr(dialog, "match_source_grid_checkbox")


@pytest.mark.parametrize("source_dt", [0.0005, 0.002])
def test_binary_input_preserves_file_timing(
    tmp_path: Path, qtbot: QtBot, source_dt: float
) -> None:
    values = np.array([1, 2, 3, 4], dtype="<f4")
    path = tmp_path / "input.ibw"
    path.write_bytes(_write_ibw_v5(values, wave_name="current", dt=source_dt))
    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog._import_file()

    stim = project.stim_arrays[dialog.selected_stim_name]
    assert stim.dt == pytest.approx(source_dt)
    np.testing.assert_array_equal(stim.values[:, 0], values)


def test_patch_clamp_data_input_import_dialog_wave_plot_hidden_initially(
    qtbot: QtBot,
) -> None:
    dialog = PatchClampDataInputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)

    assert not dialog.wave_plot.isVisible()


def test_patch_clamp_data_input_import_dialog_wave_plot_shown_after_wave_load(
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
    dialog = PatchClampDataInputImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)

    dialog._load_file(str(path))

    qtbot.waitUntil(lambda: dialog.wave_plot.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_patch_clamp_data_input_import_dialog_shows_multi_sweep_info(
    qtbot: QtBot,
) -> None:
    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.loaded_wave = IgorWave(
        name="multi",
        values=np.arange(12, dtype=np.float32),
        source_dt=0.001,
        units="pA",
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
    # Lives inside the Input Waveform Preview group, not floating at the top of the dialog.
    parent_group = label.parent()
    assert isinstance(parent_group, QtWidgets.QGroupBox)
    assert parent_group.title() == "Input Waveform Preview"


def test_patch_clamp_data_input_import_dialog_imports_into_existing_stim_array(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n10.0\n20.0\n30.0\n", encoding="utf-8")

    project = Project()
    project.add_stim_array(
        StimArray(
            name="stimulus",
            stim_type=StimType.External,
            values=np.zeros((2, 2), dtype=np.float32),
            units="pA",
            dt=0.001,
        )
    )

    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.import_existing_radio.setChecked(True)
    dialog.target_combo.setCurrentText("stimulus")
    dialog.neuron_index_spin.setValue(1)
    dialog.replace_checkbox.setChecked(True)
    dialog.extend_checkbox.setChecked(True)
    dialog._import_file()

    stim_array = project.stim_arrays["stimulus"]
    assert stim_array.values.shape == (3, 2)
    np.testing.assert_allclose(stim_array.values[:, 1], np.array([10.0, 20.0, 30.0]))
    assert dialog.selected_stim_name == "stimulus"
    assert qmessagebox_calls[-1]["name"] == "information"
    assert "import complete" in qmessagebox_calls[-1]["title"].lower()


def test_patch_clamp_data_input_import_dialog_creates_new_stim_array_on_source_grid(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert "igor_wave" in project.stim_arrays
    stim_array = project.stim_arrays["igor_wave"]
    assert stim_array.dt == pytest.approx(0.0005)
    assert stim_array.values.shape == (4, 1)
    np.testing.assert_allclose(stim_array.values[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
    assert qmessagebox_calls[-1]["name"] == "information"


def test_patch_clamp_data_input_import_dialog_imports_into_existing_stim_array_on_source_grid(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n10.0\n20.0\n30.0\n", encoding="utf-8")

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.ask_dt_mismatch_choice",
        lambda *args, **kwargs: DtMismatchChoice.REMAP_TARGET,
    )

    project = Project()
    project.add_stim_array(
        StimArray(
            name="stimulus",
            stim_type=StimType.External,
            values=np.zeros((2, 2), dtype=np.float32),
            units="pA",
            dt=0.001,
        )
    )

    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog.import_existing_radio.setChecked(True)
    dialog.target_combo.setCurrentText("stimulus")
    dialog.neuron_index_spin.setValue(1)
    dialog.replace_checkbox.setChecked(True)
    dialog.extend_checkbox.setChecked(True)
    dialog._import_file()

    stim_array = project.stim_arrays["stimulus"]
    assert stim_array.dt == pytest.approx(0.0005)
    assert stim_array.values.shape == (4, 2)
    np.testing.assert_allclose(
        stim_array.values[:, 1], np.array([10.0, 20.0, 30.0, 0.0])
    )
    assert dialog.selected_stim_name == "stimulus"
    assert qmessagebox_calls[-1]["name"] == "information"
    assert "adjusted target input grid" in qmessagebox_calls[-1]["text"].lower()
    # Remap + import is one undoable edit restoring the original grid.
    project.history.undo()
    assert stim_array.dt == pytest.approx(0.001)
    np.testing.assert_array_equal(stim_array.values, np.zeros((2, 2)))


def test_patch_clamp_data_input_import_dialog_loads_pxp_and_shows_wave_picker(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

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
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    assert dialog.wave_picker_group.isVisible()
    assert dialog.pxp_wave_list.count() == 2
    items = []
    for i in range(dialog.pxp_wave_list.count()):
        item = dialog.pxp_wave_list.item(i)
        assert item is not None
        items.append(item.text())
    assert "root/WaveA" in items
    assert "root/WaveB" in items
    assert dialog.loaded_wave is None
    assert not dialog.import_button.isEnabled()


def test_patch_clamp_data_input_import_dialog_pxp_wave_selection_sets_loaded_wave(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

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

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    assert dialog.loaded_wave is wave_a
    assert dialog.new_name_edit.text() == "WaveA"
    assert dialog.new_units_edit.text() == "pA"
    assert "Input name=WaveA" in dialog.preview_text.toPlainText()
    assert dialog.import_button.isEnabled()


def test_patch_clamp_data_input_import_dialog_pxp_multi_wave_requires_selection(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    wave_b = IgorWave(
        name="WaveB",
        values=np.array([2.0], dtype=np.float32),
        source_dt=0.001,
        units="mV",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a, "root/WaveB": wave_b}

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    assert dialog.loaded_wave is None
    assert not dialog.import_button.isEnabled()

    items = []
    for i in range(dialog.pxp_wave_list.count()):
        item = dialog.pxp_wave_list.item(i)
        assert item is not None
        items.append(item.text())
    dialog.pxp_wave_list.setCurrentRow(items.index("root/WaveB"))

    assert dialog.loaded_wave is wave_b
    assert dialog.new_name_edit.text() == "WaveB"
    assert dialog.new_units_edit.text() == "mV"
    assert "Input name=WaveB" in dialog.preview_text.toPlainText()
    assert dialog.import_button.isEnabled()


def test_patch_clamp_data_input_import_dialog_loading_csv_after_pxp_clears_wave_picker(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

    pxp_path = tmp_path / "test.pxp"
    pxp_path.write_bytes(b"fake")
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("csv_wave\n1.0\n2.0\n", encoding="utf-8")

    fake_waves = {
        "root/WaveA": IgorWave(
            name="WaveA",
            values=np.array([1.0], dtype=np.float32),
            source_dt=0.001,
            units="pA",
            axis_units="s",
            source_format="ibw",
        )
    }

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda path: str(path).lower().endswith(".pxp"),
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)

    dialog._load_file(str(pxp_path))
    assert dialog.wave_picker_group.isVisible()
    assert dialog.loaded_pxp_waves == fake_waves
    assert dialog.pxp_wave_list.count() == 1

    dialog._load_file(str(csv_path))

    assert not dialog.wave_picker_group.isVisible()
    assert dialog.loaded_pxp_waves == {}
    assert dialog.pxp_wave_list.count() == 0
    assert dialog.loaded_wave is not None
    assert dialog.loaded_wave.name == "csv_wave"
    assert "Input name=csv_wave" in dialog.preview_text.toPlainText()
    assert dialog.import_button.isEnabled()


def test_patch_clamp_data_input_import_dialog_warns_when_pxp_records_are_skipped(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

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
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=skipped),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    assert dialog.wave_picker_group.isVisible()
    assert dialog.pxp_wave_list.count() == 1
    assert dialog.loaded_wave is wave_a

    warning_calls = [call for call in qmessagebox_calls if call["name"] == "warning"]
    assert len(warning_calls) == 1
    assert warning_calls[0]["title"] == "Some waves were skipped"
    assert "could not be parsed and were skipped" in warning_calls[0]["text"]


def test_patch_clamp_data_input_import_dialog_single_wave_pxp_import_stays_open_until_close(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

    pxp_path = tmp_path / "single.pxp"
    pxp_path.write_bytes(b"fake")

    wave_a = IgorWave(
        name="WaveA",
        values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        source_dt=0.001,
        units="pA",
        axis_units="s",
        source_format="ibw",
    )
    fake_waves = {"root/WaveA": wave_a}

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    assert dialog.loaded_wave is wave_a
    assert dialog.import_button.isEnabled()

    dialog._import_file()

    assert dialog.isVisible()
    assert dialog.cancel_button.text() == "Close"
    assert "WaveA" in project.stim_arrays
    assert (
        dialog.last_import_label.text()
        == "Imported 'WaveA' into new input array 'WaveA'."
    )
    assert not any(call["name"] == "information" for call in qmessagebox_calls)

    with qtbot.waitSignal(dialog.finished, timeout=WAIT_TIMEOUT_MS):
        dialog.cancel_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    qtbot.waitUntil(lambda: not dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)


def test_patch_clamp_data_input_import_dialog_stays_open_for_multiple_pxp_imports(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two waves from the same .pxp must be importable without reopening the dialog."""
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

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
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

    items = [dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())]
    assert {it.text() for it in items if it is not None} == {"root/WaveA", "root/WaveB"}
    dialog.pxp_wave_list.setCurrentItem(
        next(it for it in items if it is not None and it.text() == "root/WaveA")
    )
    assert dialog.loaded_wave is wave_a
    assert dialog.import_button.isEnabled()
    dialog._import_file()

    assert dialog.isVisible()
    assert dialog.cancel_button.text() == "Close"
    wave_a_item = next(
        it
        for it in (
            dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())
        )
        if it is not None and "WaveA" in it.text()
    )
    assert wave_a_item.text().startswith("✓ ")
    assert not (wave_a_item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable)
    assert "WaveA" in project.stim_arrays

    wave_b_item = next(
        it
        for it in (
            dialog.pxp_wave_list.item(i) for i in range(dialog.pxp_wave_list.count())
        )
        if it is not None and "WaveB" in it.text()
    )
    dialog.pxp_wave_list.setCurrentItem(wave_b_item)
    assert dialog.loaded_wave is wave_b
    assert dialog.new_name_edit.text() == "WaveB"
    dialog._import_file()

    assert dialog.isVisible()
    assert "WaveB" in project.stim_arrays
    assert dialog.selected_stim_name == "WaveB"

    with qtbot.waitSignal(dialog.finished, timeout=WAIT_TIMEOUT_MS):
        dialog.cancel_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_patch_clamp_data_input_import_dialog_reimport_same_pxp_wave_is_blocked(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wave already imported in this .pxp session must not be importable again."""
    from neuroseq.importers.importer_patch_clamp_data import IgorWave

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
        "neuroseq.importers.patch_clamp_data_input_import_dialog.is_pxp_file",
        lambda _: True,
    )
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    dialog._load_file(str(pxp_path))

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


def test_new_input_preserves_existing_stim_and_trace_timing(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Importing a new input keeps existing datasets on their original grids."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("new_wave\n1.0\n2.0\n", encoding="utf-8")

    project = Project()
    project.add_stim_array(
        StimArray(
            "existing",
            StimType.External,
            np.zeros((10, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )

    trace = TraceData("calcium", np.array([[1, 2, 3]]), np.array([0]), "units", 1 / 30)
    project.add_experiment(Experiment("exp", 0.1, [], [trace]))
    original_values = project.stim_arrays["existing"].values
    original_traces = trace.traces

    def unexpected_question(*args, **kwargs):
        pytest.fail("Creating an independent input must not prompt to resample data")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", unexpected_question)

    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.stim_arrays["existing"].dt == pytest.approx(0.001)
    assert project.stim_arrays["existing"].values is original_values
    assert trace.dt == pytest.approx(1 / 30)
    assert trace.traces is original_traces
    assert "new_wave" in project.stim_arrays
    assert project.stim_arrays["new_wave"].dt == pytest.approx(0.0005)


def test_patch_clamp_data_input_import_dialog_aborts_import_on_remap_decline(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the remap confirmation must leave the project untouched."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("new_wave\n1.0\n2.0\n", encoding="utf-8")

    project = Project()
    project.add_stim_array(
        StimArray(
            "existing",
            StimType.External,
            np.zeros((10, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.ask_dt_mismatch_choice",
        lambda *args, **kwargs: DtMismatchChoice.CANCEL,
    )

    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.import_existing_radio.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert "new_wave" not in project.stim_arrays
    assert project.stim_arrays["existing"].dt == pytest.approx(0.001)
    assert project.stim_arrays["existing"].values.shape[0] == 10


def test_patch_clamp_data_input_import_dialog_no_remap_dialog_when_project_is_empty(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Creating the first stim at a non-default dt must NOT show a remap dialog."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()

    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert "wave" in project.stim_arrays
    assert project.stim_arrays["wave"].dt == pytest.approx(0.0005)
    question_calls = [c for c in qmessagebox_calls if c["name"] == "question"]
    assert not question_calls


def _existing_stim_project() -> Project:
    project = Project()
    project.add_stim_array(
        StimArray(
            "stimulus",
            StimType.External,
            np.zeros((2, 2), dtype=np.float32),
            "pA",
            0.001,
        )
    )
    return project


def _load_into_existing(qtbot: QtBot, project: Project, csv_path: Path):
    dialog = PatchClampDataInputImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog.import_existing_radio.setChecked(True)
    dialog.target_combo.setCurrentText("stimulus")
    dialog.neuron_index_spin.setValue(1)
    dialog.replace_checkbox.setChecked(True)
    dialog.extend_checkbox.setChecked(True)
    return dialog


def test_input_import_can_resample_the_wave_onto_the_target_grid(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n10.0\n20.0\n30.0\n40.0\n", encoding="utf-8")
    choices: list[dict] = []

    def choose(*args, **kwargs):
        choices.append(kwargs)
        return DtMismatchChoice.RESAMPLE_WAVE

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.ask_dt_mismatch_choice",
        choose,
    )
    project = _existing_stim_project()
    dialog = _load_into_existing(qtbot, project, csv_path)
    dialog._import_file()

    assert choices[0]["target_dt"] == pytest.approx(0.001)
    assert choices[0]["wave_dt"] == pytest.approx(0.0005)
    stim_array = project.stim_arrays["stimulus"]
    assert stim_array.dt == pytest.approx(0.001)
    np.testing.assert_allclose(stim_array.values[:, 1], [10.0, 30.0])
    assert "resampled the wave" in qmessagebox_calls[-1]["text"].lower()
    project.history.undo()
    np.testing.assert_array_equal(stim_array.values, np.zeros((2, 2)))


def test_failed_input_import_leaves_the_target_unremapped(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.ask_dt_mismatch_choice",
        lambda *args, **kwargs: DtMismatchChoice.REMAP_TARGET,
    )

    def fail(*args, **kwargs):
        raise ValueError("import failed")

    monkeypatch.setattr(
        "neuroseq.importers.patch_clamp_data_input_import_dialog.import_igor_wave_into_stim_array",
        fail,
    )
    project = _existing_stim_project()
    original = project.stim_arrays["stimulus"].values
    dialog = _load_into_existing(qtbot, project, csv_path)
    dialog._import_file()

    assert qmessagebox_calls[-1]["name"] == "critical"
    assert project.stim_arrays["stimulus"].values is original
    assert project.stim_arrays["stimulus"].dt == pytest.approx(0.001)
    assert not project.history.can_undo()


@pytest.mark.parametrize(
    ("button_prefix", "expected"),
    [
        (None, DtMismatchChoice.RESAMPLE_WAVE),  # the default button
        ("Remap", DtMismatchChoice.REMAP_TARGET),
        ("Cancel", DtMismatchChoice.CANCEL),
    ],
)
def test_dt_mismatch_prompt_defaults_to_resampling_the_wave(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    button_prefix: str | None,
    expected: DtMismatchChoice,
) -> None:
    def clicked(box: QtWidgets.QMessageBox):
        if button_prefix is None:
            return box.defaultButton()
        return next(
            b
            for b in box.buttons()
            if b.text().replace("&", "").startswith(button_prefix)
        )

    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", clicked)
    choice = ask_dt_mismatch_choice(
        None, target_label="input array 'x'", target_dt=0.001, wave_dt=0.00002
    )
    assert choice is expected
