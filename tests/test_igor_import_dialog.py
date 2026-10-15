from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.exporter_igor2 import _write_ibw_v5
from neuroseq.igor_import_dialog import IgorImportDialog
from neuroseq.importer_igor2 import IgorPxpLoadReport
from neuroseq.importer_igor2 import IgorWave
from neuroseq.importer_igor2 import SkippedPxpRecord
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def test_input_format_selector(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = IgorImportDialog(Project())
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

    monkeypatch.setattr("neuroseq.igor_import_dialog.getopenfilename", choose_file)
    dialog.format_combo.setCurrentIndex(1)
    assert dialog.sampling_rate_spin.isVisible()
    dialog._browse_file()
    assert filters[-1] == "CSV / Text files (*.csv *.txt)"
    dialog.format_combo.setCurrentIndex(0)
    assert not dialog.sampling_rate_spin.isVisible()
    dialog._browse_file()
    assert "*.ibw" in filters[-1]
    dialog.close()


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
    dialog = IgorImportDialog(project, csv_only=True)
    qtbot.addWidget(dialog)
    dialog._load_file(str(path))
    dialog.sampling_rate_spin.setValue(rate)

    assert dialog.source_dt_spin.value() == pytest.approx(1.0 / rate)
    dialog._import_file()

    name = "injected_current" if header else "current"
    stim = project.stim_arrays[name]
    assert stim.dt == 0.001
    expected = [1, 3] if rate == 2000.0 else [1, 1, 2, 2, 3, 3, 4, 4]
    np.testing.assert_allclose(stim.values[:, 0], expected)
    assert qmessagebox_calls[-1]["name"] == "information"


def test_igor_import_dialog_creates_new_stim_array(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert "igor_wave" in project.stim_arrays
    stim_array = project.stim_arrays["igor_wave"]
    assert stim_array.stim_type == StimType.External
    assert stim_array.dt == 0.001
    assert stim_array.values.shape == (2, 1)
    np.testing.assert_allclose(stim_array.values[:, 0], np.array([1.0, 3.0]))
    assert dialog.selected_stim_name == "igor_wave"
    assert qmessagebox_calls[-1]["name"] == "information"
    assert "import complete" in qmessagebox_calls[-1]["title"].lower()


def test_igor_import_dialog_wraps_content_in_scroll_area(qtbot: QtBot) -> None:
    dialog = IgorImportDialog(Project())
    qtbot.addWidget(dialog)

    assert isinstance(dialog.scroll_area, QtWidgets.QScrollArea)
    assert dialog.scroll_area.widget() is not None
    assert dialog.scroll_area.widgetResizable() is True


def test_igor_import_dialog_wave_plot_hidden_initially(qtbot: QtBot) -> None:
    dialog = IgorImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert not dialog.wave_plot.isVisible()


def test_igor_import_dialog_wave_plot_shown_after_wave_load(
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
    dialog = IgorImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog._load_file(str(path))

    qtbot.waitUntil(lambda: dialog.wave_plot.isVisible())


def test_igor_import_dialog_shows_multi_sweep_info(qtbot: QtBot) -> None:
    project = Project()
    dialog = IgorImportDialog(project)
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


def test_igor_import_dialog_imports_into_existing_stim_array(
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

    dialog = IgorImportDialog(project)
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


def test_igor_import_dialog_creates_new_stim_array_on_source_grid(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog._import_file()

    assert "igor_wave" in project.stim_arrays
    stim_array = project.stim_arrays["igor_wave"]
    assert stim_array.dt == pytest.approx(0.0005)
    assert stim_array.values.shape == (4, 1)
    np.testing.assert_allclose(stim_array.values[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
    assert "match source/file grid" in dialog.target_dt_label.text().lower()
    assert qmessagebox_calls[-1]["name"] == "information"


def test_igor_import_dialog_imports_into_existing_stim_array_on_source_grid(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("igor_wave\n10.0\n20.0\n30.0\n", encoding="utf-8")

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
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

    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.source_dt_spin.setValue(0.0005)
    dialog.match_source_grid_checkbox.setChecked(True)
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


def test_igor_import_dialog_loads_pxp_and_shows_wave_picker(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
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


def test_igor_import_dialog_pxp_wave_selection_sets_loaded_wave(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    dialog._load_file(str(pxp_path))

    assert dialog.loaded_wave is wave_a
    assert dialog.new_name_edit.text() == "WaveA"
    assert dialog.new_units_edit.text() == "pA"
    assert "Input name=WaveA" in dialog.preview_text.toPlainText()
    assert dialog.import_button.isEnabled()


def test_igor_import_dialog_pxp_multi_wave_requires_selection(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
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


def test_igor_import_dialog_loading_csv_after_pxp_clears_wave_picker(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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
        "neuroseq.igor_import_dialog.is_pxp_file",
        lambda path: str(path).lower().endswith(".pxp"),
    )
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

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


def test_igor_import_dialog_warns_when_pxp_records_are_skipped(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=skipped),
    )

    project = Project()
    dialog = IgorImportDialog(project)
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


def test_igor_import_dialog_single_wave_pxp_import_stays_open_until_close(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
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

    with qtbot.waitSignal(dialog.finished, timeout=1000):
        dialog.cancel_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    qtbot.waitUntil(lambda: not dialog.isVisible())


def test_igor_import_dialog_stays_open_for_multiple_pxp_imports(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two waves from the same .pxp must be importable without reopening the dialog."""
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
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

    with qtbot.waitSignal(dialog.finished, timeout=1000):
        dialog.cancel_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_igor_import_dialog_reimport_same_pxp_wave_is_blocked(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wave already imported in this .pxp session must not be importable again."""
    from neuroseq.importer_igor2 import IgorWave

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

    monkeypatch.setattr("neuroseq.igor_import_dialog.is_pxp_file", lambda _: True)
    monkeypatch.setattr(
        "neuroseq.igor_import_dialog.load_igor_pxp_file",
        lambda _: IgorPxpLoadReport(waves=fake_waves, skipped=[]),
    )

    project = Project()
    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
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


def test_igor_import_dialog_offers_remap_on_dt_change_and_remaps_on_yes(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """When a non-default dt is selected and the project has existing data,
    a confirmation dialog must be shown and all data must be remapped on Yes."""
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
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == pytest.approx(0.0005)
    # existing stim remapped: 10 samples at 1 ms → 20 samples at 0.5 ms
    assert project.stim_arrays["existing"].dt == pytest.approx(0.0005)
    assert project.stim_arrays["existing"].values.shape[0] == 20
    assert "new_wave" in project.stim_arrays
    assert project.stim_arrays["new_wave"].dt == pytest.approx(0.0005)


def test_igor_import_dialog_aborts_import_on_remap_decline(
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
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == pytest.approx(0.001)
    assert "new_wave" not in project.stim_arrays
    assert project.stim_arrays["existing"].dt == pytest.approx(0.001)
    assert project.stim_arrays["existing"].values.shape[0] == 10


def test_igor_import_dialog_no_remap_dialog_when_project_is_empty(
    tmp_path: Path,
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    """Creating the first stim at a non-default dt must NOT show a remap dialog."""
    csv_path = tmp_path / "wave.csv"
    csv_path.write_text("wave\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")

    project = Project()

    dialog = IgorImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._load_file(str(csv_path))
    dialog.match_source_grid_checkbox.setChecked(True)
    dialog.source_dt_spin.setValue(0.0005)
    dialog._import_file()

    assert project.dt == pytest.approx(0.0005)
    assert "wave" in project.stim_arrays
    question_calls = [c for c in qmessagebox_calls if c["name"] == "question"]
    assert not question_calls
