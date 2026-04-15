from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.igor_import_dialog import IgorImportDialog
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


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
    assert "adjusted target stim grid" in qmessagebox_calls[-1]["text"].lower()


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
        lambda _: fake_waves,
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
        lambda _: fake_waves,
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
    assert "Wave name=WaveA" in dialog.preview_text.toPlainText()
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
        lambda _: fake_waves,
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
    assert "Wave name=WaveB" in dialog.preview_text.toPlainText()
    assert dialog.import_button.isEnabled()
