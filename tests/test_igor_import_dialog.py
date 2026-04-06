from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot

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
