from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.project import Project
from neuroseq.startup_dialog import StartupDialog


def test_startup_dialog_example_button_loads_getting_started_project(
    qtbot: QtBot,
) -> None:
    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    dialog.btn_example.click()

    assert isinstance(dialog.chosen_project, Project)
    assert dialog.chosen_project.project_name == "Getting Started Project"
    assert dialog.chosen_filename is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_new_project_button_creates_a_truly_blank_project(
    qtbot: QtBot,
) -> None:
    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    dialog.btn_new.click()

    assert isinstance(dialog.chosen_project, Project)
    assert dialog.chosen_project.stim_arrays == {}
    assert dialog.chosen_project.b2_model.neurongroups == {}
    assert dialog.chosen_filename is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_load_button_sets_chosen_filename(
    qtbot: QtBot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a: ("/some/project.nseq", "neuroseq project (*.nseq)"),
    )
    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    dialog.btn_load.click()

    assert dialog.chosen_filename == "/some/project.nseq"
    assert dialog.chosen_project is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_load_button_cancelled_does_not_accept(
    qtbot: QtBot, monkeypatch
) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a: ("", ""),
    )
    dialog = StartupDialog()
    qtbot.addWidget(dialog)

    dialog.btn_load.click()

    assert dialog.chosen_filename is None
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
