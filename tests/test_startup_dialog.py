from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.example_projects import example_projects
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.startup_dialog import StartupDialog


def test_startup_dialog_has_no_recent_section_when_empty(
    qtbot: QtBot, tmp_config_dir
) -> None:
    settings = GuiSettings()
    dialog = StartupDialog(settings)
    qtbot.addWidget(dialog)
    assert not hasattr(dialog, "recent_list")


def test_startup_dialog_lists_recent_projects_that_still_exist(
    qtbot: QtBot, tmp_config_dir, tmp_path
) -> None:
    existing = tmp_path / "existing.nseq"
    existing.write_text("not a real archive, just needs to exist")
    settings = GuiSettings()
    settings.recent_projects = [str(existing), "/does/not/exist.nseq"]

    dialog = StartupDialog(settings)
    qtbot.addWidget(dialog)

    assert dialog.recent_paths == [str(existing)]
    assert dialog.recent_list.count() == 1
    assert dialog.recent_list.item(0).text() == "existing.nseq"


def test_startup_dialog_new_project_button_sets_chosen_project_and_accepts(
    qtbot: QtBot, tmp_config_dir
) -> None:
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    dialog.btn_new.click()

    assert isinstance(dialog.chosen_project, Project)
    assert dialog.chosen_filename is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_example_tree_matches_registry_sections(
    qtbot: QtBot, tmp_config_dir
) -> None:
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    sections = {}
    for i in range(dialog.example_tree.topLevelItemCount()):
        section_item = dialog.example_tree.topLevelItem(i)
        sections[section_item.text(0)] = [
            section_item.child(j).text(0) for j in range(section_item.childCount())
        ]

    expected = {
        section: list(projects.keys()) for section, projects in example_projects.items()
    }
    assert sections == expected


def test_startup_dialog_double_clicking_an_example_sets_chosen_project(
    qtbot: QtBot, tmp_config_dir
) -> None:
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    section_item = dialog.example_tree.topLevelItem(0)
    child_item = section_item.child(0)
    dialog._choose_example(child_item, 0)

    assert isinstance(dialog.chosen_project, Project)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_double_clicking_a_section_header_does_nothing(
    qtbot: QtBot, tmp_config_dir
) -> None:
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    section_item = dialog.example_tree.topLevelItem(0)
    dialog._choose_example(section_item, 0)

    assert dialog.chosen_project is None
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_double_clicking_recent_sets_chosen_filename(
    qtbot: QtBot, tmp_config_dir, tmp_path
) -> None:
    existing = tmp_path / "existing.nseq"
    existing.write_text("placeholder")
    settings = GuiSettings()
    settings.recent_projects = [str(existing)]

    dialog = StartupDialog(settings)
    qtbot.addWidget(dialog)
    dialog._choose_recent(dialog.recent_list.item(0))

    assert dialog.chosen_filename == str(existing)
    assert dialog.chosen_project is None
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_load_button_sets_chosen_filename(
    qtbot: QtBot, tmp_config_dir, monkeypatch
) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a: ("/some/project.nseq", "neuroseq project (*.nseq)"),
    )
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    dialog.btn_load.click()

    assert dialog.chosen_filename == "/some/project.nseq"
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_startup_dialog_load_button_cancelled_does_not_accept(
    qtbot: QtBot, tmp_config_dir, monkeypatch
) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a: ("", ""),
    )
    dialog = StartupDialog(GuiSettings())
    qtbot.addWidget(dialog)

    dialog.btn_load.click()

    assert dialog.chosen_filename is None
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
