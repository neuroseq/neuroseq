from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.save_load_options import SaveLoadOptionsDialog


def test_save_load_options(qtbot: QtBot):
    dialog = SaveLoadOptionsDialog(action="Import")
    qtbot.addWidget(dialog)

    # initially all options should be True
    assert dialog.options.include_stim_arrays is True
    assert dialog.checkbox_stim_arrays.isChecked()
    assert dialog.options.include_model is True
    assert dialog.checkbox_model.isChecked()
    assert dialog.options.include_simulation_data is True
    assert dialog.checkbox_simulation_data.isChecked()
    assert dialog.options.include_experimental_data is True
    assert dialog.checkbox_experimental_data.isChecked()

    # if model is unchecked, simulation data should also be unchecked & disabled
    dialog.checkbox_model.setChecked(False)
    assert dialog.options.include_model is False
    assert dialog.options.include_simulation_data is False
    assert dialog.checkbox_simulation_data.isChecked() is False
    assert dialog.checkbox_simulation_data.isEnabled() is False

    # re-check model should enable simulation data checkbox
    dialog.checkbox_model.setChecked(True)
    assert dialog.options.include_model is True
    assert dialog.checkbox_simulation_data.isEnabled() is True

    # uncheck all options
    dialog.checkbox_stim_arrays.setChecked(False)
    dialog.checkbox_model.setChecked(False)
    dialog.checkbox_simulation_data.setChecked(False)
    dialog.checkbox_experimental_data.setChecked(False)
    assert dialog.options.include_stim_arrays is False
    assert dialog.options.include_model is False
    assert dialog.options.include_simulation_data is False
    assert dialog.options.include_experimental_data is False


def test_save_dialog_title_refers_to_project(qtbot: QtBot) -> None:
    dialog = SaveLoadOptionsDialog(action="Save")
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Save Project"
