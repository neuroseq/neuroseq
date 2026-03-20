from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.data import Experiment
from neuroseq.import_experiment_dialog import ImportExperimentDialog
from neuroseq.project import Project


def test_import_experiment_dialog(
    qtbot: QtBot,
    test_experiment: Experiment,
    project_nmda: Project,
    test_experiments: list[Experiment],
) -> None:
    for experiment in test_experiments:
        project_nmda.add_experiment(experiment)
    test_experiment.name = test_experiments[0].name
    dialog = ImportExperimentDialog(test_experiment, project_nmda)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert dialog.cmb_experiments.count() == len(test_experiments)
    assert dialog.cmb_experiments.itemText(0) == test_experiments[0].name
    assert dialog.cmb_experiments.itemText(1) == test_experiments[1].name

    # Matching experiment name at dialog open should preselect append mode itself
    assert dialog.rad_append.isChecked()

    # Switch to option to insert into new experiment that we will input a name for ourselves...
    dialog.rad_new.setChecked(True)
    dialog.rad_new_append_toggled()
    dialog.txt_new_name.setText("Unique Experiment Name")
    dialog.new_name_text_edited(dialog.txt_new_name.text())
    assert dialog.save_button.isEnabled()

    # select append instead of new
    dialog.rad_append.setChecked(True)
    dialog.rad_new_append_toggled()
    assert dialog.append_box.isVisible()
    assert not dialog.new_box.isVisible()
    assert dialog.append_to_experiment is True
    assert dialog.cmb_experiments.count() == len(test_experiments)
    assert dialog.cmb_experiments.currentText() == test_experiments[0].name
    assert dialog.cmb_offset.count() == 2
    assert dialog.cmb_offset.itemText(0) == "0 ms (partially replace + insert)"
    assert dialog.cmb_offset.itemText(1) == "200 ms (Append)"
    # select the first offset
    dialog.cmb_offset.setCurrentIndex(0)
    assert dialog.offset == pytest.approx(0.0)
    # select the second (and last) offset
    dialog.cmb_offset.setCurrentIndex(1)
    assert dialog.offset == pytest.approx(0.2)

    # select second experiment and check offsets
    dialog.cmb_experiments.setCurrentIndex(1)
    dialog.cmb_experiments_index_changed(dialog.cmb_experiments.currentIndex())
    assert dialog.append_experiment_name == test_experiments[1].name
    assert dialog.cmb_offset.count() == 4
    assert dialog.cmb_offset.itemText(0) == "0 ms (overwrite)"
    assert dialog.cmb_offset.itemText(2) == "350 ms (overwrite)"
    assert dialog.cmb_offset.currentIndex() == 3
    assert dialog.offset == pytest.approx(0.5)
    # select 0.35 offset
    dialog.cmb_offset.setCurrentIndex(2)
    assert dialog.offset == pytest.approx(0.35)
