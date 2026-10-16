"""Tests for the Igor2 PXP stim-export dialog's optional unit / scaling fields."""

from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.export.igor2_pxp_stim_export_dialog import Igor2PxpStimExportDialog
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _project_with_one_stim() -> Project:
    project = Project()
    project.add_stim_array(
        StimArray(
            "Light",
            StimType.Excitation,
            np.ones((4, 1), dtype=np.float32),
            "pA",
            0.001,
        )
    )
    return project


def test_dialog_defaults_unit_override_none_and_scaling_one(qtbot: QtBot) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    assert dialog.unit_override is None
    assert dialog.scaling_factor == 1.0


def test_dialog_strips_whitespace_in_unit_override(qtbot: QtBot) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    dialog._unit_edit.setText("  mW/mm^2  ")
    assert dialog.unit_override == "mW/mm^2"


def test_dialog_blank_unit_returns_none(qtbot: QtBot) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    dialog._unit_edit.setText("   ")
    assert dialog.unit_override is None


def test_dialog_parses_scaling_factor_float(qtbot: QtBot) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    dialog._scale_edit.setText("2.5")
    assert dialog.scaling_factor == pytest.approx(2.5)


def test_dialog_accept_rejects_non_numeric_scaling(
    qtbot: QtBot, qmessagebox_calls: list
) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    dialog._scale_edit.setText("not a number")
    dialog.accept()

    # Accept was blocked -> dialog still visible (not "Accepted")
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    assert any(
        "scaling factor" in call.get("title", "").lower() for call in qmessagebox_calls
    )


def test_dialog_accept_passes_with_blank_fields(qtbot: QtBot) -> None:
    project = _project_with_one_stim()
    dialog = Igor2PxpStimExportDialog(project)
    qtbot.addWidget(dialog)

    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
