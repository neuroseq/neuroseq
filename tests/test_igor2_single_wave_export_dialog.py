"""Tests for the Igor2 single-wave export dialog's optional unit / scaling fields."""

from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.igor2_single_wave_export_dialog import Igor2SingleWaveExportDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


def _stim() -> StimArray:
    return StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.ones((4, 2), dtype=np.float32),
        units="pA",
        dt=0.001,
    )


def test_dialog_defaults_unit_override_none_and_scaling_one(qtbot: QtBot) -> None:
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    assert dialog.unit_override is None
    assert dialog.scaling_factor == 1.0


def test_dialog_parses_unit_override_and_scaling(qtbot: QtBot) -> None:
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog._unit_edit.setText("  mW/mm^2  ")
    dialog._scale_edit.setText("3.25")
    assert dialog.unit_override == "mW/mm^2"
    assert dialog.scaling_factor == pytest.approx(3.25)


def test_dialog_blank_fields_return_defaults(qtbot: QtBot) -> None:
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog._unit_edit.setText("   ")
    dialog._scale_edit.setText("")
    assert dialog.unit_override is None
    assert dialog.scaling_factor == 1.0


def test_dialog_accept_rejects_non_numeric_scaling(
    qtbot: QtBot, qmessagebox_calls: list
) -> None:
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog._scale_edit.setText("oops")
    dialog.accept()

    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    assert any(
        "scaling factor" in call.get("title", "").lower() for call in qmessagebox_calls
    )


def test_dialog_accept_still_rejects_empty_experiment_name(
    qtbot: QtBot, qmessagebox_calls: list
) -> None:
    """Existing validation must continue to work after the new fields are added."""
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog.experiment_name_edit.setText("   ")
    dialog.accept()

    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    assert any(
        "invalid input" in call.get("title", "").lower() for call in qmessagebox_calls
    )
