"""Tests for the Igor2 single-wave export dialog's optional unit / scaling fields."""

from __future__ import annotations

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.export.igor2_single_wave_export_dialog import Igor2SingleWaveExportDialog
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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
    assert dialog.as_sweeps is False


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


def test_dialog_accepts_power_of_ten_scaling(qtbot: QtBot) -> None:
    """Scaling factor field accepts '10^-12' / '10^(-12)' shorthand."""
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog._scale_edit.setText("10^-12")
    assert dialog.scaling_factor == pytest.approx(1e-12)

    dialog._scale_edit.setText("2.5*10^(-12)")
    assert dialog.scaling_factor == pytest.approx(2.5e-12)


def test_dialog_accept_passes_with_power_of_ten_scaling(qtbot: QtBot) -> None:
    dialog = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(dialog)

    dialog._scale_edit.setText("10^(-12)")
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


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


def test_dialog_status_empty_when_checkbox_off(qtbot: QtBot) -> None:
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.tile(np.array([[1.0], [2.0], [3.0]], dtype=np.float32), (4, 2)),
        units="pA",
        dt=0.001,
    )
    dialog = Igor2SingleWaveExportDialog("stim", stim)
    qtbot.addWidget(dialog)

    status = dialog.findChild(QtWidgets.QLabel, "sweep_detection_status")
    assert status is not None
    assert status.text() == ""


def test_dialog_status_reports_detected_period(qtbot: QtBot) -> None:
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.tile(np.array([[1.0], [2.0], [3.0]], dtype=np.float32), (4, 1)),
        units="pA",
        dt=0.001,
    )
    dialog = Igor2SingleWaveExportDialog("stim", stim)
    qtbot.addWidget(dialog)

    dialog._as_sweeps_check.setChecked(True)
    status = dialog.findChild(QtWidgets.QLabel, "sweep_detection_status")
    assert status is not None
    assert "Period detected: 3 samples × 4 sweeps" in status.text()


def test_dialog_status_reports_no_period_for_non_periodic_data(qtbot: QtBot) -> None:
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=np.arange(7, dtype=np.float32).reshape(7, 1),
        units="pA",
        dt=0.001,
    )
    dialog = Igor2SingleWaveExportDialog("stim", stim)
    qtbot.addWidget(dialog)

    dialog._as_sweeps_check.setChecked(True)
    status = dialog.findChild(QtWidgets.QLabel, "sweep_detection_status")
    assert status is not None
    assert "No periodic block detected" in status.text()


def test_dialog_notes_that_inhibitory_arrays_are_negated(qtbot: QtBot) -> None:
    stim = _stim()
    stim.stim_type = StimType.Inhibition
    dialog = Igor2SingleWaveExportDialog("stim", stim)
    qtbot.addWidget(dialog)

    texts = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
    assert any("negated" in text for text in texts)
    plain = Igor2SingleWaveExportDialog("stim", _stim())
    qtbot.addWidget(plain)
    assert not any(
        "negated" in label.text() for label in plain.findChildren(QtWidgets.QLabel)
    )
