from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim import StimType
from neuroseq.stim_intensity_dialog import StimIntensityDialog


def test_stim_intensity_dialog(qtbot: QtBot):
    draw_stim = DrawStim(DrawStimType.Linear, 0.5, 1.0)
    stim_units = "mV"
    stim_type = StimType.Excitation

    dialog = StimIntensityDialog(draw_stim, stim_units, stim_type)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert dialog.windowTitle() == "Draw designer stimuli"
    assert dialog.rad_intensity_linear.text() == "Ramp"
    assert not hasattr(dialog, "rad_intensity_log")
    assert not hasattr(dialog, "rad_intensity_random")

    # value1
    assert stim_units in dialog.label_value1.text()
    assert dialog.txt_value1.isEnabled()
    assert dialog.txt_value1.isVisible()
    assert float(dialog.txt_value1.text()) == pytest.approx(draw_stim.value1)

    # value 2
    assert stim_units in dialog.label_value2.text()
    assert dialog.txt_value2.isEnabled()
    assert dialog.txt_value2.isVisible()
    assert float(dialog.txt_value2.text()) == pytest.approx(draw_stim.value2)

    # correct radio button selected
    assert dialog.rad_intensity_linear.isChecked()

    assert dialog.btn_ok.isEnabled()

    # invalid value1
    dialog.txt_value1.setText("not a number")
    assert not dialog.btn_ok.isEnabled()
    dialog.txt_value1.setText("-1")
    assert not dialog.btn_ok.isEnabled()

    dialog.txt_value1.setText("0.1")
    assert dialog.btn_ok.isEnabled()

    # invalid value2
    dialog.txt_value2.setText("not a number")
    assert not dialog.btn_ok.isEnabled()
    dialog.txt_value2.setText("-1")
    assert not dialog.btn_ok.isEnabled()

    dialog.txt_value2.setText("0.2")
    assert dialog.btn_ok.isEnabled()

    assert dialog.draw_stim.value1 == pytest.approx(0.1)
    assert dialog.draw_stim.value2 == pytest.approx(0.2)

    # change to constant
    dialog.rad_intensity_constant.setChecked(True)
    dialog.intensity_type_changed()
    assert dialog.draw_stim.draw_stim_type == DrawStimType.Constant
    assert not dialog.label_value2.isEnabled()
    assert not dialog.label_value2.isVisible()
    assert not dialog.txt_value2.isEnabled()
    assert not dialog.txt_value2.isVisible()

    # change to ramp (DrawStimType.Linear internally)
    dialog.rad_intensity_linear.setChecked(True)
    dialog.intensity_type_changed()
    assert dialog.draw_stim.draw_stim_type == DrawStimType.Linear
    assert dialog.label_value2.isEnabled()
    assert dialog.label_value2.isVisible()
    assert dialog.txt_value2.isEnabled()
    assert dialog.txt_value2.isVisible()


def test_stim_intensity_dialog_legacy_random_falls_back_to_ramp(qtbot: QtBot):
    # An existing draw_stim that was saved as Random / Logarithmic in older
    # versions must open on the new Ramp (Linear) radio rather than crash.
    draw_stim = DrawStim(DrawStimType.Random, 0.5, 1.0)
    dialog = StimIntensityDialog(draw_stim, "mV", StimType.Excitation)
    qtbot.addWidget(dialog)

    assert dialog.draw_stim.draw_stim_type == DrawStimType.Linear
    assert dialog.rad_intensity_linear.isChecked()
