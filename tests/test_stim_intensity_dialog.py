from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from neuroseq.stim import DrawStim
from neuroseq.stim import DrawStimType
from neuroseq.stim import StimType
from neuroseq.stim_intensity_dialog import StimIntensityDialog


def test_stim_intensity_dialog(qtbot: QtBot):
    draw_stim = DrawStim(DrawStimType.Random, 0.5, 1.0)
    stim_units = "mV"
    stim_type = StimType.Excitation

    dialog = StimIntensityDialog(draw_stim, stim_units, stim_type)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

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
    assert dialog.rad_intensity_random.isChecked()

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

    # change to linear
    dialog.rad_intensity_linear.setChecked(True)
    dialog.intensity_type_changed()
    assert dialog.draw_stim.draw_stim_type == DrawStimType.Linear
    assert dialog.label_value2.isEnabled()
    assert dialog.label_value2.isVisible()
    assert dialog.txt_value2.isEnabled()
    assert dialog.txt_value2.isVisible()

    # change to logarithmic
    dialog.rad_intensity_log.setChecked(True)
    dialog.intensity_type_changed()
    assert dialog.draw_stim.draw_stim_type == DrawStimType.Logarithmic
    assert dialog.label_value2.isEnabled()
    assert dialog.label_value2.isVisible()
    assert dialog.txt_value2.isEnabled()
    assert dialog.txt_value2.isVisible()
    # zero is invalid for log scale
    assert dialog.btn_ok.isEnabled()
    dialog.txt_value1.setText("0")
    assert not dialog.btn_ok.isEnabled()
    dialog.txt_value1.setText("1e-15")
    assert dialog.btn_ok.isEnabled()
