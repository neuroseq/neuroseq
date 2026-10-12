import numpy as np
from pytestqt.qtbot import QtBot

from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stim_dialog import EditStimDialog


def test_edit_stim_dialog(qtbot: QtBot):
    stim_array = StimArray(
        name="s1",
        stim_type=StimType.Inhibition,
        values=np.array([[0.0, 0.1], [0.2, 0.3]]),
        units="mV",
        dt=0.001,
    )
    existing_names = ["s2", "s3"]

    dialog = EditStimDialog(stim_array, existing_names)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    # Check initial values
    assert dialog.txt_name.text() == "s1"
    assert dialog.stim_type == StimType.Inhibition
    assert dialog.n_neurons == 2
    assert dialog.units == "mV"
    assert dialog.dt == 0.001

    # Change name to an existing name
    assert dialog.btn_save.isEnabled()
    dialog.txt_name.setText("s2")
    assert not dialog.btn_save.isEnabled()

    # Change name to a new name and check validation
    dialog.txt_name.setText("s8")
    assert dialog.btn_save.isEnabled()
    assert dialog.txt_dt.isEnabled()

    # Change dt to an invalid value and back
    dialog.txt_dt.setText("0")
    assert not dialog.btn_save.isEnabled()
    dialog.txt_dt.setText("0.002")
    assert dialog.btn_save.isEnabled()
    assert dialog.dt == 0.002

    # Change stim type
    dialog.cmb_type.setCurrentIndex(0)
    assert dialog.stim_type == StimType.Excitation
    dialog.cmb_type.setCurrentIndex(1)
    assert dialog.stim_type == StimType.Inhibition
    dialog.cmb_type.setCurrentIndex(2)
    assert dialog.stim_type == StimType.External


def test_syscon_timestep_preset_preserves_exact_quantum(qtbot):
    import numpy as np

    from neuroseq.stim import StimArray
    from neuroseq.stim import StimType
    from neuroseq.stim_dialog import EditStimDialog

    dialog = EditStimDialog(
        StimArray("stim", StimType.Excitation, np.zeros((10, 2)), "1", 0.001), []
    )
    qtbot.addWidget(dialog)
    index = dialog.cmb_dt_presets.findData(0.0005208125)
    dialog._apply_dt_preset(index)
    assert dialog.dt == 0.0005208125
    assert dialog.txt_dt.text() == "0.0005208125"
