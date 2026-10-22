import numpy as np

from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.stimulation.dynamic_input.hardware_settings_widget import (
    HardwareSettingsWidget,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.fake_stimulation_host import FakeStimulationHost


def _widget(qtbot, **kwargs):
    widget = HardwareSettingsWidget(**kwargs)
    qtbot.addWidget(widget)
    host = FakeStimulationHost()
    widget.set_host(host)
    return widget, host


def test_settings_edit_the_hosts_export_settings(qtbot) -> None:
    widget, host = _widget(qtbot)
    widget.spn_power_mw.setValue(23.0)
    widget.cmb_laser.setCurrentIndex(widget.cmb_laser.findData(1100))
    widget.chk_spiral.setChecked(True)
    widget.chk_ten_spots.setChecked(False)
    assert host.settings.target_power_mw == 23.0
    assert host.settings.laser_wavelength_nm == 1100
    assert host.settings.spiralling
    assert not host.settings.ten_spots_per_roi
    host.settings = SysconExportSettings(config_text="", target_power_mw=5.0)
    widget.refresh()
    assert widget.spn_power_mw.value() == 5.0


def test_without_host_settings_the_box_is_disabled(qtbot) -> None:
    widget = HardwareSettingsWidget()
    qtbot.addWidget(widget)
    widget.set_host(None)
    assert not widget.isEnabled()


def test_recording_options_only_apply_to_automatic_blocks(qtbot) -> None:
    widget, _host = _widget(qtbot, response_threshold=True)
    widget.chk_automatic.setChecked(False)
    assert not widget.spn_tail_s.isEnabled()
    assert not widget.spn_channel.isEnabled()
    assert not widget.spn_spike_threshold.isEnabled()
    widget.chk_automatic.setChecked(True)
    assert widget.spn_tail_s.isEnabled()


def test_readout_shows_its_own_threshold(qtbot) -> None:
    widget, _host = _widget(qtbot, response_threshold=True)
    widget.show_readout(True)
    assert widget.spn_threshold.isVisibleTo(widget)
    assert not widget.spn_spike_threshold.isVisibleTo(widget)
    widget.show_readout(False)
    assert not widget.spn_threshold.isVisibleTo(widget)
    assert widget.spn_spike_threshold.isVisibleTo(widget)
    widget.spn_threshold.setValue(35.0)
    assert widget.response_threshold == 0.35


def test_block_recording_describes_the_block(qtbot) -> None:
    widget, _host = _widget(qtbot, response_threshold=True)
    widget.spn_tail_s.setValue(4.0)
    widget.spn_channel.setValue(2)
    widget.spn_spike_threshold.setValue(0.3)
    stim = StimArray(
        "Stim", StimType.External, np.arange(20.0).reshape(10, 2), "", 0.001
    )
    recording = widget.block_recording(stim, 2, 6, "rheobase")
    np.testing.assert_array_equal(recording.stim_block.values, stim.values[2:6])
    assert (recording.label, recording.tail_s) == ("rheobase", 4.0)
    assert (recording.functional_channel, recording.spike_threshold) == (2, 0.3)
    assert recording.duration_s == 0.004
