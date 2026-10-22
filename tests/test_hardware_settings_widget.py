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


def test_without_host_settings_the_export_options_are_disabled(qtbot) -> None:
    widget = HardwareSettingsWidget()
    qtbot.addWidget(widget)
    widget.set_host(None)
    assert not widget.spn_power_mw.isEnabled()
    assert widget.chk_automatic.isEnabled()


def test_recording_options_only_apply_to_automatic_blocks(qtbot) -> None:
    widget, _host = _widget(qtbot, response_threshold=True)
    widget.chk_automatic.setChecked(False)
    assert not widget.spn_tail_s.isEnabled()
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
    widget.spn_spike_threshold.setValue(0.3)
    stim = StimArray(
        "Stim", StimType.External, np.arange(20.0).reshape(10, 2), "", 0.001
    )
    recording = widget.block_recording(stim, 2, 6, "rheobase")
    np.testing.assert_array_equal(recording.stim_block.values, stim.values[2:6])
    assert (recording.label, recording.tail_s) == ("rheobase", 4.0)
    assert recording.spike_threshold == 0.3
    assert recording.duration_s == 0.004


def test_refresh_leaves_a_running_routines_lock_alone(qtbot) -> None:
    """The dialog disables the box while running; refreshing must not undo that
    (review finding 2)."""
    widget, host = _widget(qtbot)
    widget.setEnabled(False)
    widget.refresh()
    widget.set_host(host)
    host.export_settings_changed.emit()
    assert not widget.isEnabled()
    assert not widget.chk_automatic.isEnabled()


def test_follows_settings_changed_elsewhere_without_overwriting_them(qtbot) -> None:
    """A change in the SysCon2 panel shows up here, and editing another option here
    keeps it (review finding 3)."""
    widget, host = _widget(qtbot)
    host.settings = SysconExportSettings(config_text="", target_power_mw=40.0)
    host.export_settings_changed.emit()
    assert widget.spn_power_mw.value() == 40.0
    # Even without a notification, an edit only touches its own option.
    host.settings = SysconExportSettings(config_text="", target_power_mw=55.0)
    widget.chk_spiral.setChecked(True)
    assert host.settings.target_power_mw == 55.0
    assert host.settings.spiralling


def test_switching_hosts_stops_following_the_old_one(qtbot) -> None:
    widget, old = _widget(qtbot)
    new = FakeStimulationHost()
    new.settings = SysconExportSettings(config_text="", target_power_mw=7.0)
    widget.set_host(new)
    old.settings = SysconExportSettings(config_text="", target_power_mw=99.0)
    old.export_settings_changed.emit()
    assert widget.spn_power_mw.value() == 7.0
