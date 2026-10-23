"""Hardware settings of a dynamic input routine delivered through SysCon2 (issue #651).

Shown by the rheobase and STDP window dialogs when their target is SysCon2, so a routine
can be set up without a prior "Export with settings…": the holographic export options
(shared with the SysCon2 panel and saved with the project) and how each block is recorded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from qtpy import QtWidgets

from neuroseq.device_control.syscon2_stimulation_host import BlockRecording
from neuroseq.export.exporter_syscon2 import SUPPORTED_LASER_WAVELENGTHS_NM
from neuroseq.export.syscon2_export_service import slice_stim_array
from neuroseq.importers.importer_scanimage_traces import DEFAULT_SPIKE_THRESHOLD

if TYPE_CHECKING:
    from neuroseq.device_control.syscon2_stimulation_host import SysCon2StimulationHost
    from neuroseq.stimulation.stim import StimArray


class HardwareSettingsWidget(QtWidgets.QGroupBox):
    """Export and recording options of SysCon2 blocks.

    With "Run blocks automatically", the host uploads and runs every block itself and
    ScanImage records it (triggered by the sequence); otherwise each block is only
    imported into SysCon2 and the user runs, records and imports it.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        response_threshold: bool = False,
    ) -> None:
        super().__init__("Hardware settings (SysCon2 / ScanImage)", parent)
        self._host: SysCon2StimulationHost | None = None
        form = QtWidgets.QFormLayout(self)
        self.chk_automatic = QtWidgets.QCheckBox(
            "Run blocks automatically and record them with ScanImage", self
        )
        self.chk_automatic.setChecked(True)
        self.chk_automatic.setToolTip(
            "Upload and run each block in SysCon2 while ScanImage records it, "
            "started by the sequence's trigger, then extract ΔF/F traces with the "
            "project's Cellpose masks. Unticked, each block is only imported into "
            "SysCon2; you run, record and import it yourself."
        )
        form.addRow(self.chk_automatic)
        self.spn_power_mw = QtWidgets.QDoubleSpinBox(self)
        self.spn_power_mw.setRange(0.0, 1000.0)
        self.spn_power_mw.setDecimals(2)
        self.spn_power_mw.setSingleStep(0.5)
        self.spn_power_mw.setSuffix(" mW")
        self.spn_power_mw.setToolTip(
            "Power per spot at amplitude 1; amplitudes are fractions of it. "
            "0 disables power control."
        )
        form.addRow("Target power:", self.spn_power_mw)
        self.cmb_laser = QtWidgets.QComboBox(self)
        # The lasers the HOLO4D export can drive.
        for wavelength in SUPPORTED_LASER_WAVELENGTHS_NM:
            self.cmb_laser.addItem(f"{wavelength} nm", wavelength)
        form.addRow("Laser:", self.cmb_laser)
        self.chk_spiral = QtWidgets.QCheckBox("Spiral scanning", self)
        self.chk_ten_spots = QtWidgets.QCheckBox("10 spots per cell", self)
        spot_row = QtWidgets.QHBoxLayout()
        spot_row.addWidget(self.chk_spiral)
        spot_row.addWidget(self.chk_ten_spots)
        form.addRow("Spots:", spot_row)
        self.spn_tail_s = QtWidgets.QDoubleSpinBox(self)
        self.spn_tail_s.setRange(0.0, 600.0)
        self.spn_tail_s.setDecimals(1)
        self.spn_tail_s.setValue(3.0)
        self.spn_tail_s.setSuffix(" s")
        self.spn_tail_s.setToolTip(
            "Recording kept after the end of each block, for the calcium response to "
            "decay. The number of frames follows from the block length and this tail."
        )
        form.addRow("Record after block:", self.spn_tail_s)
        channel_note = QtWidgets.QLabel(
            "Measured on the Functional channel set in the ScanImage tab.", self
        )
        channel_note.setWordWrap(True)
        form.addRow("Calcium channel:", channel_note)
        self.spn_threshold: QtWidgets.QDoubleSpinBox | None = None
        self.spn_spike_threshold = QtWidgets.QDoubleSpinBox(self)
        self.spn_spike_threshold.setRange(0.001, 100.0)
        self.spn_spike_threshold.setDecimals(3)
        self.spn_spike_threshold.setSingleStep(0.05)
        self.spn_spike_threshold.setValue(DEFAULT_SPIKE_THRESHOLD)
        self.spn_spike_threshold.setSuffix(" ΔF/F")
        self.spn_spike_threshold.setToolTip(
            "Each recorded block's ΔF/F is deconvolved with OASIS (jGCaMP8m decay, "
            "0.5 s); an event at least this large, in ΔF/F, counts as a spike. A "
            "single action potential of jGCaMP8m gives a few tenths."
        )
        self._lbl_spike_threshold = QtWidgets.QLabel("Spike threshold:", self)
        self._lbl_spike_threshold.setVisible(response_threshold)
        self.spn_spike_threshold.setVisible(response_threshold)
        if response_threshold:
            self.spn_threshold = QtWidgets.QDoubleSpinBox(self)
            self.spn_threshold.setRange(0.1, 10_000.0)
            self.spn_threshold.setDecimals(1)
            self.spn_threshold.setValue(20.0)
            self.spn_threshold.setSuffix(" % ΔF/F")
            self.spn_threshold.setToolTip(
                "A probe counts as evoking a response when the neuron's ΔF/F rises "
                "by at least this much after the pulse."
            )
            self._lbl_threshold = QtWidgets.QLabel("Response threshold:", self)
            form.addRow(self._lbl_threshold, self.spn_threshold)
            form.addRow(self._lbl_spike_threshold, self.spn_spike_threshold)
        # Each control changes only its own field, so options changed elsewhere
        # (the SysCon2 panel) are never overwritten with what this box showed.
        self.spn_power_mw.valueChanged.connect(
            lambda value: self._store_setting(target_power_mw=float(value))
        )
        self.cmb_laser.currentIndexChanged.connect(
            lambda _index: self._store_setting(
                laser_wavelength_nm=int(self.cmb_laser.currentData())
            )
        )
        self.chk_spiral.toggled.connect(
            lambda checked: self._store_setting(spiralling=checked)
        )
        self.chk_ten_spots.toggled.connect(
            lambda checked: self._store_setting(ten_spots_per_roi=checked)
        )
        self.chk_automatic.toggled.connect(self._update_recording_controls)
        self._export_widgets: tuple[QtWidgets.QWidget, ...] = (
            self.spn_power_mw,
            self.cmb_laser,
            self.chk_spiral,
            self.chk_ten_spots,
        )

    def _update_recording_controls(self, automatic: bool) -> None:
        """Recording options only apply to blocks NeuroSeq records itself."""
        self.spn_tail_s.setEnabled(automatic)
        self.spn_spike_threshold.setEnabled(automatic)

    def show_readout(self, dff: bool) -> None:
        """Show the threshold of the routine's calcium readout: ΔF/F rise or spikes."""
        if self.spn_threshold is None:
            return
        self._lbl_threshold.setVisible(dff)
        self.spn_threshold.setVisible(dff)
        self._lbl_spike_threshold.setVisible(not dff)
        self.spn_spike_threshold.setVisible(not dff)

    @property
    def automatic(self) -> bool:
        return self.chk_automatic.isChecked()

    @property
    def response_threshold(self) -> float:
        """ΔF/F rise counted as a response (a fraction, 0.2 for 20 %)."""
        return self.spn_threshold.value() / 100.0 if self.spn_threshold else 0.0

    def set_host(self, host: SysCon2StimulationHost | None) -> None:
        if host is not self._host:
            if self._host is not None:
                self._host.export_settings_changed.disconnect(self.refresh)
            if host is not None:
                host.export_settings_changed.connect(self.refresh)
            self._host = host
        self.refresh()

    def refresh(self) -> None:
        """Show the export options currently stored for the project.

        Leaves the box's own enabled state to the dialog (disabled while running).
        """
        settings = self._host.export_settings() if self._host is not None else None
        widgets = self._export_widgets
        for widget in widgets:
            widget.setEnabled(settings is not None)
        if settings is None:
            return
        for widget in widgets:
            widget.blockSignals(True)
        self.spn_power_mw.setValue(settings.target_power_mw)
        index = self.cmb_laser.findData(settings.laser_wavelength_nm)
        if index < 0:
            self.cmb_laser.addItem(
                f"{settings.laser_wavelength_nm} nm", settings.laser_wavelength_nm
            )
            index = self.cmb_laser.count() - 1
        self.cmb_laser.setCurrentIndex(index)
        self.chk_spiral.setChecked(settings.spiralling)
        self.chk_ten_spots.setChecked(settings.ten_spots_per_roi)
        for widget in widgets:
            widget.blockSignals(False)

    def _store_setting(self, **change: object) -> None:
        if self._host is None:
            return
        settings = self._host.export_settings()
        if settings is not None:
            self._host.set_export_settings(replace(settings, **change))  # type: ignore[arg-type]

    def block_recording(
        self, stim: StimArray, start: int, stop: int, label: str
    ) -> BlockRecording:
        """How to record the block ``stim[start:stop]``."""
        return BlockRecording(
            label=label,
            stim_block=slice_stim_array(stim, start, stop),
            tail_s=self.spn_tail_s.value(),
            spike_threshold=self.spn_spike_threshold.value(),
        )
