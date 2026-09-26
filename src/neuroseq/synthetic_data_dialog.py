from __future__ import annotations

import logging
from fractions import Fraction

import brian2 as b2
import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets
from scipy.signal import resample_poly

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.project import Project


class SyntheticDataDialog(QtWidgets.QDialog):
    """Dialog to generate synthetic trace and spike data and apply to widgets.

    The dialog can be constructed with references to an existing
    `TraceWidget` and `SpikeWidget`. Generated data will be applied to those
    widgets when the user clicks "Generate & Apply".
    """

    def __init__(
        self,
        model: Project,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.data_added = False  # Track if data was added to the project

        self.setWindowTitle("Generate Synthetic Data")
        self.setModal(True)
        self.resize(520, 360)

        layout = QtWidgets.QVBoxLayout(self)

        # Patch clamp

        data_patch_clamp_group = QtWidgets.QGroupBox("Patch clamp")
        data_patch_clamp_group_layout = QtWidgets.QVBoxLayout()

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select traces:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_patch_clamp_combo = QtWidgets.QComboBox()
        self.data_patch_clamp_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(TraceData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_patch_clamp_combo.addItem("")
        for name in data_names:
            self.data_patch_clamp_combo.addItem(name)

        if data_names:
            self.data_patch_clamp_combo.setCurrentIndex(1)

        self.data_patch_clamp_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_patch_clamp_combo)
        data_patch_clamp_group_layout.addLayout(data_inner_layout)

        # Noise value selector
        noise_layout = QtWidgets.QHBoxLayout()
        noise_label = QtWidgets.QLabel("Noise (mV):")
        noise_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.noise_spinbox = QtWidgets.QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 10.0)
        self.noise_spinbox.setValue(0.5)
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setDecimals(2)
        self.noise_spinbox.setToolTip(
            "Noise level in millivolts to add to the trace data"
        )
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_spinbox)
        data_patch_clamp_group_layout.addLayout(noise_layout)

        sampling_layout = QtWidgets.QHBoxLayout()
        sampling_layout.addWidget(QtWidgets.QLabel("Output sampling rate (Hz):"))
        self.patch_sampling_rate_spinbox = QtWidgets.QDoubleSpinBox()
        self.patch_sampling_rate_spinbox.setDecimals(2)
        self.patch_sampling_rate_spinbox.setRange(0.01, 1_000_000.0)
        self.patch_sampling_rate_spinbox.setValue(1000.0)
        self.patch_sampling_rate_spinbox.setToolTip(
            "Sampling rate of the generated patch-clamp traces. Must not exceed "
            "the source rate. Low-pass filtering prevents aliasing when downsampling."
        )
        sampling_layout.addWidget(self.patch_sampling_rate_spinbox)
        data_patch_clamp_group_layout.addLayout(sampling_layout)

        # Index selector
        index_layout = QtWidgets.QHBoxLayout()
        index_label = QtWidgets.QLabel("Indexes to import:")
        index_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.index_lineedit = QtWidgets.QLineEdit()
        self.index_lineedit.setPlaceholderText("e.g., 0,1,5-15,20")
        self.index_lineedit.setToolTip(
            "Comma-separated neuron indexes and/or ranges (e.g., 0,1,5-15,20). Leave empty for all."
        )
        index_layout.addWidget(index_label)
        index_layout.addWidget(self.index_lineedit)
        data_patch_clamp_group_layout.addLayout(index_layout)

        # Buttons for patch clamp
        data_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_export_trace = QtWidgets.QPushButton("Export as a file")
        self.btn_export_trace.clicked.connect(self._export_patch_clamp_data)
        self.btn_add_trace = QtWidgets.QPushButton("Add to the project")
        self.btn_add_trace.clicked.connect(self._add_patch_clamp_to_project)
        data_buttons_layout.addWidget(self.btn_export_trace)
        data_buttons_layout.addWidget(self.btn_add_trace)
        data_patch_clamp_group_layout.addLayout(data_buttons_layout)

        data_patch_clamp_group.setLayout(data_patch_clamp_group_layout)

        layout.addWidget(data_patch_clamp_group)

        ### Imaging

        data_imaging_group = QtWidgets.QGroupBox("Imaging")
        data_imaging_group_layout = QtWidgets.QVBoxLayout()

        data_inner_layout = QtWidgets.QHBoxLayout()
        data_label = QtWidgets.QLabel("Select spikes:")
        data_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.data_imaging_combo = QtWidgets.QComboBox()
        self.data_imaging_combo.setToolTip(
            "Select a dataset from the model (simulation or experimental)"
        )
        data_names = self.model.get_data_names(SpikeData)
        # populate with all data names from model
        # data_names: list[str] = []
        # data_names.extend(self.model.get_data_names(TraceData))
        # data_names.extend(self.model.get_data_names(SpikeData))
        # data_names.extend(self.model.get_data_names(SynapseData))

        self.data_imaging_combo.addItem("")
        for name in data_names:
            self.data_imaging_combo.addItem(name)

        if data_names:
            self.data_imaging_combo.setCurrentIndex(1)

        self.data_imaging_combo.currentTextChanged.connect(self._update_preview)

        data_inner_layout.addWidget(data_label)
        data_inner_layout.addWidget(self.data_imaging_combo)
        data_imaging_group_layout.addLayout(data_inner_layout)

        # Recording duration selector
        duration_layout = QtWidgets.QHBoxLayout()
        duration_label = QtWidgets.QLabel("Recording duration (s):")
        duration_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.recording_duration_spinbox = QtWidgets.QDoubleSpinBox()
        self.recording_duration_spinbox.setRange(0.1, 1000000.0)
        self.recording_duration_spinbox.setValue(10.0)
        self.recording_duration_spinbox.setSingleStep(1.0)
        self.recording_duration_spinbox.setDecimals(2)
        self.recording_duration_spinbox.setToolTip(
            "Duration of the recording in seconds"
        )
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.recording_duration_spinbox)
        data_imaging_group_layout.addLayout(duration_layout)

        # Number of neurons selector
        num_neurons_layout = QtWidgets.QHBoxLayout()
        num_neurons_label = QtWidgets.QLabel("Number of neurons:")
        num_neurons_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.num_neurons_spinbox = QtWidgets.QSpinBox()
        self.num_neurons_spinbox.setRange(1, 1000000)
        self.num_neurons_spinbox.setValue(10)
        self.num_neurons_spinbox.setSingleStep(1)
        self.num_neurons_spinbox.setToolTip("Number of neurons in the imaging session")
        num_neurons_layout.addWidget(num_neurons_label)
        num_neurons_layout.addWidget(self.num_neurons_spinbox)
        data_imaging_group_layout.addLayout(num_neurons_layout)

        # Index selector for imaging
        imaging_index_layout = QtWidgets.QHBoxLayout()
        imaging_index_label = QtWidgets.QLabel("Indexes to import:")
        imaging_index_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.imaging_index_lineedit = QtWidgets.QLineEdit()
        self.imaging_index_lineedit.setPlaceholderText("e.g., 0,1,5-15,20")
        self.imaging_index_lineedit.setToolTip(
            "Comma-separated neuron indexes and/or ranges (e.g., 0,1,5-15,20). Leave empty for all."
        )
        imaging_index_layout.addWidget(imaging_index_label)
        imaging_index_layout.addWidget(self.imaging_index_lineedit)
        data_imaging_group_layout.addLayout(imaging_index_layout)

        # F0 selector
        f0_layout = QtWidgets.QHBoxLayout()
        f0_label = QtWidgets.QLabel("F0 (photoelectrons):")
        f0_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.f0_spinbox = QtWidgets.QDoubleSpinBox()
        self.f0_spinbox.setRange(1.0, 100000.0)
        self.f0_spinbox.setValue(1000.0)
        self.f0_spinbox.setSingleStep(100.0)
        self.f0_spinbox.setDecimals(1)
        self.f0_spinbox.setToolTip("Baseline mean fluorescence in photoelectrons")
        f0_layout.addWidget(f0_label)
        f0_layout.addWidget(self.f0_spinbox)
        data_imaging_group_layout.addLayout(f0_layout)

        # dF/F selector
        df_f_layout = QtWidgets.QHBoxLayout()
        df_f_label = QtWidgets.QLabel("dF/F:")
        df_f_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.df_f_spinbox = QtWidgets.QDoubleSpinBox()
        self.df_f_spinbox.setRange(0.0, 10.0)
        self.df_f_spinbox.setValue(0.4)
        self.df_f_spinbox.setSingleStep(0.1)
        self.df_f_spinbox.setDecimals(2)
        self.df_f_spinbox.setToolTip("Mean relative fluorescence change per spike")
        df_f_layout.addWidget(df_f_label)
        df_f_layout.addWidget(self.df_f_spinbox)
        data_imaging_group_layout.addLayout(df_f_layout)

        # dF/F variability selector
        df_f_var_layout = QtWidgets.QHBoxLayout()
        df_f_var_label = QtWidgets.QLabel("dF/F variability:")
        df_f_var_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.df_f_var_spinbox = QtWidgets.QDoubleSpinBox()
        self.df_f_var_spinbox.setRange(0.0, 1.0)
        self.df_f_var_spinbox.setValue(0.04)
        self.df_f_var_spinbox.setSingleStep(0.01)
        self.df_f_var_spinbox.setDecimals(3)
        self.df_f_var_spinbox.setToolTip("Standard deviation of dF/F per spike")
        df_f_var_layout.addWidget(df_f_var_label)
        df_f_var_layout.addWidget(self.df_f_var_spinbox)
        data_imaging_group_layout.addLayout(df_f_var_layout)

        response_layout = QtWidgets.QHBoxLayout()
        response_layout.addWidget(QtWidgets.QLabel("Indicator response:"))
        self.indicator_response_combo = QtWidgets.QComboBox()
        self.indicator_response_combo.addItem("Exponential decay", "decay")
        self.indicator_response_combo.addItem("Rise and decay", "rise_decay")
        response_layout.addWidget(self.indicator_response_combo)
        data_imaging_group_layout.addLayout(response_layout)

        rise_layout = QtWidgets.QHBoxLayout()
        rise_layout.addWidget(QtWidgets.QLabel("Rise time constant (s):"))
        self.rise_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.rise_time_spinbox.setRange(0.001, 10.0)
        self.rise_time_spinbox.setDecimals(3)
        self.rise_time_spinbox.setSingleStep(0.001)
        self.rise_time_spinbox.setValue(0.01)
        self.rise_time_spinbox.setToolTip(
            "Rise time constant, not time to peak. Must be shorter than decay time. "
            "dF/F sets the peak response to an isolated spike."
        )
        rise_layout.addWidget(self.rise_time_spinbox)
        data_imaging_group_layout.addLayout(rise_layout)

        # Decay time constant selector
        decay_layout = QtWidgets.QHBoxLayout()
        decay_label = QtWidgets.QLabel("Decay time (s):")
        decay_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.decay_time_spinbox = QtWidgets.QDoubleSpinBox()
        self.decay_time_spinbox.setRange(0.001, 10.0)
        self.decay_time_spinbox.setValue(0.05)
        self.decay_time_spinbox.setSingleStep(0.01)
        self.decay_time_spinbox.setDecimals(3)
        self.decay_time_spinbox.setToolTip("Calcium decay time constant in seconds")
        decay_layout.addWidget(decay_label)
        decay_layout.addWidget(self.decay_time_spinbox)
        data_imaging_group_layout.addLayout(decay_layout)

        # Sampling rate selector
        sampling_layout = QtWidgets.QHBoxLayout()
        sampling_label = QtWidgets.QLabel("Sampling rate (Hz):")
        sampling_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.sampling_rate_spinbox = QtWidgets.QDoubleSpinBox()
        self.sampling_rate_spinbox.setRange(0.1, 10000.0)
        self.sampling_rate_spinbox.setValue(30.0)
        self.sampling_rate_spinbox.setSingleStep(1.0)
        self.sampling_rate_spinbox.setDecimals(1)
        self.sampling_rate_spinbox.setToolTip("Imaging sampling rate in Hz")
        sampling_layout.addWidget(sampling_label)
        sampling_layout.addWidget(self.sampling_rate_spinbox)
        data_imaging_group_layout.addLayout(sampling_layout)

        # Lognormal sigma selector
        lognormal_layout = QtWidgets.QHBoxLayout()
        lognormal_label = QtWidgets.QLabel("Lognormal sigma (F0 distribution):")
        lognormal_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.lognormal_sigma_spinbox = QtWidgets.QDoubleSpinBox()
        self.lognormal_sigma_spinbox.setRange(0.0, 2.0)
        self.lognormal_sigma_spinbox.setValue(0.25)
        self.lognormal_sigma_spinbox.setSingleStep(0.05)
        self.lognormal_sigma_spinbox.setDecimals(2)
        self.lognormal_sigma_spinbox.setToolTip(
            "Lognormal distribution sigma for neuron-to-neuron fluorescence variability"
        )
        lognormal_layout.addWidget(lognormal_label)
        lognormal_layout.addWidget(self.lognormal_sigma_spinbox)
        data_imaging_group_layout.addLayout(lognormal_layout)

        # Lognormal mean selector
        lognormal_mean_layout = QtWidgets.QHBoxLayout()
        lognormal_mean_label = QtWidgets.QLabel("Lognormal mean (F0 distribution):")
        lognormal_mean_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.lognormal_mean_spinbox = QtWidgets.QDoubleSpinBox()
        self.lognormal_mean_spinbox.setRange(-5.0, 5.0)
        self.lognormal_mean_spinbox.setValue(0.0)
        self.lognormal_mean_spinbox.setSingleStep(0.1)
        self.lognormal_mean_spinbox.setDecimals(2)
        self.lognormal_mean_spinbox.setToolTip(
            "Lognormal distribution mean for neuron-to-neuron fluorescence variability"
        )
        lognormal_mean_layout.addWidget(lognormal_mean_label)
        lognormal_mean_layout.addWidget(self.lognormal_mean_spinbox)
        data_imaging_group_layout.addLayout(lognormal_mean_layout)

        # Photoelectron noise factor selector
        noise_factor_layout = QtWidgets.QHBoxLayout()
        noise_factor_label = QtWidgets.QLabel("Photoelectron noise factor:")
        noise_factor_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.photoelectron_noise_spinbox = QtWidgets.QDoubleSpinBox()
        self.photoelectron_noise_spinbox.setRange(0.0, 10.0)
        self.photoelectron_noise_spinbox.setValue(1.0)
        self.photoelectron_noise_spinbox.setSingleStep(0.1)
        self.photoelectron_noise_spinbox.setDecimals(2)
        self.photoelectron_noise_spinbox.setToolTip(
            "Multiplicative factor for Poisson noise (1.0 = pure Poisson noise)"
        )
        noise_factor_layout.addWidget(noise_factor_label)
        noise_factor_layout.addWidget(self.photoelectron_noise_spinbox)
        data_imaging_group_layout.addLayout(noise_factor_layout)

        # Buttons for imaging
        imaging_buttons_layout = QtWidgets.QHBoxLayout()
        self.btn_export_imaging = QtWidgets.QPushButton("Export as a file")
        self.btn_export_imaging.clicked.connect(self._export_imaging_data)
        self.btn_add_imaging = QtWidgets.QPushButton("Add to the project")
        self.btn_add_imaging.clicked.connect(self._add_imaging_to_project)
        imaging_buttons_layout.addWidget(self.btn_export_imaging)
        imaging_buttons_layout.addWidget(self.btn_add_imaging)
        data_imaging_group_layout.addLayout(imaging_buttons_layout)

        self.indicator_validation_label = QtWidgets.QLabel()
        data_imaging_group_layout.addWidget(self.indicator_validation_label)
        self.indicator_response_combo.currentIndexChanged.connect(
            self._update_indicator_controls
        )
        self.rise_time_spinbox.valueChanged.connect(self._update_indicator_controls)
        self.decay_time_spinbox.valueChanged.connect(self._update_indicator_controls)
        self._update_indicator_controls()

        data_imaging_group.setLayout(data_imaging_group_layout)
        layout.addWidget(data_imaging_group)

        # Data Preview
        preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        # Initial preview update
        self._update_preview()

    def _update_indicator_controls(self) -> None:
        """Enable rise settings and prevent generation with invalid kinetics."""
        use_rise = self.indicator_response_combo.currentData() == "rise_decay"
        self.rise_time_spinbox.setEnabled(use_rise)
        valid = (
            not use_rise
            or self.rise_time_spinbox.value() < self.decay_time_spinbox.value()
        )
        self.indicator_validation_label.setText(
            "" if valid else "Rise time must be shorter than decay time."
        )
        self.btn_export_imaging.setEnabled(valid)
        self.btn_add_imaging.setEnabled(valid)

    def _parse_indices(self, index_string: str) -> np.ndarray | None:
        """Parse comma-separated indices and ranges into a numpy array.
        Args:
            index_string: String containing indices and/or ranges (e.g., "0,1,5-15,20")
        Returns:
            Numpy array of parsed indices, or None if empty string
        """
        index_string = index_string.strip()
        if not index_string:
            return None

        indices: list[int] = []
        parts = index_string.split(",")

        for part in parts:
            part = part.strip()
            if "-" in part:
                # Handle range (e.g., "5-15")
                try:
                    start, end = part.split("-")
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                    if start_idx > end_idx:
                        raise ValueError(
                            f"Invalid range '{part}': start of the range must be <= than the end of the range"
                        )
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"Invalid range format: '{part}'") from None
                    raise
            else:
                # Handle single index
                try:
                    indices.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid index: '{part}'") from None

        # Remove duplicates and sort
        indices = sorted(set(indices))
        return np.array(indices, dtype=int)

    def _apply_temporal_decay(
        self, traces: np.ndarray, decay_factor: float
    ) -> np.ndarray:
        """Apply temporal decay to imaging traces.

        Each element at position t is calculated based on the previous value at t-1
        multiplied by a decay factor, plus the current spike indicator.
        This simulates calcium imaging dynamics where the signal decays over time.

        Args:
            traces: Input array of shape (n_neurons, n_timepoints) with spike indicators
            decay_factor: Constant multiplication factor for temporal decay (0 < factor <= 1)

        Returns:
            Processed array with temporal decay applied to all rows simultaneously
        """
        # Create output array with same shape
        decayed_traces = np.zeros_like(traces, dtype=float)

        # First column is the same as input (no previous value)
        decayed_traces[:, 0] = traces[:, 0]

        # For each subsequent time point, calculate based on previous value
        # Process all rows at the same time using vectorized operations
        for t in range(1, traces.shape[1]):
            decayed_traces[:, t] = (
                decayed_traces[:, t - 1] * decay_factor + traces[:, t]
            )

        return decayed_traces

    def _apply_rise_and_decay(
        self, traces: np.ndarray, dt: float, rise_time: float, decay_time: float
    ) -> np.ndarray:
        """Filter spike amplitudes with a peak-normalized biexponential kernel.

        Each spike starts at zero and reaches its supplied amplitude at the
        continuous-time kernel peak. Overlapping responses add linearly.
        """
        if not 0 < rise_time < decay_time:
            raise ValueError("Rise time must be positive and shorter than decay time.")
        rate_difference = 1 / rise_time - 1 / decay_time
        peak_time = np.log1p((decay_time - rise_time) / rise_time) / rate_difference
        peak = np.exp(-peak_time / decay_time) * (
            -np.expm1(-peak_time * rate_difference)
        )
        decay_factor = np.exp(-dt / decay_time)
        rise_factor = np.exp(-dt / rise_time)
        decay_state = np.zeros(traces.shape[0])
        rise_state = np.zeros(traces.shape[0])
        result = np.zeros_like(traces, dtype=float)
        for t in range(traces.shape[1]):
            decay_state = decay_factor * decay_state + traces[:, t]
            rise_state = rise_factor * rise_state + traces[:, t]
            result[:, t] = (decay_state - rise_state) / peak
        return result

    def _generate_synthetic_imaging_traces(
        self,
        spike_data: SpikeData,
        indices: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Generate synthetic imaging traces from spike data.

        Args:
            spike_data: The spike data to convert to imaging traces.
            indices: Array of indices to extract, or None to generate all traces.

        Returns:
            Tuple of (imaging_traces, indices, dt_downsample)
        """
        # Calculate number of time points based on recording duration
        duration = self.recording_duration_spinbox.value()  # in seconds
        dt = 0.0001  # in seconds (0.1 ms)
        n_timepoints = int(duration / dt)
        num_neurons = self.num_neurons_spinbox.value()
        imaging_traces = np.zeros((num_neurons, n_timepoints))

        # Get imaging parameters from UI widgets
        F0 = self.f0_spinbox.value()  # baseline mean fluorescence in photoelectrons
        lognornal_mean = self.lognormal_mean_spinbox.value()
        lognormal_sigma = self.lognormal_sigma_spinbox.value()
        normalization_factor = np.exp(lognornal_mean + (lognormal_sigma**2) / 2)

        dF_F = self.df_f_spinbox.value()
        dF_F_variability = self.df_f_var_spinbox.value()
        decay_time_constant = self.decay_time_spinbox.value()  # seconds
        sampling_rate = self.sampling_rate_spinbox.value()  # Hz
        dt_downsample = 1.0 / sampling_rate  # seconds

        photelectron_noise_factor = (
            self.photoelectron_noise_spinbox.value()
        )  # 1 - corresponds to Poisson noise only

        # Insert 1s at spike positions (can be improved for large N if done after the selection of indices)
        for i, spike_time in enumerate(spike_data.spike_times):
            neuron_idx = spike_data.spike_indices[i]
            if neuron_idx < num_neurons:
                time_idx = int(spike_time / dt)
                if 0 <= time_idx < n_timepoints:
                    imaging_traces[neuron_idx, time_idx] = np.random.normal(
                        dF_F, dF_F_variability
                    )  # simulate variability in calcium amplitude

        indices = np.arange(num_neurons) if indices is None else indices

        imaging_traces = imaging_traces[indices]

        # Apply indicator dynamics before sampling; photobleaching is ignored.
        if self.indicator_response_combo.currentData() == "rise_decay":
            imaging_traces = self._apply_rise_and_decay(
                imaging_traces, dt, self.rise_time_spinbox.value(), decay_time_constant
            )
        else:
            decay_factor = np.exp(-dt / decay_time_constant)
            imaging_traces = self._apply_temporal_decay(imaging_traces, decay_factor)
        imaging_traces += 1

        # Downsample to imaging sampling rate
        # The mean of the signal can vary depending on downsampling strategy
        # During experiment the exact value will likely depend on the position of the neuron within the frame
        downsample_factor = int(dt_downsample / dt)
        imaging_traces = imaging_traces[:, ::downsample_factor]

        imaging_traces = imaging_traces * F0  # Convert to fluorescence units

        for i in range(imaging_traces.shape[0]):
            # Apply lognormal variability to each neuron
            lognormal_variability = (
                np.random.lognormal(lognornal_mean, lognormal_sigma)
                / normalization_factor
            )
            imaging_traces[i, :] *= lognormal_variability

        imaging_traces = np.random.poisson(
            imaging_traces * photelectron_noise_factor
        )  # Apply Poisson noise

        return imaging_traces, indices, dt_downsample

    def _update_preview(self) -> None:
        """Update the data preview based on current selections."""
        preview_lines = []

        # Patch clamp preview
        trace_name = self.data_patch_clamp_combo.currentText()
        if trace_name:
            trace_data = self.model.get_data(trace_name, TraceData)
            if trace_data:
                shape = trace_data.traces.shape
                dt = trace_data.dt
                # units = trace_data.units
                indices = trace_data.indices
                preview_lines.append(f"Traces: {trace_name}")
                preview_lines.append(f"  Number of neurons: {shape[0]}")
                preview_lines.append(f"  Duration in seconds: {shape[1] * dt:.2f}")
                # preview_lines.append(f"  Units: {units}")
                preview_lines.append("  Units: mV")  # hardcoded
                preview_lines.append(
                    f"  Neuron indices (range): from {indices[0]} to {indices[-1]}"
                )
                preview_lines.append(f"  Neuron indices (full list): {indices}")

                # Update default values for recording duration and number of neurons
                duration_seconds = shape[1] * dt
                num_neurons = shape[0]
                self.recording_duration_spinbox.setValue(duration_seconds)
                self.num_neurons_spinbox.setValue(num_neurons)

        # Imaging preview
        spike_name = self.data_imaging_combo.currentText()
        if spike_name:
            spike_data = self.model.get_data(spike_name, SpikeData)
            if spike_data:
                if preview_lines:
                    preview_lines.append("")

                shape = spike_data.spike_times.shape
                preview_lines.append(f"Spikes: {spike_name}")
                preview_lines.append(f"  Number of spikes: {shape[0]}")

        self.preview_text.setPlainText(
            "\n".join(preview_lines) if preview_lines else "No data selected"
        )

    def _generate_synthetic_patch_clamp_traces(
        self,
        trace_data: TraceData,
        indices: np.ndarray | None,
        noise_amplitude: float,
    ) -> tuple[np.ndarray, float] | None:
        """Select and downsample traces, then add Gaussian measurement noise.

        Args:
            trace_data: The trace data to filter.
            indices: Array of indices to extract, or None to export all traces.

            noise_amplitude: Standard deviation of Gaussian noise to add (in mV).

        Returns:
            The generated traces and sample interval, or None if validation fails.
        """
        sampling_rate = self.patch_sampling_rate_spinbox.value()
        if not np.isfinite(trace_data.dt) or trace_data.dt <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Sampling Rate",
                "The source sample interval must be positive.",
            )
            return None
        if sampling_rate > 1.0 / trace_data.dt and not np.isclose(
            sampling_rate * trace_data.dt, 1.0, rtol=1e-12, atol=0
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Sampling Rate",
                f"Output sampling rate must not exceed the source rate "
                f"({1.0 / trace_data.dt:g} Hz).",
            )
            return None
        ratio = Fraction(sampling_rate * trace_data.dt).limit_denominator(1_000_000)
        if ratio.numerator == 0:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Sampling Rate", "The requested sampling rate is too low."
            )
            return None
        output_dt = trace_data.dt * ratio.denominator / ratio.numerator
        # Check if trace data has non-consecutive indices
        if len(trace_data.indices) > 1:
            indices_diff = np.diff(trace_data.indices)
            if not np.all(indices_diff == 1):
                reply = QtWidgets.QMessageBox.warning(
                    self,
                    "Non-consecutive Indices",
                    f"The trace data has non-consecutive neuron indices: {trace_data.indices}\n\n"
                    "This may cause issues when interpreting the exported data as the current implementation expects consecutive indices. "
                    "Do you want to continue with the export?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.No:
                    return None

        # Filter trace data to only include selected indices
        if indices is not None:
            # Validate that all requested indices are within bounds
            max_idx = trace_data.traces.shape[0] - 1
            invalid_indices = indices[indices > max_idx]
            if len(invalid_indices) > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Indices",
                    f"Some indices are out of bounds. Maximum index is {max_idx}.\n"
                    f"Invalid indices: {invalid_indices}",
                )
                return None
            traces_to_export = trace_data.traces[indices]
        else:
            # Export all traces if no indices specified
            traces_to_export = trace_data.traces

        if ratio.numerator != ratio.denominator:
            traces_to_export = resample_poly(
                traces_to_export,
                ratio.numerator,
                ratio.denominator,
                axis=1,
                padtype="edge",
            )

        # Add measurement noise at the output rate, preserving its requested SD.
        noise_amplitude_with_units = noise_amplitude * b2.mV
        traces_to_export = traces_to_export + np.random.normal(
            0, noise_amplitude_with_units, traces_to_export.shape
        )

        return traces_to_export, output_dt

    def _export_patch_clamp_data(self) -> None:
        """Export selected trace data to a file."""
        trace_name = self.data_patch_clamp_combo.currentText()
        if not trace_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select trace data."
            )
            return

        trace_data = self.model.get_data(trace_name, TraceData)
        if not trace_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {trace_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        generated = self._generate_synthetic_patch_clamp_traces(
            trace_data, indices, self.noise_spinbox.value()
        )
        if generated is None:
            return
        traces_to_export, output_dt = generated

        # Open file dialog

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Patch Clamp Data",
            f"{trace_name}_patch_clamp.npy",
            "NumPy Array (*.npy);;CSV File (*.csv)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npy"):
                np.save(file_path, traces_to_export)
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, traces_to_export, delimiter=",")
            else:
                np.savetxt(file_path, traces_to_export)

            self.logger.info(f"Exported patch clamp data to {file_path}")
            QtWidgets.QMessageBox.information(
                self,
                "Export Successful",
                f"Patch clamp data exported to:\n{file_path}\n"
                f"Sampling rate: {1.0 / output_dt:g} Hz",
            )
        except Exception as e:
            self.logger.error(f"Failed to export patch clamp data: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export patch clamp data:\n{str(e)}"
            )

    def _add_patch_clamp_to_project(self) -> None:
        """Add generated synthetic trace data to the project as experimental data."""
        trace_name = self.data_patch_clamp_combo.currentText()
        if not trace_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select trace data."
            )
            return

        trace_data = self.model.get_data(trace_name, TraceData)
        if not trace_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find trace data: {trace_name}"
            )
            return

        # Parse indices from the index selector
        try:
            indices = self._parse_indices(self.index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Filter and add noise to traces
        generated = self._generate_synthetic_patch_clamp_traces(
            trace_data, indices, self.noise_spinbox.value()
        )
        if generated is None:
            return
        traces_to_add, output_dt = generated

        # Determine the indices for the new data
        new_indices = indices if indices is not None else trace_data.indices

        # Create a unique experiment name
        base_experiment_name = "Synthetic data"
        experiment_name = base_experiment_name
        counter = 1
        while experiment_name in self.model.experiments:
            experiment_name = f"{base_experiment_name} {counter}"
            counter += 1

        # Create new TraceData object with the synthetic data
        new_trace_data = TraceData(
            name="Patch clamp",
            traces=traces_to_add,
            indices=new_indices,
            dt=output_dt,
            units=trace_data.units,
        )

        # Add to model
        try:
            self.model.add_experimental_data(experiment_name, [new_trace_data])
            self.data_added = True  # Mark that data was added
            self.logger.info(
                f"Added synthetic trace data to experiment '{experiment_name}'"
            )
            QtWidgets.QMessageBox.information(
                self,
                "Data Added",
                f"Synthetic trace data added to experiment:\n{experiment_name}",
            )
        except Exception as e:
            self.logger.error(f"Failed to add trace data to project: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Add Failed",
                f"Failed to add trace data to project:\n{str(e)}",
            )

    def _export_imaging_data(self) -> None:
        """Export synthetic imaging data generated from spike data to a file."""
        spike_name = self.data_imaging_combo.currentText()
        if not spike_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select spike data."
            )
            return

        spike_data = self.model.get_data(spike_name, SpikeData)
        if not spike_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find spike data: {spike_name}"
            )
            return

        # Parse indices from the imaging index selector
        try:
            indices = self._parse_indices(self.imaging_index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Generate synthetic imaging traces
        imaging_traces, indices, dt_downsample = (
            self._generate_synthetic_imaging_traces(spike_data, indices)
        )

        # Open file dialog
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Imaging Data",
            "F.npy",  # TODO: better to creat a separate folder for imaging data
            "NumPy Array (*.npy);;CSV File (*.csv)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npy"):
                np.save(file_path, imaging_traces)
            elif file_path.endswith(".csv"):
                np.savetxt(file_path, imaging_traces, delimiter=",")
            else:
                np.savetxt(file_path, imaging_traces)

            self.logger.info(f"Exported imaging data to {file_path}")
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Imaging data exported to:\n{file_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to export imaging data: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export imaging data:\n{str(e)}"
            )

    def _add_imaging_to_project(self) -> None:
        """Add spike data to the project as experimental data."""
        spike_name = self.data_imaging_combo.currentText()
        if not spike_name:
            QtWidgets.QMessageBox.warning(
                self, "No Data Selected", "Please select spike data."
            )
            return

        spike_data = self.model.get_data(spike_name, SpikeData)
        if not spike_data:
            QtWidgets.QMessageBox.warning(
                self, "Data Not Found", f"Could not find spike data: {spike_name}"
            )
            return

        # Parse indices from the imaging index selector
        try:
            indices = self._parse_indices(self.imaging_index_lineedit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Indices", f"Invalid index format:\n{str(e)}"
            )
            return

        # Create a unique experiment name
        base_experiment_name = "Synthetic data"
        experiment_name = base_experiment_name
        counter = 1
        while experiment_name in self.model.experiments:
            experiment_name = f"{base_experiment_name} {counter}"
            counter += 1

        # Generate synthetic imaging traces
        imaging_traces, indices, dt_downsample = (
            self._generate_synthetic_imaging_traces(spike_data, indices)
        )

        # Create new TraceData object with the synthetic data
        imaging_data = TraceData(
            name="Imaging",
            traces=imaging_traces,
            indices=indices,
            dt=dt_downsample,
            units="1",
        )

        # Add to model
        try:
            self.model.add_experimental_data(experiment_name, [imaging_data])
            self.data_added = True  # Mark that data was added
            self.logger.info(
                f"Added synthetic spike data to experiment '{experiment_name}'"
            )
            QtWidgets.QMessageBox.information(
                self,
                "Data Added",
                f"Synthetic spike data added to experiment:\n{experiment_name}",
            )
        except Exception as e:
            self.logger.error(f"Failed to add spike data to project: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Add Failed",
                f"Failed to add spike data to project:\n{str(e)}",
            )
