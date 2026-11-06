"""File > Export > Traces/Spikes as NumPy array: pick a dataset and save it as a
(neurons, time points) array (see :mod:`neuroseq.export.exporter_numpy`)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from qtpy import QtWidgets
from qtpy.compat import getsavefilename

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.export.exporter_numpy import MAX_MATRIX_ELEMENTS
from neuroseq.export.exporter_numpy import NeuronTimeMatrix
from neuroseq.export.exporter_numpy import save_matrix
from neuroseq.export.exporter_numpy import spike_bin_count
from neuroseq.export.exporter_numpy import spike_matrix
from neuroseq.export.exporter_numpy import trace_matrix

if TYPE_CHECKING:
    from neuroseq.project import Project

#: Bin size offered for spikes when their data group has no trace to take it from.
DEFAULT_SPIKE_BIN_MS = 1.0
NPY_FILTER = "NumPy array (*.npy)"
NPZ_FILTER = "NumPy archive with neuron ids and times (*.npz)"


class NumpyExportDialog(QtWidgets.QDialog):
    """Choose traces or spikes and save them as a neurons x time points array."""

    def __init__(
        self,
        project: Project,
        kind: Literal["traces", "spikes"],
        current_name: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.kind = kind
        self.saved_path: str | None = None
        self.setWindowTitle(f"Export {kind} as NumPy array")
        data_type = TraceData if kind == "traces" else SpikeData
        form = QtWidgets.QFormLayout()
        self.cmb_data = QtWidgets.QComboBox(self)
        self.cmb_data.addItems(project.get_data_names(data_type))
        if current_name:
            self.cmb_data.setCurrentText(current_name)
        form.addRow("Data:", self.cmb_data)
        self.spn_bin_ms = QtWidgets.QDoubleSpinBox(self)
        self.spn_bin_ms.setRange(0.001, 60000.0)
        self.spn_bin_ms.setDecimals(3)
        self.spn_bin_ms.setSuffix(" ms")
        # Only once the value is entered: a partly typed "0.005" is not a bin size.
        self.spn_bin_ms.setKeyboardTracking(False)
        self.spn_bin_ms.setToolTip(
            "Width of each column: the spikes of a neuron in that time are counted. "
            "Defaults to the frame period of a trace in the same data group."
        )
        if kind == "spikes":
            form.addRow("Bin size:", self.spn_bin_ms)
        else:
            self.spn_bin_ms.hide()
        self.lbl_shape = QtWidgets.QLabel(self)
        self.lbl_shape.setWordWrap(True)
        form.addRow("Array:", self.lbl_shape)
        note = QtWidgets.QLabel(
            "Rows are neurons (by id), columns are time points. A .npy file holds "
            "only the array; a .npz archive also holds 'neuron_ids', 'times' (s) and "
            "'units', with the array as 'values'.",
            self,
        )
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_export = buttons.addButton(
            "Export…", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        assert btn_export is not None
        self.btn_export: QtWidgets.QPushButton = btn_export
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.cmb_data.currentTextChanged.connect(self._on_data_changed)
        self.spn_bin_ms.valueChanged.connect(self._update_shape)
        self._on_data_changed(self.cmb_data.currentText())

    def _on_data_changed(self, name: str) -> None:
        if self.kind == "spikes":
            self.spn_bin_ms.blockSignals(True)
            self.spn_bin_ms.setValue(self._default_bin_ms(name))
            self.spn_bin_ms.blockSignals(False)
        self._update_shape()

    def _default_bin_ms(self, name: str) -> float:
        """The frame period of a trace in the spikes' data group, else 1 ms."""
        group = name.split(" :: ", 1)[0]
        experiment = self.project.experiments.get(group)
        for datum in experiment.data if experiment is not None else ():
            if isinstance(datum, TraceData) and datum.dt > 0:
                return float(datum.dt) * 1000.0
        return DEFAULT_SPIKE_BIN_MS

    def _spike_neuron_ids(self, spikes: SpikeData) -> np.ndarray:
        """Rows of a spike matrix: every neuron of the project (at least up to the
        highest spiking one), so row ``i`` is neuron ``i``."""
        n_neurons = int(self.project.max_neurons)
        if spikes.spike_indices.size:
            n_neurons = max(n_neurons, int(spikes.spike_indices.max()) + 1)
        return np.arange(n_neurons)

    def matrix(self) -> NeuronTimeMatrix | None:
        """The selected data as a neurons x time points matrix (None if missing)."""
        name = self.cmb_data.currentText()
        if not name:
            return None
        if self.kind == "traces":
            trace = self.project.get_full_resolution_trace(name)
            return trace_matrix(trace) if trace is not None else None
        spikes = self.project.get_data(name, SpikeData)
        if spikes is None:
            return None
        return spike_matrix(
            spikes,
            self.spn_bin_ms.value() / 1000.0,
            self.project.data_duration_seconds(name),
            neuron_ids=self._spike_neuron_ids(spikes),
        )

    def _shape(self) -> tuple[int, int, float, str] | None:
        """(neurons, time points, seconds between them, units) of :meth:`matrix`,
        worked out without building it; None if there is no data."""
        name = self.cmb_data.currentText()
        if not name:
            return None
        if self.kind == "traces":
            trace = self.project.get_full_resolution_trace(name)
            if trace is None:
                return None
            neurons, points = np.shape(trace.traces)[:2]
            return int(neurons), int(points), float(trace.dt), trace.units
        spikes = self.project.get_data(name, SpikeData)
        if spikes is None:
            return None
        bin_s = self.spn_bin_ms.value() / 1000.0
        points = spike_bin_count(
            spikes, bin_s, self.project.data_duration_seconds(name)
        )
        return len(self._spike_neuron_ids(spikes)), points, bin_s, "spikes"

    def _update_shape(self, *_args: object) -> None:
        shape = self._shape()
        if shape is None:
            self.btn_export.setEnabled(False)
            self.lbl_shape.setText(f"No {self.kind} to export.")
            return
        neurons, points, step, units = shape
        text = (
            f"{neurons} neurons × {points} time points"
            + (f" ({step * 1000.0:g} ms apart)" if step and points > 1 else "")
            + f", in {units or 'no units'}."
        )
        too_large = neurons * points > MAX_MATRIX_ELEMENTS
        if too_large:
            text += " Too large to export: use a larger bin size."
        self.btn_export.setEnabled(not too_large)
        self.lbl_shape.setText(text)

    def _export(self) -> None:
        try:
            matrix = self.matrix()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot export", str(exc))
            return
        if matrix is None:
            return
        name = self.cmb_data.currentText()
        suggested = re.sub(r"[^\w.-]+", "_", name).strip("_") or self.kind
        filename, chosen_filter = getsavefilename(
            self,
            f"Export {self.kind} as NumPy array",
            f"{suggested}.npy",
            f"{NPY_FILTER};;{NPZ_FILTER}",
        )
        if not filename:
            return
        filename = _with_suffix_for_filter(filename, chosen_filter)
        try:
            save_matrix(matrix, filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not write {filename}:\n{exc}"
            )
            return
        self.saved_path = filename
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {matrix.values.shape[0]} × {matrix.values.shape[1]} array of "
            f"'{name}' to:\n{filename}",
        )
        self.accept()


def _with_suffix_for_filter(filename: str, chosen_filter: str) -> str:
    """``filename`` with the suffix of the chosen file type. The suggested name ends
    in ``.npy``, so choosing the archive type turns a ``.npy`` name into ``.npz``;
    a typed ``.npz`` name is kept whichever type is chosen."""
    lower = filename.lower()
    if chosen_filter == NPZ_FILTER:
        if lower.endswith(".npy"):
            return filename[:-4] + ".npz"
        return filename if lower.endswith(".npz") else filename + ".npz"
    return filename if lower.endswith((".npy", ".npz")) else filename + ".npy"
