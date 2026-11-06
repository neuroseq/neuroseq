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
from neuroseq.export.exporter_numpy import NeuronTimeMatrix
from neuroseq.export.exporter_numpy import save_matrix
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
        self.btn_export = buttons.addButton(
            "Export…", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
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
        n_neurons = int(self.project.max_neurons)
        if spikes.spike_indices.size:
            n_neurons = max(n_neurons, int(spikes.spike_indices.max()) + 1)
        return spike_matrix(
            spikes,
            self.spn_bin_ms.value() / 1000.0,
            self.project.data_duration_seconds(name),
            neuron_ids=np.arange(n_neurons),
        )

    def _update_shape(self, *_args: object) -> None:
        matrix = self.matrix()
        self.btn_export.setEnabled(matrix is not None)
        if matrix is None:
            self.lbl_shape.setText(f"No {self.kind} to export.")
            return
        neurons, points = matrix.values.shape
        step = matrix.times[1] - matrix.times[0] if points > 1 else 0.0
        self.lbl_shape.setText(
            f"{neurons} neurons × {points} time points"
            + (f" ({step * 1000.0:g} ms apart)" if step else "")
            + f", in {matrix.units or 'no units'}."
        )

    def _export(self) -> None:
        matrix = self.matrix()
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
        if not filename.lower().endswith((".npy", ".npz")):
            filename += ".npz" if chosen_filter == NPZ_FILTER else ".npy"
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
